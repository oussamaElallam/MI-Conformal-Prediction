/**
 * ESP32-S3 MI Detection — ON-DEVICE BENCHMARK FIRMWARE (Reviewer #6)
 * ------------------------------------------------------------------
 * Drop-in replacement for esp32_deployment/src/main.cpp.
 * Produces the MEASURED numbers needed for Table 7:
 *   (1) Inference latency  : mean / std / min / median / max over NUM_INFERENCES
 *   (2) Preprocessing time : normalization + int8 quantization, measured separately
 *   (3) Tensor-arena RAM   : interpreter->arena_used_bytes()  <-- the real working set
 *   (4) DRAM high-water    : how much internal SRAM the whole program actually used
 *   (5) Model flash size   : g_model_data_len
 *   (6) Active power        : POWER_BENCH=1 -> tight continuous loop; read W on a USB-C meter
 *
 * Keeps your existing model_data.h and cp_params.h (regenerate with:
 *     cd esp32_deployment && python setup_deployment.py ).
 * lib_deps / platformio.ini are unchanged (tanakamasayuki/TensorFlowLite_ESP32).
 *
 * IMPORTANT — settles the paper's "fits in internal SRAM, no external PSRAM" claim:
 *   USE_PSRAM_ARENA defaults to 0, so the arena is allocated in INTERNAL SRAM.
 *   - If "AllocateTensors OK" prints, the model genuinely fits without PSRAM:
 *     report arena_used_bytes() as the RAM figure and keep the claim.
 *   - If allocation/AllocateTensors FAILS, set USE_PSRAM_ARENA 1, raise ARENA_KB,
 *     and CORRECT the paper to state the arena lives in PSRAM. Do not keep an
 *     unverified "no PSRAM" claim.
 */

#include <Arduino.h>
#undef DEFAULT
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"
#include <math.h>

#include "model_data.h"   // g_model_data[], g_model_data_len
#include "cp_params.h"     // CP_TAU_NORM, CP_TAU_MI, CP_LEAD_MEAN[12], CP_LEAD_STD[12], CP_EPSILON

// ---------------- Benchmark configuration ----------------
#define SAMPLES            1000
#define LEADS              12
#define N_INPUTS           (SAMPLES * LEADS)

#define NUM_WARMUP         50      // discarded warmup inferences
#define NUM_INFERENCES     1000    // timed inferences (reduce to ~200 if a run is too slow)

#define USE_PSRAM_ARENA    0       // 0 = internal SRAM (tests the "no PSRAM" claim); 1 = PSRAM
#define ARENA_KB           300     // arena size to TRY (shrink toward arena_used_bytes after measuring)

#define POWER_BENCH        0       // 1 = after benchmark, loop inference forever (no prints) for a USB-C power meter
// ---------------------------------------------------------

namespace {
  const tflite::Model*       g_model       = nullptr;
  tflite::MicroInterpreter*  g_interpreter = nullptr;
  TfLiteTensor*              g_input       = nullptr;
  TfLiteTensor*              g_output      = nullptr;
  uint8_t*                   g_tensor_arena = nullptr;
  float g_in_scale = 1.0f;  int g_in_zp  = 0;
  float g_out_scale = 1.0f; int g_out_zp = 0;
}

static float* ecg_raw  = nullptr;   // raw demo signal (mV)
static float* ecg_norm = nullptr;   // normalized signal
static uint32_t* lat_us = nullptr;  // per-inference latency samples (us)

static size_t g_dram_free_start = 0;

// Fill a deterministic demo ECG (raw mV). Replace with real samples if available.
static void make_demo_raw(float* dst) {
  for (int t = 0; t < SAMPLES; t++) {
    float phase = 2.0f * PI * (float)t / 50.0f;   // ~2 Hz @ 100 Hz fs
    for (int l = 0; l < LEADS; l++) {
      dst[t * LEADS + l] = 0.1f * sinf(phase + 0.1f * l);
    }
  }
}

// Preprocessing step 1: z-score normalization (raw -> normalized float)
static inline void normalize(const float* raw, float* norm) {
  for (int t = 0; t < SAMPLES; t++)
    for (int l = 0; l < LEADS; l++) {
      int i = t * LEADS + l;
      norm[i] = (raw[i] - CP_LEAD_MEAN[l]) / CP_LEAD_STD[l];
    }
}

// Preprocessing step 2: quantize normalized float -> int8 input tensor
static inline void quantize_into_input(const float* norm) {
  int8_t* qin = g_input->data.int8;
  for (int i = 0; i < N_INPUTS; i++) {
    float q = roundf(norm[i] / g_in_scale) + (float)g_in_zp;
    if (q < -128.0f) q = -128.0f;
    if (q >  127.0f) q =  127.0f;
    qin[i] = (int8_t)q;
  }
}

static void summarize(const char* label, uint32_t* x, int n) {
  // mean / std (population) / min / max
  double sum = 0, sumsq = 0; uint32_t mn = 0xFFFFFFFF, mx = 0;
  for (int i = 0; i < n; i++) { sum += x[i]; sumsq += (double)x[i] * x[i]; if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; }
  double mean = sum / n;
  double var  = (sumsq / n) - (mean * mean); if (var < 0) var = 0;
  double sd   = sqrt(var);
  // median via insertion sort on a copy (n is small)
  static uint32_t tmp[NUM_INFERENCES];
  for (int i = 0; i < n; i++) tmp[i] = x[i];
  for (int i = 1; i < n; i++) { uint32_t k = tmp[i]; int j = i - 1; while (j >= 0 && tmp[j] > k) { tmp[j+1] = tmp[j]; j--; } tmp[j+1] = k; }
  double med = (n % 2) ? tmp[n/2] : 0.5 * (tmp[n/2 - 1] + tmp[n/2]);
  Serial.printf("%s  mean=%.3f ms  std=%.3f ms  min=%.3f  median=%.3f  max=%.3f  (n=%d)\n",
                label, mean/1000.0, sd/1000.0, mn/1000.0, med/1000.0, mx/1000.0, n);
}

void setup() {
  Serial.begin(115200);
  delay(2500);
  Serial.println("\n===== ESP32-S3 MI / Conformal — BENCHMARK BUILD =====");
  Serial.printf("CPU freq: %d MHz\n", getCpuFrequencyMhz());

  g_dram_free_start = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  Serial.printf("Internal SRAM free (start): %u B\n", (unsigned)g_dram_free_start);
  Serial.printf("PSRAM free (start):         %u B\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Serial.printf("Model flash footprint:      %u B (%.1f KB)\n", g_model_data_len, g_model_data_len / 1024.0);

  // Buffers (these are working memory, not the arena)
  ecg_raw  = (float*)heap_caps_malloc(N_INPUTS * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  ecg_norm = (float*)heap_caps_malloc(N_INPUTS * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  lat_us   = (uint32_t*)heap_caps_malloc(NUM_INFERENCES * sizeof(uint32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!ecg_raw || !ecg_norm || !lat_us) { Serial.println("ERROR: buffer alloc failed"); while (1) delay(1000); }

  // ---- Tensor arena ----
#if USE_PSRAM_ARENA
  g_tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, (size_t)ARENA_KB * 1024, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  const char* arena_loc = "PSRAM";
#else
  g_tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, (size_t)ARENA_KB * 1024, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  const char* arena_loc = "INTERNAL SRAM";
#endif
  if (!g_tensor_arena) {
    Serial.printf("ERROR: %d KB arena alloc in %s FAILED.\n", ARENA_KB, arena_loc);
    Serial.println("-> If you targeted internal SRAM, set USE_PSRAM_ARENA 1 (and the paper must say PSRAM).");
    while (1) delay(1000);
  }
  Serial.printf("Arena: requested %d KB in %s\n", ARENA_KB, arena_loc);

  g_model = tflite::GetModel(g_model_data);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroErrorReporter err;
  static tflite::MicroInterpreter interp(g_model, resolver, g_tensor_arena, (size_t)ARENA_KB * 1024, &err, nullptr);
  g_interpreter = &interp;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors FAILED — increase ARENA_KB (or switch to PSRAM).");
    while (1) delay(1000);
  }
  Serial.println("AllocateTensors OK");
  Serial.printf(">>> arena_used_bytes = %u B (%.1f KB)  <-- REPORT THIS as tensor-arena RAM\n",
                (unsigned)g_interpreter->arena_used_bytes(), g_interpreter->arena_used_bytes() / 1024.0);

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  g_in_scale  = g_input->params.scale;  g_in_zp  = g_input->params.zero_point;
  g_out_scale = g_output->params.scale; g_out_zp = g_output->params.zero_point;
  Serial.printf("Input dims: [%d,%d,%d]  in_q(scale=%.6f, zp=%d)\n",
                g_input->dims->data[0], g_input->dims->data[1], g_input->dims->data[2], g_in_scale, g_in_zp);
  Serial.printf("Output q(scale=%.6f, zp=%d)\n", g_out_scale, g_out_zp);

  // ---- Prepare one input ----
  make_demo_raw(ecg_raw);
  normalize(ecg_raw, ecg_norm);
  quantize_into_input(ecg_norm);

  // ---- Warmup ----
  for (int i = 0; i < NUM_WARMUP; i++) g_interpreter->Invoke();

  // ---- (1) Inference-only latency ----
  Serial.printf("\nTiming %d inferences (inference only)...\n", NUM_INFERENCES);
  for (int i = 0; i < NUM_INFERENCES; i++) {
    uint32_t t0 = micros();
    g_interpreter->Invoke();
    lat_us[i] = micros() - t0;
  }
  summarize("[inference only]      ", lat_us, NUM_INFERENCES);

  // ---- (2) Preprocessing (normalize + quantize) ----
  Serial.printf("Timing %d preprocessing passes (normalize + int8 quantize)...\n", NUM_INFERENCES);
  for (int i = 0; i < NUM_INFERENCES; i++) {
    uint32_t t0 = micros();
    normalize(ecg_raw, ecg_norm);
    quantize_into_input(ecg_norm);
    lat_us[i] = micros() - t0;
  }
  summarize("[preprocessing]       ", lat_us, NUM_INFERENCES);

  // ---- (3) End-to-end (preprocess + inference) ----
  Serial.printf("Timing %d end-to-end passes (preprocess + inference)...\n", NUM_INFERENCES);
  for (int i = 0; i < NUM_INFERENCES; i++) {
    uint32_t t0 = micros();
    normalize(ecg_raw, ecg_norm);
    quantize_into_input(ecg_norm);
    g_interpreter->Invoke();
    lat_us[i] = micros() - t0;
  }
  summarize("[preprocess+inference]", lat_us, NUM_INFERENCES);

  // ---- RAM high-water ----
  size_t dram_min = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
  Serial.println("\n----- MEMORY SUMMARY (for Table 7) -----");
  Serial.printf("Model flash footprint : %u B (%.1f KB)\n", g_model_data_len, g_model_data_len / 1024.0);
  Serial.printf("Tensor arena used     : %u B (%.1f KB)\n", (unsigned)g_interpreter->arena_used_bytes(), g_interpreter->arena_used_bytes() / 1024.0);
  Serial.printf("Internal SRAM low-water: %u B free (lowest seen)\n", (unsigned)dram_min);
  Serial.printf("Peak internal SRAM used: %u B (%.1f KB)  [= start_free - low_water]\n",
                (unsigned)(g_dram_free_start - dram_min), (g_dram_free_start - dram_min) / 1024.0);
  Serial.printf("Arena located in       : %s\n", arena_loc);
  Serial.println("----------------------------------------");

  // ---- One labeled CP demo so the prediction path is exercised ----
  g_interpreter->Invoke();
  {
    int8_t q = g_output->data.int8[0];
    float p_mi = g_out_scale * ((int)q - g_out_zp); if (p_mi < 0) p_mi = 0; if (p_mi > 1) p_mi = 1;
    bool in_norm = (p_mi <= CP_TAU_NORM);
    bool in_mi   = ((1.0f - p_mi) <= CP_TAU_MI);
    Serial.printf("CP demo: P(MI)=%.4f  set={%s%s%s}  (eps=%.2f)\n",
                  p_mi, in_norm ? "Normal" : "", (in_norm && in_mi) ? "," : "", in_mi ? "MI" : "", CP_EPSILON);
  }

#if POWER_BENCH
  Serial.println("\n>>> POWER MODE: continuous inference, no prints. Read steady-state W on the USB-C meter now.");
  Serial.flush();
  for (;;) { g_interpreter->Invoke(); }
#else
  Serial.println("\nBenchmark complete. (Set POWER_BENCH 1 to measure active power with a USB-C meter.)");
#endif
}

void loop() {
  delay(60000);
}
