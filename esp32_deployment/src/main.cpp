/**
 * ESP32-S3 MI Detection with Conformal Prediction
 * 
 * This code runs a lightweight CNN model with Mondrian split-conformal
 * prediction on ESP32-S3 for real-time MI detection from 12-lead ECG.
 *
 * Uses TensorFlow Lite Micro (atomic14/tensorflow-lite-esp32) with a PSRAM-backed
 * tensor arena. Input/output quantization params are read from tensors.
 */

#include <Arduino.h>
#undef DEFAULT
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model and parameters
#include "model_data.h"
#include "cp_params.h"

// Inference settings
#define SAMPLES 1000
#define LEADS   12
// Use PSRAM-backed tensor arena (adjust if AllocateTensors fails)
#define TENSOR_ARENA_SIZE (1024 * 1024)  // 1 MB tensor arena in PSRAM
#define N_INPUTS (SAMPLES * LEADS)
#define N_OUTPUTS 1

// TFLM globals
namespace {
  const tflite::Model* g_model = nullptr;
  tflite::MicroInterpreter* g_interpreter = nullptr;
  TfLiteTensor* g_input = nullptr;
  TfLiteTensor* g_output = nullptr;
  uint8_t* g_tensor_arena = nullptr;  // allocated in PSRAM
  // Quantization params
  float g_in_scale = 1.0f; int g_in_zp = 0;
  float g_out_scale = 1.0f; int g_out_zp = 0;
}

// Single buffer for normalized ECG (stored as float)
float* ecg_normalized = nullptr;


// No separate preprocess needed when generating normalized demo directly


void generate_demo_signal_normalized(float* ecg_norm) {
  for (int t = 0; t < SAMPLES; t++) {
    // Simple sinusoid at ~2 Hz (100 samples/sec, period = 50 samples)
    float phase = 2.0f * PI * (float)t / 50.0f;
    float amplitude = 0.1f;  // generate directly in mV-scale
    for (int lead = 0; lead < LEADS; lead++) {
      float lead_phase = phase + (float)lead * 0.1f;
      float x_raw = amplitude * sinf(lead_phase);
      // Normalize inline
      int idx = t * LEADS + lead;
      ecg_norm[idx] = (x_raw - CP_LEAD_MEAN[lead]) / CP_LEAD_STD[lead];
    }
  }
}


void run_inference_and_cp(float* ecg_norm) {
  // Quantize normalized input into INT8 tensor
  int8_t* qin = g_input->data.int8;
  for (int i = 0; i < N_INPUTS; i++) {
    float q = roundf(ecg_norm[i] / g_in_scale) + (float)g_in_zp;
    if (q < -128.0f) q = -128.0f;
    if (q > 127.0f) q = 127.0f;
    qin[i] = (int8_t)q;
  }

  // Run inference
  unsigned long t_start = micros();
  TfLiteStatus st = g_interpreter->Invoke();
  unsigned long t_end = micros();
  if (st != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return;
  }
  
  float latency_ms = (t_end - t_start) / 1000.0f;
  
  // Dequantize output to probability
  int8_t q_out = g_output->data.int8[0];
  float p_mi = g_out_scale * ((int)q_out - g_out_zp);
  p_mi = constrain(p_mi, 0.0f, 1.0f);
  
  // Mondrian conformal prediction
  float s_norm = p_mi;        // Score for Normal hypothesis
  float s_mi = 1.0f - p_mi;   // Score for MI hypothesis
  
  bool in_normal = (s_norm <= CP_TAU_NORM);
  bool in_mi = (s_mi <= CP_TAU_MI);
  
  // Print results
  Serial.println("\n========== INFERENCE RESULTS ==========");
  Serial.printf("Inference latency: %.2f ms\n", latency_ms);
  Serial.printf("P(MI) = %.4f\n", p_mi);
  Serial.println("\nConformal Prediction:");
  Serial.printf("  s_norm = %.4f <= tau_norm = %.4f ? %s\n", 
                s_norm, CP_TAU_NORM, in_normal ? "YES" : "NO");
  Serial.printf("  s_mi   = %.4f <= tau_mi   = %.4f ? %s\n", 
                s_mi, CP_TAU_MI, in_mi ? "YES" : "NO");
  
  Serial.print("Prediction Set: {");
  if (in_normal) Serial.print("Normal");
  if (in_normal && in_mi) Serial.print(", ");
  if (in_mi) Serial.print("MI");
  Serial.println("}");
  
  // Interpretation
  if (in_normal && !in_mi) {
    Serial.println("→ LIKELY NORMAL (high confidence)");
  } else if (!in_normal && in_mi) {
    Serial.println("→ LIKELY MI (high confidence)");
  } else if (in_normal && in_mi) {
    Serial.println("→ UNCERTAIN (both classes possible)");
  } else {
    Serial.println("→ OUT OF DISTRIBUTION (neither class fits)");
  }
  Serial.println("========================================\n");
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // Wait for serial to initialize
  
  Serial.println("\n\n========================================");
  Serial.println("ESP32-S3 MI Detection with Conformal Prediction");
  Serial.println("========================================\n");
  
  // Report memory before initialization
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  
  // Allocate buffers (prefer PSRAM if available, else fall back to DRAM heap)
  size_t buf_bytes = N_INPUTS * sizeof(float);
  if (ESP.getFreePsram() > 0) {
    Serial.println("\nAllocating ECG buffers in PSRAM...");
    ecg_normalized = (float*)ps_malloc(buf_bytes);
    if (ecg_normalized) {
      Serial.printf("✓ Allocated %d bytes for ECG buffer in PSRAM\n", (int)(buf_bytes));
    }
  }
  if (ecg_normalized == nullptr) {
    Serial.println("\nPSRAM not available or allocation failed. Falling back to DRAM heap...");
    ecg_normalized = (float*)malloc(buf_bytes);
  }
  if (ecg_normalized == nullptr) {
    Serial.println("ERROR: Failed to allocate ECG buffers in DRAM heap!");
    Serial.printf("Requested per-buffer: %d bytes\n", (int)buf_bytes);
    while (1) delay(1000);
  }
  Serial.printf("✓ Buffer ready. Allocated: %d bytes\n", (int)(buf_bytes));
  
  // Initialize TensorFlow Lite Micro
  Serial.println("\nInitializing TensorFlow Lite Micro model...");
  g_model = tflite::GetModel(g_model_data);
#ifdef TFLITE_SCHEMA_VERSION
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR: Model schema %d != supported %d\n", (int)g_model->version(), (int)TFLITE_SCHEMA_VERSION);
    while (1) delay(1000);
  }
#else
  Serial.printf("Model schema version: %d (no compile-time reference)\n", (int)g_model->version());
#endif

  // Allocate PSRAM tensor arena aligned to 16 bytes
  g_tensor_arena = (uint8_t*) heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!g_tensor_arena) {
    Serial.println("PSRAM aligned alloc failed, falling back to ps_malloc...");
    g_tensor_arena = (uint8_t*) ps_malloc(TENSOR_ARENA_SIZE);
  }
  if (!g_tensor_arena) {
    Serial.println("ERROR: Failed to allocate tensor arena in PSRAM");
    while (1) delay(1000);
  }
  Serial.printf("✓ Allocated tensor arena: %d KB (PSRAM)\n", TENSOR_ARENA_SIZE / 1024);

  // Use AllOpsResolver with Espressif TFLM port
  static tflite::AllOpsResolver resolver;
  static tflite::MicroErrorReporter error_reporter;
  static tflite::MicroInterpreter static_interpreter(
    g_model, resolver, g_tensor_arena, TENSOR_ARENA_SIZE, &error_reporter, nullptr);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    Serial.println("Try increasing TENSOR_ARENA_SIZE.");
    while (1) delay(1000);
  }
  Serial.println("✓ Tensors allocated successfully");

  // Get IO tensors
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  Serial.printf("✓ Input dims: %d dims\n", g_input->dims->size);
  if (g_input->dims->size >= 3)
    Serial.printf("  shape: [%d, %d, %d]\n", g_input->dims->data[0], g_input->dims->data[1], g_input->dims->data[2]);
  Serial.printf("✓ Output dims: %d\n", g_output->dims->data[0]);

  // Read quantization params
  g_in_scale = g_input->params.scale; 
  g_in_zp    = g_input->params.zero_point;
  g_out_scale= g_output->params.scale; 
  g_out_zp   = g_output->params.zero_point;
  Serial.printf("Input q: scale=%.6f zp=%d\n", g_in_scale, g_in_zp);
  Serial.printf("Output q: scale=%.6f zp=%d\n", g_out_scale, g_out_zp);
  
  // Report conformal prediction parameters
  Serial.println("\nConformal Prediction Parameters:");
  Serial.printf("  Epsilon (ε): %.2f (%.0f%% coverage target)\n", 
                CP_EPSILON, (1.0f - CP_EPSILON) * 100.0f);
  Serial.printf("  τ_normal: %.4f\n", CP_TAU_NORM);
  Serial.printf("  τ_MI:     %.4f\n", CP_TAU_MI);
  
  Serial.printf("\nFree heap after init: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM after init: %d bytes\n", ESP.getFreePsram());
  
  Serial.println("\n========================================");
  Serial.println("Setup complete! Running demo inference...\n");
  delay(1000);
}

void loop() {
  // Generate demo ECG signal
  // TODO: Replace with real ECG data from your source
  generate_demo_signal_normalized(ecg_normalized);
  
  Serial.println("Running inference on demo ECG signal...");
  run_inference_and_cp(ecg_normalized);
  
  // Wait before next inference
  delay(5000);  // Run every 5 seconds
  
  // TODO: For continuous monitoring, implement sliding window:
  // 1. Acquire new ECG samples (e.g., 100 samples = 1 second)
  // 2. Shift window by 100 samples
  // 3. Run inference on new window
  // 4. Repeat
}
