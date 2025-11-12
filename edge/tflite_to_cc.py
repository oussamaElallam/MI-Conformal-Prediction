import sys
from pathlib import Path

template = """
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>
#if defined(__has_attribute)
#if __has_attribute(aligned)
#define DATA_ALIGN(x) __attribute__((aligned(x)))
#else
#define DATA_ALIGN(x)
#endif
#else
#define DATA_ALIGN(x)
#endif

DATA_ALIGN(16) const unsigned char g_model_data[] = {
%s
};
const unsigned int g_model_data_len = sizeof(g_model_data);

#endif // MODEL_DATA_H
"""


def to_hex_array(data: bytes, cols: int = 12) -> str:
    lines = []
    for i in range(0, len(data), cols):
        chunk = data[i:i+cols]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {line}")
    return ",\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python tflite_to_cc.py <model.tflite> <out_header.h>")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    data = in_path.read_bytes()
    out_path.write_text(template % to_hex_array(data))
    print(f"Wrote {out_path}")

if __name__ == '__main__':
    main()
