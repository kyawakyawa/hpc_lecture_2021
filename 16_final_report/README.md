# Final Report

## final_report_cpu.cpp

- AVX 対応の cpu を使っている場合、`-DUSE_SIMD -march=native` で SIMD 化出来ます
- MPI の並列数で、N が割り切れる必要があります

## final_report_gpu.cu

- N は 2 のべき定数のみに対応しています
