// Copyright GNU GPLv3 (c) 2026 MoneroOcean
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include "hip-shape.inc"

int main() {
  for (const auto& shape : {std::initializer_list<int>{256, 128, 2048, 128},
       {131072, 131072, 2048, 128}, {524288, 131072, 2048, 128},
       {262144, 262144, 2048, 128}, {131072, 131072, 4096, 128},
       {131072, 131072, 4096, 256}, {131072, 131072, 8192, 128},
       {128, 4194304, 2048, 128}, {524288, 524288, 2048, 128},
       {524288, 262144, 2048, 128}}) {
    const int* p = shape.begin();
    if (!supports_shape(p[0], p[1], p[2], p[3])) return 1;
  }
  for (const auto& shape : {std::initializer_list<int>{0, 128, 2048, 128},
       {256, -128, 2048, 128}, {64, 64, 2048, 128},
       {256, 128, 2048, 0}, {256, 128, 2048, 16}, {256, 128, 4096, 0},
       {256, 128, 3072, 48}}) {
    const int* p = shape.begin();
    if (supports_shape(p[0], p[1], p[2], p[3])) return 2;
  }
  std::puts("PearlHash HIP shape admission supports bounded row slices");
}
