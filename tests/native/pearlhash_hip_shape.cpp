// Copyright GNU GPLv3 (c) 2026 MoneroOcean
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "hip-shape.inc"

int main() {
  for (const auto& shape : {std::initializer_list<int>{256, 128, 2048, 128},
       {131072, 131072, 2048, 128}, {524288, 131072, 2048, 128},
       {262144, 262144, 2048, 128}, {131072, 131072, 4096, 256}}) {
    const int* p = shape.begin();
    if (!supports_shape(p[0], p[1], p[2], p[3])) return 1;
  }
  for (const auto& shape : {std::initializer_list<int>{524288, 524288, 2048, 128},
       {524288, 262144, 2048, 128}, {0, 128, 2048, 128},
       {256, -128, 2048, 128}, {64, 64, 2048, 128},
       {256, 128, 2048, 0}, {256, 128, 4096, 128}}) {
    const int* p = shape.begin();
    if (supports_shape(p[0], p[1], p[2], p[3])) return 2;
  }
  std::puts("PearlHash HIP transcript indexing stays within 32-bit word offsets");
}
