// Compile the actual tiny device-selection methods without SYCL/CUDA libraries or hardware.
#include <cassert>
#include <cstdio>
#include <stdexcept>
using CUdevice = int;
namespace sycl {
enum class backend { cuda, ext_oneapi_cuda };
struct device { int ordinal; bool fail = false; };
template <backend Kind> int get_native(const device &selected) {
  (void)Kind;
  if (selected.fail) {throw std::runtime_error("interop failed");}
  return selected.ordinal;
}
}
#include "selection.inc"

int main() {
  for (int ordinal : {5, 11}) {
    assert(CnSelection::native_device(sycl::device{ordinal}) == ordinal);
    assert(PearlSelection::native_device(sycl::device{ordinal}) == ordinal);
  }
  unsigned failures = 0;
  try {CnSelection::native_device(sycl::device{7, true});} catch (const std::runtime_error &) {++failures;}
  try {PearlSelection::native_device(sycl::device{7, true});} catch (const std::runtime_error &) {++failures;}
  assert(failures == 2);
  std::puts("Selected native CUDA ordinals and interop errors preserved; GPU calls=0");
}
