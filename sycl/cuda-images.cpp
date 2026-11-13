// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "cuda-images.h"

#include "cuda-images.inc"

namespace mom::cuda_images {

// Validate the narrow bridge before constructing a local C++ view. All returned bytes belong
// to this immutable application table and remain valid across compiler-profile lifetimes.
extern "C" void mom_cuda_images_select_embedded(const char* profile, const std::size_t length,
                                                 const int sm, Candidates* output) noexcept {
  if (!output)
    return;
  *output = {};
  if (!profile || !length || length > 128)
    return;
  const auto lookup = [](const std::string_view name) { return ::embedded_resource(name); };
  *output = select(std::string_view(profile, length), sm, ::kCudaImageRows,
                   sizeof(::kCudaImageRows) / sizeof(::kCudaImageRows[0]), lookup);
}

} // namespace mom::cuda_images
