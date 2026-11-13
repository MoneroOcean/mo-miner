// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace mom::cuda_images {

struct CudaImageRow {
  const char* profile;
  int arch;
  bool cubin;
  bool compatibility;
  const char* resource;
};

enum class Kind { cubin, ptx };

struct Blob {
  const char* data = nullptr;
  std::size_t size = 0;
  Kind kind = Kind::ptx;
  int arch = 0;
};

struct Candidates {
  Blob primary;
  Blob compatibility;
};

// Pure selection policy: exact cubin, then newest primary PTX, then newest compatibility PTX.
// The second candidate is the newest distinct compatibility resource at or below the device SM.
template <typename Lookup>
Candidates select(std::string_view profile, int sm, const CudaImageRow* rows, std::size_t count,
                  Lookup&& lookup) {
  Candidates result;
  if (!rows || sm < 50 || sm > 999)
    return result;

  const CudaImageRow* exact_cubin = nullptr;
  const CudaImageRow* primary_ptx = nullptr;
  const CudaImageRow* compatibility_ptx = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    const CudaImageRow& row = rows[i];
    if (!row.profile || !row.resource || profile != row.profile || row.arch < 50 ||
        row.arch > sm || row.arch > 999)
      continue;
    if (row.cubin) {
      if (!row.compatibility && row.arch == sm && !exact_cubin)
        exact_cubin = &row;
    } else if (row.compatibility) {
      if (!compatibility_ptx || row.arch > compatibility_ptx->arch)
        compatibility_ptx = &row;
    } else if (!primary_ptx || row.arch > primary_ptx->arch) {
      primary_ptx = &row;
    }
  }

  const CudaImageRow* first = exact_cubin ? exact_cubin : primary_ptx ? primary_ptx : compatibility_ptx;
  if (first) {
    const std::string_view bytes = lookup(first->resource);
    if (!bytes.empty())
      result.primary = {bytes.data(), bytes.size(), first->cubin ? Kind::cubin : Kind::ptx,
                        first->arch};
  }

  if (profile != "pearl-compact") {
    const CudaImageRow* second = nullptr;
    for (std::size_t i = 0; i < count; ++i) {
      const CudaImageRow& row = rows[i];
      if (!row.profile || !row.resource || profile != row.profile || row.cubin ||
          !row.compatibility || row.arch < 50 || row.arch > sm || row.arch > 999 ||
          (first && std::string_view(row.resource) == first->resource))
        continue;
      if (!second || row.arch > second->arch)
        second = &row;
    }
    if (second) {
      const std::string_view bytes = lookup(second->resource);
      if (!bytes.empty())
        result.compatibility = {bytes.data(), bytes.size(), Kind::ptx, second->arch};
    }
  }

  // Compact Pearl's feature-specific image has no safe PTX or older-architecture fallback.
  if (profile == "pearl-compact" &&
      (sm != 120 || !result.primary.data || result.primary.kind != Kind::cubin ||
       result.primary.arch != 120))
    return {};
  return result;
}

// Isolated compiler profiles share one application-owned table. The C bridge passes only
// pointer/length inputs and a fixed-layout output, never C++ standard-library objects.
static_assert(std::is_standard_layout_v<Candidates> && std::is_trivially_copyable_v<Candidates>);
static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8 && sizeof(int) == 4);
static_assert(sizeof(Blob) == 24 && offsetof(Blob, size) == 8 &&
              offsetof(Blob, kind) == 16 && offsetof(Blob, arch) == 20 && sizeof(Candidates) == 48);
extern "C" void mom_cuda_images_select_embedded(const char* profile, std::size_t length, int sm,
                                                 Candidates* output) noexcept;

inline Candidates select_embedded(std::string_view profile, int sm) noexcept {
  Candidates result;
  mom_cuda_images_select_embedded(profile.data(), profile.size(), sm, &result);
  return result;
}

} // namespace mom::cuda_images
