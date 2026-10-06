// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <string_view>

namespace mom::amd {

// These int8 WMMA kernels require RDNA4's wave32 operand layout, not just any matrix hardware.
// HIP may append feature flags; never infer support for an unknown or future gfx target.
constexpr bool has_gfx12_int8_wmma(std::string_view architecture) {
  architecture = architecture.substr(0, architecture.find(':'));
  return architecture == "gfx1200" || architecture == "gfx1201";
}

} // namespace mom::amd
