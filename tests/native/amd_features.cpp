// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "../../sycl/amd-features.h"

#include <cassert>
#include <initializer_list>
#include <string_view>

int main() {
  using mom::amd::has_gfx12_int8_wmma;
  for (const std::string_view architecture : {
           "gfx1200", "gfx1201", "gfx1200:xnack-", "gfx1201:sramecc-:xnack+"})
    assert(has_gfx12_int8_wmma(architecture));
  for (const std::string_view architecture : {
           "", "unknown", "gfx803", "gfx900", "gfx906", "gfx908", "gfx90a", "gfx942",
           "gfx1010", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151",
           "gfx12", "gfx120", "gfx1202", "gfx12000", "gfx1200junk", "gfx1300",
           "gfx1100:xnack-", ":gfx1200", " gfx1200", "gfx1200 "})
    assert(!has_gfx12_int8_wmma(architecture));
  static_assert(has_gfx12_int8_wmma("gfx1200"));
  static_assert(!has_gfx12_int8_wmma("gfx1100"));
}
