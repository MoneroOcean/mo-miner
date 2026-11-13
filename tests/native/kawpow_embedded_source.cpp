// Execute the actual source-loader body against generated resources, never a SYCL runtime.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include "resources.inc"
static unsigned mode, reads;
static std::string_view embedded_resource(std::string_view name) {
  ++reads;
  if ((mode == 1 && name == "kawpow-device.inc") || (mode == 2 && name == "kawpow-keccak.inc")) return {};
  if (mode == 3 && name == "kawpow-device.inc") return "invalid source without include marker";
  return fixture_resource(name);
}
#include "source.inc"

int main(int argc, char** argv) try {
  if (argc != 2) return 2;
  mode = std::strtoul(argv[1], nullptr, 10);
  if (mode > 3) return 2;
  if (mode) {
    bool rejected = false;
    try {(void)kawpow_device_src();} catch (const std::string&) {rejected = true;}
    if (!rejected) throw std::runtime_error("Malformed/missing embedded source was accepted");
    mode = 0; // C++ retries a failed static initializer; a bad attempt must not poison its cache.
  }
  const auto& source = kawpow_device_src();
  auto expected = std::string(fixture_resource("kawpow-device.inc"));
  const auto marker = expected.find("#include \"keccak.inc\"");
  if (marker == std::string::npos) throw std::runtime_error("Fixture lost the original marker");
  expected.replace(marker, sizeof("#include \"keccak.inc\"") - 1, fixture_resource("kawpow-keccak.inc"));
  if (source != expected) throw std::runtime_error("Embedded source differs from maintained fragments");
  const auto before = reads;
  if (&kawpow_device_src() != &source || reads != before) throw std::runtime_error("Source cache was reread or replaced");
  std::puts("Original KawPow fragments assembled byte-for-byte, cached and retryable without disk input");
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
} catch (const std::string& error) {
  std::fprintf(stderr, "%s\n", error.c_str()); return 1;
}
