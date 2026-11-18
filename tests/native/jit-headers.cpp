// Actual memory-only decoder test. No CUDA SDK, GPU, compiler subprocess or filesystem resolver.
#include "../../native/jit-headers.h"
#include <iostream>
#include <string>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<mom::JitHeaders>);
static_assert(!std::is_move_constructible_v<mom::JitHeaders>);
int main(int argc, char **argv) {
  if (argc != 3) {return 2;}
  try {
    const std::string hex(argv[1]); std::string compressed;
    if (hex.size() % 2) {return 2;}
    for (size_t i = 0; i < hex.size(); i += 2) {compressed.push_back(static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));}
    const mom::JitHeaders headers(compressed, std::stoull(argv[2]));
    if (!headers.find("absent").empty()) {return 3;}
    std::cout << headers.count() << '\n';
    for (int i = 0; i < headers.count(); ++i) {
      const auto value = headers.find(headers.names()[i]);
      if (value.data() != headers.sources()[i]) {return 3;}
      std::cout << headers.names()[i] << ' ';
      constexpr char digits[] = "0123456789abcdef";
      for (const unsigned char c : value) {std::cout << digits[c >> 4] << digits[c & 15];}
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception &error) {std::cerr << error.what() << '\n'; return 1;}
}
