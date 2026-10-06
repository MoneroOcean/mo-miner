#include "../../sycl/verthash/data_file.hpp"

#include <array>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>

using mom_verthash::data_file::Node;

static void check(const bool condition) {
  if (!condition)
    throw std::runtime_error("Verthash data test failed");
}

static Node bytes(const char* hex) {
  Node result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>(std::stoul(std::string(hex + i * 2, 2), nullptr, 16));
  return result;
}

int main(const int argc, char** argv) {
  using namespace mom_verthash::data_file;
  if (argc == 2) {
    const std::filesystem::path path = argv[1];
    generate(path);
    return has_expected_hash(path) ? 0 : 1;
  }
  if (argc != 1)
    return 2;

  check(sha256(nullptr, 0) ==
         bytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  static constexpr std::array<std::uint8_t, 3> abc = {'a', 'b', 'c'};
  check(sha256(abc.data(), abc.size()) ==
         bytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  static constexpr std::array<std::uint8_t, 32> seed = {
      'V','e','r','t','h','a','s','h',' ','P','r','o','o','f','-','o','f','-','S','p','a','c','e',' ',
      'D','a','t','a','f','i','l','e'};
  check(sha3_256(seed.data(), seed.size()) ==
         bytes("78841f08de415556b429b0084e2e97a99b49f56f151e7be9ccc5874261c22786"));

  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("mom-verthash-data-" + std::to_string(std::random_device{}()));
  const std::filesystem::path path = directory / "verthash.dat";
  try {
    generate(path, 2);
    check(std::filesystem::file_size(path) == graph_nodes(2) * Node{}.size());
    check(std::distance(std::filesystem::directory_iterator(directory),
                        std::filesystem::directory_iterator{}) == 1);

    const std::filesystem::path invalid = directory / "invalid.dat";
    std::ofstream(invalid, std::ios::binary).put('\0');
    bool rejected = false;
    try {
      generate(invalid, 2);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check(rejected);
    check(std::filesystem::file_size(invalid) == 1);

    const std::filesystem::path concurrent = directory / "concurrent.dat";
    std::array<std::exception_ptr, 2> failures;
    std::thread first([&] {
      try {
        generate(concurrent, 2);
      } catch (...) {
        failures[0] = std::current_exception();
      }
    });
    std::thread second([&] {
      try {
        generate(concurrent, 2);
      } catch (...) {
        failures[1] = std::current_exception();
      }
    });
    first.join();
    second.join();
    for (const std::exception_ptr& failure : failures) {
      if (failure)
        std::rethrow_exception(failure);
    }
    check(std::filesystem::file_size(concurrent) == graph_nodes(2) * Node{}.size());
    for (const auto& entry : std::filesystem::directory_iterator(directory))
      check(entry.path().filename().string().find("concurrent.dat.tmp.") != 0);

    rejected = false;
    try {
      generate(directory / "unsupported.dat", 0);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check(rejected);
  } catch (...) {
    std::filesystem::remove_all(directory);
    throw;
  }
  std::filesystem::remove_all(directory);
}
