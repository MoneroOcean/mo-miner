// Exercise the real patched AdaptiveCpp storage implementation, without a GPU.
#include "hipSYCL/common/filesystem.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" bool mom_acpp_set_appdb_cache_tag(const unsigned char *) noexcept;

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  require(argc == 3, "expected mode and isolated storage root");
  const std::string mode = argv[1], root = argv[2];
  require(mode == "untagged" || mode == "a" || mode == "b", "unknown mode");
#ifdef _WIN32
  require(_putenv_s("ACPP_APPDB_DIR", root.c_str()) == 0, "cannot isolate storage");
#else
  require(setenv("ACPP_APPDB_DIR", root.c_str(), 1) == 0, "cannot isolate storage");
#endif
  std::array<unsigned char, 32> tag{};
  tag.fill(mode == "b" ? 0xbb : 0xaa);
  require(!mom_acpp_set_appdb_cache_tag(nullptr), "null tag accepted");
  if (mode != "untagged") {
    require(mom_acpp_set_appdb_cache_tag(tag.data()), "initial tag rejected");
    require(mom_acpp_set_appdb_cache_tag(tag.data()), "identical early tag rejected");
    tag[0] ^= 1;
    require(!mom_acpp_set_appdb_cache_tag(tag.data()), "conflicting tag accepted");
  }
  auto &storage = hipsycl::common::filesystem::persistent_storage::get();
  require(storage.get_base_dir() == root, "explicit appdb root changed");
  require(!mom_acpp_set_appdb_cache_tag(tag.data()), "late tag accepted");
  require(storage.generate_app_dir("") == root + "/apps/global", "global directory changed");
  const std::filesystem::path directory = storage.get_this_app_dir();
  if (mode == "untagged") {
    require(directory.parent_path() == std::filesystem::path(root) / "apps", "untagged layout changed");
  } else {
    const std::string suffix = "jit-" + std::string(64, mode == "b" ? 'b' : 'a');
    require(directory.filename() == suffix, "identity not copied into directory");
    require(directory.parent_path().parent_path() == std::filesystem::path(root) / "apps", "tag escaped app root");
  }
  require(std::filesystem::is_directory(storage.get_jit_cache_dir()), "cache not created");
  std::cout << directory.string() << '\n';
}
