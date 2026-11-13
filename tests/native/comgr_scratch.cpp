// Test the exact native scratch-acquisition block with deterministic filesystem outcomes.
// Real compiler/container tests independently check actual permissions and cleanup.
#include <cassert>
#include <cstdlib>
#include <string>
#include <system_error>

constexpr int AMD_COMGR_STATUS_SUCCESS = 0, AMD_COMGR_STATUS_ERROR = 1;
static unsigned calls = 0, collisions = 0;
static bool denied = false;
namespace fs {
constexpr int owner_all = 0700;
void createUniquePath(const char*, std::string& output, bool absolute) {
  assert(absolute); output = "/private/comgr-" + std::to_string(calls);
}
std::error_code create_directory(const std::string&, bool ignore_existing, int permissions) {
  assert(!ignore_existing && permissions == owner_all); ++calls;
  if (calls <= collisions) return std::make_error_code(std::errc::file_exists);
  if (denied) return std::make_error_code(std::errc::permission_denied);
  return {};
}
}
struct Scratch {
  std::string TmpDir = "must-not-survive-a-failed-acquisition";
  int acquire() {
#include "comgr-scratch.inc"
    return AMD_COMGR_STATUS_SUCCESS;
  }
};

int main(int argc, char** argv) {
  assert(argc == 3);
  collisions = static_cast<unsigned>(std::atoi(argv[1])); denied = std::atoi(argv[2]) != 0;
  Scratch scratch;
  const bool success = collisions < 16 && !denied;
  assert(scratch.acquire() == (success ? AMD_COMGR_STATUS_SUCCESS : AMD_COMGR_STATUS_ERROR));
  assert(scratch.TmpDir.empty() != success);
  assert(calls == (collisions >= 16 ? 16 : collisions + 1));
}
