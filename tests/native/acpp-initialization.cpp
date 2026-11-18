// CPU-only lifecycle fixture: compile the exact pre-main statements from our SDK patch.
// The settings singleton deliberately snapshots visibility, as AdaptiveCpp's real one does.
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace common::settings {
template<class T> bool try_retrieve_settings_variable(const std::string &name, T &out) {
  const char *value = std::getenv(("ACPP_" + name).c_str());
  if (!value) value = std::getenv(("HIPSYCL_" + name).c_str());
  if (!value || !*value) value = std::getenv(("MOM_TEST_CONFIG_" + name).c_str());
  if (!value) return false;
  T parsed;
  std::istringstream input{value};
  if (!(input >> parsed)) return false;
  out = parsed;
  return true;
}
}
namespace rt {
enum class setting {debug_level, hcf_dump_directory, persistent_runtime};
template<setting> struct setting_trait;
template<> struct setting_trait<setting::debug_level> {static constexpr auto str = "DEBUG_LEVEL";};
template<> struct setting_trait<setting::hcf_dump_directory> {static constexpr auto str = "HCF_DUMP_DIRECTORY";};
template<> struct setting_trait<setting::persistent_runtime> {static constexpr auto str = "PERSISTENT_RUNTIME";};
struct settings {
  std::string visibility, dump;
  int debug = 2;
  bool persistent = false;
  settings() {
    common::settings::try_retrieve_settings_variable("VISIBILITY_MASK", visibility);
    common::settings::try_retrieve_settings_variable("DEBUG_LEVEL", debug);
    common::settings::try_retrieve_settings_variable("HCF_DUMP_DIRECTORY", dump);
    common::settings::try_retrieve_settings_variable("PERSISTENT_RUNTIME", persistent);
  }
  template<setting S> auto get() const {
    if constexpr (S == setting::debug_level) return debug;
    else if constexpr (S == setting::persistent_runtime) return persistent;
    else return dump;
  }
};
namespace application {
int runtime_initializations = 0;
settings &get_settings() {static settings value; return value;}
int get_runtime_pointer() {get_settings(); return ++runtime_initializations;}
}
}
namespace common {
struct output_stream {
  int _debug_level = 2;
  output_stream() {
#include "debug-setting.inc"
  }
};
}
using namespace rt;
std::string register_hcf() {
#include "hcf-setting.inc"
  return hcf_dump_dir;
}
// SYCL's persistent-runtime object and HCF diagnostics all run before the controller.
struct persistent_runtime {
  int _rt = 0;
  persistent_runtime() {
#include "persistent-setting.inc"
      _rt = rt::application::get_runtime_pointer();
    }
  }
} persistence;
common::output_stream diagnostics;
const auto dump_path = register_hcf();
int main() {
  if (const char *expected = std::getenv("MOM_TEST_ASSERT_PERSISTENCE")) {
    if (rt::application::runtime_initializations != std::stoi(expected)) return 2;
  }
  if (setenv("ACPP_VISIBILITY_MASK", "hip", 1) != 0) return 1;
  std::cout << application::get_settings().visibility << '|' << diagnostics._debug_level << '|' << dump_path << '\n';
}
