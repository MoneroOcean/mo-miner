// Host-only source-derived test for the Verthash bounded-data admission boundary.
// The compiler-policy test extracts the production state_for() definition into the include below;
// this file supplies only a lightweight device, State, and real DeviceStateRegistry harness.

#include "../../sycl/device-state.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace sycl {
namespace info::device {
struct host_unified_memory {};
} // namespace info::device

struct device {
  bool cpu = false;
  bool integrated = false;

  bool is_cpu() const {
    return cpu;
  }

  bool is_gpu() const {
    return !cpu;
  }

  template <typename Info>
  bool get_info() const {
    (void)sizeof(Info);
    return integrated;
  }
};
} // namespace sycl

static bool fixture_cpu = false;
static bool fixture_integrated = false;
static unsigned fixture_device_queries = 0;
static unsigned fixture_states_constructed = 0;

struct State {
  std::string device_name;
  bool compact_test_data;

  State(const std::string& name, const bool compact)
      : device_name(name), compact_test_data(compact) {
    ++fixture_states_constructed;
  }
};

sycl::device get_dev(const std::string&) {
  ++fixture_device_queries;
  return sycl::device{fixture_cpu, fixture_integrated};
}

#include "verthash_state_for.inc"

static void require(const bool condition, const char* const reason) {
  if (!condition)
    throw std::runtime_error(reason);
}

static void configure_env(const char* const value) {
#ifdef _WIN32
  require(_putenv_s("MOM_SYCL_PORTABLE_TEST", value ? value : "") == 0,
          "failed to configure Verthash test environment");
#else
  if (value)
    require(setenv("MOM_SYCL_PORTABLE_TEST", value, 1) == 0,
            "failed to configure Verthash test environment");
  else
    require(unsetenv("MOM_SYCL_PORTABLE_TEST") == 0,
            "failed to clear Verthash test environment");
#endif
}

// MOM_SYCL_PORTABLE_TEST uses the shared on/off rule: empty, "0", "false" and "off" stay off.
static bool requested(const char* const env) {
  return env && *env && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
         std::strcmp(env, "off") != 0;
}

static bool expected_compact(const bool cpu, const bool integrated, const bool is_test,
                             const bool is_benchmark, const char* const env) {
  return (cpu || integrated) && is_test && !is_benchmark && requested(env);
}

static void test_admission_matrix() {
  constexpr const char* environments[] = {nullptr, "0", "false", "1", "2", "true"};
  for (const char* const env : environments)
    for (const bool cpu : {false, true})
      for (const bool integrated : {false, true})
        for (const bool is_test : {false, true})
          for (const bool is_benchmark : {false, true}) {
            fixture_cpu = cpu;
            fixture_integrated = integrated;
            configure_env(env);
            persistentDeviceStates<State>().clear();
            const unsigned before_queries = fixture_device_queries;
            State& state = state_for("matrix", is_test, is_benchmark);
            require(state.compact_test_data ==
                        expected_compact(cpu, integrated, is_test, is_benchmark, env),
                    "bounded Verthash admission mismatch");
            require(fixture_device_queries == before_queries +
                        ((is_test && !is_benchmark && requested(env)) ? 1u : 0u),
                    "unexpected device query for inactive Verthash test switch");
          }
}

static void test_state_keys() {
  fixture_cpu = true;
  fixture_integrated = false;
  configure_env(nullptr);
  persistentDeviceStates<State>().clear();
  State* const normal = &state_for("normal", false, false);
  require(normal == &state_for("normal", false, false), "normal Verthash state key is not reused");
  require(normal == &state_for("normal", true, false),
          "official test state should reuse the normal Verthash state");
  require(!normal->compact_test_data, "official Verthash state became compact without switch");

  configure_env("1");
  persistentDeviceStates<State>().clear();
  State* const compact = &state_for("transition", true, false);
  require(compact->compact_test_data, "CPU test did not select compact Verthash state");
  require(compact == &state_for("transition", true, false),
          "compact Verthash state key is not reused");
  State* const mining = &state_for("transition", false, false);
  require(mining != compact, "compact Verthash state leaked into mining");
  require(!mining->compact_test_data, "mining reused compact Verthash state");
  require(mining == &state_for("transition", false, false),
          "mining Verthash state key is not reused");
  require(compact == &state_for("transition", true, false),
          "compact Verthash state was not retained across mining admission");

  fixture_cpu = false;
  fixture_integrated = false;
  persistentDeviceStates<State>().clear();
  State* const gpu_test = &state_for("gpu", true, false);
  require(!gpu_test->compact_test_data, "GPU inherited test switch selected compact data");
  require(gpu_test == &state_for("gpu", false, false),
          "GPU inherited test switch created a second official state");

  fixture_integrated = true;
  persistentDeviceStates<State>().clear();
  State* const integrated_compact = &state_for("integrated", true, false);
  require(integrated_compact->compact_test_data,
          "integrated GPU test did not select compact Verthash state");
  require(integrated_compact == &state_for("integrated", true, false),
          "integrated compact Verthash state key is not reused");
  State* const integrated_full = &state_for("integrated", true, true);
  require(integrated_full != integrated_compact,
          "integrated benchmark reused compact Verthash state");
  require(!integrated_full->compact_test_data,
          "integrated benchmark selected compact Verthash data");
}

int main() {
  try {
    test_admission_matrix();
    test_state_keys();
    require(fixture_states_constructed != 0, "Verthash route fixture constructed no states");
    configure_env(nullptr);
    persistentDeviceStates<State>().clear();
    std::puts("verthash-test-route:passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "verthash-test-route: %s\n", error.what());
    return 1;
  }
}
