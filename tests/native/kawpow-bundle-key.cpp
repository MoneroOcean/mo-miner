// A KawPow period bundle bakes the DAG modulus. The portable compact-DAG test and a full-DAG job
// can share one period and epoch, so the bundle cache must rebuild when only the DAG size changes.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

enum class SealMode { KAWPOW };
struct FastModData {
  uint32_t reciprocal, increment, shift, divisor;
};
struct MomExecutableKernelBundle {
  uint32_t dag_items;
};

struct State {
  std::unique_ptr<MomExecutableKernelBundle> period_bundle;
  uint64_t period_bundle_period = UINT64_MAX;
  uint32_t period_bundle_epoch = UINT32_MAX;
  uint32_t period_bundle_dag_items = 0;
  SealMode period_bundle_seal = SealMode::KAWPOW;
  uint32_t period_bundle_magic[15] = {};
  uint32_t period_bundle_regs = 32, period_bundle_cnt_cache = 11, period_bundle_cnt_math = 18;
  std::unique_ptr<MomExecutableKernelBundle> next_bundle;
  uint64_t next_bundle_period = UINT64_MAX;
  uint32_t next_bundle_epoch = UINT32_MAX;
  uint32_t next_bundle_dag_items = 0;
  SealMode next_bundle_seal = SealMode::KAWPOW;
  uint32_t next_bundle_magic[15] = {};
  uint32_t next_bundle_regs = 32, next_bundle_cnt_cache = 11, next_bundle_cnt_math = 18;
  std::thread prefetch_thread;
  SealMode seal = SealMode::KAWPOW;
  uint32_t magic[15] = {};
  uint32_t regs = 32, cnt_cache = 11, cnt_math = 18;
  unsigned builds = 0;

  std::unique_ptr<MomExecutableKernelBundle> build_period_bundle(
      uint64_t, const FastModData dag_mod, SealMode, const uint32_t*, uint32_t, uint32_t,
      uint32_t) {
    ++builds;
    return std::make_unique<MomExecutableKernelBundle>(MomExecutableKernelBundle{dag_mod.divisor});
  }

  // Production bundle_key_eq and ensure_period_bundle from sycl/kawpow/state.inc.
#include "kawpow_bundle_key.inc"
};

int main() {
  constexpr FastModData compact{0, 0, 0, 320};
  constexpr FastModData full{0, 0, 0, 16777213};
  State state;
  state.ensure_period_bundle(7, 1, compact, false);
  state.ensure_period_bundle(7, 1, compact, false);
  assert(state.builds == 1);
  state.ensure_period_bundle(7, 1, full, false);
  assert(state.builds == 2 && state.period_bundle->dag_items == full.divisor);
  std::printf("KawPow bundle key: a DAG size change rebuilds the period bundle\n");
}
