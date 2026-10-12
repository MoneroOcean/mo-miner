// Compile against the actual selector extracted from state.inc, without loading a GPU runtime.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "autolykos_pipeline_selector.inc"

int main() {
  using Selector = AutolykosPipelineSelection;
  assert(Selector::eligible(true, true, false, false, false));
  assert(!Selector::eligible(false, true, false, false, false));
  assert(!Selector::eligible(true, false, false, false, false));
  assert(!Selector::eligible(true, true, true, false, false));
  assert(!Selector::eligible(true, true, false, true, false));
  assert(!Selector::eligible(true, true, false, false, true));
  const auto select = [](Selector& state) {return state.select(70464240, 65536, 64, true, true);};
  const bool order[] = {true, false, true, false, false, true};
  for (const double mono : {100.0, 97.0, 95.3, 100.0 / 1.05, 90.0}) {
    Selector state;
    for (unsigned i = 0; i < 6; ++i) {
      assert(select(state) == order[i]);
      assert(select(state) == order[i]); // Selection alone (or an exception) never advances.
      state.observe(i < 2 ? 1000000 : order[i] ? 100 : mono);
    }
    assert(state.phase == 6);
    assert(select(state) == !(mono <= 100.0 / 1.05));
    state.observe(1); assert(state.phase == 6);
    assert(state.select(70464240, 65536, 64, false, true));
    assert(!state.select(70464240, 65536, 64, false, false));
    assert(state.phase == 6); // Explicit/backend bypass leaves the learned state unchanged.
    assert(state.select(70464241, 65536, 64, true, true)); assert(state.phase == 0);
    state.observe(1);
    assert(state.select(70464241, 65537, 64, true, true)); assert(state.phase == 0);
    state.observe(1);
    assert(state.select(70464241, 65537, 128, true, true)); assert(state.phase == 0);
  }
  Selector state;
  for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    assert(select(state)); state.observe(invalid); assert(state.phase == 0);
  }
  state.observe(1); state.observe(1); state.observe(100);
  for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    state.observe(invalid); assert(state.phase == 3 && state.split_us == 100 && state.mono_us == 0);
  }
  state = {};
  for (unsigned i = 0; i < 5; ++i) {
    select(state); state.observe(i == 2 ? std::numeric_limits<double>::max() : 1);
  }
  assert(state.phase == 5); state.observe(std::numeric_limits<double>::max());
  assert(state.phase == 5); // Finite individual samples may still overflow the accumulated total.
  state.observe(1); assert(state.phase == 6);
  return 0;
}
