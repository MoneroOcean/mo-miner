"use strict";

// Correctness tests operate exclusively on committed mining fixtures. Keep the pool guard in
// NODE_OPTIONS so miner/test subprocesses inherit it as well. Ordinary network access remains
// available to tooling; pool.js rejects only real mining-pool socket creation.
const guardOption = `--require=${__filename}`;
if (!(process.env.NODE_OPTIONS || "").split(/\s+/).includes(guardOption)) {
  process.env.NODE_OPTIONS = [process.env.NODE_OPTIONS, guardOption].filter(Boolean).join(" ");
}
process.env.MOM_TEST_NO_POOL_NETWORK = "1";
