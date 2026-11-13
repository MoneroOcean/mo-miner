"use strict";

// All topology/timing data comes from libuv rather than approximating CPU count from affinity.
module.exports = {cpus: () => globalThis.__momCpus()};
