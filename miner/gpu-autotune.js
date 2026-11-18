"use strict";

// Optional first-run empirical tuner. The C++ side supplies safe device-derived defaults; this
// layer only compares a small neighborhood and persists a materially faster device string.
/**
 * @param {{
 *   h: {
 *     formatHashrate(rate: number): string,
 *     log(message: string): void,
 *     log_err(message: string): void,
 *     repeat(callback: (next: () => void) => unknown): void,
 *   },
 *   opt: MinerOptions,
 *   gpuTuning: typeof import("../gpu-tuning"),
 *   benchAlgo(algo: string, callback: (rate: number) => void, dev: string, samples: number): void,
 * }} dependencies
 */
module.exports = function({h, opt, gpuTuning, benchAlgo}) {
  const MIN_SWITCH_GAIN = 1.02; // do not save benchmark noise as hardware-specific configuration

  /** @param {string} algo @param {DeviceEntry} entry @param {(entry: DeviceEntry) => void} done */
  function tuneEntry(algo, entry, done) {
    const candidates = gpuTuning.autotuneCandidates(algo, entry);
    const baseline = candidates[0];
    if (!baseline || candidates.length < 2) {return done(entry);}
    let index = 0;
    let fastest = baseline;
    let baselineRate = 0;
    let bestRate = 0;
    h.log(`Empirically tuning ${algo} on ${entry.device} (${candidates.length} candidates)...`);
    h.repeat(function(/** @type {() => void} */ next) {
      const candidate = candidates[index];
      if (!candidate) {
        const selected = bestRate >= baselineRate * MIN_SWITCH_GAIN ? fastest : baseline;
        const selectedRate = selected === fastest ? bestRate : baselineRate;
        h.log(`Selected ${algo} tuning ${gpuTuning.formatDeviceEntry(selected)} (${h.formatHashrate(selectedRate)})`);
        return done(selected);
      }
      const isBaseline = index++ === 0;
      const dev = gpuTuning.formatDeviceEntry(candidate);
      benchAlgo(algo, function(/** @type {number} */ rate) {
        const measuredRate = typeof rate === "number" && Number.isFinite(rate) && rate >= 0
          ? rate : 0;
        if (measuredRate !== rate) {
          h.log_err(`Ignoring invalid ${algo} tuning rate for ${dev}`);
        }
        if (isBaseline) {baselineRate = measuredRate;}
        if (measuredRate > bestRate) {
          fastest = candidate;
          bestRate = measuredRate;
        }
        setImmediate(next);
      }, dev, 2);
    });
  }

  /** @param {string} algo @param {() => void} done */
  function tuneAlgo(algo, done) {
    const algoParam = opt.algo_params[algo];
    if (!algoParam) {
      h.log_err(`Skipping unknown ${algo} GPU tuning`);
      return done();
    }
    let entries;
    try {
      entries = gpuTuning.parseDeviceList(algoParam.dev, algo);
    } catch (error) {
      h.log_err(`Skipping ${algo} GPU tuning: ${error instanceof Error ? error.message : String(error)}`);
      return done();
    }
    const gpuEntries = entries.map((entry, index) => ({entry, index}))
      .filter(({entry}) => entry.device.startsWith("gpu"));
    let nextGpu = 0;
    h.repeat(function(/** @type {() => void} */ next) {
      const current = gpuEntries[nextGpu++];
      if (!current) {
        algoParam.dev = gpuTuning.formatDeviceList(entries);
        return done();
      }
      tuneEntry(algo, current.entry, function(best) {
        entries[current.index] = best;
        next();
      });
    });
  }

  return {tuneAlgo};
};
