"use strict";

// Optional first-run empirical tuner. The C++ side supplies safe device-derived defaults; this
// layer only compares a small neighborhood and persists a materially faster device string.
module.exports = function({h, opt, gpuTuning, benchAlgo}) {
  const MIN_SWITCH_GAIN = 1.02; // do not save benchmark noise as hardware-specific configuration

  function tuneEntry(algo, entry, done) {
    const candidates = gpuTuning.autotuneCandidates(algo, entry);
    if (candidates.length < 2) {return done(entry);}
    let index = 0;
    let fastest = candidates[0];
    let baselineRate = 0;
    let bestRate = 0;
    h.log(`Empirically tuning ${algo} on ${entry.device} (${candidates.length} candidates)...`);
    h.repeat(function(next) {
      if (index >= candidates.length) {
        const selected = bestRate >= baselineRate * MIN_SWITCH_GAIN ? fastest : candidates[0];
        const selectedRate = selected === fastest ? bestRate : baselineRate;
        h.log(`Selected ${algo} tuning ${gpuTuning.formatDeviceEntry(selected)} (${h.formatHashrate(selectedRate)})`);
        return done(selected);
      }
      const isBaseline = index === 0;
      const candidate = candidates[index++];
      const dev = gpuTuning.formatDeviceEntry(candidate);
      benchAlgo(algo, function(rate) {
        if (isBaseline) {baselineRate = rate;}
        if (rate > bestRate) {
          fastest = candidate;
          bestRate = rate;
        }
        setImmediate(next);
      }, dev, 2);
    });
  }

  function tuneAlgo(algo, done) {
    let entries;
    try {
      entries = gpuTuning.parseDeviceList(opt.algo_params[algo].dev, algo);
    } catch (error) {
      h.log_err(`Skipping ${algo} GPU tuning: ${error.message}`);
      return done();
    }
    const gpuIndexes = entries.map((entry, index) =>
      entry.device.startsWith("gpu") ? index : -1).filter((index) => index >= 0);
    let nextGpu = 0;
    h.repeat(function(next) {
      if (nextGpu >= gpuIndexes.length) {
        opt.algo_params[algo].dev = gpuTuning.formatDeviceList(entries);
        return done();
      }
      const entryIndex = gpuIndexes[nextGpu++];
      tuneEntry(algo, entries[entryIndex], function(best) {
        entries[entryIndex] = best;
        next();
      });
    });
  }

  return {tuneAlgo};
};
