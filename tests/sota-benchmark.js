"use strict";

const crypto = require("node:crypto");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const miners = require("../scripts/sota-miners");
const {
  benchmarkRate,
  benchmarkSpec,
  buildContainerScript,
  buildDockerArgs,
  comparatorAlgos,
  comparatorsFor,
  deviceArgs,
  downloadArchive,
  exitAccepted,
  lastRate,
  qualificationForVendor,
  qualificationResult,
} = require("../scripts/sota-benchmark");

/** @typedef {{momRate: number, unit: string, minRatio: number}} QualificationInput */
/** @typedef {Parameters<typeof buildContainerScript>[0]} TestMiner */
/** @typedef {NonNullable<ReturnType<typeof benchmarkSpec>>} TestBenchmarkSpec */
const testMiners = miners;

/** @typedef {{epoch?: string, devices?: string, seconds?: string}} TestBenchmarkOptions */
/** @param {TestMiner} miner @param {string} algo @param {string} vendor
 * @param {TestBenchmarkOptions} [options] @returns {string} */
function buildTestContainerScript(miner, algo, vendor, options) {
  return buildContainerScript(miner, algo, vendor, options);
}

/** @param {TestMiner} miner @param {string} algo @returns {TestBenchmarkSpec} */
function requiredBenchmarkSpec(miner, algo) {
  const spec = benchmarkSpec(miner, algo);
  if (!spec) {throw new Error(`${miner.name} has no benchmark for ${algo}`);}
  return spec;
}

/** @param {QualificationInput | null | undefined} qualification
 * @param {{value: number, unit: string}} rate @returns {{ratio: number, pass: boolean}} */
function requiredQualificationResult(qualification, rate) {
  const result = qualificationResult(qualification, rate);
  if (!result) {throw new Error("Expected a positive qualification result");}
  return result;
}

/** @param {RegExpMatchArray | null} matches @returns {RegExpMatchArray} */
function requiredMatch(matches) {
  if (!matches) {throw new Error("Expected the benchmark output to contain the pattern");}
  return matches;
}

test("SOTA comparator metadata is pinned and pool-only where no offline rate exists", () => {
  /** @type {Record<string, [string, string[]]>} */
  const expected = {
    xelishashv3: ["rigel", ["rigel", "--algorithm", "xelishashv3", "--url", "stratum+tcp://POOL:PORT",
      "--username", "WALLET.WORKER", "--password", "x"]],
    verthash: ["teamblackminer", ["TBMiner", "--algo", "verthash", "--hostname", "POOL", "--port", "PORT",
      "--wallet", "WALLET", "--worker-name", "WORKER", "--verthash-data", "/data/verthash.dat"]],
    walahash: ["srbminer", ["SRBMiner-MULTI", "--algorithm", "walahash", "--pool", "POOL:PORT",
      "--wallet", "WALLET.WORKER", "--password", "x"]],
    hoohash: ["hoominer", ["hoo_gpu", "-o", "stratum+tcp://mainnet-node-1.hoosat.fi:5555",
      "-u", "hoosat:README_DONATION_ADDRESS", "-p", "x", "--same-stratum", "--no-share-hashrate",
      "--gpu-id", "0"]],
    equihash192_7: ["miniz", ["miniZ", "--url=stratum+tcp://WALLET.WORKER@POOL:PORT",
      "--par=192,7", "--pers=ZcashPoW"]],
    zhash: ["miniz", ["miniZ", "--url=stratum+tcp://WALLET.WORKER@POOL:PORT", "--par=144,5"]],
  };
  for (const [algo, [id, command]] of Object.entries(expected)) {
    const candidate = comparatorsFor(algo).find((entry) => entry.id === id);
    if (!candidate || !candidate.poolOnly || typeof candidate.miner.url !== "string" ||
        typeof candidate.miner.sha256 !== "string") {
      throw new Error(`${id} should be metadata-only for ${algo}`);
    }
    assert.equal(candidate.spec, undefined);
    assert.match(candidate.miner.url, /^https:\/\/(github\.com\/[^/]+\/[^/]+\/releases\/download\/|htn\.foztor\.net\/)/);
    assert.match(candidate.miner.sha256, /^[0-9a-f]{64}$/);
    assert.deepEqual(candidate.poolOnly.command, command);
  }
  const hooCandidate = comparatorsFor("hoohash").find(({id}) => id === "hoominer");
  if (!hooCandidate?.poolOnly) {throw new Error("Hoominer should be metadata-only for hoohash");}
  assert.equal(hooCandidate.poolOnly.network, "bridge");
  assert.deepEqual(comparatorAlgos(), ["c29", "c30", "equihash192_7", "hoohash", "nexapow",
    "octopus", "verthash", "walahash", "xelishashv3", "zhash"]);
});

test("SRBMiner metadata pins WalaHash and 3.5.6 cross-platform evidence", () => {
  assert.deepEqual(testMiners.srbminer.windowsAsset, {
    url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/2.9.3/" +
      "SRBMiner-Multi-2-9-3-win64.zip",
    sha256: "0b62e1e6447c456ca04e8bf4ccf3cea23c3077a92185d15bada06afdc9797398",
  });
  assert.deepEqual(testMiners.srbminer.linuxEvidence.walahash, [
    {
      platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
      samples: [1034.62, 1030.15, 1014.56],
      referenceRate: 1030.15, unit: "MH/s",
      reason: "median of three final pool-rate samples",
    },
    {
      platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
      samples: [784.37, 782.93, 769.28],
      referenceRate: 782.93, unit: "MH/s",
      reason: "median of three final pool-rate samples",
    },
  ]);
  assert.deepEqual(testMiners.srbminer.windowsEvidence.walahash, [
    {
      platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
      samples: [974.89, 1025.77, 1027.06],
      referenceRate: 1025.77, unit: "MH/s",
      reason: "median of three final pool-rate samples",
    },
    {
      platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
      samples: [782.3517920891048, 782.4291863086698, 780.9717604646505],
      referenceRate: 782.3517920891048, unit: "MH/s",
      reason: "median of three final pool-rate samples",
    },
  ]);

  const miner = testMiners.srbminer356;
  assert.equal(miner.name, "SRBMiner-Multi 3.5.6");
  assert.equal(miner.url,
    "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.5.6/" +
    "SRBMiner-Multi-3-5-6-Linux.tar.gz");
  assert.equal(miner.sha256,
    "9e28c43dc16e687a97e666643bd453f6f94432d132d3d8d21388ffda835f63fc");
  assert.equal(miner.strip, 1);
  assert.equal(miner.exe, "SRBMiner-MULTI");
  assert.deepEqual(miner.windowsAsset, {
    url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.5.6/" +
      "SRBMiner-Multi-3-5-6-win64.zip",
    sha256: "de2c450019b7e10bd7a824e42d1ff28e3e87d559adca3c1f010ce9066470c6c1",
  });
  assert.deepEqual(miner.linuxEvidence, {
    xelishashv3: {
      platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
      samples: [2.63524, 2.63534, 2.63487], referenceRate: 2.63524, unit: "KH/s",
      reason: "median of three final pool-rate samples",
    },
    verthash: {
      platform: "linux", vendor: "intel", gpu: "Intel B580",
      samples: [316.98933, 316.6901, 316.62288], referenceRate: 316.6901, unit: "KH/s",
      reason: "median of three final pool-rate samples",
    },
  });
  assert.deepEqual(miner.windowsEvidence, {
    xelishashv3: {
      platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
      samples: [2.27814, 2.26737, 2.26749], referenceRate: 2.26749, unit: "KH/s",
      reason: "median of three final pool-rate samples",
    },
    verthash: {
      platform: "windows", vendor: "intel", gpu: "Intel B580",
      samples: [317.35542, 317.24746, 317.17253], referenceRate: 317.24746, unit: "KH/s",
      reason: "median of three final pool-rate samples",
    },
  });
  assert.deepEqual(miner.poolOnly.xelishashv3.command, [
    "SRBMiner-MULTI", "--algorithm", "xelishashv3", "--pool", "de.xelis.herominers.com:1225",
    "--wallet", "WALLET.WORKER", "--password", "x",
  ]);
  assert.deepEqual(miner.poolOnly.verthash.command, [
    "SRBMiner-MULTI", "--algorithm", "verthash", "--pool", "vertcoin.cedric-crispin.com:3334",
    "--wallet", "WALLET.WORKER", "--password", "x", "--verthash-dat-path", "/data/verthash.dat",
  ]);
  assert.equal(Object.hasOwn(miner.poolOnly.xelishashv3, "evidence"), false);
  assert.equal(Object.hasOwn(miner.poolOnly.verthash, "evidence"), false);
});

test("retained SOTA qualifications have pinned 80% evidence", () => {
  const expected = {
    xelishashv3: {
      id: "rigel", evidence: [{reference: "Rigel 1.23.2", momRate: 5.88, sotaRate: 6.95,
        unit: "KH/s", platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti"}],
    },
    verthash: {id: "teamblackminer", evidence: [
      {reference: "TeamBlackMiner 2.44", momRate: 908.06, sotaRate: 925.88,
        unit: "KH/s", platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti"},
      {reference: "TeamBlackMiner 2.44", momRate: 293.48, sotaRate: 291.85,
        unit: "KH/s", platform: "linux", vendor: "amd", gpu: "RX 9060 XT"},
    ]},
    walahash: {id: "srbminer", evidence: [
      {reference: "SRBMiner-Multi 2.9.3", momRate: 623.38, sotaRate: 362.76,
        unit: "MH/s", platform: "linux", vendor: "intel", gpu: "Intel B580"},
      {reference: "SRBMiner-Multi 2.9.3", momRate: 532.93, sotaRate: 404.39,
        unit: "MH/s", platform: "windows", vendor: "intel", gpu: "Intel B580"},
      {reference: "SRBMiner-Multi 2.9.3", momRate: 679.50, sotaRate: 782.93,
        unit: "MH/s", platform: "linux", vendor: "amd", gpu: "RX 9060 XT"},
      {reference: "SRBMiner-Multi 2.9.3", momRate: 667.68, sotaRate: 782.3517920891048,
        unit: "MH/s", platform: "windows", vendor: "amd", gpu: "RX 9060 XT"},
    ]},
  };
  for (const [algo, {id, evidence: expectedEvidence}] of Object.entries(expected)) {
    const candidate = comparatorsFor(algo).find((entry) => entry.id === id);
    assert.ok(candidate?.poolOnly, `${id} should retain pool-only status`);
    const evidence = Array.isArray(candidate.poolOnly.evidence) ? candidate.poolOnly.evidence :
      [candidate.poolOnly.evidence];
    assert.equal(candidate.spec, undefined);
    assert.deepEqual(evidence, expectedEvidence.map((entry) => ({...entry, minRatio: 0.8})));
    for (const entry of evidence) {
      assert.equal(requiredQualificationResult(entry, {
        value: entry.sotaRate, unit: entry.unit,
      }).pass, true);
    }
  }
});

test("existing lolMiner benchmark mappings remain offline where vendor-qualified", () => {
  assert.deepEqual(testMiners.lolminer.benchmark, {
    c30: "C30CTX",
    c29: {
      args: ["--benchmark", "CR29", "--no-oc-reset", "1", "--no-cl", "1"],
      rate: "average-speed-runs",
      unit: "g/s",
      runs: 3,
      vendors: ["nvidia"],
    },
    equihash192_7: {
      args: ["--benchmark", "EQUI192_7"],
      rate: "lolminer-input-iterations",
      unit: "I/s",
      runs: 3,
    },
    nexapow: {args: ["--benchmark", "NEXA"], rate: "gpu-speed", unit: "Mh/s", runs: 12},
    octopus: {args: ["--benchmark", "OCTOPUS"], rate: "average-speed-runs", unit: "Mh/s",
      runs: 3, longStats: 60, tailSamples: 3, vendors: ["nvidia"]},
    zhash: "EQUI144_5",
  });
  assert.deepEqual(requiredBenchmarkSpec(testMiners.lolminer, "zhash"), {args: ["--benchmark", "EQUI144_5"]});
  const script = buildTestContainerScript(testMiners.lolminer, "zhash", "amd",
    {epoch: "440", seconds: "90", devices: "0"});
  assert.match(script, /'tar' '-xzf' '\/archive'/);
  assert.match(script, /'--no-same-owner' '--no-same-permissions'; cd \/tmp\/miner; .*'\.\/lolMiner'/);
  assert.match(script, /'--benchmark' 'EQUI144_5'/);
  assert.match(script, /'--longstats' '10'/);
  assert.match(script, /OCL_ICD_FILENAMES=\/sota\/libamdocl64\.so/);
  assert.doesNotMatch(script, /--pool|stratum/);

  const octopusScript = buildTestContainerScript(testMiners.lolminer, "octopus", "nvidia",
    {epoch: "440", seconds: "90", devices: "0"});
  assert.equal(requiredMatch(octopusScript.match(/'--benchmark' 'OCTOPUS'/g)).length, 3);
  assert.equal(requiredMatch(octopusScript.match(/'--longstats' '60'/g)).length, 3);
  assert.equal(requiredMatch(octopusScript.match(/'--shortstats' '10'/g)).length, 3);
  assert.throws(() => buildTestContainerScript(testMiners.lolminer, "octopus", "amd"),
    /no offline benchmark for octopus/);
  assert.equal(benchmarkSpec(testMiners.lolminer, "octopus", "amd"), null);
  assert.ok(benchmarkSpec(testMiners.lolminer, "octopus", "nvidia"));

  const amd = comparatorsFor("octopus", "amd").find(({id}) => id === "lolminer");
  assert.equal(amd?.spec, undefined);
  assert.ok(amd?.poolOnly);
  const nvidia = comparatorsFor("octopus", "nvidia").find(({id}) => id === "lolminer");
  assert.ok(nvidia?.spec);
  assert.equal(nvidia?.poolOnly, undefined);
});

test("lolMiner CR29 evidence parses three sparse rates and stays NVIDIA-only", () => {
  assert.deepEqual(testMiners.lolminer.linuxEvidence.c29, {
    platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
    samples: [7.40, 7.40, 7.40], referenceRate: 7.40, unit: "g/s",
    reason: "three official lolMiner 1.98a offline 10-second runs at 150 W; " +
      "host nvidia-smi captured compute-active windows",
  });
  const spec = requiredBenchmarkSpec(testMiners.lolminer, "c29");
  assert.deepEqual(spec, {
    args: ["--benchmark", "CR29", "--no-oc-reset", "1", "--no-cl", "1"],
    rate: "average-speed-runs", unit: "g/s", runs: 3, vendors: ["nvidia"],
  });
  /** @param {number | string} rate @param {string} [unit] @returns {string} */
  const block = (rate, unit = "g/s") =>
    "Start Benchmark... \n" +
    "Average speed (10s): " + rate + " " + unit + "\n";
  const threeRuns = [7.40, 7.40, 7.40].map((rate) =>
    block(rate.toFixed(2))).join("\n");
  assert.deepEqual(benchmarkRate(spec, threeRuns), {value: 7.4, unit: "g/s"});
  const script = buildTestContainerScript(testMiners.lolminer, "c29", "nvidia",
    {epoch: "440", seconds: "90", devices: "0"});
  assert.equal(requiredMatch(script.match(/'--benchmark' 'CR29'/g)).length, 3);
  assert.equal(requiredMatch(script.match(/'--no-oc-reset' '1' '--no-cl' '1'/g)).length, 3);

  const incomplete = [7.40, 7.40].map((rate) => block(rate.toFixed(2))).join("\n");
  assert.equal(benchmarkRate(spec, incomplete), null);
  const wrongUnits = [7.40, 7.40, 7.40].map(() =>
    block("7.40", "MH/s")).join("\n");
  assert.equal(benchmarkRate(spec, wrongUnits), null);
  for (const vendor of ["amd", "intel"]) {
    assert.equal(benchmarkSpec(testMiners.lolminer, "c29", vendor), null);
    assert.equal(comparatorsFor("c29", vendor).some(({id}) => id === "lolminer"), false);
  }
});

test("WildRig NexaPow metadata reproduces the local three-run AMD reference", () => {
  assert.equal(testMiners.wildrig.name, "WildRig Multi 0.50.3");
  assert.match(testMiners.wildrig.url,
    /^https:\/\/github\.com\/andru-kun\/wildrig-multi\/releases\/download\/0\.50\.3\//);
  assert.match(testMiners.wildrig.sha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(testMiners.wildrig.windowsAsset, {
    url: "https://github.com/andru-kun/wildrig-multi/releases/download/0.50.3/" +
      "wildrig-multi-windows-0.50.3.zip",
    sha256: "a2634e061e4e20b4826903a28a53bc5add8d7b1e6e1427a8ff8537d9f9231e27",
  });
  assert.deepEqual(testMiners.wildrig.windowsEvidence.nexapow, {
    platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
    samples: [31.29, 31.22, 31.18], referenceRate: 31.22, unit: "MH/s",
    reason: "median of three final 60-second rates from 90-second offline benchmarks",
  });
  const spec = requiredBenchmarkSpec(testMiners.wildrig, "nexapow");
  assert.deepEqual(spec, {
    runner: "wildrig",
    args: ["--algo", "nexapow", "--benchmark"],
    rate: "wildrig-60s",
    unit: "MH/s",
    runs: 3,
  });

  const script = buildTestContainerScript(testMiners.wildrig, "nexapow", "amd",
    {epoch: "440", seconds: "90", devices: "0"});
  assert.match(script, /'--no-same-owner'/);
  assert.match(script, /'--algo' 'nexapow' '--benchmark'/);
  assert.match(script, /'--benchmark-timeout' '90'/);
  assert.match(script, /'--opencl-platforms' 'amd' '-d' '0'/);
  assert.match(script, /'--no-adl' '--no-color' '--print-time' '5'/);
  assert.match(script, /OCL_ICD_FILENAMES=\/sota\/libamdocl64\.so/);
  assert.doesNotMatch(script, /--url|stratum/);

  const output = [32.51, 32.41, 32.34].map((rate) =>
    ` 60s: ${rate.toFixed(2)} MH/s\nBenchmark finished...`).join("\n");
  assert.deepEqual(benchmarkRate(spec, output), {value: 32.41, unit: "MH/s"});
  assert.deepEqual(benchmarkRate({...spec, runs: 1},
    " 60s: 32.51 MH/s\nBenchmark finished..."), {value: 32.51, unit: "MH/s"});
  const windowsOutput = [31.29, 31.22, 31.18].map((rate) =>
    ` 60s: ${rate.toFixed(2)} MH/s\r\nBenchmark finished...`).join("\r\n");
  assert.deepEqual(benchmarkRate(spec, windowsOutput), {value: 31.22, unit: "MH/s"});
  assert.equal(exitAccepted(testMiners.wildrig, "nexapow", 1, true), true);
  assert.equal(benchmarkRate(spec, output.replaceAll("MH/s", "KH/s")), null);
  const qualification = requiredQualificationResult(testMiners.wildrig.qualification.nexapow,
    {value: 32.41, unit: "MH/s"});
  assert.equal(qualification.pass, true);
  assert.ok(qualification.ratio > 0.84 && qualification.ratio < 0.85);
});

test("AMD lolMiner Equihash evidence uses local input iterations on both platforms", () => {
  assert.deepEqual(testMiners.lolminer.windowsAsset, {
    url: "https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.98a/lolMiner_v1.98a_Win64.zip",
    sha256: "f2bbda2d2255155d50935967b8c55105b9aeefbd27cda3e8d01beaf535a16762",
  });
  const linuxEquihash = testMiners.lolminer.linuxEvidence.equihash192_7;
  assert.deepEqual(linuxEquihash.samples, [20.2, 20.2, 20.2]);
  assert.equal(linuxEquihash.referenceRate, 20.2);
  assert.equal(linuxEquihash.unit, "I/s");
  assert.match(linuxEquihash.reason, /input-iteration rows/);
  const windowsEquihash = testMiners.lolminer.windowsEvidence.equihash192_7;
  assert.deepEqual(windowsEquihash.samples, [20.3, 20.2, 20.2]);
  assert.equal(windowsEquihash.referenceRate, 20.2);
  assert.equal(windowsEquihash.unit, "I/s");
  assert.match(windowsEquihash.reason, /input-iteration column/);

  const spec = requiredBenchmarkSpec(testMiners.lolminer, "equihash192_7");
  const output = [
    "      Name       Speed   Pool  Iter.  Shares   Best     Eff.  Power",
    "                 sol/s  sol/s   it/s   A/S/R  Share  sol/s/W      W",
    "GPU 0 RX 9060 XT  42.0    0.0   20.2   0/0/0    0.0    3.818   11.0",
    "Total             42.0    0.0   20.2   0/0/0    0.0    3.818   11.0",
    "GPU 0 RX 9060 XT  42.1    0.0   20.3   0/0/0    0.0    3.508   12.0",
    "Total             42.1    0.0   20.3   0/0/0    0.0    3.508   12.0",
    "GPU 0 RX 9060 XT  42.0    0.0   20.2   0/0/0    0.0    3.500   12.0",
    "Total             42.0    0.0   20.2   0/0/0    0.0    3.500   12.0",
  ].join("\r\n");
  assert.deepEqual(benchmarkRate(spec, output), {value: 20.2, unit: "I/s"});
  assert.equal(benchmarkRate(spec, output.split("\r\n").slice(0, 6).join("\n")), null);
  const nexapow = testMiners.lolminer.windowsEvidence.nexapow;
  assert.deepEqual(nexapow.samples, [0, 0, 0]);
  assert.equal(nexapow.status, "unsupported");
  assert.match(nexapow.reason, /error -11/);
  const zhash = testMiners.lolminer.windowsEvidence.zhash;
  assert.deepEqual(zhash.samples, [83.0, 82.9, 83.0]);
  assert.equal(zhash.referenceRate, 83.0);
  assert.equal(zhash.unit, "Sol/s");
  const linuxZhash = testMiners.lolminer.linuxEvidence.zhash;
  assert.deepEqual(linuxZhash.samples, [83.2, 83.1, 83.1]);
  assert.equal(linuxZhash.referenceRate, 83.1);
  assert.equal(linuxZhash.unit, "Sol/s");
});

test("official Cortex PoolMiner metadata records the exhausted RTX 5060 Ti candidate", () => {
  const poolMiner = testMiners.cortexpoolminer;
  assert.equal(poolMiner.name, "Cortex PoolMiner 1.0.0");
  assert.equal(poolMiner.url,
    "https://github.com/CortexFoundation/PoolMiner/releases/download/" +
    "cortex_miner_1.0.0/cortex_miner.tar.gz");
  assert.equal(poolMiner.sha256,
    "4ec195a9060956c490038fec578bc6fb61fd22034dc8ea8b9f9565c5bc61d4d4");
  assert.deepEqual(poolMiner.linuxEvidence.c30, {
    platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
    samples: [0], status: "unsupported",
    reason: "The unchanged release crashed in CuckooInitialize before producing a rate.",
  });
  assert.deepEqual(poolMiner.poolOnly.c30.command, [
    "cortex_miner", "-pool_uri=POOL:PORT", "-worker=WORKER", "-devices=0", "-account=ADDRESS",
  ]);
  assert.match(poolMiner.poolOnly.c30.reason, /no offline benchmark/);
});

test("GMiner metadata records the exhausted Windows RTX C30 candidate", () => {
  const gminer = testMiners.gminer;
  assert.equal(gminer.name, "GMiner 3.44");
  assert.deepEqual(gminer.windowsAsset, {
    url: "https://github.com/develsoftware/GMinerRelease/releases/download/3.44/" +
      "gminer_3_44_windows64.zip",
    sha256: "dfa02b289f555d7055342967b1d1c390f5c93108c2b73beee1853ce0281bbe04",
  });
  assert.deepEqual(gminer.windowsEvidence.c30, {
    platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
    samples: [0], status: "unsupported",
    reason: "The unchanged release identified the GPU, then reported no compatible kernel image.",
  });
});

test("AMD hoo_gpu evidence pins the controlled Linux HooHash median", () => {
  const miner = testMiners.hoogpuamd;
  assert.equal(miner.name, "hoo_gpu_amd 1.4.22");
  assert.equal(miner.url, "https://htn.foztor.net/hoo_gpu_amd-1.4.22.tar.gz");
  assert.equal(miner.sha256,
    "d1e72920b06ff4577460bb2c9022ee40c8aa8d889c6851f163adc94ed834ff7f");
  assert.deepEqual(miner.linuxEvidence.hoohash, {
    platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
    samples: [1.49, 1.49, 1.49], referenceRate: 1.49, unit: "MH/s",
    reason: "median of three per-run final-five 30-second pool-rate medians",
  });
  assert.deepEqual(miner.poolOnly.hoohash.evidence, {
    platform: "linux", vendor: "amd", gpu: "RX 9060 XT", momRate: 1.71,
    sotaRate: 1.49, unit: "MH/s", minRatio: 0.8, reference: "hoo_gpu_amd 1.4.22",
  });
  assert.equal(requiredQualificationResult(miner.poolOnly.hoohash.evidence,
    {value: 1.49, unit: "MH/s"}).pass, true);
});

test("Hoominer metadata records the exhausted Windows AMD HooHash candidate", () => {
  const miner = testMiners.hoosathoominer;
  assert.equal(miner.name, "Hoominer 0.4.1");
  assert.deepEqual(miner.windowsAsset, {
    url: "https://github.com/HoosatNetwork/hoominer/releases/download/0.4.1/" +
      "hoominer-0.4.1-windows.zip",
    sha256: "0db7fb5e59a8b115460a11e8f4143bdca5c89fce748432fd6a713afa1933fc3d",
  });
  assert.deepEqual(miner.windowsEvidence.hoohash, {
    platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
    samples: [0, 0], status: "unsupported",
    reason: "Both bundled- and system-OpenCL-loader runs crashed in the packaged " +
      "pthreadVC3.dll before GPU enumeration.",
  });
});

test("AMD lolMiner Octopus evidence pins controlled Linux and Windows medians", () => {
  assert.deepEqual(testMiners.lolminer.linuxEvidence.octopus, {
    platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
    samples: [38.28, 38.27, 38.27], referenceRate: 38.27, unit: "MH/s",
    reason: "median of three steady 10-second real-pool rates; a 119.81G share was accepted",
  });
  assert.deepEqual(testMiners.lolminer.poolOnly.octopus.evidence, [
    {
      platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
      samples: [38.28, 38.27, 38.27], referenceRate: 38.27, unit: "MH/s",
      momRate: 29.18, sotaRate: 38.27, minRatio: 0.8, reference: "lolMiner 1.98a",
      reason: "A 119.81G share was accepted during the controlled direct-pool run.",
    },
    {
      platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
      samples: [32.18, 32.21, 32.18], referenceRate: 32.18, unit: "MH/s",
      momRate: 26.49, sotaRate: 32.18, minRatio: 0.8, reference: "lolMiner 1.98a",
      reason: "The bounded 175-second run authorized and received current jobs without errors.",
    },
  ]);
  assert.deepEqual(testMiners.lolminer.windowsEvidence.octopus, {
    platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
    samples: [32.18, 32.21, 32.18], referenceRate: 32.18, unit: "MH/s",
    reason: "median of the final three warmed 10-second rates from a bounded real-pool run",
  });
});

test("BZMiner metadata retains the 100.36 local candidate measurements", () => {
  const miner = testMiners.bzminer10036;
  assert.equal(miner.name, "BZMiner 100.36");
  assert.equal(miner.url, "https://github.com/bzminer/bzminer/releases/download/v100.36/" +
    "bzminer_v100.36_linux.tar.gz");
  assert.equal(miner.sha256,
    "8a665ddd7138a87b94cabba851cc6897e86c65bfcd8f0639ebcb99c1b7c3f889");
  assert.deepEqual(miner.windowsAsset, {
    url: "https://github.com/bzminer/bzminer/releases/download/v100.36/" +
      "bzminer_v100.36_windows.zip",
    sha256: "85b07dafea1e61a76711fcbd5ca06a4d3671698963283bd829a67f54060fdd49",
  });
  const evidence = miner.linuxEvidence;
  if (!evidence) {throw new Error("Expected BZMiner Linux evidence");}
  const linuxPearlAmd = evidence.pearlhash.find((row) => row.vendor === "amd");
  assert.ok(linuxPearlAmd);
  assert.equal(linuxPearlAmd.status, "candidate");
  assert.match(linuxPearlAmd.reason, /KRig/);
  assert.deepEqual(evidence.c29.map((row) => row.referenceRate), [7.68, 6.64]);
  assert.deepEqual(evidence.autolykos2.map((row) => row.referenceRate), [119.28, 40.51]);
  assert.deepEqual(evidence.pearlhash.map((row) => row.referenceRate), [91.46, 37.39, 35.79]);
  const windows = miner.windowsEvidence;
  if (!windows) {throw new Error("Expected BZMiner Windows evidence");}
  const windowsPearlAmd = windows.pearlhash.find((row) => row.vendor === "amd");
  assert.ok(windowsPearlAmd);
  assert.equal(windowsPearlAmd.status, "candidate");
  assert.match(windowsPearlAmd.reason, /KRig/);
  assert.deepEqual(windows.autolykos2.map((row) => row.referenceRate),
    [37.94, 119.25, 36.57]);
  assert.equal(windows["cn/gpu"].referenceRate, 2.63);
  assert.equal(windows.c29.referenceRate, 6.21);
  assert.equal(windows.etchash.referenceRate, 13.85);
  assert.deepEqual(windows.pearlhash.map((row) => row.referenceRate), [37.60, 91.19, 35.27]);
  assert.equal(windows.xelishashv3.referenceRate, 2.37);
});

test("KRig metadata retains the qualified AMD Pearl reference", () => {
  assert.deepEqual(testMiners.krig, {
    name: "KRig 1.5.2",
    linuxEvidence: {
      pearlhash: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        referenceRate: 46.29, unit: "TH/s",
        reason: "verified final-five median at M=8192, N=32768, K=2048, rank=128; " +
          "six accepted shares, no rejects, and 1170 seconds of warm samples at 145 W",
      },
    },
  });
});

test("container invocation remains read-only and network-isolated", () => {
  for (const [vendor, devices] of Object.entries({
    intel: ["--device", "/dev/dri"],
    amd: ["--device", "/dev/dri", "--device", "/dev/kfd"],
    nvidia: ["--gpus", "device=0"],
  })) {
    const args = buildDockerArgs({archive: "/tmp/lol.tar.gz", image: "test-image", script: "true", vendor});
    const gpuStart = args.indexOf("bash") + 1;
    assert.deepEqual(args.slice(gpuStart, gpuStart + devices.length), devices);
    assert.ok(!args.includes("--privileged"));
    assert.ok(args.includes("--network") && args.includes("none"));
    assert.ok(args.includes("--read-only"));
    assert.deepEqual(args.slice(args.indexOf("--user"), args.indexOf("--user") + 2),
      ["--user", "65534:65534"]);
    assert.deepEqual(args.slice(args.indexOf("--pull"), args.indexOf("--pull") + 2),
      ["--pull", "never"]);
    assert.ok(args.includes("--memory") && args.includes("4g"));
    assert.ok(args.includes("--memory-swap") && args.includes("4g"));
    assert.ok(args.includes("--cpus") && args.includes("4"));
    assert.ok(args.includes("--pids-limit") && args.includes("256"));
    assert.ok(args.includes("--cap-drop") && args.includes("ALL"));
    assert.ok(args.includes("--security-opt") && args.includes("no-new-privileges"));
    assert.ok(args.includes("/tmp/lol.tar.gz:/archive:ro"));
    assert.deepEqual(deviceArgs(vendor), devices);
  }
  assert.deepEqual(deviceArgs("nvidia", "0,1"), ["--gpus", "device=0,1"]);
  const multiGpuArgs = buildDockerArgs({archive: "/tmp/lol.tar.gz", image: "test-image",
    script: "true", vendor: "nvidia", devices: "0,1"});
  const multiGpuStart = multiGpuArgs.indexOf("bash") + 1;
  assert.deepEqual(multiGpuArgs.slice(multiGpuStart, multiGpuStart + 2),
    ["--gpus", "device=0,1"]);
  const script = buildTestContainerScript(testMiners.lolminer, "zhash", "nvidia",
    {epoch: "440", seconds: "90", devices: "1,3"});
  assert.match(script, /'--devices' '0,1'/);
  assert.deepEqual(deviceArgs("nvidia", "1,3"), ["--gpus", "device=1,3"]);
});

test("benchmark inputs reject injection and are shell-quoted", () => {
  const valid = {epoch: "440", seconds: "90", devices: "0,1"};
  const script = buildTestContainerScript(testMiners.lolminer, "zhash", "intel", valid);
  assert.match(script, /'--devices' '0,1'/);
  assert.match(script, /'--benchepoch' '440'/);
  assert.match(script, /'timeout' '-s' 'INT' '-k' '10s' '90'/);
  for (const epoch of ["0", "-1", "1.5", "1; touch /tmp/pwned", "1e3", ""]) {
    assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel", {...valid, epoch}),
      /epoch must be a positive integer/);
  }
  assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel",
    {...valid, epoch: "1000001"}), /epoch must be a positive integer/);
  for (const seconds of ["0", "-1", "1.5", "90; touch /tmp/pwned", "1e3", ""]) {
    assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel", {...valid, seconds}),
      /seconds must be a positive integer/);
  }
  assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel",
    {...valid, seconds: "3601"}), /seconds must be a positive integer/);
  for (const devices of ["", "-1", "0,1; touch /tmp/pwned", "0,,1", "0 1", "1.5"]) {
    assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel", {...valid, devices}),
      /devices must be comma-separated nonnegative integers/);
  }
  assert.throws(() => buildTestContainerScript(testMiners.lolminer, "zhash", "intel",
    {...valid, devices: "1024"}), /devices must be comma-separated nonnegative integers/);
});

test("downloads publish only verified archives", () => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sota-test-"));
  const content = "verified archive";
  const reference = {
    url: "https://example.test/miner.tar.gz",
    sha256: crypto.createHash("sha256").update(content).digest("hex"),
  };
  /** @type {string[][]} */
  const downloads = [];
  /** @param {string} data @param {number} [status]
   * @returns {(args: string[]) => {status: number}} */
  const writeDownload = (data, status = 0) => (args) => {
    downloads.push(args);
    const outputPath = args[args.indexOf("-o") + 1];
    if (!outputPath) {throw new Error("Download callback did not receive an output path");}
    fs.writeFileSync(outputPath, data);
    return {status};
  };
  try {
    assert.equal(downloadArchive(reference, cache, writeDownload("partial", 22)), 22);
    assert.deepEqual(fs.readdirSync(cache), []);
    const archive = downloadArchive(reference, cache, writeDownload(content));
    if (typeof archive !== "string") {throw new Error("Verified download did not publish an archive");}
    assert.equal(fs.readFileSync(archive, "utf8"), content);
    assert.deepEqual(fs.readdirSync(cache), [path.basename(archive)]);
    for (const args of downloads) {
      assert.deepEqual(args.slice(args.indexOf("--connect-timeout"), args.indexOf("--connect-timeout") + 2),
        ["--connect-timeout", "30"]);
      assert.deepEqual(args.slice(args.indexOf("--max-time"), args.indexOf("--max-time") + 2),
        ["--max-time", "600"]);
    }
  } finally {
    fs.rmSync(cache, {recursive: true, force: true});
  }
});

test("replaces a corrupt cached archive with one verified download", () => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sota-test-"));
  const content = "verified replacement";
  const reference = {
    url: "https://example.test/miner.tar.gz",
    sha256: crypto.createHash("sha256").update(content).digest("hex"),
  };
  const archive = path.join(cache, path.basename(new URL(reference.url).pathname));
  /** @type {string[]} */
  const downloads = [];
  fs.writeFileSync(archive, "corrupt archive");
  try {
    const result = downloadArchive(reference, cache, (args) => {
      const outputPath = args[args.indexOf("-o") + 1];
      if (!outputPath) {throw new Error("Download callback did not receive an output path");}
      downloads.push(outputPath);
      fs.writeFileSync(outputPath, content);
      return {status: 0};
    });
    assert.equal(result, archive);
    assert.equal(downloads.length, 1);
    assert.notEqual(downloads[0], archive);
    assert.equal(crypto.createHash("sha256").update(fs.readFileSync(archive)).digest("hex"),
      reference.sha256);
  } finally {
    fs.rmSync(cache, {recursive: true, force: true});
  }
});

test("Nexa exit 15 is accepted only with its recorded positive-rate evidence", () => {
  const rate = lastRate("Average speed (10s): 96.62 Mh/s\nAverage speed (10s): 0.0 Mh/s");
  if (!rate) {throw new Error("Expected a positive rate in the benchmark output");}
  assert.deepEqual(rate, {value: 96.62, unit: "Mh/s"});
  assert.equal(exitAccepted(testMiners.lolminer, "nexapow", 15, rate.value > 0), true);
  assert.equal(exitAccepted(testMiners.lolminer, "nexapow", 15, false), false);
  assert.equal(exitAccepted(testMiners.lolminer, "zhash", 15, true), false);
  assert.equal(exitAccepted(testMiners.lolminer, "nexapow", 124, true), true);
  assert.equal(exitAccepted(testMiners.lolminer, "nexapow", 124, false), false);
});

test("Nexa qualification uses the warmed per-GPU speed median", () => {
  const spec = requiredBenchmarkSpec(testMiners.lolminer, "nexapow");
  const rows = [96.97, 96.60, 96.15, 95.87, 95.50, 95.05, 94.80, 95.40, 94.72, 94.82, 94.68, 94.77]
    .map((rate) => `GPU 0 RTX 5060 Ti ${rate}  0.00   0/0/0    0.0   1.849   40.7  2835  14001    59   66`)
    .join("\n");
  assert.deepEqual(benchmarkRate(spec, rows), {value: 94.785, unit: "Mh/s"});
  assert.equal(benchmarkRate(spec, rows.split("\n").slice(0, 5).join("\n")), null);
  assert.deepEqual(benchmarkRate({...spec, runs: 1},
    "GPU 0 RTX 5060 Ti 96.97  0.00   0/0/0    0.0"), {value: 96.97, unit: "Mh/s"});
  const script = buildTestContainerScript(testMiners.lolminer, "nexapow", "nvidia");
  assert.equal(requiredMatch(script.match(/'--benchmark' 'NEXA'/g)).length, 12);
  assert.equal(requiredMatch(script.match(/benchmark_status=0/g)).length, 12);
  assert.equal(requiredMatch(script.match(/\[ "\$benchmark_status" -eq 124 \]/g)).length, 12);
  assert.equal(requiredMatch(script.match(/exit "\$benchmark_status"/g)).length, 12);
});

test("ZHash qualification uses exact solution throughput", () => {
  const evidence = testMiners.miniz.poolOnly.zhash.evidence;
  assert.equal(evidence.momRate, 87.63);
  assert.equal(evidence.sotaRate, 107.68);
  assert.equal(evidence.unit, "Sol/s");
  assert.equal(evidence.minRatio, 0.8);
  assert.equal(requiredQualificationResult(evidence, {value: 107.68, unit: "Sol/s"}).pass, true);
  assert.equal(requiredQualificationResult(evidence, {value: 116, unit: "Sol/s"}).pass, false);
  assert.equal(requiredQualificationResult(evidence, {value: 87.63, unit: "I/s"}).pass, false);
});

test("Equihash 192,7 qualification records the qualified miniZ input rate", () => {
  const evidence = testMiners.miniz.poolOnly.equihash192_7.evidence;
  assert.equal(evidence.momRate, 26.85);
  assert.equal(evidence.sotaRate, 30.92);
  assert.equal(evidence.unit, "I/s");
  assert.equal(requiredQualificationResult(evidence, {value: 30.92, unit: "I/s"}).pass, true);
  assert.equal(requiredQualificationResult(evidence, {value: 33.57, unit: "I/s"}).pass, false);
});

test("Octopus qualification records the current SYCL-native rate", () => {
  const qualification = testMiners.lolminer.qualification.octopus;
  assert.equal(qualification.status, "qualified");
  assert.equal(qualification.momRate, 45.62);
  assert.equal(qualification.sotaRate, 51.65);
  const output = [
    [0, 51.02, 51.04, 51.06],
    [0, 51.03, 51.04, 51.05],
    [0, 51.00, 51.04, 51.08],
  ].map((rates) => `Start Benchmark...\n${rates.map((rate) =>
    `Average speed (10s): ${rate.toFixed(2)} Mh/s`).join("\n")}`).join("\n");
  assert.deepEqual(benchmarkRate(requiredBenchmarkSpec(testMiners.lolminer, "octopus"), output),
    {value: 51.04, unit: "Mh/s"});
  assert.equal(requiredQualificationResult(qualification, {value: 51.65, unit: "MH/s"}).pass, true);
  assert.equal(requiredQualificationResult(qualification, {value: 57.03, unit: "MH/s"}).pass, false);
});

test("Octopus average-speed runs use positive matching-unit tail medians", () => {
  const spec = requiredBenchmarkSpec(testMiners.lolminer, "octopus");
  /** @param {string[]} rates @returns {string} */
  const block = (rates) => `Start Benchmark...\n${rates.join("\n")}`;
  const output = [
    block([
      "Average speed (10s): 0.00 Mh/s",
      "Average speed (10s): 49.00 KH/s",
      "Average speed (10s): 50.00 Mh/s",
      "Average speed (10s): 51.00 MH/S",
      "Average speed (10s): 52.00 Mh/s",
    ]),
    block([
      "Average speed (10s): 0.00 Mh/s",
      "Average speed (10s): 60.00 Mh/s",
      "Average speed (10s): 61.00 Mh/s",
      "Average speed (10s): 62.00 Mh/s",
    ]),
    block([
      "Average speed (10s): 70.00 Mh/s",
      "Average speed (10s): 71.00 Mh/s",
      "Average speed (10s): 72.00 Mh/s",
    ]),
  ].join("\n");
  assert.deepEqual(benchmarkRate(spec, output), {value: 61, unit: "Mh/s"});
});

test("Octopus average-speed runs reject wrong units and incomplete runs or tails", () => {
  const spec = requiredBenchmarkSpec(testMiners.lolminer, "octopus");
  /** @param {string} unit @param {number} count @returns {string} */
  const rates = (unit, count) => Array.from({length: count}, (_, index) =>
    `Average speed (10s): ${50 + index}.00 ${unit}`).join("\n");
  const twoRuns = [rates("Mh/s", 3), rates("Mh/s", 3)].map((body) =>
    `Start Benchmark...\n${body}`).join("\n");
  assert.equal(benchmarkRate(spec, twoRuns), null);
  const shortRuns = Array.from({length: 3}, () => `Start Benchmark...\n${rates("Mh/s", 2)}`).join("\n");
  assert.equal(benchmarkRate(spec, shortRuns), null);
  const wrongUnits = Array.from({length: 3}, () => `Start Benchmark...\n${rates("GH/s", 4)}`).join("\n");
  assert.equal(benchmarkRate(spec, wrongUnits), null);
});

test("Nexa qualification records the sustained staged SYCL-native gate", () => {
  const qualification = testMiners.lolminer.qualification.nexapow;
  assert.equal(qualification.status, "qualified");
  assert.equal(qualification.momRate, 76.10);
  assert.equal(qualification.sotaRate, 94.59);
  assert.equal(qualification.unit, "MH/s");
  assert.equal(requiredQualificationResult(qualification, {value: 94.59, unit: "MH/s"}).pass, true);
});

test("vendor-scoped qualifications only apply to their requested vendor", () => {
  const qualification = testMiners.lolminer.qualification.octopus;
  const selected = qualificationForVendor(qualification, "nvidia");
  assert.equal(selected, qualification);
  assert.equal(requiredQualificationResult(selected, {value: 51.65, unit: "MH/s"}).pass, true);
  assert.equal(qualificationForVendor(qualification, "amd"), null);
});

test("unscoped qualifications remain applicable to every vendor", () => {
  const qualification = {momRate: 80, unit: "MH/s", minRatio: 0.8};
  const selected = qualificationForVendor(qualification, "amd");
  assert.equal(selected, qualification);
  assert.equal(requiredQualificationResult(selected, {value: 100, unit: "MH/s"}).pass, true);
});

test("container builder rejects option and bind escapes without arbitrary GPU arguments", () => {
  const valid = {archive: "/tmp/reference.tar.gz", image: "local/reference:latest", script: "true", vendor: "amd"};
  // Deliberately supply an unknown runtime property without widening the public options type.
  const args = buildDockerArgs(/** @type {Parameters<typeof buildDockerArgs>[0] & {gpuArgs: string[]}} */
    ({...valid, amdOpencl: "/opt/opencl/libamdocl64.so",
      gpuArgs: ["--privileged", "--network", "host", "-v", "/:/host"]}));
  assert.ok(!args.includes("--privileged"));
  assert.ok(!args.includes("host"));
  assert.ok(!args.includes("/:/host"));
  assert.ok(args.includes("/opt/opencl/libamdocl64.so:/sota/libamdocl64.so:ro"));
  for (const image of ["--privileged", "image\n--network=host", "image name", ""]) {
    assert.throws(() => buildDockerArgs({...valid, image}), /Invalid container image/);
  }
  for (const archive of ["relative.tar.gz", "/tmp/a:/host", "/tmp/a,readonly=false", "/tmp/a\n"]) {
    assert.throws(() => buildDockerArgs({...valid, archive}), /Invalid container bind path/);
  }
  assert.throws(() => buildDockerArgs({...valid, vendor: "unknown"}), /Unknown GPU vendor/);
  assert.throws(() => deviceArgs("unknown"), /Unknown GPU vendor/);
  assert.throws(() => buildDockerArgs({...valid, vendor: "nvidia", amdOpencl: "/opt/loader.so"}),
    /AMD loader requires/);
  assert.throws(() => buildDockerArgs({...valid, archive: "/"}), /Container binds must/);
  assert.throws(() => buildDockerArgs({...valid, amdOpencl: "/"}), /Container binds must/);
});

test("container script rejects malformed executable, count and exit metadata", () => {
  const valid = {epoch: "440", seconds: "1", devices: "0"};
  for (const exe of ["../miner", "/miner", "a/../miner", "a\nminer"]) {
    assert.throws(() => buildTestContainerScript({...testMiners.lolminer, exe}, "zhash", "intel", valid),
      /invalid offline archive metadata/);
  }
  for (const runs of [0, -1, 1.5, 13, Infinity]) {
    assert.throws(() => buildTestContainerScript({...testMiners.lolminer,
      benchmark: {zhash: {args: ["--benchmark", "SAFE"], runs}}}, "zhash", "intel", valid),
    /invalid offline archive metadata/);
  }
  for (const accepted of [[-1], [256], ["0; echo unsafe"]]) {
    assert.throws(() => buildTestContainerScript({...testMiners.lolminer,
      // This negative intentionally supplies malformed exit metadata, including a string status.
      exitPolicy: {zhash: /** @type {import("../scripts/sota-miners").ExitPolicy} */
        (/** @type {unknown} */ ({accepted}))}}, "zhash", "intel", valid), /invalid offline archive metadata/);
  }
  const script = buildTestContainerScript({...testMiners.lolminer,
    benchmark: {zhash: {args: ["--benchmark", "safe'$(printf NOT_EXECUTED);*"]}}},
  "zhash", "intel", valid);
  assert.ok(script.includes("'safe'\\''$(printf NOT_EXECUTED);*'"));
  assert.match(script, /'timeout' '-s' 'INT' '-k' '10s' '1'/);
});

test("downloads reject insecure references, caches and symlink archives before any fake download", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sota-safety-"));
  const cache = path.join(root, "cache");
  const content = "harmless archive fixture";
  const reference = {url: "https://example.test/miner.tar.gz",
    sha256: crypto.createHash("sha256").update(content).digest("hex")};
  let calls = 0;
  const forbiddenDownload = () => {calls++; throw new Error("Unexpected download");};
  try {
    for (const url of ["http://example.test/miner.tar.gz", "https://user:pass@example.test/miner.tar.gz",
      "https://example.test/", "https://example.test/.tar.gz"]) {
      assert.throws(() => downloadArchive({...reference, url}, cache, forbiddenDownload),
        /Invalid archive reference/);
    }
    assert.throws(() => downloadArchive({...reference, sha256: "bad"}, cache, forbiddenDownload),
      /Invalid archive reference/);
    fs.mkdirSync(cache, {mode: 0o755});
    assert.throws(() => downloadArchive(reference, cache, forbiddenDownload), /owned private directory/);
    fs.chmodSync(cache, 0o700);
    const target = path.join(root, "target");
    fs.writeFileSync(target, content);
    const archive = path.join(cache, "miner.tar.gz");
    fs.symlinkSync(target, archive);
    assert.throws(() => downloadArchive(reference, cache, forbiddenDownload), /regular file/);
    assert.equal(fs.readFileSync(target, "utf8"), content);
    fs.unlinkSync(archive);
    const alias = path.join(root, "alias");
    fs.symlinkSync(cache, alias);
    assert.throws(() => downloadArchive(reference, alias, forbiddenDownload), /owned private directory/);
    assert.equal(calls, 0);
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("failed replacement preserves the prior corrupt cache and HTTPS-only size limits", () => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sota-preserve-"));
  const content = "verified archive";
  const reference = {url: "https://example.test/miner.tar.gz",
    sha256: crypto.createHash("sha256").update(content).digest("hex")};
  const archive = path.join(cache, "miner.tar.gz");
  fs.writeFileSync(archive, "old corrupt cache");
  try {
    const result = downloadArchive(reference, cache, (args) => {
      assert.equal(args[0], "-q");
      for (const flag of ["--proto", "--proto-redir"]) {
        assert.equal(args[args.indexOf(flag) + 1], "=https");
      }
      assert.equal(args[args.indexOf("--max-filesize") + 1], "536870912");
      return {status: 22};
    });
    assert.equal(result, 22);
    assert.equal(fs.readFileSync(archive, "utf8"), "old corrupt cache");
    assert.deepEqual(fs.readdirSync(cache), ["miner.tar.gz"]);
    assert.equal(downloadArchive(reference, cache, () => ({status: 0, signal: "SIGTERM"})), 1);
    assert.equal(fs.readFileSync(archive, "utf8"), "old corrupt cache");
  } finally {
    fs.rmSync(cache, {recursive: true, force: true});
  }
});

test("source-bound CLI fakes execute only the constrained container and never a pool miner", () => {
  const vm = require("node:vm");
  const source = fs.readFileSync(require.resolve("../scripts/sota-benchmark"), "utf8");
  function runFixture({algo = "zhash", vendor = "intel", env = {}, platform = "linux", result = {}} = {}) {
    /** @type {{command: string, args: string[], options: import("node:child_process").SpawnSyncOptions}[]} */
    const calls = [];
    /** @type {string[]} */
    const messages = [];
    const fixtureModule = {exports: {}};
    const fakeFs = {
      mkdirSync() {/* The fixture cache already exists. */},
      lstatSync(/** @type {string} */ file) {
        return {mode: 0o700, uid: 1234, size: 1,
          isDirectory: () => file === "/fixture-cache", isFile: () => file !== "/fixture-cache"};
      },
      realpathSync(/** @type {string} */ file) {return file;},
      statSync() {return {isFile: () => true};},
      existsSync() {return true;},
      readFileSync() {return Buffer.from("harmless fake archive");},
    };
    /** @type {{argv: string[], env: NodeJS.ProcessEnv, platform: string, getuid: () => number,
     * stdout: {write: (value: string) => number}, stderr: {write: (value: string) => number},
     * exitCode?: number}} */
    const fixtureProcess = {argv: ["node", "sota-benchmark.js", algo, vendor],
      env: {MOM_SOTA_CACHE: "/fixture-cache", ...env}, platform, getuid: () => 1234,
      stdout: {write: (value) => messages.push(value)}, stderr: {write: (value) => messages.push(value)}};
    const fakeRequire = (/** @type {string} */ name) => {
      if (name === "node:fs") {return fakeFs;}
      if (name === "node:crypto") {
        return {createHash: () => ({update: () => ({digest: () => testMiners.lolminer.sha256})})};
      }
      if (name === "node:child_process") {
        return {spawnSync: (/** @type {string} */ command, /** @type {string[]} */ args,
          /** @type {import("node:child_process").SpawnSyncOptions} */ options) => {
          calls.push({command, args, options});
          if (command !== "docker") {throw new Error("Unexpected host command");}
          return {status: 0, signal: null, stdout: "Average speed (10s): 83.1 Sol/s\n",
            stderr: "", ...result};
        }};
      }
      if (name === "../miner/algorithms") {return require("../miner/algorithms");}
      if (name === "./sota-miners") {return testMiners;}
      if (["node:path", "node:os"].includes(name)) {return require(name);}
      throw new Error("Unexpected fixture import: " + name);
    };
    fakeRequire.main = fixtureModule;
    vm.runInNewContext(source, {require: fakeRequire, module: fixtureModule, process: fixtureProcess,
      console: {log: (/** @type {string} */ value) => messages.push(value),
        error: (/** @type {string} */ value) => messages.push(value)}, URL, Buffer});
    return {calls, messages, status: fixtureProcess.exitCode};
  }
  for (const vendor of ["intel", "amd", "nvidia"]) {
    const fixture = runFixture({vendor});
    assert.equal(fixture.status, 0);
    assert.equal(fixture.calls.length, 1);
    assert.equal(/** @type {NonNullable<typeof fixture.calls[0]>} */ (fixture.calls[0]).command, "docker");
    const args = /** @type {NonNullable<typeof fixture.calls[0]>} */ (fixture.calls[0]).args;
    assert.equal(args[args.indexOf("--network") + 1], "none");
    assert.equal(args[args.indexOf("--user") + 1], "65534:65534");
    assert.match(/** @type {string} */ (args.at(-1)), /'timeout' '-s' 'INT' '-k' '10s'/);
  }
  const pool = runFixture({algo: "xelishashv3", env: {MOM_SOTA_MINER: "rigel"}});
  assert.equal(pool.status, 3);
  assert.equal(pool.calls.length, 0);
  assert.match(pool.messages.join("\n"), /metadata only/);
  assert.doesNotMatch(pool.messages.join("\n"), /Pool command:|WALLET|stratum/);
  for (const platform of ["win32", "darwin"]) {
    const fixture = runFixture({platform});
    assert.equal(fixture.status, 2);
    assert.equal(fixture.calls.length, 0);
  }
  assert.throws(() => runFixture({result: {status: 0, signal: "SIGTERM"}}), /terminated by signal/);
  assert.throws(() => runFixture({result: {error: new Error("fake daemon failure")}}), /fake daemon failure/);
  assert.equal(runFixture({result: {status: 99}}).status, 99);
  assert.equal(runFixture({result: {stdout: "Average speed (10s): 0 Sol/s\n"}}).status, 1);
});
