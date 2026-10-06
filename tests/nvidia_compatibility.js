"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const helper = require("../helper");
const opts = require("../opts");
const gpuTuning = require("../gpu-tuning");
const createJobApi = require("../miner/jobs");
const {normalizeAlgoName} = require("../miner/algorithms");
const {hexWithoutPrefix} = require("../miner/submission");
const {windowsCmdArgs} = require("../scripts/windows-command");

/** @typedef {typeof import("../compiler-policy")} CompilerPolicy */
/** @typedef {{command: string, args: string[], options: object}} QueryCall */
/** @typedef {{status: number | null, stdout?: string, stderr?: string, error?: unknown}} SpawnFixture */
/** @typedef {{policy: CompilerPolicy, calls: QueryCall[]}} PolicyContext */

const policyPath = require.resolve("../compiler-policy");
const originalSpawnSync = childProcess.spawnSync;

/**
 * Load compiler-policy with its imported nvidia-smi boundary mocked. The module cache and
 * child_process binding are restored before returning the fresh module context.
 * @param {SpawnFixture} fixture
 * @returns {PolicyContext}
 */
function loadPolicy(fixture) {
  const previousModule = require.cache[policyPath];
  /** @type {QueryCall[]} */
  const calls = [];
  delete require.cache[policyPath];
  childProcess.spawnSync = /** @type {typeof childProcess.spawnSync} */ ((command, args, options) => {
    calls.push({command, args: args ? [...args] : [], options: options || {}});
    return /** @type {ReturnType<typeof childProcess.spawnSync>} */ (fixture);
  });
  try {
    /** @type {CompilerPolicy} */
    const policy = require(policyPath);
    return {policy, calls};
  } finally {
    childProcess.spawnSync = originalSpawnSync;
    delete require.cache[policyPath];
    if (previousModule) {
      require.cache[policyPath] = previousModule;
    }
  }
}

/**
 * Keep missing-addon tests independent of repository artifacts and restore the filesystem boundary
 * even when the synchronous worker-policy call throws.
 * @param {typeof fs.existsSync} implementation
 * @param {() => void} callback
 */
function withExistsSync(implementation, callback) {
  const originalExistsSync = fs.existsSync;
  fs.existsSync = implementation;
  try {
    callback();
  } finally {
    fs.existsSync = originalExistsSync;
  }
}

/** @param {CompilerPolicy} policy @param {NodeJS.ProcessEnv} env
 * @param {NodeJS.Platform} platform @param {string} backend */
function pearlTestRuntime(policy, env, platform, backend) {
  const opt = opts.create_default_opts();
  opt.job = {...opt.job, dev: "gpu1", backend};
  const mockProcess = Object.assign(Object.create(process), {env});
  Object.defineProperty(mockProcess, "platform", {value: platform});
  const api = createJobApi({
    h: helper, opt, process: mockProcess, compilerPolicy: policy, gpuTuning,
    normalizeAlgoName, hexWithoutPrefix,
    messageHandler: () => undefined,
    isExiting: () => false,
    getComputeCore: () => null,
    getLastJob: () => null,
    setLastJob: () => undefined,
  });
  return {
    job: api.prepareTestJob({...opt.job, algo: "pearlhash"}),
    worker: api.workerRuntimeEnv("pearlhash", "gpu1"),
  };
}

/** @type {[number, string, string][]} */
const pearlNvidiaBoundaries = [
  [0, "acpp-cuda", "sycl"],
  [50, "acpp-cuda", "sycl"],
  [60, "acpp-cuda", "sycl"],
  [61, "acpp-cuda", "sycl-native"],
  [70, "acpp-cuda", "sycl-native"],
  [79, "acpp-cuda", "sycl-native"],
  [80, "dpcpp", "native"],
  [120, "dpcpp", "native"],
];

for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
  for (const [sm, compiler, backend] of pearlNvidiaBoundaries) {
    test(`${platform} Pearl SM${sm} routes worker and wire backend without changing profile`, () => {
      const {policy, calls} = loadPolicy({status: 0, stdout: `${Math.floor(sm / 10)}.${sm % 10}\n`});
      const nativeDir = path.join(os.tmpdir(), "mom-pearl-routing-fixture");
      const addon = path.join(nativeDir, compiler, "mom.node");
      const env = {MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: nativeDir, MOM_GPU_INDEX: "2"};
      const nativeModules = Object.keys(require.cache).filter(file => file.endsWith(".node"));
      const selected = policy.selection("pearlhash", "nvidia", platform, sm);
      assert.ok(selected);
      assert.equal(selected.key, compiler);
      assert.equal(selected.addon, `${compiler}/mom.node`);
      assert.equal(selected.backend, backend);
      assert.deepEqual(selected.pearlhashProfile, {m: 131072, n: 524288, k: 8192, rank: 128});
      withExistsSync(file => String(file) === addon, () => {
        for (const requested of ["auto", "sycl"]) {
          const {job, worker} = pearlTestRuntime(policy, env, platform, requested);
          assert.equal(job.backend_request, requested);
          assert.equal(job.backend, requested === "auto" ? backend : "sycl");
          assert.deepEqual([job.intensity, job.pearlhash_n, job.pearlhash_k, job.pearlhash_rank],
            [131072, 524288, 8192, 128]);
          assert.equal(worker["MOM_NATIVE_PATH"], addon);
          assert.equal(worker["MOM_SYCL_COMPILER"], compiler);
          assert.equal(worker["CUDA_VISIBLE_DEVICES"], "2");
          assert.equal(worker["MOM_GPU_INDEX"], "0");
          assert.equal(worker["ACPP_VISIBILITY_MASK"], compiler === "acpp-cuda" ? "cuda" : undefined);
          assert.equal(worker["ONEAPI_DEVICE_SELECTOR"], compiler === "dpcpp" ? "cuda:0" : undefined);
          assert.deepEqual(Object.keys(worker).filter(name => name.startsWith("MOM_PEARLHASH_")), []);
        }
      });
      assert.equal(calls.length, 1);
      assert.equal(calls[0]?.command, "nvidia-smi");
      assert.deepEqual(Object.keys(require.cache).filter(file => file.endsWith(".node")), nativeModules);
    });
  }
}

/** @type {[string, SpawnFixture][]} */
const pearlUnknownQueries = [
  ["query failure", {status: 1, stdout: "12.0\n", stderr: "query failed"}],
  ["query timeout", {status: null, stdout: "12.0\n", error: {code: "ETIMEDOUT"}}],
  ["invalid row", {status: 0, stdout: "12.0\ninvalid\n7.0\n"}],
];
for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
  for (const [reason, fixture] of pearlUnknownQueries) {
    test(`${platform} Pearl ${reason} keeps unknown capability on scalar AdaptiveCpp`, () => {
      const {policy, calls} = loadPolicy(fixture);
      const nativeDir = path.join(os.tmpdir(), "mom-pearl-routing-fixture");
      const addon = path.join(nativeDir, "acpp-cuda", "mom.node");
      const env = {MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: nativeDir, MOM_GPU_INDEX: "0"};
      assert.equal(policy.nvidiaComputeCapability(env), null);
      withExistsSync(file => String(file) === addon, () => {
        const {job, worker} = pearlTestRuntime(policy, env, platform, "auto");
        assert.equal(job.backend_request, "auto");
        assert.equal(job.backend, "sycl");
        assert.deepEqual([job.intensity, job.pearlhash_n, job.pearlhash_k, job.pearlhash_rank],
          [131072, 524288, 8192, 128]);
        assert.equal(worker["MOM_NATIVE_PATH"], addon);
        assert.equal(worker["MOM_SYCL_COMPILER"], "acpp-cuda");
        assert.equal(worker["ACPP_VISIBILITY_MASK"], "cuda");
        assert.equal(worker["CUDA_VISIBLE_DEVICES"], "0");
        assert.equal(worker["MOM_GPU_INDEX"], "0");
        assert.equal(worker["ONEAPI_DEVICE_SELECTOR"], undefined);
        assert.deepEqual(Object.keys(worker).filter(name => name.startsWith("MOM_PEARLHASH_")), []);
      });
      assert.equal(calls.length, 1);
    });
  }
}

test("NVIDIA capability overrides accept canonical forms and reject malformed values", () => {
  /** @type {[string, number][]} */
  const accepted = [
    ["7", 70], ["7.0", 70], ["70", 70], ["12.0", 120], ["120", 120],
    ["  7.5 \t", 75],
  ];
  /** @type {string[]} */
  const rejected = [
    "0", "0.0", "0.7", "-7", "7.", "7.00", "7.5.0", ".5", "1e2",
    "9007199254740992",
  ];
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  /** @type {Record<string, number | null>} */
  const acceptedValues = {};
  for (const [value, expected] of accepted) {
    acceptedValues[value] = policy.nvidiaComputeCapability({
      MOM_NVIDIA_COMPUTE_CAPABILITY: value,
    });
    assert.equal(acceptedValues[value], expected, value);
  }
  for (const value of rejected) {
    assert.throws(() => policy.nvidiaComputeCapability({
      MOM_NVIDIA_COMPUTE_CAPABILITY: value,
    }), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
  }
  assert.throws(() => policy.nvidiaComputeCapability(/** @type {NodeJS.ProcessEnv} */ (/** @type {unknown} */ ({
    MOM_NVIDIA_COMPUTE_CAPABILITY: {toString: () => "8.0"},
  }))), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
  const sentinel = policy.selection("etchash", "nvidia", "linux", 0);
  assert.ok(sentinel);
  assert.equal(sentinel.key, "acpp-cuda");
  assert.equal(calls.length, 0);
});

test("NVIDIA query keeps invalid rows unknown independent of logical indices", () => {
  {
    const {policy, calls} = loadPolicy({status: 0, stdout: "\n7.0\n8.0", stderr: ""});
    assert.equal(policy.nvidiaComputeCapability({}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "0"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "1"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "2"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "3"}), null);
    assert.equal(calls.length, 1);
  }

  {
    const {policy, calls} = loadPolicy({status: 0, stdout: "7.0\nbad\n8.0\n\n", stderr: ""});
    assert.equal(policy.nvidiaComputeCapability({}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "0"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "1"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "2"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "3"}), null);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "4"}), null);
    assert.equal(calls.length, 1);
    const [call] = calls;
    assert.ok(call);
    assert.equal(call.command, "nvidia-smi");
    assert.deepEqual(call.args, [
      "--query-gpu=compute_cap", "--format=csv,noheader,nounits",
    ]);
    assert.deepEqual(call.options, {
      encoding: "utf8", timeout: 2000, windowsHide: true,
    });
  }
});

test("NVIDIA capability minimum ignores CUDA masks, ordering and physical-row indices", () => {
  for (const rows of ["12.0\n7.0\n", "7.0\n12.0\n", "12.0\n8.0\n", "12.0\n"]) {
    const {policy, calls} = loadPolicy({status: 0, stdout: rows});
    const expected = rows.includes("7.0") ? 70 : rows.includes("8.0") ? 80 : 120;
    for (const index of [undefined, "0", "1", "1023"]) {
      for (const mask of [undefined, "", "-1", "1", "1,0", "GPU-1234", "0,2,-1,1"]) {
        for (const order of [undefined, "FASTEST_FIRST", "PCI_BUS_ID"]) {
          assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: index,
            CUDA_VISIBLE_DEVICES: mask, CUDA_DEVICE_ORDER: order}), expected);
        }
      }
    }
    assert.equal(calls.length, 1);
    assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "0",
      CUDA_VISIBLE_DEVICES: "1", MOM_NVIDIA_COMPUTE_CAPABILITY: "120"}), 120);
    assert.equal(calls.length, 1);
  }
});

test("vendor isolation normalizes once and never widens inherited CUDA or HIP masks", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "12.0\n7.0\n"});
  for (const gpu of ["nvidia", "amd"]) {
    const visible = gpu === "nvidia" ? "CUDA_VISIBLE_DEVICES" : "HIP_VISIBLE_DEVICES";
    const original = {MOM_GPU_BACKEND: gpu, MOM_GPU_INDEX: "0002"};
    const isolated = policy.vendorDeviceEnv(original);
    assert.deepEqual(isolated, {[visible]: "2", MOM_GPU_INDEX: "0"});
    assert.deepEqual(original, {MOM_GPU_BACKEND: gpu, MOM_GPU_INDEX: "0002"});
    assert.deepEqual(policy.vendorDeviceEnv({...original, ...isolated}), {MOM_GPU_INDEX: "0"});
    const aliases = gpu === "amd" ? [visible, "CUDA_VISIBLE_DEVICES"] : [visible];
    for (const name of aliases) {
      for (const mask of ["", "-1", "0,1", "1,0", "0,2,-1,1", "99,0", "GPU-one,GPU-two"]) {
        const env = {MOM_GPU_BACKEND: gpu, [name]: mask};
        assert.deepEqual(policy.vendorDeviceEnv(env), {});
        withExistsSync(() => true, () => {
          const worker = policy.workerEnv("octopus", env);
          assert.equal(Object.hasOwn(worker, name), false);
        });
        assert.throws(() => policy.vendorDeviceEnv({...env, MOM_GPU_INDEX: "0"}),
          /cannot safely select within inherited .*single-device visibility mask/);
        assert.equal(env[name], mask);
      }
      for (const mask of ["2", "GPU-1234", "MIG-GPU-1234/1/2", "bad", "9999"]) {
        const env = {MOM_GPU_BACKEND: gpu, MOM_GPU_INDEX: "0", [name]: mask};
        assert.deepEqual(policy.vendorDeviceEnv(env), {MOM_GPU_INDEX: "0"});
        assert.throws(() => policy.vendorDeviceEnv({...env, MOM_GPU_INDEX: "1"}),
          /cannot safely select within inherited/);
        assert.equal(env[name], mask);
      }
    }
    for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
      withExistsSync(() => true, () => {
        const worker = policy.workerEnv("octopus", original, platform);
        assert.equal(worker[visible], "2");
        assert.equal(worker["MOM_GPU_INDEX"], "0");
        const repeated = policy.workerEnv("octopus", {
          ...original, ...worker, MOM_NATIVE_PATH: undefined,
        }, platform);
        assert.equal(repeated[visible], undefined);
        assert.equal(repeated["MOM_GPU_INDEX"], "0");
        assert.equal(worker["MOM_SYCL_COMPILER"], gpu === "nvidia" ? "acpp-cuda" : "acpp-hip");
      });
    }
  }
  assert.throws(() => policy.vendorDeviceEnv({MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "0",
    HIP_VISIBLE_DEVICES: "2", CUDA_VISIBLE_DEVICES: "0,1"}), /inherited CUDA_VISIBLE_DEVICES/);
  const modern = loadPolicy({status: 0, stdout: "12.0\n8.0\n"}).policy;
  withExistsSync(() => true, () => {
    const worker = modern.workerEnv("octopus", {
      MOM_GPU_BACKEND: "nvidia", MOM_GPU_INDEX: "2", CUDA_DEVICE_ORDER: "FASTEST_FIRST",
    });
    assert.equal(worker["MOM_SYCL_COMPILER"], "dpcpp");
    assert.equal(worker["CUDA_VISIBLE_DEVICES"], "2");
    assert.equal(worker["ONEAPI_DEVICE_SELECTOR"], "cuda:0");
    assert.equal(worker["CUDA_DEVICE_ORDER"], undefined);
  });
  assert.equal(calls.length, 1);
});

test("NVIDIA query failure and timeout keep unknown SM conservative", () => {
  for (const fixture of [
    {status: 1, stdout: "80\n", stderr: "nvidia-smi failed"},
    {status: 0, stdout: "80\n", stderr: "", error: {code: "EPERM"}},
    {status: null, stdout: "80\n", stderr: "", error: {code: "ETIMEDOUT"}},
    {status: null, stdout: "", stderr: "", error: {code: "ENOENT"}},
  ]) {
    const {policy, calls} = loadPolicy(fixture);
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvidia-compat-"));
    fs.mkdirSync(path.join(root, "acpp-cuda"), {recursive: true});
    fs.writeFileSync(path.join(root, "acpp-cuda", "mom.node"), "test");
    try {
      assert.equal(policy.nvidiaComputeCapability({}), null);
      assert.equal(policy.nvidiaComputeCapability({MOM_GPU_INDEX: "0"}), null);
      const worker = policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
      }, "linux");
      assert.equal(worker["MOM_SYCL_COMPILER"], "acpp-cuda");
      const nativePath = worker["MOM_NATIVE_PATH"];
      assert.ok(nativePath);
      assert.match(nativePath, /[\\/]acpp-cuda[\\/]mom\.node$/);
      assert.equal(calls.length, 1);
    } finally {
      fs.rmSync(root, {recursive: true, force: true});
    }
  }
});

test("missing NVIDIA legacy and unknown workers fail closed on Linux and Windows", () => {
  /** @type {NodeJS.Platform[]} */
  const platforms = ["linux", "win32"];
  for (const platform of platforms) {
    const {policy, calls} = loadPolicy({status: 1, stdout: "80\n", stderr: "nvidia-smi failed"});
    withExistsSync(() => false, () => {
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NVIDIA_COMPUTE_CAPABILITY: "75",
        MOM_NATIVE_DIR: "/missing-native-dir",
      }, platform), /Missing .* for nvidia\/etchash: .*SM75.*SM80/);
    });
    withExistsSync(() => false, () => {
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NATIVE_DIR: "/missing-native-dir",
      }, platform), /Missing .* for nvidia\/etchash: .*unknown NVIDIA compute capability.*SM80/);
    });
    assert.equal(calls.length, 1);
  }
});

test("missing modern NVIDIA workers keep non-strict and strict behavior on Linux and Windows", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  /** @type {NodeJS.Platform[]} */
  const platforms = ["linux", "win32"];
  for (const platform of platforms) {
    withExistsSync(() => false, () => {
      assert.deepEqual(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NVIDIA_COMPUTE_CAPABILITY: "80",
        MOM_NATIVE_DIR: "/missing-native-dir",
      }, platform), {});
    });
    withExistsSync(() => false, () => {
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NVIDIA_COMPUTE_CAPABILITY: "80",
        MOM_NATIVE_DIR: "/missing-native-dir",
        MOM_COMPILER_POLICY_STRICT: "1",
      }, platform), /Missing .* for nvidia\/etchash$/);
    });
  }
  assert.equal(calls.length, 0);
});

test("explicit NVIDIA addon override still bypasses missing-addon policy", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  let existsCalls = 0;
  withExistsSync(() => {
    existsCalls += 1;
    return false;
  }, () => {
    assert.deepEqual(policy.workerEnv("etchash", {
      MOM_GPU_BACKEND: "nvidia",
      MOM_NVIDIA_COMPUTE_CAPABILITY: "75",
      MOM_NATIVE_PATH: "/explicit/mom.node",
      MOM_NATIVE_PATH_LAUNCHER_DEFAULT: "/launcher/default/mom.node",
      MOM_COMPILER_POLICY_STRICT: "1",
    }, "linux"), {});
    assert.deepEqual(policy.workerEnv("etchash", {
      MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_PATH: "/explicit/mom.node", MOM_GPU_INDEX: "2",
    }, "linux"), {CUDA_VISIBLE_DEVICES: "2", MOM_GPU_INDEX: "0"});
    assert.throws(() => policy.workerEnv("etchash", {
      MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_PATH: "/explicit/mom.node", MOM_GPU_INDEX: "0",
      CUDA_VISIBLE_DEVICES: "",
    }, "linux"), /cannot safely select within inherited/);
  });
  assert.equal(existsCalls, 0);
  assert.equal(calls.length, 0);
});

test("invalid explicit NVIDIA indices never fall back to another GPU", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "7.0\n8.0\n", stderr: ""});
  for (const index of ["-1", "no", "1024"]) {
    assert.throws(() => policy.nvidiaComputeCapability({MOM_GPU_INDEX: index}),
      /Invalid MOM_GPU_INDEX/);
  }
  assert.equal(calls.length, 0);
});

test("selection validates explicit NVIDIA SM values while retaining null and zero", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  for (const sm of [NaN, Infinity, -1, 70.5, "70"]) {
    assert.throws(() => Reflect.apply(policy.selection, null, [
      "etchash", "nvidia", "linux", sm,
    ]), /Invalid NVIDIA compute capability/);
  }
  const sentinel = policy.selection("etchash", "nvidia", "linux", 0);
  assert.ok(sentinel);
  assert.equal(sentinel.key, "acpp-cuda");
  assert.ok(policy.selection("etchash", "nvidia", "linux", null));
  assert.equal(calls.length, 0);
});

test("XelisHashV3 keeps native backend across NVIDIA SM compatibility choices", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvidia-xelis-"));
  for (const addon of ["acpp-cuda/mom.node", "dpcpp/mom.node"]) {
    const target = path.join(root, addon);
    fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(target, "test");
  }
  /** @type {NodeJS.Platform[]} */
  const platforms = ["linux", "win32"];
  const sms = [50, 70, 75, 80, 120];
  try {
    for (const platform of platforms) {
      const baseline = policy.selection("xelishashv3", "nvidia", platform, null);
      assert.ok(baseline);
      for (const sm of sms) {
        const selected = policy.selection("xelishashv3", "nvidia", platform, sm);
        /** @type {string} */
        const expectedCompiler = sm < 80 ? "acpp-cuda" : baseline.key;
        assert.ok(selected);
        assert.equal(selected.key, expectedCompiler);
        assert.equal(selected.backend, baseline.backend);
        const worker = policy.workerEnv("xelishashv3", {
          MOM_GPU_BACKEND: "nvidia",
          MOM_NVIDIA_COMPUTE_CAPABILITY: String(sm),
          MOM_NATIVE_DIR: root,
        }, platform, "sycl-native");
        assert.equal(worker["MOM_SYCL_COMPILER"], expectedCompiler);
        assert.equal(worker["MOM_XELISHASHV3_SYCL_NATIVE"], "1");
      }
    }
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
  assert.equal(calls.length, 0);
});

test("NVIDIA routing keeps legacy and modern compiler choices across configured algorithms", () => {
  const {policy, calls} = loadPolicy({status: 0, stdout: "", stderr: ""});
  const config = policy.parse();
  const algorithms = [...new Set(config.policies
    .filter(({gpu}) => gpu === "nvidia")
    .flatMap(({overrides}) => Object.keys(overrides)))].sort();
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvidia-routing-"));
  for (const artifact of Object.values(config.artifacts)) {
    for (const addon of [artifact.linux, artifact.win32]) {
      const target = path.join(root, addon);
      fs.mkdirSync(path.dirname(target), {recursive: true});
      fs.writeFileSync(target, "test");
    }
  }
  /** @type {NodeJS.Platform[]} */
  const platforms = ["linux", "win32"];
  const oldSms = [50, 52, 60, 61, 70, 75];
  const modernSms = [80, 89, 120];
  try {
    for (const platform of platforms) {
      for (const algo of algorithms) {
        const baseline = policy.selection(algo, "nvidia", platform, null);
        assert.ok(baseline);
        for (const sm of oldSms) {
          const selected = policy.selection(algo, "nvidia", platform, sm);
          const env = {
            MOM_GPU_BACKEND: "nvidia",
            MOM_NVIDIA_COMPUTE_CAPABILITY: String(sm),
            MOM_NATIVE_DIR: root,
          };
          const worker = policy.workerEnv(algo, env, platform);
          const nativeWorker = policy.workerEnv(algo, env, platform, "sycl-native");
          assert.equal(selected?.key, "acpp-cuda");
          assert.equal(worker["MOM_SYCL_COMPILER"], "acpp-cuda");
          assert.equal(nativeWorker["MOM_SYCL_COMPILER"], "acpp-cuda");
          const nativePath = worker["MOM_NATIVE_PATH"];
          assert.ok(nativePath);
          assert.doesNotMatch(nativePath, /dpcpp/);
        }
        for (const sm of modernSms) {
          const selected = policy.selection(algo, "nvidia", platform, sm);
          const worker = policy.workerEnv(algo, {
            MOM_GPU_BACKEND: "nvidia",
            MOM_NVIDIA_COMPUTE_CAPABILITY: String(sm),
            MOM_NATIVE_DIR: root,
          }, platform);
          assert.ok(selected);
          assert.equal(selected.key, baseline.key);
          assert.equal(selected.backend, baseline.backend);
          assert.equal(selected.addon, baseline.addon);
          assert.equal(worker["MOM_SYCL_COMPILER"], baseline.key);
        }
      }
    }
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
  assert.equal(algorithms.length > 0, true);
  assert.equal(calls.length, 0);
});

/** @typedef {{nativePath: string, selector: string, cudaVisible: string, hipVisible: string, gpuIndex: string, cudaOrder: string, acppMask: string}} LauncherFields */
/** @typedef {{status: number | null, stdout: string, stderr: string, error: Error | undefined}} LauncherRun */
/** @typedef {{root: string, bin: string, launcher: string}} LinuxLauncherFixture */
/** @typedef {{line: string, expression: string}} WindowsBootstrap */

const linuxLauncherSourcePath = path.join(
  __dirname, "..", ".github", "workflows", "scripts", "package-linux-combined.sh",
);
const windowsLauncherSourcePath = path.join(
  __dirname, "..", ".github", "workflows", "scripts", "package-windows.ps1",
);

/** @returns {string} */
function extractLinuxLauncher() {
  const source = fs.readFileSync(linuxLauncherSourcePath, "utf8");
  const match = source.match(/cat >"\$package_dir\/mom" <<'EOF'\r?\n([\s\S]*?)\r?\nEOF/);
  if (!match || match[1] === undefined) {
    throw new Error("Linux release launcher heredoc was not found");
  }
  return match[1];
}

/** @returns {string} */
function extractWindowsLauncher() {
  const source = fs.readFileSync(windowsLauncherSourcePath, "utf8");
  const hereString = source.match(
    /@'\r?\n([\s\S]*?)\r?\n'@ \| Set-Content -Encoding ascii "\$packageDir\/mom\.cmd"/,
  );
  if (!hereString || hereString[1] === undefined) {
    throw new Error("Windows mom.cmd here-string was not found");
  }
  return hereString[1];
}

/** @returns {WindowsBootstrap} */
function extractWindowsBootstrap() {
  const line = extractWindowsLauncher().split(/\r?\n/).find((candidate) =>
    candidate.includes("MOM_VENDOR_RUNTIME") && candidate.includes(' -e "'));
  if (!line) {throw new Error("Windows NVIDIA -e launcher line was not found");}
  const expression = line.match(/-e "([^"]+)"/);
  if (!expression || expression[1] === undefined) {
    throw new Error("Windows NVIDIA -e expression was not found");
  }
  return {line, expression: expression[1]};
}

/** @param {string} root @returns {LinuxLauncherFixture} */
function createLinuxLauncherFixture(root) {
  const bin = path.join(root, "bin");
  const packageRoot = path.join(root, "package");
  fs.mkdirSync(bin, {recursive: true});
  fs.mkdirSync(packageRoot, {recursive: true});
  const fakeMomBin = `#!/usr/bin/env sh
printf 'MOM_NATIVE_PATH=%s\\n' "\${MOM_NATIVE_PATH-}"
printf 'ONEAPI_DEVICE_SELECTOR=%s\\n' "\${ONEAPI_DEVICE_SELECTOR-}"
printf 'CUDA_VISIBLE_DEVICES=%s\\n' "\${CUDA_VISIBLE_DEVICES-}"
printf 'HIP_VISIBLE_DEVICES=%s\\n' "\${HIP_VISIBLE_DEVICES-}"
printf 'MOM_GPU_INDEX=%s\\n' "\${MOM_GPU_INDEX-}"
printf 'CUDA_DEVICE_ORDER=%s\\n' "\${CUDA_DEVICE_ORDER-}"
printf 'ACPP_VISIBILITY_MASK=%s\\n' "\${ACPP_VISIBILITY_MASK-}"
`;
  const fakeNvidiaSmi = `#!/usr/bin/env sh
if [ -n "\${FAKE_SMI_DELAY:-}" ]; then sleep "$FAKE_SMI_DELAY"; fi
if [ "\${FAKE_SMI_OUTPUT+x}" = x ]; then printf '%s' "$FAKE_SMI_OUTPUT"; fi
exit "\${FAKE_SMI_STATUS:-0}"
`;
  const momBin = path.join(packageRoot, "mom-bin");
  const nvidiaSmi = path.join(bin, "nvidia-smi");
  fs.writeFileSync(path.join(packageRoot, "mom"), extractLinuxLauncher(), {mode: 0o755});
  fs.writeFileSync(momBin, fakeMomBin, {mode: 0o755});
  fs.writeFileSync(nvidiaSmi, fakeNvidiaSmi, {mode: 0o755});
  fs.chmodSync(path.join(packageRoot, "mom"), 0o755);
  fs.chmodSync(momBin, 0o755);
  fs.chmodSync(nvidiaSmi, 0o755);
  return {root, bin, launcher: path.join(packageRoot, "mom")};
}

/** @param {LinuxLauncherFixture} fixture @param {Record<string, string | undefined>} overrides
 * @returns {LauncherRun} */
function runLinuxLauncher(fixture, overrides = {}) {
  const env = {...process.env};
  for (const key of [
    "MOM_GPU_BACKEND", "MOM_GPU_INDEX", "MOM_NVIDIA_COMPUTE_CAPABILITY", "MOM_NATIVE_PATH",
    "MOM_NATIVE_PATH_LAUNCHER_DEFAULT", "MOM_RELEASE_RUNTIME_KEY", "ONEAPI_DEVICE_SELECTOR",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "ACPP_VISIBILITY_MASK", "FAKE_SMI_OUTPUT", "FAKE_SMI_STATUS",
    "FAKE_SMI_DELAY",
  ]) {
    delete env[key];
  }
  env["PATH"] = `${fixture.bin}${path.delimiter}${process.env["PATH"] || ""}`;
  env["MOM_GPU_BACKEND"] = "nvidia";
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      delete env[key];
    } else {
      env[key] = value;
    }
  }
  const result = childProcess.spawnSync(fixture.launcher, [], {
    cwd: fixture.root, encoding: "utf8", env, timeout: 10000,
  });
  return {
    status: result.status,
    stdout: typeof result.stdout === "string" ? result.stdout : String(result.stdout || ""),
    stderr: typeof result.stderr === "string" ? result.stderr : String(result.stderr || ""),
    error: result.error,
  };
}

/** @param {LauncherRun} result @returns {LauncherFields} */
function launcherFields(result) {
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.equal(result.error, undefined);
  /** @type {Map<string, string>} */
  const fields = new Map();
  for (const line of result.stdout.trimEnd().split(/\r?\n/)) {
    const separator = line.indexOf("=");
    assert.ok(separator > 0, `Unexpected launcher output line: ${line}`);
    fields.set(line.slice(0, separator), line.slice(separator + 1));
  }
  assert.deepEqual([...fields.keys()], [
    "MOM_NATIVE_PATH", "ONEAPI_DEVICE_SELECTOR", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "MOM_GPU_INDEX", "CUDA_DEVICE_ORDER", "ACPP_VISIBILITY_MASK",
  ]);
  return {
    nativePath: fields.get("MOM_NATIVE_PATH") || "",
    selector: fields.get("ONEAPI_DEVICE_SELECTOR") || "",
    cudaVisible: fields.get("CUDA_VISIBLE_DEVICES") || "",
    hipVisible: fields.get("HIP_VISIBLE_DEVICES") || "",
    gpuIndex: fields.get("MOM_GPU_INDEX") || "",
    cudaOrder: fields.get("CUDA_DEVICE_ORDER") || "",
    acppMask: fields.get("ACPP_VISIBILITY_MASK") || "",
  };
}

/** @param {LauncherFields} fields @param {string} addon @param {string} label */
function assertLauncherAddon(fields, addon, label) {
  assert.match(fields.nativePath, new RegExp(`[\\/]${addon}[\\/]mom\\.node$`), label);
}

/** @param {string} expression @param {Record<string, string | undefined>} overrides @returns {LauncherRun} */
function runWindowsBootstrapExpression(expression, overrides) {
  /** @type {NodeJS.ProcessEnv} */
  const env = {...process.env,
    MOM_DIR: `${path.resolve(__dirname, "..")}${path.sep}`,
    MOM_GPU_BACKEND: "nvidia",
  };
  for (const name of ["MOM_GPU_INDEX", "MOM_NVIDIA_COMPUTE_CAPABILITY", "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "FAKE_SMI_OUTPUT", "FAKE_SMI_STATUS", "FAKE_SMI_ERROR"]) {
    delete env[name];
  }
  for (const [name, value] of Object.entries(overrides)) {
    if (value === undefined) {delete env[name];} else {env[name] = value;}
  }
  // Execute the real emitted expression with only its hardware query boundary replaced.
  const queryFixture = "require('node:child_process').spawnSync=()=>({" +
    "status:Number(process.env.FAKE_SMI_STATUS||0),stdout:process.env.FAKE_SMI_OUTPUT||''," +
    "error:process.env.FAKE_SMI_ERROR?{code:process.env.FAKE_SMI_ERROR}:undefined});";
  const result = childProcess.spawnSync(process.execPath, ["-e", queryFixture + expression], {
    encoding: "utf8", env, timeout: 10000,
  });
  return {
    status: result.status,
    stdout: typeof result.stdout === "string" ? result.stdout : String(result.stdout || ""),
    stderr: typeof result.stderr === "string" ? result.stderr : String(result.stderr || ""),
    error: result.error,
  };
}

/** @param {LauncherRun} result @returns {Record<string, string>} */
function bootstrapFields(result) {
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
  return Object.fromEntries(result.stdout.trimEnd().split(/\r?\n/).map((line) => {
    const separator = line.indexOf("=");
    assert.ok(separator > 0, `Unexpected bootstrap assignment: ${line}`);
    return [line.slice(0, separator), line.slice(separator + 1)];
  }));
}

test("Linux release launcher follows compiler-policy capability parsing", {
  skip: process.platform === "win32" ? "Linux shell launcher requires POSIX /bin/sh" : false,
}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvidia-launcher-"));
  try {
    const fixture = createLinuxLauncherFixture(root);
    /** @type {Array<[string, string]>} */
    const legacyHints = [
      ["5.0", "acpp-cuda"], ["50", "acpp-cuda"], ["6.0", "acpp-cuda"], ["60", "acpp-cuda"],
      ["6.1", "acpp-cuda"], ["61", "acpp-cuda"], ["7", "acpp-cuda"], ["7.0", "acpp-cuda"],
      ["70", "acpp-cuda"], ["7.5", "acpp-cuda"], ["75", "acpp-cuda"],
      ["8.0", "dpcpp"], ["80", "dpcpp"], ["12.0", "dpcpp"], ["120", "dpcpp"],
    ];
    for (const [hint, addon] of legacyHints) {
      const fields = launcherFields(runLinuxLauncher(fixture, {
        MOM_NVIDIA_COMPUTE_CAPABILITY: hint,
      }));
      assertLauncherAddon(fields, addon, `capability ${hint}`);
      if (addon === "acpp-cuda") {
        assert.equal(fields.selector, "", hint);
        assert.equal(fields.acppMask, "cuda", hint);
      } else {
        assert.equal(fields.selector, "cuda:*", hint);
        assert.equal(fields.acppMask, "", hint);
      }
    }

    const malformed = [
      "0", "7.00", "n/a", "8junk", "9007199254740992", String.raw`8\.0`, String.raw`\x38.0`,
    ];
    for (const hint of malformed) {
      const result = runLinuxLauncher(fixture, {
        MOM_NVIDIA_COMPUTE_CAPABILITY: hint,
      });
      assert.equal(result.error, undefined);
      assert.equal(result.status, 2, hint);
      assert.match(result.stderr, /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
      assert.equal(result.stdout, "");
    }

    const modernStdoutFailure = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_STATUS: "1",
    }));
    assertLauncherAddon(modernStdoutFailure, "acpp-cuda", "failed query with modern stdout");

    for (const rows of ["\n12.0\n", "12.0\n\n", "N/A\n12.0\n", "12.0\nN/A\n"]) {
      const fields = launcherFields(runLinuxLauncher(fixture, {FAKE_SMI_OUTPUT: rows}));
      assertLauncherAddon(fields, "acpp-cuda", `unknown rows ${JSON.stringify(rows)}`);
    }

    const selectedKnown = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "7.0\n12.0\n", MOM_GPU_INDEX: "1",
    }));
    assertLauncherAddon(selectedKnown, "acpp-cuda", "mixed physical rows remain conservative");
    assert.equal(selectedKnown.cudaVisible, "1");
    assert.equal(selectedKnown.gpuIndex, "0");

    const selectedMissing = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "12.0\n", MOM_GPU_INDEX: "1",
    }));
    assertLauncherAddon(selectedMissing, "dpcpp", "logical index is not a physical-row lookup");
    assert.equal(selectedMissing.cudaVisible, "1");
    assert.equal(selectedMissing.gpuIndex, "0");
    assert.equal(selectedMissing.selector, "cuda:0");

    const selectedPadded = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "7.0\n12.0\n", MOM_GPU_INDEX: "0001",
    }));
    assertLauncherAddon(selectedPadded, "acpp-cuda", "zero-padded logical index");
    assert.equal(selectedPadded.cudaVisible, "1");
    assert.equal(selectedPadded.gpuIndex, "0");

    const selectedEight = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "7.0\n12.0\n", MOM_GPU_INDEX: "08",
    }));
    assertLauncherAddon(selectedEight, "acpp-cuda", "zero-padded missing index");
    assert.equal(selectedEight.cudaVisible, "8");
    assert.equal(selectedEight.gpuIndex, "0");

    const delayed = launcherFields(runLinuxLauncher(fixture, {
      FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_DELAY: "3",
    }));
    assertLauncherAddon(delayed, "acpp-cuda", "two-second query timeout");

    const explicitAddon = runLinuxLauncher(fixture, {
      MOM_NVIDIA_COMPUTE_CAPABILITY: "120",
      MOM_NATIVE_PATH: "/explicit/mom.node",
    });
    const explicitFields = launcherFields(explicitAddon);
    assert.equal(explicitFields.nativePath, "/explicit/mom.node");
    assert.equal(explicitFields.selector, "cuda:*");
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("Linux emitted launcher preserves inherited vendor masks and isolates indices once", {
  skip: process.platform === "win32" ? "Linux shell launcher requires POSIX /bin/sh" : false,
}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-vendor-launcher-"));
  try {
    const fixture = createLinuxLauncherFixture(root);
    for (const gpu of ["nvidia", "amd"]) {
      const visible = gpu === "nvidia" ? "CUDA_VISIBLE_DEVICES" : "HIP_VISIBLE_DEVICES";
      for (const rows of ["7.0\n12.0\n", "12.0\n8.0\n"]) {
        const base = {MOM_GPU_BACKEND: gpu, FAKE_SMI_OUTPUT: rows};
        const isolated = launcherFields(runLinuxLauncher(fixture, {...base, MOM_GPU_INDEX: "2"}));
        assert.equal(isolated.gpuIndex, "0");
        assert.equal(gpu === "nvidia" ? isolated.cudaVisible : isolated.hipVisible, "2");
        if (gpu === "nvidia" && !rows.includes("7.0")) {
          assert.equal(isolated.selector, "cuda:0");
        }
        for (const mask of ["", "-1", "1,0", "0,2,-1,1", "999,0", "GPU-a,GPU-b"]) {
          const unchanged = launcherFields(runLinuxLauncher(fixture, {...base, [visible]: mask}));
          assert.equal(gpu === "nvidia" ? unchanged.cudaVisible : unchanged.hipVisible, mask);
          const rejected = runLinuxLauncher(fixture, {...base, [visible]: mask, MOM_GPU_INDEX: "0"});
          assert.equal(rejected.error, undefined);
          assert.equal(rejected.status, 2, mask);
          assert.match(rejected.stderr, /cannot safely select within inherited/);
          assert.equal(rejected.stdout, "");
        }
        for (const mask of ["2", "GPU-1234", "MIG-GPU-1234/1/2", "invalid", "9999"]) {
          const preserved = launcherFields(runLinuxLauncher(fixture, {
            ...base, [visible]: mask, MOM_GPU_INDEX: "0000", CUDA_DEVICE_ORDER: "PCI_BUS_ID",
          }));
          assert.equal(gpu === "nvidia" ? preserved.cudaVisible : preserved.hipVisible, mask);
          assert.equal(preserved.gpuIndex, "0");
          assert.equal(preserved.cudaOrder, "PCI_BUS_ID");
          const outOfVisibleRange = runLinuxLauncher(fixture, {
            ...base, [visible]: mask, MOM_GPU_INDEX: "1",
          });
          assert.equal(outOfVisibleRange.error, undefined);
          assert.equal(outOfVisibleRange.status, 2);
          assert.equal(outOfVisibleRange.stdout, "");
        }
      }
    }
    for (const mask of ["", "-1", "0,1"]) {
      const alias = runLinuxLauncher(fixture, {
        MOM_GPU_BACKEND: "amd", CUDA_VISIBLE_DEVICES: mask, MOM_GPU_INDEX: "0",
      });
      assert.equal(alias.error, undefined);
      assert.equal(alias.status, 2);
      assert.match(alias.stderr, /inherited CUDA_VISIBLE_DEVICES/);
    }
    const aliasPreserved = launcherFields(runLinuxLauncher(fixture, {
      MOM_GPU_BACKEND: "amd", CUDA_VISIBLE_DEVICES: "GPU-1234", MOM_GPU_INDEX: "0",
    }));
    assert.equal(aliasPreserved.cudaVisible, "GPU-1234");
    assert.equal(aliasPreserved.hipVisible, "");
    for (const index of ["-1", "bad", "1024", "9".repeat(400)]) {
      const invalid = runLinuxLauncher(fixture, {MOM_GPU_INDEX: index});
      assert.equal(invalid.error, undefined);
      assert.equal(invalid.status, 2, index);
      assert.match(invalid.stderr, /MOM_GPU_INDEX/);
      assert.equal(invalid.stdout, "");
    }
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test("Windows release launcher evaluates the exact compiler-policy expression", () => {
  const {expression} = extractWindowsBootstrap();

  /** @type {Array<[string, string, string]>} */
  const pairs = [
    ["7", "7.0", "acpp-cuda"], ["7.0", "70", "acpp-cuda"],
    ["6.1", "61", "acpp-cuda"], ["8.0", "80", "dpcpp"],
    ["12.0", "120", "dpcpp"],
  ];
  for (const [first, second, expectedKey] of pairs) {
    const firstResult = bootstrapFields(runWindowsBootstrapExpression(expression, {
      MOM_NVIDIA_COMPUTE_CAPABILITY: first,
    }));
    const secondResult = bootstrapFields(runWindowsBootstrapExpression(expression, {
      MOM_NVIDIA_COMPUTE_CAPABILITY: second,
    }));
    assert.equal(firstResult["MOM_VENDOR_RUNTIME"], expectedKey, first);
    assert.deepEqual(firstResult, secondResult, `${first}/${second}`);
  }

  for (const capability of [
    "0", "7.00", "n/a", "8junk", "9007199254740992", String.raw`8\.0`, String.raw`\x38.0`,
  ]) {
    const result = runWindowsBootstrapExpression(expression, {MOM_NVIDIA_COMPUTE_CAPABILITY: capability});
    assert.equal(result.error, undefined, capability);
    assert.notEqual(result.status, 0, capability);
    assert.match(`${result.stdout}\n${result.stderr}`, /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
  }

  for (const rows of ["7.0\n12.0\n", "12.0\n7.0\n", "12.0\n8.0\n", "12.0\n"]) {
    for (const order of ["FASTEST_FIRST", "PCI_BUS_ID"]) {
      const fields = bootstrapFields(runWindowsBootstrapExpression(expression, {
        MOM_GPU_INDEX: "0001", FAKE_SMI_OUTPUT: rows, CUDA_DEVICE_ORDER: order,
      }));
      assert.equal(fields["MOM_VENDOR_RUNTIME"], rows.includes("7.0") ? "acpp-cuda" : "dpcpp");
      assert.equal(fields["CUDA_VISIBLE_DEVICES"], "1");
      assert.equal(fields["MOM_GPU_INDEX"], "0");
      assert.equal(fields["CUDA_DEVICE_ORDER"], undefined);
    }
  }
  for (const fixture of [
    {FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_STATUS: "1"},
    {FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_ERROR: "ETIMEDOUT"},
    {FAKE_SMI_OUTPUT: "12.0\n\n"},
    {FAKE_SMI_OUTPUT: "N/A\n12.0\n"},
    {FAKE_SMI_OUTPUT: ""},
  ]) {
    const fields = bootstrapFields(runWindowsBootstrapExpression(expression, fixture));
    assert.equal(fields["MOM_VENDOR_RUNTIME"], "acpp-cuda");
  }
  for (const gpu of ["nvidia", "amd"]) {
    const visible = gpu === "nvidia" ? "CUDA_VISIBLE_DEVICES" : "HIP_VISIBLE_DEVICES";
    for (const mask of ["", "-1", "1,0", "0,2,-1,1", "999,0"]) {
      const base = {MOM_GPU_BACKEND: gpu, [visible]: mask, FAKE_SMI_OUTPUT: "12.0\n7.0\n"};
      const preserved = bootstrapFields(runWindowsBootstrapExpression(expression, base));
      assert.equal(preserved[visible], undefined);
      const rejected = runWindowsBootstrapExpression(expression, {...base, MOM_GPU_INDEX: "0"});
      assert.equal(rejected.error, undefined);
      assert.equal(rejected.status, 2);
      assert.match(rejected.stderr, /cannot safely select within inherited/);
      assert.equal(rejected.stdout, "MOM_VENDOR_RUNTIME=error\n");
    }
    for (const mask of ["2", "GPU-1234", "MIG-GPU-1234/1/2", "invalid", "9999"]) {
      const base = {MOM_GPU_BACKEND: gpu, [visible]: mask, MOM_GPU_INDEX: "0"};
      const fields = bootstrapFields(runWindowsBootstrapExpression(expression, base));
      assert.equal(fields[visible], undefined);
      assert.equal(fields["MOM_GPU_INDEX"], "0");
      const rejected = runWindowsBootstrapExpression(expression, {...base, MOM_GPU_INDEX: "1"});
      assert.equal(rejected.error, undefined);
      assert.equal(rejected.status, 2);
    }
  }
  for (const mask of ["", "-1", "0,1", "0,2,-1,1"]) {
    const rejected = runWindowsBootstrapExpression(expression, {
      MOM_GPU_BACKEND: "amd", CUDA_VISIBLE_DEVICES: mask, MOM_GPU_INDEX: "0",
    });
    assert.equal(rejected.error, undefined);
    assert.equal(rejected.status, 2);
    assert.match(rejected.stderr, /inherited CUDA_VISIBLE_DEVICES/);
  }
  const hipAlias = bootstrapFields(runWindowsBootstrapExpression(expression, {
    MOM_GPU_BACKEND: "amd", CUDA_VISIBLE_DEVICES: "GPU-1234", MOM_GPU_INDEX: "0",
  }));
  assert.equal(hipAlias["HIP_VISIBLE_DEVICES"], undefined);
  assert.equal(hipAlias["CUDA_VISIBLE_DEVICES"], undefined);
  assert.equal(hipAlias["MOM_GPU_INDEX"], "0");
});

test("Windows release launcher routes workers through cmd.exe in a path with spaces", {
  skip: process.platform !== "win32",
}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvidia-windows-"));
  const packageRoot = path.join(root, "package with spaces & (punctuation)");
  const launcher = path.join(packageRoot, "mom.cmd");
  try {
    fs.mkdirSync(path.join(packageRoot, "helper"), {recursive: true});
    for (const file of ["compiler-policy.js", "gpu-tuning.js", "helper/hash.js", "GPU-CONFIG.md"]) {
      fs.copyFileSync(path.join(__dirname, "..", file), path.join(packageRoot, file));
    }
    fs.copyFileSync(process.execPath, path.join(packageRoot, "mom-node.exe"));
    fs.writeFileSync(launcher, extractWindowsLauncher().replace(/\r?\n/g, "\r\n"));
    for (const family of ["oneapi", "dpcpp"]) {
      fs.mkdirSync(path.join(packageRoot, "libs", family), {recursive: true});
      fs.writeFileSync(path.join(packageRoot, "libs", family, "sycl-jit.dll"), family);
    }
    // A CPU-only bundle reports the real launcher's environment without loading any GPU library.
    fs.writeFileSync(path.join(packageRoot, "mom.bundle.cjs"),
      "require('node:fs').writeFileSync(require('node:path').join(__dirname,'fixture-launched.txt'),'launched');" +
      "const fs=require('node:fs'),path=require('node:path');" +
      "fs.writeFileSync(path.join(__dirname,'fixture-jit.txt'),(process.env.Path||'')" +
      ".split(path.delimiter).map(dir=>path.join(dir,'sycl-jit.dll')).find(fs.existsSync)||'');" +
      "require('node:fs').writeFileSync(1,JSON.stringify({args:process.argv.slice(2)," +
      "nativePath:process.env.MOM_NATIVE_PATH,runtimeDir:process.env.MOM_RUNTIME_DIR," +
      "selector:process.env.ONEAPI_DEVICE_SELECTOR||''," +
      "urAdapters:process.env.UR_ADAPTERS_FORCE_LOAD||''," +
      "cudaVisible:process.env.CUDA_VISIBLE_DEVICES||''," +
      "hipVisible:process.env.HIP_VISIBLE_DEVICES||'',gpuIndex:process.env.MOM_GPU_INDEX||''," +
      "cudaOrder:process.env.CUDA_DEVICE_ORDER||''," +
      "acppMask:process.env.ACPP_VISIBILITY_MASK||''}));\n");
    const queryFixture = path.join(packageRoot, "query-fixture.cjs");
    fs.writeFileSync(queryFixture,
      "const cp=require('node:child_process');const original=cp.spawnSync;" +
      "cp.spawnSync=(command,...args)=>command==='nvidia-smi'?{" +
      "status:Number(process.env.FAKE_SMI_STATUS||0),stdout:process.env.FAKE_SMI_OUTPUT||''," +
      "error:process.env.FAKE_SMI_ERROR?{code:process.env.FAKE_SMI_ERROR}:undefined}" +
      ":original(command,...args);\n");
    /** @type {NodeJS.ProcessEnv} */
    const env = {...process.env, MOM_GPU_BACKEND: "nvidia", VSCMD_VER: "fixture"};
    env["Path"] = [path.join(packageRoot, "libs", "dpcpp"), env["Path"] || env["PATH"]]
      .filter(Boolean).join(path.delimiter);
    delete env["PATH"];
    for (const key of ["MOM_GPU_INDEX", "MOM_NATIVE_PATH", "MOM_NATIVE_PATH_LAUNCHER_DEFAULT",
      "MOM_RUNTIME_DIR", "ONEAPI_DEVICE_SELECTOR", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
      "CUDA_DEVICE_ORDER", "MOM_NVIDIA_COMPUTE_CAPABILITY", "MOM_VENDOR_RUNTIME",
      "FAKE_SMI_OUTPUT", "FAKE_SMI_STATUS", "FAKE_SMI_ERROR", "ACPP_VISIBILITY_MASK",
      "UR_ADAPTERS_FORCE_LOAD", "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"]) {
      delete env[key];
    }
    // NODE_OPTIONS consumes backslashes inside quoted values, so use Windows-compatible slashes.
    env["NODE_OPTIONS"] = `--require "${queryFixture.replaceAll("\\", "/")}"`;
    /** @param {Record<string, string | undefined>} overrides */
    function launch(overrides) {
      fs.rmSync(path.join(packageRoot, "fixture-launched.txt"), {force: true});
      fs.rmSync(path.join(packageRoot, "fixture-jit.txt"), {force: true});
      return childProcess.spawnSync(process.env["ComSpec"] || "cmd.exe",
        windowsCmdArgs([launcher, "algorithms", "two words", "paren(value)"]), {
          cwd: root, encoding: "utf8", timeout: 30000, windowsHide: true,
          windowsVerbatimArguments: true, env: {...env, ...overrides},
        });
    }
    // A false vendor branch must still parse; vendor paths must keep isolated indexes and argv.
    const cpu = launch({MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu"});
    assert.equal(cpu.error, undefined);
    assert.equal(cpu.status, 0, cpu.stderr);
    assert.equal(JSON.parse(cpu.stdout).nativePath, path.join(packageRoot, "libs", "dpcpp-opencl", "mom.node"));
    assert.equal(JSON.parse(cpu.stdout).selector, "opencl:cpu");
    assert.equal(JSON.parse(cpu.stdout).urAdapters,
      '"' + path.join(packageRoot, "libs", "dpcpp-opencl", "ur_adapter_opencl.dll") + '"');
    assert.deepEqual(JSON.parse(cpu.stdout).args, ["algorithms", "two words", "paren(value)"]);
    assert.equal(fs.readFileSync(path.join(packageRoot, "fixture-jit.txt"), "utf8"),
      path.join(packageRoot, "libs", "oneapi", "sycl-jit.dll"));
    /** @type {Array<[string, string, string, string]>} */
    const indexed = [["nvidia", "2", "dpcpp", "cudaVisible"], ["amd", "3", "acpp-hip", "hipVisible"]];
    for (const [gpu, index, compiler, visible] of indexed) {
      const result = launch({MOM_GPU_BACKEND: gpu, MOM_GPU_INDEX: index, MOM_NVIDIA_COMPUTE_CAPABILITY: "90"});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stderr);
      const fields = JSON.parse(result.stdout);
      assert.equal(fields.nativePath, path.join(packageRoot, "libs", compiler, "mom.node"));
      assert.equal(fields.runtimeDir, path.join(packageRoot, "libs", compiler));
      assert.equal(fields[visible], index);
      assert.equal(fields.gpuIndex, "0");
      assert.equal(fields.urAdapters, compiler === "dpcpp"
        ? '"' + path.join(packageRoot, "libs", "dpcpp", "ur_adapter_cuda.dll") + '"' : "");
      assert.deepEqual(fields.args, ["algorithms", "two words", "paren(value)"]);
      assert.equal(fs.readFileSync(path.join(packageRoot, "fixture-jit.txt"), "utf8"),
        path.join(packageRoot, "libs", "dpcpp", "sycl-jit.dll"),
        "vendor launcher PATH must remain unchanged");
    }
    const intel = launch({MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "1"});
    assert.equal(intel.error, undefined);
    assert.equal(intel.status, 0, intel.stderr);
    assert.deepEqual(JSON.parse(intel.stdout), {
      args: ["algorithms", "two words", "paren(value)"],
      nativePath: path.join(packageRoot, "libs", "oneapi", "mom.node"),
      runtimeDir: path.join(packageRoot, "libs", "oneapi"), selector: "level_zero:gpu",
      cudaVisible: "", hipVisible: "", gpuIndex: "1", cudaOrder: "", acppMask: "",
      urAdapters: ["ur_adapter_level_zero_v2.dll", "ur_adapter_opencl.dll"]
        .map(name => '"' + path.join(packageRoot, "libs", "oneapi", name) + '"').join(","),
    });
    const urOverride = '"C:\\external & (adapters)\\custom.dll"';
    for (const gpu of ["opencl", "nvidia", "intel"]) {
      const result = launch({MOM_GPU_BACKEND: gpu, MOM_OPENCL_DEVICE_TYPE: "cpu",
        MOM_NVIDIA_COMPUTE_CAPABILITY: "90", UR_ADAPTERS_FORCE_LOAD: urOverride});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(JSON.parse(result.stdout).urAdapters, urOverride);
    }
    const callerRuntime = path.join(packageRoot, "caller runtime");
    fs.mkdirSync(callerRuntime);
    fs.writeFileSync(path.join(callerRuntime, "sycl-jit.dll"), "caller");
    const custom = launch({MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu",
      MOM_RUNTIME_DIR: callerRuntime, MOM_NATIVE_PATH: "caller-addon", UR_ADAPTERS_FORCE_LOAD: urOverride});
    assert.equal(custom.error, undefined);
    assert.equal(custom.status, 0, custom.stderr);
    assert.equal(JSON.parse(custom.stdout).nativePath, "caller-addon");
    assert.equal(JSON.parse(custom.stdout).runtimeDir, callerRuntime);
    assert.equal(JSON.parse(custom.stdout).urAdapters, urOverride);
    assert.equal(fs.readFileSync(path.join(packageRoot, "fixture-jit.txt"), "utf8"),
      path.join(callerRuntime, "sycl-jit.dll"));
    for (const gpu of ["nvidia", "amd"]) {
      for (const index of ["bad", "-1"]) {
        const rejected = launch({MOM_GPU_BACKEND: gpu, MOM_GPU_INDEX: index});
        assert.equal(rejected.error, undefined);
        assert.equal(rejected.status, 2, rejected.stderr);
        assert.match(rejected.stdout, /MOM_GPU_INDEX must be a non-negative integer/);
        assert.equal(fs.existsSync(path.join(packageRoot, "fixture-launched.txt")), false);
      }
    }
    /** @type {Array<[string, string]>} */
    const variants = [["7", "acpp-cuda"], ["7.0", "acpp-cuda"], ["70", "acpp-cuda"],
      ["61", "acpp-cuda"], ["8.0", "dpcpp"], ["80", "dpcpp"], ["120", "dpcpp"]];
    for (const [capability, compiler] of variants) {
      const result = launch({MOM_NVIDIA_COMPUTE_CAPABILITY: capability});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stderr);
      assert.deepEqual(JSON.parse(result.stdout), {
        args: ["algorithms", "two words", "paren(value)"],
        nativePath: path.join(packageRoot, "libs", compiler, "mom.node"),
        runtimeDir: path.join(packageRoot, "libs", compiler),
        selector: compiler === "dpcpp" ? "cuda:gpu" : "",
        cudaVisible: "", hipVisible: "", gpuIndex: "", cudaOrder: "",
        acppMask: compiler === "acpp-cuda" ? "cuda" : "",
        urAdapters: compiler === "dpcpp"
          ? '"' + path.join(packageRoot, "libs", "dpcpp", "ur_adapter_cuda.dll") + '"' : "",
      }, capability);
    }
    for (const rows of ["12.0\n7.0\n", "7.0\n12.0\n", "12.0\n8.0\n", "12.0\n\n"]) {
      for (const order of ["FASTEST_FIRST", "PCI_BUS_ID"]) {
        const result = launch({FAKE_SMI_OUTPUT: rows, MOM_GPU_INDEX: "0001", CUDA_DEVICE_ORDER: order});
        assert.equal(result.error, undefined);
        assert.equal(result.status, 0, result.stderr);
        const fields = JSON.parse(result.stdout);
        const key = rows.includes("7.0") || rows.endsWith("\n\n") ? "acpp-cuda" : "dpcpp";
        assert.equal(fields.nativePath, path.join(packageRoot, "libs", key, "mom.node"));
        assert.equal(fields.cudaVisible, "1");
        assert.equal(fields.gpuIndex, "0");
        assert.equal(fields.cudaOrder, order);
        assert.equal(fields.selector, key === "dpcpp" ? "cuda:0" : "");
      }
    }
    for (const gpu of ["nvidia", "amd"]) {
      const visible = gpu === "nvidia" ? "CUDA_VISIBLE_DEVICES" : "HIP_VISIBLE_DEVICES";
      for (const mask of ["", "-1", "1,0", "0,2,-1,1", "999,0"]) {
        const base = {MOM_GPU_BACKEND: gpu, [visible]: mask, FAKE_SMI_OUTPUT: "12.0\n7.0\n"};
        const preserved = launch(base);
        assert.equal(preserved.error, undefined);
        assert.equal(preserved.status, 0, preserved.stderr);
        assert.equal(JSON.parse(preserved.stdout)[gpu === "nvidia" ? "cudaVisible" : "hipVisible"], mask);
        const rejected = launch({...base, MOM_GPU_INDEX: "0"});
        assert.equal(rejected.error, undefined);
        assert.equal(rejected.status, 2);
        assert.match(rejected.stderr, /cannot safely select within inherited/);
        assert.equal(rejected.stdout, "");
      }
      const single = launch({MOM_GPU_BACKEND: gpu, [visible]: "GPU-1234", MOM_GPU_INDEX: "0"});
      assert.equal(single.error, undefined);
      assert.equal(single.status, 0, single.stderr);
      assert.equal(JSON.parse(single.stdout)[gpu === "nvidia" ? "cudaVisible" : "hipVisible"], "GPU-1234");
      const outOfRange = launch({MOM_GPU_BACKEND: gpu, [visible]: "2", MOM_GPU_INDEX: "1"});
      assert.equal(outOfRange.error, undefined);
      assert.equal(outOfRange.status, 2);
      assert.equal(outOfRange.stdout, "");
    }
    for (const fixture of [
      {FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_STATUS: "1"},
      {FAKE_SMI_OUTPUT: "12.0\n", FAKE_SMI_ERROR: "ETIMEDOUT"},
    ]) {
      const result = launch(fixture);
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(JSON.parse(result.stdout).nativePath, path.join(packageRoot, "libs", "acpp-cuda", "mom.node"));
    }
    for (const mask of ["", "-1", "0,1"]) {
      const alias = launch({MOM_GPU_BACKEND: "amd", CUDA_VISIBLE_DEVICES: mask, MOM_GPU_INDEX: "0"});
      assert.equal(alias.error, undefined);
      assert.equal(alias.status, 2);
      assert.match(alias.stderr, /inherited CUDA_VISIBLE_DEVICES/);
      assert.equal(alias.stdout, "");
    }
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});
