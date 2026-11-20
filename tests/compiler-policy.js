"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {test} = require("node:test");
const {runInNewContext} = require("node:vm");
const policyModule = require("../compiler-policy");
const opts = require("../opts");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const gpuTuning = require("../gpu-tuning");
const helper = require("../helper");
const createJobApi = require("../miner/jobs");
const {normalizeAlgoName} = require("../miner/algorithms");
const {hexWithoutPrefix} = require("../miner/submission");
const {hashTests} = require("./vectors");
const {addDiscreteCases, addGenericCases, addIntegratedCases} = require("./common/gpu-test-matrix");
const {
  cloneForDiscreteGpu, cloneForIntelIntegrated, cloneForOpenclSycl, openclSyclEnv, requestedVendors,
} = require("./common/gpu-test-modes");
const {
  createAlgoParamsReportCache,
  isMissingGpuOutput,
  maxCapturedOutputBytes,
  parseDiscreteGpuDevices, parseGpuDevices,
  runNode, wrapWindowsCmd,
} = require("./common/miner-command");

const {readSyclSource, extractCppBlock} = require("./common/cpp-source");

/**
 * @template T
 * @param {import("node:test").TestContext} parent
 * @param {T[]} cases
 * @param {(item: T) => string} heading
 * @param {(group: import("node:test").TestContext, item: T) => Promise<void>} runCase
 */
async function runGroupedCases(parent, cases, heading, runCase) {
  // Keep the original case order and await each child before reusing fixture files.
  for (let first = 0; first < cases.length;) {
    const item = cases[first];
    assert.ok(item !== undefined);
    const name = heading(item);
    let last = first + 1;
    while (last < cases.length && last - first < 20) {
      const next = cases[last];
      assert.ok(next !== undefined);
      if (heading(next) !== name) {break;}
      ++last;
    }
    await parent.test(name, {concurrency: false}, async (group) => {
      for (const item of cases.slice(first, last)) {
        await runCase(group, item);
      }
    });
    first = last;
  }
}

test.describe("Compiler policy and GPU contracts", {concurrency: false}, () => {
  test.describe("C++ fixture source extraction", () => {
    test("extracts nested bodies without counting comment or literal braces", () => {
      const body = String.raw`int probe() {
  // } ignored comment
  /* { ignored block */
  const char* text = "}";
  const char* raw = R"body({ ignored raw })body";
  if (text[0] == '}') { return raw[0]; }
  return 0;
}`;
      assert.equal(extractCppBlock(body + "\n// Changed documentation\nint next() {}", "int probe("), body + "\n");
    });

    test("rejects missing or duplicate declaration anchors", () => {
      assert.throws(() => extractCppBlock("int other() {}", "int probe("));
      assert.throws(() => extractCppBlock("int probe() {}\nint probe() {}", "int probe("));
    });

    test("rejects an unfinished body rather than returning a partial fixture", () => {
      assert.throws(() => extractCppBlock("int probe() { if (true) { }", "int probe("));
    });

    test("matches anchor tokens regardless of spacing or line wrapping", () => {
      const source = "static bool selected(int m,\n                     int n) { return m < n; }\n";
      assert.equal(extractCppBlock(source, "static bool selected(int m, int n) {"), source);
      assert.throws(() => extractCppBlock(source, "static bool selectedx("));
    });

    test("keeps anchor indentation significant to tell nested members apart", () => {
      const source = "struct A {\n  void run() { inner(); }\n};\nvoid run() { outer(); }\n";
      assert.equal(extractCppBlock(source, "void run("), "void run() { outer(); }\n");
      assert.equal(extractCppBlock(source, "  void run("), "  void run() { inner(); }\n");
    });

    test("includes a template header and ignores anchors in comments but not raw-string sources", () => {
      const source = "template <typename T>\n// Purpose comment.\nT twice(T v) { return v + v; }\n" +
        "/*\nT twice(T v) {}\n*/\nconst char* kernel = R\"K(\nint device_probe() { return 1; }\n)K\";\n";
      assert.equal(extractCppBlock(source, "T twice("),
        "template <typename T>\n// Purpose comment.\nT twice(T v) { return v + v; }\n");
      assert.equal(extractCppBlock(source, "int device_probe("), "int device_probe() { return 1; }\n");
    });

    test("command line prints each requested block in order", () => {
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cpp-source-cli-"));
      try {
        const file = path.join(fixture, "source.cpp");
        fs.writeFileSync(file, "int a() { return 1; }\nint b() { return 2; }\n");
        const cli = path.join(__dirname, "common/cpp-source.js");
        const result = spawnSync(process.execPath, [cli, file, "int b(", "int a("], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "int b() { return 2; }\nint a() { return 1; }\n");
        assert.notEqual(spawnSync(process.execPath, [cli, file, "int c("], {encoding: "utf8"}).status, 0);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  });

  test.describe("GPU vector coverage and availability diagnostics", {concurrency: false}, () => {
    test("C29 keeps CPU coverage bounded and discrete coverage end to end", () => {
      const portable = hashTests.filter(({job, syclCpu}) => job.algo === "c29" && syclCpu);
      assert.equal(portable.length, 1);
      const portableVector = portable[0];
      assert.ok(portableVector);
      assert.equal(portableVector.portableOnly, true);
      assert.equal(portableVector.env?.["MOM_C29_TEST_EDGE"], "1");

      const discrete = hashTests.filter(
        ({gpu, job, portableOnly}) => gpu && job.algo === "c29" && !portableOnly
      );
      assert.deepEqual(discrete.map(({job}) => job.proofsize).sort(), [32, 42]);
      assert.ok(discrete.every(({env}) => !env?.["MOM_C29_TEST_EDGE"]));
    });

    test("GPU availability detection preserves actionable SYCL diagnostics", () => {
    /** @param {string} stderr */
      const result = (stderr) => ({code: 1, signal: null, error: null, stdout: "", stderr});
      assert.equal(isMissingGpuOutput(result("No device of requested type was found")), true);
      assert.equal(isMissingGpuOutput(result("No SYCL GPU device is available")), true);
      assert.equal(isMissingGpuOutput(result("SYCL GPU device unavailable")), true);
      assert.equal(isMissingGpuOutput(result("No GPUs detected")), true);
      assert.equal(isMissingGpuOutput(result("XelisHashV3 SYCL error on device submission")), false);
      assert.equal(isMissingGpuOutput(result("No GPU kernel was emitted after compiler error")), false);
    });
  });

  /** @typedef {{key: string, addon: string, backend: string, pearlhashProfile: {m: number, n: number, k: number, rank: number} | null}} CompilerSelection */
  /** @typedef {Record<string, string | undefined>} TestEnv */
  /** @typedef {NodeJS.ProcessEnv & {MOM_SYCL_COMPILER?: string | undefined, ACPP_VISIBILITY_MASK?: string | undefined, CUDA_VISIBLE_DEVICES?: string | undefined, HIP_VISIBLE_DEVICES?: string | undefined, MOM_NATIVE_PATH?: string | undefined, MOM_RUNTIME_DIR?: string | undefined, LD_LIBRARY_PATH?: string | undefined, PATH?: string | undefined, ONEAPI_DEVICE_SELECTOR?: string | undefined, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS?: string | undefined, MOM_NEXAPOW_NATIVE?: string | undefined, MOM_OCTOPUS_SYCL_NATIVE?: string | undefined, MOM_XELISHASHV3_SYCL_NATIVE?: string | undefined, MOM_WALAHASH_SYCL_NATIVE?: string | undefined}} CompilerEnv */

  /** @param {string} algo @param {string} gpu @param {NodeJS.Platform} [platform] @param {number | null} [nvidiaSm] @returns {CompilerSelection} */
  function selectPolicy(algo, gpu, platform = process.platform, nvidiaSm = null) {
    const selected = policyModule.selection(algo, gpu, platform, nvidiaSm);
    if (!selected) {throw new Error(`No compiler policy for ${algo}/${gpu}/${platform}`);}
    return selected;
  }

  /** @param {string} algo @param {NodeJS.ProcessEnv} [env] @param {NodeJS.Platform} [platform] @param {string} [requestedBackend] @returns {CompilerEnv} */
  function workerEnvironment(algo, env = process.env, platform = process.platform, requestedBackend = "auto") {
    return policyModule.workerEnv(algo, env, platform, requestedBackend);
  }

  const policy = {...policyModule, selection: selectPolicy, workerEnv: workerEnvironment};

  /** @param {NodeJS.ProcessEnv} [env] */
  function testJobApi(env = process.env) {
    const opt = opts.create_default_opts();
    const testProcess = Object.assign(Object.create(process), {env});
    return createJobApi({
      h: helper,
      opt,
      process: testProcess,
      compilerPolicy: policyModule,
      gpuTuning,
      hexWithoutPrefix,
      normalizeAlgoName,
      messageHandler: () => undefined,
      isExiting: () => false,
      getComputeCore: () => null,
      getLastJob: () => null,
      setLastJob: () => undefined,
    });
  }

  test.describe("Offline tooling, command output, and release launcher safety", {concurrency: false}, () => {
    test("npm tooling freezes exact dependencies in a maintained lockfile", () => {
      const root = path.join(__dirname, "..");
      const packageJson = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
      const npmrc = Object.fromEntries(fs.readFileSync(path.join(root, ".npmrc"), "utf8").split(/\r?\n/)
        .filter(line => line.includes("=")).map(line => line.split("=").map(part => part.trim())));
      const lock = JSON.parse(fs.readFileSync(path.join(root, "package-lock.json"), "utf8"));
      assert.equal(lock.lockfileVersion, 3); assert.deepEqual(lock.packages[""].devDependencies, packageJson.devDependencies);
      assert.equal(npmrc["package-lock"], "true");
      assert.equal(npmrc["save-exact"], "true");
      for (const version of Object.values(packageJson.devDependencies)) {
        assert.match(version, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
      }
    });

    test("miner command output capture retains bounded stdout and stderr tails", async () => {
      const stdoutMarker = "stdout-tail-marker";
      const stderrMarker = "stderr-tail-marker";
      // Emit the exact fixture bytes even when the runner replaces Node's stdio streams.
      const source = [
        `process.stdout.write("old-output\\n" + "x".repeat(${maxCapturedOutputBytes}) + ${JSON.stringify(stdoutMarker)}, () => {`,
        `  process.stderr.write("old-error\\n" + "x".repeat(${maxCapturedOutputBytes}) + ${JSON.stringify(stderrMarker)}, () => {});`,
        "});",
      ].join("");
      const result = await runNode(["-e", source], {timeoutMs: 30 * 1000});
      assert.equal(result.error, null);
      assert.equal(result.code, 0);
      assert.ok(Buffer.byteLength(result.stdout) <= maxCapturedOutputBytes);
      assert.ok(Buffer.byteLength(result.stderr) <= maxCapturedOutputBytes);
      assert.ok(result.stdout.endsWith(stdoutMarker));
      assert.ok(result.stderr.endsWith(stderrMarker));
      assert.doesNotMatch(result.stdout, /old-output/);
      assert.doesNotMatch(result.stderr, /old-error/);
    });

    test("miner command output capture bounds malformed UTF-8 without losing its tail", async () => {
      const marker = "malformed-output-tail-marker";
      const source = [
        `require("node:fs").writeFileSync(1, Buffer.alloc(${maxCapturedOutputBytes + 1024}, 0xff));`,
        `require("node:fs").writeFileSync(1, ${JSON.stringify(marker)});`,
      ].join("");
      const result = await runNode(["-e", source], {timeoutMs: 30 * 1000});
      assert.equal(result.error, null);
      assert.equal(result.code, 0);
      assert.ok(Buffer.byteLength(result.stdout) <= maxCapturedOutputBytes);
      assert.ok(result.stdout.endsWith(marker));
    });

    test("Windows cmd wrapper rejects percent-expansion arguments", () => {
      assert.throws(() => wrapWindowsCmd(["mom.cmd", "gpu%0"]), /cannot safely preserve/);
    });

    test("failed output buffer replays a bounded formatted tail", () => {
      const root = path.join(__dirname, "..");
      const source = [
        "for (let i = 0; i < 400; ++i) { console.error(\"debug %d %s\", i, \"x\".repeat(1024)); }",
        "console.error(\"output-buffer-tail-marker %s\", \"literal\");",
        "process.exitCode = 1;",
      ].join("");
      const result = spawnSync(process.execPath, [
        "--require", path.join(root, "tests/common/test-output-buffer.js"), "-e", source,
      ], {
        cwd: root,
        encoding: "utf8",
        env: {...process.env, NODE_TEST_FLUSH_BUFFERED_OUTPUT: "1"},
        maxBuffer: 2 * 1024 * 1024,
      });
      assert.equal(result.status, 1, result.error?.message);
      const header = "Suppressed debug output:\n";
      const headerIndex = result.stderr.indexOf(header);
      assert.ok(headerIndex >= 0);
      assert.match(result.stderr, /output-buffer-tail-marker literal/);
      assert.doesNotMatch(result.stderr, /debug 0/);
      assert.ok(result.stderr.length - headerIndex <= 256 * 1024 + header.length + 2);
    });

  });

  // Extract one top-level shell function by name, through its column-0 closing brace or paren.
  /** @param {string} source @param {string} name @returns {string} */
  function shellFunction(source, name) {
    const matches = [["{", "}"], ["(", ")"]].flatMap(([open, close]) => [...source.matchAll(
      new RegExp(`^${name}\\(\\)\\s*\\${open}\\n[\\s\\S]*?^\\${close}$`, "gm"))]);
    assert.equal(matches.length, 1, `Missing or ambiguous shell function: ${name}`);
    const [match] = matches;
    assert.ok(match);
    return match[0] + "\n";
  }

  // Run the real AdaptiveCpp staging function; only runtime bookkeeping and ELF closure are stubbed.
  /** @param {string} packager @returns {string} */
  function acppRuntimeProgram(packager) {
    return [
      "set -euo pipefail",
      'begin_runtime() { mkdir -p "$runtime_dest"; }',
      "copy_dependency_closure() { :; }",
      shellFunction(packager, "runtime_matches"),
      shellFunction(packager, "copy_acpp_runtime"),
    ].join("\n");
  }

  // Provide fixture LLVM 21 JIT tools at <root>/llvm-21/bin for a redirected staging program.
  /** @param {string} root */
  function acppLlvmFixture(root) {
    const bin = path.join(root, "llvm-21/bin");
    fs.mkdirSync(bin, {recursive: true});
    for (const tool of ["opt", "llc", "lld", "ld.lld"]) {
      fs.writeFileSync(path.join(bin, tool), "#!/bin/sh\nexit 0\n", {mode: 0o755});
    }
  }

  /** @param {string} directory @returns {NodeJS.ProcessEnv} */
  function packagerValidationEnv(directory) {
    const bin = path.join(directory, "bin");
    fs.mkdirSync(bin);
    // Early safety checks must work without either ELF packaging tool installed.
    for (const tool of ["bash", "realpath", "awk", "sha256sum", "grep"]) {
      const resolved = spawnSync("bash", ["-c", 'command -v "$1"', "fixture", tool], {encoding: "utf8"});
      assert.equal(resolved.status, 0, `Validation fixture requires ${tool}`);
      fs.symlinkSync(resolved.stdout.trim(), path.join(bin, tool));
    }
    const env = {...process.env, PATH: bin};
    assert.equal(spawnSync(path.join(bin, "bash"), ["-c", "! command -v patchelf && ! command -v readelf"],
      {env}).status, 0, "Fixture must exclude both packaging tools");
    return env;
  }

  test.describe("Linux runtime packaging and native SDK installers", {concurrency: false}, () => {
    test("Linux combined packager protects cleanup roots before build checks", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const script = path.join(root, "scripts/package-linux-combined.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-safety-test-"));
      const marker = path.join(temp, "release-combined", "marker");
      const archive = path.join(temp, "release-combined", "nested", "archive.tgz");
      fs.mkdirSync(path.dirname(marker), {recursive: true});
      fs.writeFileSync(marker, "keep");
      try {
        const result = spawnSync("bash", [script, "1.2.3", archive], {
          cwd: temp, encoding: "utf8", env: packagerValidationEnv(temp),
        });
        assert.equal(result.status, 2, result.stderr);
        assert.match(result.stderr, /Archive path must not be inside cleanup directory/);
        assert.doesNotMatch(result.stdout + result.stderr, /missing; run|docker image inspect/i);
        assert.equal(fs.readFileSync(marker, "utf8"), "keep");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Linux combined packager refuses a directory archive without removing it", {
      skip: process.platform === "win32",
    }, () => {
      const script = path.join(__dirname, "../scripts/package-linux-combined.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-archive-dir-"));
      const marker = path.join(temp, "published", "marker");
      fs.mkdirSync(path.dirname(marker));
      fs.writeFileSync(marker, "keep");
      try {
        const result = spawnSync("bash", [script, "1.2.3", "published"], {
          cwd: temp, encoding: "utf8", env: packagerValidationEnv(temp),
        });
        assert.equal(result.status, 1, result.stderr);
        assert.match(result.stderr, /Archive path must not be a directory/);
        assert.equal(fs.readFileSync(marker, "utf8"), "keep");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Linux native packager checks every compiler worker before collecting runtimes", {
      skip: process.platform === "win32",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const compilers = source.match(/^compilers=\([^\n]+\)$/m)?.[0];
      // The pre-collection guard is the worker loop that only checks readability (no copy/profile).
      const loops = [...source.matchAll(/^for key in [^\n]*compilers[^\n]*; do\n[\s\S]*?^done$/gm)]
        .map(([loop]) => loop).filter(loop => loop.includes("[ -s ") && !/\bcp\b|profile/.test(loop));
      assert.ok(compilers);
      assert.equal(loops.length, 1);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-workers-"));
      const keys = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
      const guard = ["set -euo pipefail", compilers, "compiler_build_dir=build/lin/Release", loops[0]].join("\n");
      try {
        for (const key of keys) {
          const addon = path.join(temp, "build/lin/Release", key, "mom.node");
          fs.mkdirSync(path.dirname(addon), {recursive: true});
          fs.writeFileSync(addon, "fixture");
        }
        assert.equal(spawnSync("bash", ["-c", guard], {cwd: temp}).status, 0);
        for (const key of keys) {
          const addon = path.join(temp, "build/lin/Release", key, "mom.node");
          fs.writeFileSync(addon, "");
          assert.equal(spawnSync("bash", ["-c", guard], {cwd: temp}).status, 1,
            "Every compiler worker must be nonempty before SDK collection");
          fs.writeFileSync(addon, "fixture");
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Native runtime patterns expand files without evaluating commands", {
      skip: process.platform === "win32",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-patterns-"));
      const driver = shellFunction(source, "runtime_matches") + "runtime_matches \"$1\"";
      try {
        for (const name of ["one.so", "two.bc"]) {fs.writeFileSync(path.join(temp, name), "fixture");}
        fs.mkdirSync(path.join(temp, "directory.so"));
        const matched = spawnSync("bash", ["-c", driver, "fixture", temp + "/*.so\n" + temp + "/*.bc"], {encoding: "utf8"});
        assert.equal(matched.status, 0);
        assert.deepEqual(matched.stdout.trim().split("\n").sort(), ["one.so", "two.bc"].map(name => path.join(temp, name)).sort());
        const marker = path.join(temp, "unexpected-command");
        const injected = spawnSync("bash", ["-c", driver, "fixture", temp + "/*.missing; touch " + marker], {encoding: "utf8"});
        assert.equal(injected.status, 0);
        assert.equal(injected.stdout, "");
        assert.equal(fs.existsSync(marker), false);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Native packaging includes indirectly loaded oneAPI OpenCL runtime files", {
      skip: process.platform === "win32",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-opencl-"));
      const sdk = path.join(temp, "sdk");
      const dest = path.join(temp, "runtime");
      const files = [
        "compiler/latest/lib/libintelocl.so",
        "compiler/latest/lib/libocl_svml_test.so",
        "compiler/latest/lib/clbltfn_test.rtl",
        "compiler/latest/lib/cllibrary_test.o",
        "tbb/2026.0/lib/libtbbmalloc.so.2",
        "tcm/latest/lib/libtcm.so.1",
        "umf/latest/lib/libiomp5.so",
      ];
      try {
        fs.mkdirSync(dest);
        for (const name of files) {
          const file = path.join(sdk, name);
          fs.mkdirSync(path.dirname(file), {recursive: true});
          fs.writeFileSync(file, "fixture");
        }
        const driver = [
          "set -euo pipefail",
          'runtime_dest="$1"',
          'copy_runtime_file() { cp -L -- "$1" "$runtime_dest/"; }',
          shellFunction(source, "runtime_matches"), shellFunction(source, "copy_matches"),
          shellFunction(source, "copy_oneapi_opencl_runtime").replaceAll("/opt/intel/oneapi", sdk),
          "copy_oneapi_opencl_runtime",
        ].join("\n");
        const result = spawnSync("bash", ["-c", driver, "fixture", dest], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.deepEqual(fs.readdirSync(dest).sort(), files.map(file => path.basename(file)).sort());
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    for (const fixture of [
      {name: "staging preserves the HIP alias and CUDA tools and passes validation", alias: "valid"},
      {name: "validator rejects a missing HIP linker alias", alias: "missing"},
      {name: "validator rejects a nonexecutable HIP linker alias", alias: "nonexec"},
      {name: "validator rejects a wrong executable HIP linker alias", alias: "wrong"},
    ]) {
      test(`AdaptiveCpp release LLVM ${fixture.name}`, {skip: process.platform === "win32"}, () => {
        const scripts = path.join(__dirname, "../scripts");
        const packager = fs.readFileSync(path.join(scripts, "package-linux-combined.sh"), "utf8");
        const validator = fs.readFileSync(path.join(scripts, "test-release-linux.sh"), "utf8");
        // The release validator's LLVM checks are top-level statements, bounded by the next loop.
        const validationStart = validator.indexOf("for tool in opt llc lld ld.lld; do");
        const validationEnd = validator.indexOf("for bitcode in", validationStart);
        assert.ok(validationStart >= 0 && validationEnd > validationStart);
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-llvm-test-"));
        const image = path.join(temp, "image");
        const libs = path.join(temp, "libs");
        const tools = ["opt", "llc", "lld", "ld.lld"];
        const env = {...process.env, MOM_FIXTURE_IMAGE: image, MOM_FIXTURE_LIBS: libs, MOM_ROCM_VERSION: ""};
        try {
          for (const version of [20, 21]) {
            const directory = path.join(image, `opt/llvm${version}-ubuntu24/bin`);
            fs.mkdirSync(directory, {recursive: true});
            for (const tool of tools) {
              fs.writeFileSync(path.join(directory, tool), `#!/bin/sh\n# LLVM${version} ${tool}\nexit 0\n`, {mode: 0o755});
            }
            fs.mkdirSync(path.join(image, "usr/lib"), {recursive: true});
            fs.symlinkSync(path.dirname(directory), path.join(image, `usr/lib/llvm-${version}`));
            const backend = version === 20 ? "hip" : "cuda";
            const prefix = path.join(image, `prefix-acpp-${backend}`);
            fs.mkdirSync(path.join(prefix, "lib"), {recursive: true});
            fs.writeFileSync(path.join(prefix, ".mom-build-identity"),
              `${"a".repeat(40)}:${version}:${backend}:${backend === "hip" ? "7.1.1" : "12-6"}:1:none\n`);
          }
          const bitcode = path.join(image, "prefix-acpp-hip/lib/hipSYCL/ext/bitcode/amdgcn");
          fs.mkdirSync(bitcode, {recursive: true});
          fs.writeFileSync(path.join(bitcode, "ockl.bc"), "fixture");
          const libdevice = path.join(image, "usr/local/cuda-12.6/nvvm/libdevice");
          fs.mkdirSync(libdevice, {recursive: true});
          fs.writeFileSync(path.join(libdevice, "libdevice.10.bc"), "fixture");
          // Execute the real metadata selection and staging function against only fixture SDKs.
          const staging = [
            acppRuntimeProgram(packager).replaceAll("/usr/lib/llvm-", "$MOM_FIXTURE_IMAGE/usr/lib/llvm-")
              .replaceAll("/usr/local/", "$MOM_FIXTURE_IMAGE/usr/local/"),
            "container=fixture; closure_queue=(); libs_dir=$MOM_FIXTURE_LIBS",
            "for key in acpp-cuda acpp-hip; do",
            '  runtime_dest="$libs_dir/$key"',
            '  copy_acpp_runtime "$key" "$MOM_FIXTURE_IMAGE/prefix-$key"',
            "done",
          ].join("\n");
          const staged = spawnSync("bash", ["-c", staging], {encoding: "utf8", env});
          assert.equal(staged.error, undefined);
          assert.equal(staged.signal, null);
          assert.equal(staged.status, 0, staged.stdout + staged.stderr);
          const hip = path.join(libs, "acpp-hip/hipSYCL/ext/llvm/bin");
          const cuda = path.join(libs, "acpp-cuda/hipSYCL/ext/llvm/bin");
          const linker = path.join(hip, "ld.lld");
          const alias = path.join(hip, "ld.lld-20");
          assert.equal(fs.lstatSync(alias).isFile(), true);
          assert.equal(fs.statSync(alias).ino, fs.statSync(linker).ino);
          assert.equal(fs.statSync(alias).dev, fs.statSync(linker).dev);
          assert.notEqual(fs.statSync(alias).mode & 0o111, 0);
          assert.equal(fs.statSync(linker).size,
            fs.statSync(path.join(image, "opt/llvm20-ubuntu24/bin/ld.lld")).size,
            "LLVM staging fixtures must not read host SDK binaries");
          assert.equal(fs.readFileSync(alias).equals(fs.readFileSync(linker)), true,
            "Staged LLVM bytes must match the fixture");
          assert.deepEqual(fs.readdirSync(cuda).sort(), [...tools].sort());
          for (const tool of tools) {
            assert.equal(fs.readFileSync(path.join(hip, tool)).equals(fs.readFileSync(path.join(image, "opt/llvm20-ubuntu24/bin", tool))), true,
              "Staged LLVM bytes must match the fixture");
            assert.equal(fs.readFileSync(path.join(cuda, tool)).equals(fs.readFileSync(path.join(image, "opt/llvm21-ubuntu24/bin", tool))), true,
              "Staged LLVM bytes must match the fixture");
          }
          if (fixture.alias !== "valid") {
          // Replace the alias inode so a negative cannot accidentally modify the linked original.
            fs.unlinkSync(alias);
            if (fixture.alias !== "missing") {
              fs.writeFileSync(alias, fixture.alias === "wrong" ? "wrong linker\n" : fs.readFileSync(linker),
                {mode: fixture.alias === "nonexec" ? 0o644 : 0o755});
            }
          }
          const validation = ["set -euo pipefail", "libs_dir=$MOM_FIXTURE_LIBS",
            'die() { echo "$1" >&2; exit 1; }', validator.slice(validationStart, validationEnd)].join("\n");
          const checked = spawnSync("bash", ["-c", validation], {encoding: "utf8", env});
          assert.equal(checked.error, undefined);
          assert.equal(checked.signal, null);
          assert.equal(checked.status, fixture.alias === "valid" ? 0 : 1, checked.stdout + checked.stderr);
          if (fixture.alias === "wrong") {
            assert.match(checked.stderr, /Release HIP linker alias differs from the staged LLVM linker/);
          } else if (fixture.alias !== "valid") {
            assert.match(checked.stderr, /Release is missing executable acpp-hip\/hipSYCL\/ext\/llvm\/bin\/ld\.lld-20/);
          }
        } finally {
          fs.rmSync(temp, {recursive: true, force: true});
        }
      });
    }

    test("Linux SEA configuration preserves quoted and escaped checkout paths", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const command = packager.match(/^"\$node_bin" - .*<<'NODE_SEA_CONFIG'\n[\s\S]*?^NODE_SEA_CONFIG$/m)?.[0];
      assert.ok(command, "Exercise the actual serialized SEA configuration command");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sea-paths-"));
      try {
        const directory = path.join(temp, "quoted\" back\\slash Δ\nline");
        fs.mkdirSync(directory);
        const main = path.join(directory, "mom.bundle.cjs");
        const output = path.join(directory, "mom.blob");
        const result = spawnSync("bash", ["-c", "set -euo pipefail\n" + command], {
          encoding: "utf8", env: {...process.env, node_bin: process.execPath,
            build_dir: directory, bundle_path: main, blob_path: output},
        });
        assert.equal(result.status, 0, "SEA configuration generation must preserve tokenized paths");
        assert.deepEqual(JSON.parse(fs.readFileSync(path.join(directory, "sea-config.json"), "utf8")), {
          main, output, disableExperimentalSEAWarning: true, useCodeCache: false, useSnapshot: false,
        });
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native SDK installer keeps public files readable with a restrictive caller umask", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      // Run the installer's top-level setup, which precedes its first function definition.
      const end = installer.search(/^[a-z_]+\(\)\s*[{(]$/m);
      assert.ok(end > 0);
      const setup = installer.slice(0, end);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sdk-permissions-"));
      try {
        for (const mask of ["077", "027"]) {
          const destination = path.join(temp, mask);
          const result = spawnSync("bash", ["-c", [
            `umask ${mask}`, setup,
            'mkdir "$MOM_FIXTURE_DEST"',
            'printf "public repository key\\n" > "$MOM_FIXTURE_DEST/key.asc"',
          ].join("\n")], {encoding: "utf8", env: {...process.env, MOM_FIXTURE_DEST: destination}});
          assert.equal(result.status, 0, "Installer setup must work without invoking any SDK or network");
          assert.equal(fs.statSync(destination).mode & 0o777, 0o755);
          assert.equal(fs.statSync(path.join(destination, "key.asc")).mode & 0o777, 0o644,
            "The unprivileged apt user must be able to read public repository keys");
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native DPC++ installer publishes a traversable SDK root", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-permissions-"));
      try {
        const payload = path.join(temp, "payload");
        fs.mkdirSync(path.join(payload, "bin"), {recursive: true});
        fs.mkdirSync(path.join(payload, "lib"));
        fs.writeFileSync(path.join(payload, "bin/clang++"), "#!/bin/sh\nexit 0\n", {mode: 0o755});
        // Archive permissions model the published SDK, independent of the test caller's umask.
        for (const directory of [payload, path.join(payload, "bin"), path.join(payload, "lib")]) {
          fs.chmodSync(directory, 0o755);
        }
        fs.chmodSync(path.join(payload, "bin/clang++"), 0o755);
        for (const library of ["libsycl.so.9", "libsycl-jit.so"]) {
          fs.writeFileSync(path.join(payload, "lib", library), "fixture\n");
          fs.chmodSync(path.join(payload, "lib", library), 0o644);
        }
        const archive = path.join(temp, "fixture.tgz");
        assert.equal(spawnSync("tar", ["-czf", archive, "-C", payload, "."]).status, 0);
        const program = [
          "set -euo pipefail; umask 022",
          "dpcpp_ready() { return 1; }",
          'curl() { cp -- "$MOM_FIXTURE_ARCHIVE" "$WORKSPACE/dpcpp-fixture.tar.gz"; }',
          'verify_sha256() { test "$(sha256sum "$1" | cut -d " " -f1)" = "$2"; }',
          'replace_staged_directory() { mv -- "$1" "$2"; }',
          shellFunction(installer, "install_dpcpp").replaceAll("/opt/", `${temp}/`),
          "install_dpcpp",
        ].join("\n");
        const result = spawnSync("bash", ["-c", program], {encoding: "utf8", env: {...process.env,
          WORKSPACE: temp, DPCPP_RELEASE: "fixture", DPCPP_ASSET: "fixture.tgz",
          DPCPP_SHA256: createHash("sha256").update(fs.readFileSync(archive)).digest("hex"),
          MOM_FIXTURE_ARCHIVE: archive}});
        assert.equal(result.status, 0, "Actual installer staging must publish the verified fixture SDK");
        assert.equal(fs.statSync(path.join(temp, "dpcpp")).mode & 0o777, 0o755,
          "mktemp's private staging root must not become a root-only installed compiler");
        assert.equal(fs.statSync(path.join(temp, "dpcpp/bin/clang++")).mode & 0o111, 0o111);
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native LLVM installer uses minimal mode while retaining development packages", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-llvm-mode-"));
      try {
        // Redirect SDK readiness probes: this fixture never inspects a real host SDK.
        const body = shellFunction(installer, "ensure_llvm")
          .replaceAll("/usr/bin/", `${temp}/bin/`).replaceAll("/usr/lib/", `${temp}/lib/`);
        const payload = "offline LLVM installer fixture\n";
        const digest = createHash("sha256").update(payload).digest("hex");
        const run = (/** @type {string} */ source, /** @type {number} */ version,
        /** @type {string} */ hash = digest) => {
          const result = spawnSync("bash", ["-c", [
            "set -euo pipefail",
            'record() { printf "%s\\t" "$@"; printf "\\n"; }',
            'apt_install() { record apt "$@"; }',
            'curl() { record curl "$@"; printf "offline LLVM installer fixture\\n" > "$WORKSPACE/llvm.sh"; }',
            'verify_sha256() { record verify "$@"; test "$(sha256sum "$1" | cut -d " " -f1)" = "$2"; }',
            'bash() { record installer "$@"; }',
            source, "ensure_llvm",
          ].join("\n")], {encoding: "utf8", timeout: 5000, env: {...process.env,
            WORKSPACE: temp, LLVM_VERSION: String(version), LLVM_INSTALLER_URL: "fixture://llvm.sh",
            LLVM_INSTALLER_SHA256: hash}});
          return {status: result.status, calls: result.stdout.trim().split("\n")
            .filter(Boolean).map(line => line.trimEnd().split("\t"))};
        };
        for (const version of [20, 21]) {
          const installerArgs = ["installer", `${temp}/llvm.sh`, String(version)];
          const expected = [
            ["apt", "software-properties-common"],
            ["curl", "-fsSL", "--retry", "5", "fixture://llvm.sh", "-o", `${temp}/llvm.sh`],
            ["verify", `${temp}/llvm.sh`, digest],
            installerArgs,
            ["apt", `libclang-${version}-dev`, `libomp-${version}-dev`, `llvm-${version}-dev`],
          ];
          const actual = run(body, version);
          assert.equal(actual.status, 0);
          assert.deepEqual(actual.calls, expected, `LLVM ${version} must verify before invoking minimal mode`);
          const unverified = run(body, version, "0".repeat(64));
          assert.notEqual(unverified.status, 0);
          assert.deepEqual(unverified.calls, expected.slice(0, 3)
            .map(call => call[0] === "verify" ? ["verify", `${temp}/llvm.sh`, "0".repeat(64)] : call));
          fs.mkdirSync(path.join(temp, "bin"), {recursive: true});
          fs.mkdirSync(path.join(temp, `lib/llvm-${version}/lib/cmake/llvm`), {recursive: true});
          fs.writeFileSync(path.join(temp, `bin/clang-${version}`), "fixture\n", {mode: 0o755});
          assert.deepEqual(run(body, version).calls, expected,
            "A compiler and CMake directory without development files must be repaired");
          const required = [`bin/clang++-${version}`, ...[
            "include/llvm/Support/Casting.h", "include/clang/AST/ASTContext.h",
            "lib/libLLVMCore.a", "lib/libclangBasic.a", "lib/cmake/llvm/LLVMConfig.cmake",
            "lib/cmake/clang/ClangConfig.cmake",
          ].map(file => `lib/llvm-${version}/${file}`)];
          for (const file of required) {
            fs.mkdirSync(path.dirname(path.join(temp, file)), {recursive: true});
            fs.writeFileSync(path.join(temp, file), "fixture\n", {mode: 0o755});
          }
          const ready = run(body, version);
          assert.equal(ready.status, 0);
          assert.deepEqual(ready.calls, [], "An already-ready fake SDK must not invoke network or apt");
          for (const file of required) {
            const absolute = path.join(temp, file);
            fs.unlinkSync(absolute);
            const partial = run(body, version);
            assert.equal(partial.status, 0);
            assert.deepEqual(partial.calls, expected, `A missing ${file} must request repair`);
            fs.writeFileSync(absolute, "fixture\n", {mode: 0o755});
          }
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native base installer includes native compression and TLS development packages once", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const result = spawnSync("bash", ["-c", ["set -euo pipefail", "BASE_INSTALLED=0",
        'apt_install() { printf "%s\\n" "$@"; }', shellFunction(installer, "install_base"),
        "install_base", "install_base"].join("\n")], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr);
      for (const name of ["zlib1g-dev", "libssl-dev"]) {
        assert.equal(result.stdout.trim().split("\n").filter(value => value === name).length, 1,
          `Native builds and source-level diagnostics need ${name}`);
      }
    });

    test("Native Node installer rejects mismatched SEA, build headers or node-gyp", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const extract = (/** @type {string} */ name) => shellFunction(installer, name);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-node-readiness-"));
      try {
        const bin = path.join(temp, "bin");
        const headers = path.join(temp, "include/node");
        fs.mkdirSync(bin); fs.mkdirSync(headers, {recursive: true});
        const cases = ["ready", "missing-fuse", "missing-header", "wrong-headers", "wrong-gyp", "broken-gyp"];
        for (const name of cases) {
          fs.writeFileSync(path.join(bin, "node-gyp"), name === "broken-gyp" ? "#!/bin/sh\nexit 7\n" :
            `#!/bin/sh\nprintf 'v${name === "wrong-gyp" ? "12.4.0" : "12.2.0"}\\n'\n`, {mode: 0o755});
          fs.writeFileSync(path.join(bin, "node"), "#!/bin/sh\nprintf 'v24.15.0\\n'\n" +
          (name === "missing-fuse" ? "" : "# NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2\n"), {mode: 0o755});
          for (const file of ["node.h", "common.gypi", "config.gypi"]) {
            fs.writeFileSync(path.join(headers, file), "fixture\n");
          }
          fs.writeFileSync(path.join(headers, "node_version.h"),
            `#define NODE_MAJOR_VERSION ${name === "wrong-headers" ? 22 : 24}\n#define NODE_MINOR_VERSION 15\n#define NODE_PATCH_VERSION 0\n`);
          if (name === "missing-header") {fs.unlinkSync(path.join(headers, "node.h"));}
          const program = ["set -euo pipefail", extract("node_ready"), extract("install_node"),
            "curl() { return 23; }", "install_node"].join("\n").replaceAll("/usr/local", temp);
          const result = spawnSync("bash", ["-c", program], {encoding: "utf8", env: {...process.env,
            PATH: `${bin}:${process.env["PATH"]}`, NODE_VERSION: "24.15.0", WORKSPACE: temp}});
          assert.equal(result.status, name === "ready" ? 0 : 23, `Node readiness must reject ${name} before reuse`);
          const validate = ["set -euo pipefail", extract("node_ready"), extract("validate_component"),
            "validate_component node"].join("\n").replaceAll("/usr/local", temp);
          assert.equal(spawnSync("bash", ["-c", validate], {env: {...process.env,
            PATH: `${bin}:${process.env["PATH"]}`, NODE_VERSION: "24.15.0"}}).status,
          name === "ready" ? 0 : 1, `Validate-only must reject ${name}`);
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native Node installer exposes bundled node-gyp without registry access", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const extract = (/** @type {string} */ name) => shellFunction(installer, name);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-bundled-node-gyp-"));
      try {
        for (const variant of ["ready", "wrong-version", "missing-cli"]) {
          const root = path.join(temp, variant), prefix = path.join(root, "installed"), payload = path.join(root, "payload");
          const headers = path.join(payload, "include/node"), bin = path.join(payload, "bin"),
            bundled = path.join(payload, "lib/node_modules/npm/node_modules/node-gyp/bin");
          fs.mkdirSync(headers, {recursive: true}); fs.mkdirSync(bin); fs.mkdirSync(bundled, {recursive: true});
          fs.writeFileSync(path.join(bin, "node"), "#!/bin/sh\nprintf 'v24.15.0\\n'\n" +
            "# NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2\n", {mode: 0o755});
          for (const file of ["node.h", "common.gypi", "config.gypi"]) {fs.writeFileSync(path.join(headers, file), "fixture\n");}
          fs.writeFileSync(path.join(headers, "node_version.h"),
            "#define NODE_MAJOR_VERSION 24\n#define NODE_MINOR_VERSION 15\n#define NODE_PATCH_VERSION 0\n");
          if (variant !== "missing-cli") {
            fs.writeFileSync(path.join(bundled, "node-gyp.js"),
              `#!/bin/sh\nprintf 'v${variant === "wrong-version" ? "12.4.0" : "12.2.0"}\\n'\n`, {mode: 0o755});
          }
          const program = ["set -euo pipefail", extract("node_ready"), extract("install_node"),
            "curl() { :; }", "verify_sha256() { :; }",
            'tar() { mkdir -p "$MOM_FIXTURE_PREFIX"; cp -a "$MOM_FIXTURE_PAYLOAD/." "$MOM_FIXTURE_PREFIX/"; }',
            "npm() { echo 'unexpected registry install' >&2; return 43; }", "install_node",
            // The second call must reuse the validated installation without downloading again.
            "curl() { return 29; }", "install_node"].join("\n").replaceAll("/usr/local", prefix);
          const result = spawnSync("bash", ["-c", program], {encoding: "utf8", env: {...process.env,
            PATH: `${path.join(prefix, "bin")}:${process.env["PATH"]}`, NODE_VERSION: "24.15.0", NODE_SHA256: "fixture",
            WORKSPACE: root, MOM_FIXTURE_PREFIX: prefix, MOM_FIXTURE_PAYLOAD: payload}});
          assert.equal(result.status, variant === "ready" ? 0 : 1, `${variant}: ${result.stderr}`);
          assert.equal(fs.readlinkSync(path.join(prefix, "bin/node-gyp")),
            "../lib/node_modules/npm/node_modules/node-gyp/bin/node-gyp.js");
        }
      } finally {fs.rmSync(temp, {recursive: true, force: true});}
    });

    test("Native base validation requires both Linux packaging tools", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const bash = spawnSync("bash", ["-c", "command -v bash"], {encoding: "utf8"});
      assert.equal(bash.status, 0);
      const program = `${shellFunction(installer, "validate_component")}validate_component base`;
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-base-validation-"));
      try {
        for (const missing of ["", "patchelf", "readelf"]) {
          const bin = path.join(temp, missing || "complete");
          fs.mkdirSync(bin);
          for (const tool of ["cmake", "ninja", "git", "patchelf", "readelf"]) {
            if (tool !== missing) {
              fs.writeFileSync(path.join(bin, tool), "#!/bin/sh\nexit 0\n", {mode: 0o755});
            }
          }
          const result = spawnSync(bash.stdout.trim(), ["-c", program], {
            encoding: "utf8", env: {PATH: bin},
          });
          assert.equal(result.status, missing ? 1 : 0,
            `Base validation must ${missing ? `reject missing ${missing}` : "accept complete tools"}`);
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native HIP installer linker resolves in the packaged LLVM21 runtime", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const installer = fs.readFileSync(path.join(root, "scripts/install-dev.sh"), "utf8");
      const packager = fs.readFileSync(path.join(root,
        "scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-hip-linker-"));
      const sdk = path.join(temp, "sdk");
      const llvm = path.join(sdk, "usr/lib/llvm-21/bin");
      const runtime = path.join(temp, "runtime");
      const env = {...process.env, LLVM_VERSION: "21", MOM_LLVM_VERSION: "21", JOBS: "1",
        MOM_FIXTURE_SDK: sdk, MOM_FIXTURE_RUNTIME: runtime, MOM_ROCM_VERSION: ""};
      try {
        fs.mkdirSync(llvm, {recursive: true});
        const prefix = path.join(sdk, "adaptivecpp-hip");
        const bitcode = path.join(prefix, "lib/hipSYCL/ext/bitcode/amdgcn");
        fs.mkdirSync(bitcode, {recursive: true});
        fs.writeFileSync(path.join(bitcode, "ockl.bc"), "fixture");
        fs.writeFileSync(path.join(prefix, ".mom-build-identity"), `${"a".repeat(40)}:21:hip:7.1.1:1-hip-lld1:none\n`);
        const rocm = path.join(sdk, "opt/rocm-7.1.1/lib");
        fs.mkdirSync(rocm, {recursive: true});
        fs.writeFileSync(path.join(rocm, "libamdhip64.so"), "fixture");
        // An unrelated legacy SDK must never choose the toolchain for this installed compiler.
        fs.mkdirSync(path.join(sdk, "opt/llvm20-ubuntu24/bin"), {recursive: true});
        for (const tool of ["opt", "llc", "lld", "ld.lld"]) {
          fs.writeFileSync(path.join(llvm, tool), "#!/bin/sh\nprintf '%s\\n' fixture-llvm21\n", {mode: 0o755});
        }
        // Run the real build function up to its configure step; cmake prints and stops there.
        const configured = spawnSync("bash", ["-c", [
          "set -euo pipefail",
          "adaptivecpp_ready() { return 1; }", "ensure_llvm() { :; }", "git() { :; }",
          'cmake() { printf "%s\\n" "$@"; exit 0; }',
          shellFunction(installer, "build_adaptivecpp").replaceAll("/opt/rocm-", "$MOM_FIXTURE_SDK/opt/rocm-"),
          'build_adaptivecpp hip "$MOM_FIXTURE_SDK/unused"',
        ].join("\n")], {encoding: "utf8", env: {...env, ROCM_VERSION: "7.1.1", CUDA_VERSION: "12-6", ADAPTIVECPP_COMMIT: "fixture",
          WORKSPACE: path.join(temp, "workspace")}});
        assert.equal(configured.status, 0, "Real installer options must evaluate without SDK access: " + configured.stderr);
        const linkerOption = configured.stdout.split("\n").find(line => line.startsWith("-DACPP_LLD_PATH="));
        assert.ok(linkerOption, "The installer must configure the HIP linker");
        const staging = acppRuntimeProgram(packager)
          .replaceAll("/opt/llvm", "$MOM_FIXTURE_SDK/opt/llvm")
          .replaceAll("/usr/lib/llvm-", "$MOM_FIXTURE_SDK/usr/lib/llvm-");
        const staged = spawnSync("bash", ["-c", [
          staging,
          'closure_queue=(); runtime_dest="$MOM_FIXTURE_RUNTIME"',
          'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_SDK/adaptivecpp-hip"',
        ].join("\n")], {encoding: "utf8", env});
        assert.equal(staged.status, 0, "Real packaging must stage only the fixture tools: " + staged.stderr);
        const bin = path.join(runtime, "hipSYCL/ext/llvm/bin");
        const linker = path.join(bin, path.basename(linkerOption.slice("-DACPP_LLD_PATH=".length)));
        const executed = spawnSync(linker, [], {encoding: "utf8"});
        assert.equal(executed.status, 0, "The configured HIP linker basename must survive relocation");
        assert.equal(executed.stdout.trim(), "fixture-llvm21");
        assert.equal(fs.statSync(path.join(bin, "ld.lld-20")).ino, fs.statSync(path.join(bin, "ld.lld")).ino,
          "Existing Docker LLVM20 packages retain their compatibility alias");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });
  });

  test.describe("AdaptiveCpp toolchain identity and portable release workflows", {concurrency: false}, () => {
    test("AdaptiveCpp packaging rejects missing, malformed, or wrong-backend LLVM identity", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-llvm-identity-"));
      const marker = path.join(temp, ".mom-build-identity");
      try {
        for (const identity of [null, "", `${"a".repeat(40)}:21:cuda:12-6:1:none`,
          `${"a".repeat(40)}:21;exit 0:hip:7.1.1:1:none`, `${"a".repeat(40)}:21:hip:7.1.1:1:none\nextra`]) {
          if (identity !== null) {fs.writeFileSync(marker, identity);}
          const result = spawnSync("bash", ["-c", [acppRuntimeProgram(packager),
            'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_PREFIX"',
          ].join("\n")], {encoding: "utf8", env: {...process.env, MOM_FIXTURE_PREFIX: temp,
            runtime_dest: path.join(temp, "runtime")}});
          assert.equal(result.status, 1, "Invalid SDK identity must fail before staging any tools");
          assert.match(result.stderr, /valid matching \.mom-build-identity/);
          assert.equal(fs.existsSync(path.join(temp, "runtime")), false);
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("AdaptiveCpp HIP bitcode stays with the recorded SDK cohort", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-cohort-"));
      try {
        acppLlvmFixture(temp);
        const opt = path.join(temp, "opt");
        for (const [relative, value] of [
          ["rocm-7.1.1/amdgcn/bitcode", "unrelated"],
          ["rocm-7.2.0/amdgcn/bitcode", "matching"],
          ["rocm-device-libs-ubuntu24", "legacy"],
          ["rocm/amdgcn/bitcode", "mutable"],
        ]) {
          assert.ok(relative && value);
          fs.mkdirSync(path.join(opt, relative), {recursive: true});
          fs.writeFileSync(path.join(opt, relative, "ockl.bc"), value);
        }
        for (const fixture of [
          {name: "recorded", version: "7.2.0", requested: "", bundled: false, expected: "matching"},
          {name: "bundled", version: "7.2.0", requested: "7.2.0", bundled: true, expected: "bundled"},
          {name: "wrong-env", version: "7.2.0", requested: "7.1.1", bundled: true, expected: ""},
          {name: "missing", version: "7.3.0", requested: "", bundled: false, expected: ""},
          {name: "invalid", version: "7-2", requested: "", bundled: true, expected: ""},
        ]) {
          const prefix = path.join(temp, fixture.name);
          const bundled = path.join(prefix, "lib/hipSYCL/ext/bitcode/amdgcn");
          const runtime = path.join(temp, `${fixture.name}-runtime`);
          const destination = path.join(runtime, "hipSYCL/ext/bitcode/amdgcn");
          fs.mkdirSync(bundled, {recursive: true});
          fs.mkdirSync(destination, {recursive: true});
          fs.writeFileSync(path.join(prefix, ".mom-build-identity"),
            `${"a".repeat(40)}:21:hip:${fixture.version}:1-hip-lld1:none\n`);
          if (fixture.bundled) {fs.writeFileSync(path.join(bundled, "ockl.bc"), "bundled");}
          const program = [acppRuntimeProgram(packager), 'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_PREFIX"']
            .join("\n").replaceAll("/opt/rocm", `${opt}/rocm`).replaceAll("/usr/lib/llvm-", `${temp}/llvm-`);
          const result = spawnSync("bash", ["-c", program], {encoding: "utf8", env: {...process.env,
            MOM_FIXTURE_PREFIX: prefix, MOM_ROCM_VERSION: fixture.requested, runtime_dest: runtime}});
          assert.equal(result.status, fixture.expected ? 0 : 1, `Wrong bitcode selection for ${fixture.name}`);
          if (fixture.expected) {
            assert.equal(fs.readFileSync(path.join(destination, "ockl.bc"), "utf8"), fixture.expected);
          } else {
            assert.match(result.stderr, /bitcode/);
            // Packaging aborts; only the cohort's own bundled payload may have been copied so far.
            for (const file of fs.readdirSync(destination)) {
              assert.equal(fs.readFileSync(path.join(destination, file), "utf8"), "bundled",
                "No mismatched host bitcode may be packaged");
            }
          }
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("AdaptiveCpp CUDA bitcode stays with the recorded SDK cohort", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../scripts/package-linux-combined.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-cohort-"));
      try {
        acppLlvmFixture(temp);
        const sdkRoot = path.join(temp, "sdk");
        for (const fixture of [{version: "12.6", bytes: "matching"}, {version: "13.0", bytes: "unrelated"}]) {
          const dir = path.join(sdkRoot, `cuda-${fixture.version}/nvvm/libdevice`);
          fs.mkdirSync(dir, {recursive: true});
          fs.writeFileSync(path.join(dir, "libdevice.10.bc"), fixture.bytes);
        }
        for (const fixture of [
          {name: "default", version: "12-6", alias: "12.6", expected: "matching"},
          {name: "moved-alias", version: "12-6", alias: "13.0", expected: "matching"},
          {name: "missing", version: "12-7", alias: "13.0", expected: ""},
          {name: "invalid", version: "12.6", alias: "12.6", expected: ""},
        ]) {
          const alias = path.join(sdkRoot, "cuda");
          if (fs.existsSync(alias)) {fs.unlinkSync(alias);}
          fs.symlinkSync(`cuda-${fixture.alias}`, alias);
          const prefix = path.join(temp, fixture.name);
          const runtime = path.join(temp, `${fixture.name}-runtime`);
          fs.mkdirSync(path.join(prefix, "lib"), {recursive: true});
          fs.writeFileSync(path.join(prefix, ".mom-build-identity"),
            `${"a".repeat(40)}:21:cuda:${fixture.version}:1:none\n`);
          const program = [acppRuntimeProgram(packager), 'copy_acpp_runtime acpp-cuda "$MOM_FIXTURE_PREFIX"']
            .join("\n").replaceAll("/usr/local/", `${sdkRoot}/`).replaceAll("/usr/lib/llvm-", `${temp}/llvm-`);
          const result = spawnSync("bash", ["-c", program], {encoding: "utf8", timeout: 5000,
            env: {...process.env, MOM_FIXTURE_PREFIX: prefix, runtime_dest: runtime}});
          const output = path.join(runtime, "hipSYCL/ext/bitcode/ptx/libdevice.10.bc");
          assert.equal(result.status, fixture.expected ? 0 : 1, `Wrong CUDA bitcode selection for ${fixture.name}`);
          if (fixture.expected) {
            assert.equal(fs.readFileSync(output, "utf8"), fixture.expected);
          } else {
            assert.equal(fs.existsSync(output), false, "No missing or malformed cohort may fall back to the alias");
          }
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native HIP linker recipe invalidates only the HIP toolchain identity", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const installer = fs.readFileSync(path.join(root, "scripts/install-dev.sh"), "utf8");
      const patchHash = createHash("sha256").update(fs.readFileSync(path.join(root,
        "scripts/patches/adaptivecpp-cuda-unloading.patch"))).digest("hex");
      const identity = spawnSync("bash", ["-c", [
        "set -euo pipefail",
        shellFunction(installer, "adaptivecpp_identity"),
        "adaptivecpp_identity cuda",
        "adaptivecpp_identity hip",
      ].join("\n")], {encoding: "utf8", env: {...process.env, SCRIPT_DIR: path.join(root, "scripts"),
        ADAPTIVECPP_COMMIT: "fixture", LLVM_VERSION: "21", CUDA_VERSION: "12-6",
        ROCM_VERSION: "7.1.1", ADAPTIVECPP_RECIPE_VERSION: "1"}});
      assert.equal(identity.status, 0, "Toolchain identity must evaluate without an SDK");
      const identities = identity.stdout.trim().split("\n");
      assert.equal(identities.length, 2, "Both compiler identities must be emitted");
      const [cuda, hip] = identities;
      assert.equal(cuda, `fixture:21:cuda:12-6:1:${patchHash}`, "CUDA identity must remain byte-identical");
      assert.ok(hip?.startsWith("fixture:21:hip:7.1.1:"), "HIP identity must retain its SDK binding");
      assert.notEqual(hip, "fixture:21:hip:7.1.1:1:none", "HIP must reject the old versioned-linker SDK");
    });

    test("Linux combined packager rejects invalid workers or missing tools before mutation", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const script = path.join(root, "scripts/package-linux-combined.sh");
      const workers = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
      const contents = "worker fixture\n";
      const sha256 = createHash("sha256").update(contents).digest("hex");
      /** @typedef {{name: string, target: string, kind: string, expected: RegExp}} ProfileCase */
      /** @type {ProfileCase[]} */
      const profileCases = [
        ...workers.map((worker) => ({
          name: `${worker} portable=0`, target: worker, kind: "portable-zero",
          expected: /is not a portable Engineering worker[\s\S]*npm run build:release -- --engineering-archive --clean/,
        })),
        {name: "missing worker", target: "oneapi", kind: "missing-worker",
          expected: /mom\.node is missing; run 'npm run build:release -- --engineering-archive --clean'/},
        {name: "missing sidecar", target: "oneapi", kind: "missing-sidecar",
          expected: /\.build-profile is missing/},
        {name: "missing field", target: "oneapi", kind: "missing-field",
          expected: /has missing or duplicate fields/},
        {name: "duplicate field", target: "oneapi", kind: "duplicate-field",
          expected: /has missing or duplicate fields/},
        {name: "wrong worker key", target: "oneapi", kind: "wrong-worker",
          expected: /does not identify .*mom\.node exactly/},
        {name: "invalid hash", target: "oneapi", kind: "invalid-hash",
          expected: /does not identify .*mom\.node exactly/},
        {name: "mismatched hash", target: "oneapi", kind: "mismatched-hash",
          expected: /does not identify .*mom\.node exactly/},
        {name: "cpu=native", target: "oneapi", kind: "cpu-native",
          expected: /is not a portable Engineering worker[\s\S]*npm run build:release -- --engineering-archive --clean/},
        {name: "cpu=x86-64-v4", target: "oneapi", kind: "cpu-v4",
          expected: /is not a portable Engineering worker[\s\S]*npm run build:release -- --engineering-archive --clean/},
        {name: "missing patchelf", target: "oneapi", kind: "missing-patchelf",
          expected: /Engineering Linux archive packaging requires patchelf/},
        {name: "missing readelf", target: "oneapi", kind: "missing-readelf",
          expected: /Engineering Linux archive packaging requires readelf/},
      ];

      for (const profileCase of profileCases) {
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-profile-test-"));
        const fakeBin = path.join(temp, "bin");
        const dockerSentinel = path.join(temp, "docker-used");
        const env = packagerValidationEnv(temp);
        const fakeDocker = path.join(fakeBin, "docker");
        fs.writeFileSync(fakeDocker, [
          "#!/bin/sh",
          'printf "%s\\n" invoked > "$MOM_TEST_DOCKER_SENTINEL"',
          "exit 0",
          "",
        ].join("\n"));
        fs.chmodSync(fakeDocker, 0o755);

        try {
          const node = path.join(temp, "sea-node");
          fs.writeFileSync(node, "#!/bin/sh\n# NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2\nexit 91\n", {mode: 0o755});
          if (profileCase.kind === "missing-readelf") {
            fs.writeFileSync(path.join(fakeBin, "patchelf"), "#!/bin/sh\nexit 91\n", {mode: 0o755});
          }
          for (const directory of ["release-combined", "release-combined-build"]) {
            fs.mkdirSync(path.join(temp, directory));
            fs.writeFileSync(path.join(temp, directory, "marker"), "keep staging");
          }
          fs.writeFileSync(path.join(temp, "published.tgz"), "keep archive");
          for (const worker of workers) {
            if (profileCase.kind === "missing-worker" && worker === profileCase.target) {
              continue;
            }
            const addon = path.join(temp, "build/lin/Release", worker, "mom.node");
            fs.mkdirSync(path.dirname(addon), {recursive: true});
            fs.writeFileSync(addon, contents);
            if (profileCase.kind === "missing-sidecar" && worker === profileCase.target) {
              continue;
            }
            const lines = [
              "schema=1",
              `worker=${profileCase.kind === "wrong-worker" && worker === profileCase.target
                ? "dpcpp" : worker}`,
              `sha256=${profileCase.kind === "invalid-hash" && worker === profileCase.target
                ? "not-a-digest" : profileCase.kind === "mismatched-hash" && worker === profileCase.target
                  ? "0".repeat(64) : sha256}`,
              `portable=${profileCase.kind === "portable-zero" && worker === profileCase.target
                ? "0" : "1"}`,
              `cpu=${profileCase.kind === "cpu-native" && worker === profileCase.target
                ? "native" : profileCase.kind === "cpu-v4" && worker === profileCase.target
                  ? "x86-64-v4" : "unset"}`,
            ];
            if (profileCase.kind === "missing-field" && worker === profileCase.target) {
              lines.pop();
            }
            if (profileCase.kind === "duplicate-field" && worker === profileCase.target) {
              lines.push("portable=1");
            }
            fs.writeFileSync(`${addon}.build-profile`, `${lines.join("\n")}\n`);
          }

          const result = spawnSync("bash", [script, "1.2.3", "published.tgz"], {
            cwd: temp,
            encoding: "utf8",
            env: {
              ...env,
              NODE_BIN: node,
              MOM_TEST_DOCKER_SENTINEL: dockerSentinel,
            },
          });
          assert.equal(result.status, 1, `${profileCase.name}: ${result.stdout}${result.stderr}`);
          assert.match(result.stderr, profileCase.expected);
          assert.equal(fs.existsSync(dockerSentinel), false, `${profileCase.name}: Docker was used`);
          for (const directory of ["release-combined", "release-combined-build"]) {
            assert.equal(fs.readFileSync(path.join(temp, directory, "marker"), "utf8"), "keep staging");
          }
          assert.equal(fs.readFileSync(path.join(temp, "published.tgz"), "utf8"), "keep archive");
        } finally {
          fs.rmSync(temp, {recursive: true, force: true});
        }
      }
    });

    test("portable x86 CPU flags use the baseline while developer builds stay native", {
      skip: process.platform === "win32" || process.arch !== "x64",
    }, () => {
      const root = path.join(__dirname, "..");
      const script = path.join(root, "scripts/cpu-cflags.sh");
      /** @type {NodeJS.ProcessEnv} */
      const portableEnv = {...process.env, MOM_PORTABLE_BUILD: "1"};
      delete portableEnv["MOM_CPU_MARCH"];
      const portable = spawnSync("bash", [script], {
        cwd: root, encoding: "utf8", env: portableEnv,
      });
      assert.equal(portable.status, 0, portable.stderr);
      assert.equal(portable.stdout.trim(), "-march=x86-64 -mtune=generic -maes");
      assert.doesNotMatch(portable.stdout, /native|avx(?:2|512)|multiversion|multipath/i);

      /** @type {NodeJS.ProcessEnv} */
      const developerEnv = {...process.env, MOM_PORTABLE_BUILD: "0"};
      delete developerEnv["MOM_CPU_MARCH"];
      const developer = spawnSync("bash", [script], {
        cwd: root, encoding: "utf8", env: developerEnv,
      });
      assert.equal(developer.status, 0, developer.stderr);
      assert.equal(developer.stdout.trim(), "-march=native");
    });

    test("Linux release scripts reject traversal versions before build or deploy phases", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const traversalVersion = "../release-pwn";
      const linux = spawnSync("bash", [
        path.join(root, "scripts/package-linux-combined.sh"), traversalVersion,
      ], {cwd: root, encoding: "utf8"});
      assert.equal(linux.status, 2, linux.error?.message);
      assert.match(linux.stderr, /^Invalid release version:/);
      assert.doesNotMatch(linux.stdout + linux.stderr, /missing; run|▶|docker/i);

      const deploy = spawnSync("bash", [path.join(root, "scripts/test-deploy.sh")], {
        cwd: root,
        encoding: "utf8",
        env: {...process.env, MOM_DEPLOY_TARGET: "linux", MOM_RELEASE_VERSION: traversalVersion},
      });
      assert.equal(deploy.status, 2, deploy.error?.message);
      assert.match(deploy.stderr, /^Invalid release version:/);
      assert.doesNotMatch(deploy.stdout + deploy.stderr, /▶|docker|Building|Packaging/i);
    });

    test("Windows release packager rejects traversal versions before build phases", {
      skip: process.platform !== "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const windows = spawnSync("powershell.exe", [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        path.join(root, "scripts/package-windows.ps1"),
        "-Version", "../release-pwn",
      ], {cwd: root, encoding: "utf8"});
      assert.notEqual(windows.status, 0, windows.error?.message);
      assert.match(windows.stdout + windows.stderr, /Invalid release version:/);
      assert.doesNotMatch(windows.stdout + windows.stderr, /Building|Packaging/i);
    });

    test("scripts/dev-container.sh keeps process locks in private Git metadata", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-r-sh-lock-test-"));
      const fakeBin = path.join(temp, "bin");
      fs.mkdirSync(fakeBin);
      fs.mkdirSync(path.join(temp, "scripts"), {recursive: true});
      fs.copyFileSync(path.join(root, "scripts/dev-container.sh"), path.join(temp, "scripts/dev-container.sh"));
      fs.writeFileSync(path.join(fakeBin, "docker"), "#!/bin/sh\nexit 1\n");
      fs.chmodSync(path.join(fakeBin, "docker"), 0o755);
      try {
        const initialized = spawnSync("git", ["init", "-q"], {cwd: temp, encoding: "utf8"});
        assert.equal(initialized.status, 0, initialized.stderr);
        const result = spawnSync("bash", [path.join(temp, "scripts/dev-container.sh")], {
          cwd: temp,
          encoding: "utf8",
          env: {...process.env, PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`},
        });
        assert.equal(result.status, 1, result.stderr);
        assert.match(result.stderr, /Docker buildx is required/);
        assert.equal(fs.existsSync(path.join(temp, ".git", "mom-locks",
          `mom-r-sh-${process.getuid?.()}.build.lock`)), true);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("scripts/dev-container.sh keeps OpenCL out of GPU vendors and forwards controls once", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-r-sh-opencl-test-"));
      const fakeBin = path.join(temp, "bin");
      const dockerArgs = path.join(temp, "docker-args");
      const dockerEvents = path.join(temp, "docker-events");
      const dockerProbe = path.join(temp, "docker-probe");
      const dockerBuild = path.join(temp, "docker-build");
      const verthashData = path.join(temp, "verthash.dat");
      fs.mkdirSync(fakeBin);
      fs.writeFileSync(verthashData, "fixture");
      fs.mkdirSync(path.join(temp, "scripts"), {recursive: true});
      fs.copyFileSync(path.join(root, "scripts/dev-container.sh"), path.join(temp, "scripts/dev-container.sh"));
      const docker = path.join(fakeBin, "docker");
      fs.writeFileSync(docker, [
        "#!/bin/sh",
        'printf "%s\\n" "$1:$2" >> "$MOM_TEST_DOCKER_EVENTS"',
        'case "$1:$2" in',
        "  buildx:version|buildx:prune) exit 0 ;;",
        '  buildx:build) printf \'%s\\n\' "$@" > "$MOM_TEST_DOCKER_BUILD"; exit 0 ;;',
        '  image:inspect) exit "$MOM_TEST_IMAGE_MISSING" ;;',
        "  container:inspect) exit 1 ;;",
        "  run:*)",
        '    shift; printf \'%s\\n\' "$@" > "$MOM_TEST_DOCKER_ARGS"',
        '    case " $* " in',
        '      *" --entrypoint "*) printf \'%s\\n\' "$@" > "$MOM_TEST_DOCKER_PROBE"; exit "$MOM_TEST_TOOLS_MISSING" ;;',
        "    esac",
        "    exit 0 ;;",
        "esac",
        "exit 1",
      ].join("\n"));
      fs.chmodSync(docker, 0o755);
      try {
        const initialized = spawnSync("git", ["init", "-q"], {cwd: temp, encoding: "utf8"});
        assert.equal(initialized.status, 0, initialized.stderr);
        const env = Object.fromEntries(Object.entries(process.env)
          .filter(([key]) => !key.startsWith("MOM_")));
        Object.assign(env, {
          MOM_DOCKER_GPUS: "0",
          MOM_CONTAINER_CPUS: "2",
          MOM_CONTAINER_MEMORY: "4g",
          MOM_CONTAINER_NETWORK: "none",
          MOM_CONTAINER_PIDS: "256",
          MOM_GPU_BACKEND: "opencl",
          MOM_NEXAPOW_PROFILE: "1",
          MOM_TEST_DOCKER_ARGS: dockerArgs,
          MOM_TEST_DOCKER_EVENTS: dockerEvents,
          MOM_TEST_DOCKER_PROBE: dockerProbe,
          MOM_TEST_DOCKER_BUILD: dockerBuild,
          MOM_TEST_IMAGE_MISSING: "0",
          MOM_TEST_TOOLS_MISSING: "0",
          MOM_VERTHASH_DATA: verthashData,
          PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
        });
        const result = spawnSync("bash", [path.join(temp, "scripts/dev-container.sh"), "npm", "test", "--", "gpu"], {
          cwd: temp, encoding: "utf8", env,
        });
        assert.equal(result.status, 0, result.stderr);
        const args = fs.readFileSync(dockerArgs, "utf8").trim().split(/\r?\n/);
        assert.deepEqual(args.filter((arg) => arg.startsWith("MOM_GPU_BACKEND")),
          ["MOM_GPU_BACKEND=opencl"]);
        assert.equal(args.some((arg) => arg.startsWith("MOM_GPU_TEST_VENDORS")), false);
        assert.equal(args.includes("MOM_NEXAPOW_PROFILE"), true);
        assert.deepEqual(args.filter((arg) => arg.startsWith("MOM_VERTHASH_DATA")),
          ["MOM_VERTHASH_DATA=/verthash.dat"]);
        assert.equal(args.includes("--device"), false);
        assert.equal(args.includes(
          `type=bind,source=${verthashData},target=/verthash.dat,readonly`), true);
        assert.equal(args.includes("--privileged"), false);
        /** @param {string} flag */
        const flagValue = (flag) => args[args.indexOf(flag) + 1];
        assert.equal(flagValue("--cap-drop"), "ALL");
        assert.deepEqual(args.filter((_arg, index) => args[index - 1] === "--cap-add"),
          ["CHOWN", "DAC_OVERRIDE", "FOWNER"]);
        assert.equal(flagValue("--security-opt"), "no-new-privileges:true");
        assert.equal(flagValue("--cpus"), "2");
        assert.equal(flagValue("--memory"), "4g");
        assert.equal(flagValue("--memory-swap"), "4g");
        assert.equal(flagValue("--pids-limit"), "256");
        assert.equal(flagValue("--network"), "none");
        assert.equal(args.some((arg) => arg.startsWith("MOM_CONTAINER_")), false);
        // CI consumes imported caches before packaging re-enters this wrapper. Explicit worker
        // reuse must keep the loaded image, but normal/forced/missing-image paths still rebuild.
        for (const variant of ["reuse-ci-cache", "ordinary-ci-cache", "forced-image", "missing-image",
          "missing-tools", "ordinary-image"]) {
          fs.writeFileSync(dockerEvents, "");
          const reuse = !variant.startsWith("ordinary");
          const runEnv = {...env, MOM_GPU_BACKEND: "all", MOM_REUSE_BUILT_WORKER: reuse ? "1" : "0",
            MOM_BUILDX_CACHE_DIR: variant === "ordinary-image" ? "" : path.join(temp, "consumed-cache"),
            MOM_BUILDX_CACHE_SCOPES: "acpp-cuda,acpp-hip",
            MOM_BUILDX_CACHE_READ_ONLY: "1", MOM_BUILDX_CACHE_DELETE_AFTER_LOAD: "1",
            MOM_REBUILD_DEV_IMAGE: variant === "forced-image" ? "1" : "0",
            MOM_TEST_IMAGE_MISSING: variant === "missing-image" ? "1" : "0",
            MOM_TEST_TOOLS_MISSING: variant === "missing-tools" ? "1" : "0"};
          const invoked = spawnSync("bash", [path.join(temp, "scripts/dev-container.sh"),
            "npm", "run", "build:release", "--", "--package-only"], {
            cwd: temp, encoding: "utf8", timeout: 10000, env: runEnv,
          });
          assert.equal(invoked.error, undefined);
          assert.equal(invoked.status, 0, `${variant}: ${invoked.stderr}`);
          const events = fs.readFileSync(dockerEvents, "utf8").trim().split("\n");
          const rebuild = ["ordinary-ci-cache", "forced-image", "missing-image"].includes(variant);
          assert.equal(events.filter(event => event === "buildx:build").length,
            rebuild ? 4 : variant === "missing-tools" ? 1 : 0, variant);
          assert.equal(events.includes("buildx:prune"), rebuild, variant);
          assert.equal(events.filter(event => event.startsWith("run:")).length, rebuild ? 1 : 2, variant);
          if (!rebuild) {
            const probe = fs.readFileSync(dockerProbe, "utf8").trim().split("\n");
            assert.ok(probe.includes("--read-only") && probe.includes("no-new-privileges:true"));
            assert.equal(probe[probe.indexOf("--network") + 1], "none");
            assert.deepEqual(probe.slice(-2), ["-ec", "command -v patchelf >/dev/null && command -v readelf >/dev/null"]);
          }
          if (variant === "missing-tools") {
            const repair = fs.readFileSync(dockerBuild, "utf8").trim().split("\n");
            assert.equal(repair[repair.indexOf("--target") + 1], "release-tools");
            assert.ok(repair.includes("MOM_RELEASE_TOOLS_BASE=mom-build-multicompiler"));
          }
          const launched = fs.readFileSync(dockerArgs, "utf8").trim().split("\n");
          assert.deepEqual(launched.slice(-6), ["mom-build-multicompiler", "npm", "run", "build:release", "--", "--package-only"]);
          assert.ok(launched.includes("MOM_REUSE_BUILT_WORKER"));
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("deployment lanes require a test summary before passing", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const deploySource = fs.readFileSync(path.join(root, "scripts", "test-deploy.sh"), "utf8");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-summary-test-"));
      const scripts = path.join(temp, "scripts");
      const fakeBin = path.join(temp, "bin");
      fs.mkdirSync(scripts);
      fs.mkdirSync(fakeBin);
      fs.copyFileSync(path.join(root, "scripts", "test-deploy.sh"),
        path.join(scripts, "test-deploy.sh"));
      for (const command of ["docker", "nvidia-smi"]) {
        const executable = path.join(fakeBin, command);
        fs.writeFileSync(executable, "#!/bin/sh\nexit 0\n");
        fs.chmodSync(executable, 0o755);
      }
      fs.writeFileSync(path.join(temp, "mom-v1.2.3-lin.tgz"), "fixture");
      try {
        const result = spawnSync("bash", [path.join(scripts, "test-deploy.sh")], {
          cwd: temp,
          encoding: "utf8",
          env: {
            ...process.env,
            PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
            MOM_DEPLOY_TARGET: "linux-nvidia",
            MOM_DEPLOY_REUSE_ARCHIVE: "1",
            MOM_RELEASE_VERSION: "1.2.3",
          },
        });
        assert.equal(result.status, 1, result.stdout + result.stderr);
        assert.match(result.stdout, /nvidia-linux \(test summary missing\)/);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
      const multiRelease = shellFunction(deploySource, "test_windows_multi_release");
      for (const status of [0, 37]) {
      /** @type {import("node:child_process").SpawnSyncReturns<string>} */
        const result = spawnSync("bash", ["-c", [
          "set +e",
          "mkdir() { :; }",
          "cp() { :; }",
          "run_windows_root() { printf 'ℹ tests 5\\r\\nℹ pass 3\\r\\nℹ fail 0\\r\\nℹ skipped 2\\r\\n'; return \"$MOM_DEPLOY_FIXTURE_EXIT\"; }",
          multiRelease,
          "test_windows_multi_release",
        ].join("\n")], {
          encoding: "utf8",
          env: {...process.env, MOM_DEPLOY_FIXTURE_EXIT: String(status),
            DEPLOY_SKIP_VECTORS: "0", MOM_DEPLOY_ALGO: "", WINDOWS_STAGE: "fixture",
            WINDOWS_ARCHIVE: "fixture.zip", WIN_RUN: "fixture-only"},
        });
        assert.equal(result.status, status, result.stdout + result.stderr);
        assert.match(result.stdout, /MOM_TEST_SUMMARY 5 3 0 2/);
      }
    });

    test("Linux mixed-vendor archive tests admit only one NVIDIA and one AMD GPU", {
      skip: process.platform !== "linux",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/test-deploy.sh"), "utf8");
      const functionSource = shellFunction(source, "test_linux_release");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-multi-test-"));
      try {
        const sys = path.join(temp, "sys");
        const dri = path.join(temp, "dri");
        fs.mkdirSync(dri);
        /** @type {[string, string][]} */
        const cards = [["card0", "0x8086"], ["card1", "0x1002"], ["card2", "0x1002"]];
        for (const [card, vendor] of cards) {
          const device = path.join(sys, card, "device");
          fs.mkdirSync(path.join(device, "drm", card), {recursive: true});
          fs.writeFileSync(path.join(device, "vendor"), vendor);
          fs.symlinkSync("/dev/null", path.join(dri, card));
        }
        fs.mkdirSync(path.join(sys, "card1/device/drm/renderD129"));
        fs.symlinkSync("/dev/null", path.join(dri, "renderD129"));
        // Replace only discovery roots: the real admission grammar and character-device check run.
        const fixtureFunction = functionSource.replaceAll("/sys/class/drm", sys).replaceAll("/dev/dri", dri);
        for (const controls of [{}, {DEPLOY_SKIP_VECTORS: "1"}, {MOM_DEPLOY_ALGO: "octopus"},
          {FIXTURE_GPU_MISSING: "1"}, {FIXTURE_GPU_BROKEN: "1"}, {FIXTURE_RUNTIME_FAIL: "37"},
          {FIXTURE_UUID: "invalid"}]) {
          const result = spawnSync("bash", ["-c", [
            "set -euo pipefail",
            "skip() { echo SKIP; }",
            'has_drm_vendor() { [ "$FIXTURE_GPU_MISSING" != 1 ]; }',
            'linux_gpu_available() { [ "$FIXTURE_GPU_MISSING" != 1 ] && [ "$FIXTURE_GPU_BROKEN" != 1 ]; }',
            'nvidia-smi() { echo "$FIXTURE_UUID"; }',
            'docker() { printf "DOCKER\\n"; printf "%s\\n" "$@"; return "$FIXTURE_RUNTIME_FAIL"; }',
            fixtureFunction,
            "test_linux_release multi",
          ].join("\n")], {encoding: "utf8", env: {...process.env,
            DEPLOY_SKIP_VECTORS: "0", MOM_DEPLOY_ALGO: "", FIXTURE_GPU_MISSING: "0", FIXTURE_GPU_BROKEN: "0",
            FIXTURE_RUNTIME_FAIL: "0",
            FIXTURE_UUID: "GPU-12345678-1234-1234-1234-123456789abc",
            MOM_VERTHASH_DATA: "", MOM_RELEASE_PERF_EVIDENCE_DIR: "",
            ROOT_DIR: temp, LINUX_ARCHIVE: "fixture.tgz", RELEASE_DIR: "fixture", ...controls}});
          if (controls.FIXTURE_UUID || controls.FIXTURE_GPU_BROKEN) {
            assert.equal(result.status, 1);
            assert.doesNotMatch(result.stdout, /DOCKER/);
            assert.doesNotMatch(result.stdout, /^SKIP$/m);
          } else if (controls.FIXTURE_RUNTIME_FAIL) {
            assert.equal(result.status, 37);
            assert.match(result.stdout, /DOCKER/);
            assert.match(result.stdout, /MOM_REQUIRE_MULTI_GPU_TESTS=1/);
            assert.doesNotMatch(result.stdout, /^SKIP$/m);
          } else if (Object.keys(controls).length) {
            assert.equal(result.status, 0, result.stderr);
            assert.match(result.stdout, /SKIP/);
            assert.doesNotMatch(result.stdout, /DOCKER/);
          } else {
            assert.equal(result.status, 0, result.stderr);
            assert.match(result.stdout, /device=GPU-12345678-1234-1234-1234-123456789abc/);
            assert.ok(result.stdout.includes(`--device=${dri}/card1`));
            assert.ok(result.stdout.includes(`--device=${dri}/renderD129`));
            assert.ok(!result.stdout.includes(`--device=${dri}/card0`));
            assert.ok(!result.stdout.includes(`--device=${dri}/card2`));
            assert.match(result.stdout, /MOM_GPU_TEST_VENDORS="nvidia,amd"/);
            assert.match(result.stdout, /MOM_REQUIRE_MULTI_GPU_TESTS=1/);
            assert.match(result.stdout, /test-release-linux\.sh "\$MOM_RELEASE_ARCHIVE" gpu-multi/);
            assert.match(result.stdout, /\[ "gpu-multi" != gpu-multi \] \|\| exit 0/);
          }
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Linux release builds force portable compiler mode", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-portable-test-"));
      const scripts = path.join(temp, "scripts");
      const fakeBin = path.join(temp, "bin");
      const rLog = path.join(temp, "r-env");
      const dockerLog = path.join(temp, "docker-invocations");
      fs.mkdirSync(scripts);
      fs.mkdirSync(fakeBin);
      fs.copyFileSync(path.join(root, "scripts", "test-deploy.sh"),
        path.join(scripts, "test-deploy.sh"));
      const fakeR = path.join(temp, "scripts/dev-container.sh");
      fs.writeFileSync(fakeR, [
        "#!/bin/sh",
        'printf "%s\\n" "$MOM_PORTABLE_BUILD" > "$MOM_TEST_R_LOG"',
        'printf "%s\\n" "$MOM_GPU_BACKEND" >> "$MOM_TEST_R_LOG"',
        "exit 42",
      ].join("\n"));
      fs.chmodSync(fakeR, 0o755);
      const fakeDocker = path.join(fakeBin, "docker");
      fs.writeFileSync(fakeDocker, [
        "#!/bin/sh",
        'printf "%s\\n" "$*" >> "$MOM_TEST_DOCKER_LOG"',
        "exit 0",
      ].join("\n"));
      fs.chmodSync(fakeDocker, 0o755);
      try {
        const result = spawnSync("bash", [path.join(scripts, "test-deploy.sh")], {
          cwd: temp,
          encoding: "utf8",
          env: {
            ...process.env,
            PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
            MOM_DEPLOY_TARGET: "linux-nvidia",
            MOM_DEPLOY_REUSE_ARCHIVE: "0",
            MOM_PORTABLE_BUILD: "0",
            MOM_RELEASE_VERSION: "1.2.3",
            MOM_TEST_DOCKER_LOG: dockerLog,
            MOM_TEST_R_LOG: rLog,
          },
        });
        assert.equal(result.status, 1, result.stdout + result.stderr);
        assert.equal(fs.readFileSync(rLog, "utf8"), "1\nall\n");
        assert.equal(fs.existsSync(dockerLog), false);
        assert.equal(fs.existsSync(path.join(temp, "mom-v1.2.3-lin.tgz")), false);
        assert.match(result.stdout, /Build Linux release \(exit 42\)/);
        assert.doesNotMatch(result.stdout + result.stderr, /Packaging Linux release archive/);
        assert.doesNotMatch(result.stdout + result.stderr, /Testing every nvidia GPU/);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });
  });

  test.describe("Bounded vectors and integrated GPU coverage", {concurrency: false}, () => {
    test("GPU vendor selections keep canonical order, subsets and deduplication", (t) => {
      const previousVendors = process.env["MOM_GPU_TEST_VENDORS"];
      const previousBackend = process.env["MOM_GPU_BACKEND"];
      t.after(() => {
        if (previousVendors === undefined) {delete process.env["MOM_GPU_TEST_VENDORS"];} else {
          process.env["MOM_GPU_TEST_VENDORS"] = previousVendors;
        }
        if (previousBackend === undefined) {delete process.env["MOM_GPU_BACKEND"];} else {
          process.env["MOM_GPU_BACKEND"] = previousBackend;
        }
      });
      process.env["MOM_GPU_BACKEND"] = "opencl";
      for (const [selected, expected] of [
        ["AMD, intel,NVIDIA,AMD", ["intel", "nvidia", "amd"]],
        ["nvidia,amd,nvidia", ["nvidia", "amd"]],
        ["amd,intel,intel", ["intel", "amd"]],
        ["all", ["intel", "nvidia", "amd"]],
      ]) {
        assert.equal(typeof selected, "string");
        process.env["MOM_GPU_TEST_VENDORS"] = /** @type {string} */ (selected);
        assert.deepEqual(requestedVendors(), expected);
      }
      process.env["MOM_GPU_TEST_VENDORS"] = "amd,unknown";
      assert.throws(requestedVendors, /Unknown MOM_GPU_TEST_VENDORS/);
      delete process.env["MOM_GPU_TEST_VENDORS"];
      assert.deepEqual(requestedVendors(), []);
      process.env["MOM_GPU_BACKEND"] = "intel";
      assert.deepEqual(requestedVendors(), ["intel"]);
    });

    test("GPU discrete matrix canonicalizes shuffled vendors without renumbering devices", async (t) => {
      t.mock.method(policyModule, "selection", () => ({backend: "sycl"}));
      /** @type {Parameters<typeof addDiscreteCases>[0]} */
      const plan = new Map();
      /** @type {string[]} */
      const discovered = [];
      /** @type {Record<string, string>} */
      const ids = {intel: "gpu8", nvidia: "gpu4", amd: "gpu2"};
      await addDiscreteCases(plan, "octopus", ["amd", "nvidia", "intel", "amd"], async (algo, vendor) => {
        discovered.push(vendor);
        const dev = ids[vendor];
        assert.ok(dev);
        return {skipped: false, devices: [{dev, description: `${vendor} GPU`, integrated: false}],
          params: {[algo]: dev}};
      });
      assert.deepEqual(discovered, ["intel", "nvidia", "amd"]);
      const cases = plan.get("sycl")?.discrete;
      assert.ok(cases);
      assert.deepEqual(cases.map(({name}) => name.split(":")[0]), ["intel gpu8", "nvidia gpu4", "amd gpu2"]);
      for (const entry of cases) {
        const definition = entry.definitions[0];
        assert.ok(definition && entry.select);
        const selected = entry.select(definition);
        assert.ok(selected.job.dev?.startsWith(entry.name.split(":")[0]?.split(" ")[1] || "missing"));
        assert.equal(selected.job["backend"], "sycl");
      }
      await assert.rejects(addDiscreteCases(new Map(), "octopus", ["unknown"], async () => {
        throw new Error("Unknown vendors must not reach discovery");
      }), /Unknown GPU matrix vendors/);
    });

    test("GPU generic matrix sorts shuffled discoveries while preserving IDs and within-vendor order", async (t) => {
      t.mock.method(policyModule, "selection", () => ({backend: "sycl"}));
      t.mock.method(policyModule, "workerEnv", () => ({}));
      const devices = [
        {dev: "gpu3", description: "AMD Radeon fixture", integrated: false},
        {dev: "gpu9", description: "NVIDIA fixture", integrated: false},
        {dev: "gpu8", description: "Intel(R) fixture", integrated: false},
        {dev: "gpu0", description: "Other fixture", integrated: false},
        {dev: "gpu2", description: "Intel fixture", integrated: false},
      ];
      const original = devices.map(({dev}) => dev);
      /** @type {Parameters<typeof addGenericCases>[0]} */
      const plan = new Map();
      await addGenericCases(plan, "octopus", {
        openclGpu: async (algo) => ({skipped: false, devices,
          params: {[algo]: devices.map(({dev}) => dev).join(",")}}),
        openclCpu: async () => ({skipped: true, reason: "Fixture CPU unavailable"}),
        intelIntegrated: async () => ({skipped: true, reason: "Fixture integrated GPU unavailable"}),
      });
      const cases = plan.get("sycl")?.serial.filter(({name}) => name.startsWith("OpenCL "));
      assert.ok(cases);
      assert.deepEqual(cases.map(({name}) => name.split(":")[0]), [
        "OpenCL intel gpu8", "OpenCL intel gpu2", "OpenCL nvidia gpu9", "OpenCL amd gpu3", "OpenCL other gpu0",
      ]);
      assert.deepEqual(devices.map(({dev}) => dev), original);
      const selectedIds = cases.map((entry) => {
        const definition = entry.definitions[0];
        assert.ok(definition && entry.select);
        const selected = entry.select(definition);
        assert.equal(selected.job["backend"], "sycl-opencl");
        assert.equal(selected.env?.["MOM_GPU_BACKEND"], "opencl");
        assert.equal(typeof selected.job.dev, "string");
        return /** @type {string} */ (selected.job.dev).match(/^gpu\d+/)?.[0];
      });
      assert.deepEqual(selectedIds, ["gpu8", "gpu2", "gpu9", "gpu3", "gpu0"]);
    });

    test("Discrete Octopus vectors select the full DAG test path", (/** @type {import("node:test").TestContext} */ t) => {
      t.mock.method(policyModule, "workerEnv", (
      /** @type {string} */ algo, /** @type {NodeJS.ProcessEnv} */ env,
      ) => {
        assert.equal(algo, "__control__");
        assert.equal(env["MOM_GPU_BACKEND"], "opencl");
        assert.equal(env["MOM_OPENCL_DEVICE_TYPE"], "cpu");
        return {ONEAPI_DEVICE_SELECTOR: "opencl:cpu"};
      });
      const octopus = hashTests.find(({job}) => job.algo === "octopus");
      assert.ok(octopus, "the real Octopus vector must exist");
      const original = JSON.stringify(octopus);
      for (const vendor of ["intel", "nvidia", "amd"]) {
        for (const backend of ["sycl", "sycl-native", "native"]) {
          const copy = cloneForDiscreteGpu(octopus, vendor, "gpu2", backend);
          assert.equal(copy.env?.["MOM_OCTOPUS_TEST_FULL_DAG"], "1");
          assert.equal(copy.env?.["MOM_OCTOPUS_TEST_NATIVE"], backend === "sycl-native" ? "1" : undefined);
          assert.equal(copy.env?.["MOM_GPU_BACKEND"], vendor);
          assert.ok(Object.hasOwn(copy.env || {}, "MOM_NATIVE_PATH"));
          assert.equal(copy.env?.["MOM_NATIVE_PATH"], undefined);
          assert.equal(copy.env?.["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], undefined);
          assert.equal(copy.job["backend"], backend);
        }
      }

      const openclCpu = cloneForOpenclSycl(octopus, "gpu2", "cpu");
      assert.equal(openclCpu.env?.["MOM_OCTOPUS_TEST_FULL_DAG"], undefined);
      assert.equal(openclCpu.env?.["MOM_OCTOPUS_TEST_NATIVE"], undefined);
      assert.equal(openclCpu.env?.["ONEAPI_DEVICE_SELECTOR"], "opencl:cpu");
      const intelIntegrated = cloneForIntelIntegrated(octopus, "gpu2", "sycl-native");
      assert.equal(intelIntegrated.env?.["MOM_GPU_BACKEND"], "intel_igpu");
      assert.equal(intelIntegrated.env?.["MOM_OCTOPUS_TEST_FULL_DAG"], undefined);
      assert.equal(intelIntegrated.env?.["MOM_COMPILER_POLICY_STRICT"], "1");
      assert.equal(intelIntegrated.job["backend"], "sycl-native");

      for (const algo of ["kawpow", "firopow", "evrprogpow", "meowpow"]) {
        const definition = hashTests.find(({job, syclCpu}) => job.algo === algo && syclCpu);
        assert.ok(definition, `the portable ${algo} vector must exist`);
        const integrated = cloneForIntelIntegrated(definition, "gpu2", "sycl-native");
        assert.equal(integrated.timeoutMs, 3 * 60 * 60 * 1000);
      }

      const etchash = hashTests.find(({job}) => job.algo === "etchash");
      assert.ok(etchash, "the real Etchash vector must exist");
      const etchashCopy = cloneForDiscreteGpu(etchash, "amd", "gpu2", "sycl");
      assert.equal(etchashCopy.env?.["MOM_OCTOPUS_TEST_FULL_DAG"], undefined);
      assert.equal(etchashCopy.env?.["MOM_OCTOPUS_TEST_NATIVE"], undefined);
      const integratedEtchash = cloneForIntelIntegrated(etchash, "gpu2", "sycl");
      assert.equal(integratedEtchash.timeoutMs, 3 * 60 * 60 * 1000);
      assert.equal(JSON.stringify(octopus), original);
    });

    test("bounded CPU gold and geometry never replace GPU or integrated vectors", (
    /** @type {import("node:test").TestContext} */ t,
    ) => {
      t.mock.method(policyModule, "workerEnv", () => ({}));
      const definitions = hashTests.filter(({syclCpuExpected}) => syclCpuExpected !== undefined);
      assert.equal(definitions.length, 6);
      for (const definition of definitions) {
        const original = JSON.stringify(definition);
        const cpu = cloneForOpenclSycl(definition, "cpu1", "cpu");
        assert.equal(cpu.expected, definition.syclCpuExpected);
        assert.equal(cpu.env?.["MOM_SYCL_PORTABLE_TEST"], "1");
        for (const gpu of [cloneForOpenclSycl(definition, "gpu2", "gpu"),
          cloneForDiscreteGpu(definition, "intel", "gpu2", "sycl"),
          cloneForIntelIntegrated(definition, "gpu2", "sycl")]) {
          assert.equal(gpu.expected, definition.expected);
          assert.notEqual(gpu.env?.["MOM_SYCL_PORTABLE_TEST"], "1");
        }
        assert.equal(JSON.stringify(definition), original);
      }
    });

    test("Verthash bounded CPU/iGPU fixture checks its route and independent scalar gold", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const source = fs.readFileSync(path.join(root, "sycl/verthash/verthash.cpp"), "utf8");
      const internal = fs.readFileSync(path.join(root, "sycl/lib-internal.h"), "utf8");
      const compactHelpers = ["inline bool mom_parse_env_ulong(", "inline bool is_integrated_gpu(",
        "inline bool mom_compact_correctness_test("].map(anchor => extractCppBlock(internal, anchor)).join("\n");
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-verthash-small-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/verthash-test-route.cpp"),
          path.join(directory, "route.cpp"));
        fs.writeFileSync(path.join(directory, "verthash_state_for.inc"),
          `${compactHelpers}\n${extractCppBlock(source, "static State& state_for(")}`);
        /** @type {[string, string][]} */
        const cases = [
          [path.join(directory, "route.cpp"), "verthash-test-route:passed"],
          [path.join(__dirname, "reference/verthash-small.cpp"), "verthash-small-reference:passed"],
        ];
        for (const [file, marker] of cases) {
          const executable = path.join(directory, path.basename(file, ".cpp"));
          const built = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", "-I", path.join(__dirname, "native"), file, "-o", executable],
          {encoding: "utf8", timeout: 30000});
          assert.equal(built.status, 0, built.stderr);
          const run = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
          assert.equal(run.status, 0, run.stderr);
          assert.equal(run.stdout.trim(), marker);
        }
      } finally {
        fs.rmSync(directory, {recursive: true, force: true});
      }
    });

    test("Etchash bounded CPU gold matches the independent scalar reference", {
      skip: process.platform === "win32" ? "requires host C/C++ compilers" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-etchash-small-"));
      try {
        const object = path.join(directory, "ethash.o");
        const executable = path.join(directory, "reference");
        const c = spawnSync("cc", ["-std=c11", "-O2", "-I", path.join(root, "xmrig"),
          "-c", path.join(root, "xmrig/3rdparty/libethash/ethash_internal.c"), "-o", object],
        {encoding: "utf8", timeout: 30000});
        assert.equal(c.status, 0, c.stderr);
        const cpp = spawnSync("c++", ["-std=c++17", "-O2", "-I", path.join(root, "xmrig"),
          path.join(__dirname, "reference/etchash-small.cpp"),
          path.join(root, "xmrig/base/crypto/sha3.cpp"),
          path.join(root, "xmrig/base/crypto/keccak.cpp"), object, "-o", executable],
        {encoding: "utf8", timeout: 30000});
        assert.equal(cpp.status, 0, cpp.stderr);
        const run = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
        assert.equal(run.status, 0, run.stderr);
        assert.equal(run.stdout.trim(), "etchash-small-reference:passed");
      } finally {
        fs.rmSync(directory, {recursive: true, force: true});
      }
    });

    test("portable CPU emulation timeout is isolated from native CPU and GPU clones", (
    /** @type {import("node:test").TestContext} */ t,
    ) => {
      const definition = hashTests.find(({job, syclCpu}) => syclCpu && job.algo === "kawpow");
      assert.ok(definition, "the real portable KawPow vector must exist");
      const original = JSON.stringify(definition);
      const expectedHash = definition.expected;
      const expectedBlob = definition.job.blob_hex;
      const baseTimeout = Math.max(definition.timeoutMs || 0, 15 * 60 * 1000);
      const previousEmulation = process.env["MOM_RELEASE_EMULATE_INTEL_CPU"];
      t.mock.method(policyModule, "workerEnv", () => ({}));

      try {
        delete process.env["MOM_RELEASE_EMULATE_INTEL_CPU"];
        const nativeCpu = cloneForOpenclSycl(definition, "gpu2", "cpu");
        assert.equal(nativeCpu.timeoutMs, baseTimeout);

        process.env["MOM_RELEASE_EMULATE_INTEL_CPU"] = "1";
        const emulatedCpu = cloneForOpenclSycl(definition, "gpu2", "cpu");
        assert.equal(emulatedCpu.timeoutMs, 60 * 60 * 1000);
        assert.equal(emulatedCpu.gpu, false);

        const openclGpu = cloneForOpenclSycl(definition, "gpu2", "gpu");
        assert.equal(openclGpu.timeoutMs, baseTimeout);
        assert.equal(openclGpu.gpu, true);

        const discreteGpu = cloneForDiscreteGpu(definition, "nvidia", "gpu2", "sycl");
        assert.equal(discreteGpu.timeoutMs, definition.timeoutMs);
        const nativeIntegrated = cloneForIntelIntegrated(definition, "gpu2", "sycl-native");
        assert.equal(nativeIntegrated.timeoutMs, 3 * 60 * 60 * 1000);

        for (const copy of [nativeCpu, emulatedCpu]) {
          assert.equal(copy.expected, definition.syclCpuExpected);
          assert.equal(copy.job.blob_hex, expectedBlob);
        }
        for (const copy of [openclGpu, discreteGpu, nativeIntegrated]) {
          assert.equal(copy.expected, expectedHash);
          assert.equal(copy.job.blob_hex, expectedBlob);
        }
      } finally {
        if (previousEmulation === undefined) {
          delete process.env["MOM_RELEASE_EMULATE_INTEL_CPU"];
        } else {
          process.env["MOM_RELEASE_EMULATE_INTEL_CPU"] = previousEmulation;
        }
        assert.equal(JSON.stringify(definition), original,
          "cloning a portable vector must not mutate its input or hashes");
      }
    });

    test("RandomX resume fixture tolerates cold setup but still diagnoses a stalled phase", async () => {
      const source = fs.readFileSync(path.join(__dirname, "native-addon.js"), "utf8");
      const start = source.indexOf('test("native RandomX labels its first result and resumes the same job"');
      assert(start >= 0, "RandomX resume fixture must exist");
      const closing = "\n  });";
      const end = source.indexOf(closing, start);
      assert(end > start, "RandomX resume fixture must close inside its suite");
      const fixture = source.slice(start, end + closing.length);

      /** @param {string} body @param {boolean} stall */
      const run = async (body, stall) => {
        let now = 0;
        let nextId = 0;
        let closeCount = 0;
        let outcome = Promise.resolve();
        /** @type {Map<number, {at: number, callback: () => void}>} */
        const timers = new Map();
        /** @param {() => void} callback @param {number} delay */
        const schedule = (callback, delay) => {
          const id = ++nextId;
          timers.set(id, {at: now + delay, callback});
          return id;
        };
        class Worker {
        /** @param {(name: string, values: Record<string, string>) => void} progress
         * @param {() => void} complete */
          constructor(progress, complete) {
            this.progress = progress;
            this.complete = complete;
          }
          /** @param {string} name @param {Record<string, string>} [job] */
          sendToCpp(name, job = {}) {
            if (name === "test" && !stall) {
              schedule(() => this.progress("test", {result: "fixture-hash"}), 13000);
            } else if (name === "job") {
              schedule(() => this.progress("result", {
                job_id: job["job_id"] || "", nonce: "00000000", hash: "fixture-hash",
              }), 100);
            } else if (name === "close") {
              ++closeCount;
              schedule(this.complete, 0);
            }
          }
        }
        runInNewContext(body, {
          assert, core: {AsyncWorker: Worker}, setTimeout: schedule,
          clearTimeout: (/** @type {number} */ id) => timers.delete(id),
          test: (/** @type {string} */ name,
          /** @type {{timeout: number}} */ options,
            /** @type {() => Promise<void>} */ execute) => {
            assert.match(name, /RandomX labels/);
            assert(options.timeout > 30000 && options.timeout <= 60000);
            outcome = execute();
          },
        });
        for (let count = 0; timers.size; ++count) {
          assert(count < 10, "fixture must finish with bounded work");
          const next = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
          assert(next);
          timers.delete(next[0]);
          now = next[1].at;
          next[1].callback();
        }
        try {
          await outcome;
        } finally {
          assert.equal(closeCount, 1, "fixture closes exactly once on success or failure");
        }
      };
      await run(fixture, false);
      await assert.rejects(run(fixture.replace("}, 30000);", "}, 10000);"), false),
        /timed out in phase test/);
      await assert.rejects(run(fixture, true), /timed out in phase test/);
    });

    test("PearlHash AMD discovery keeps the ordinary profile when capacity metadata throws", {
      skip: process.platform === "win32",
    }, () => {
      const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
      const capacityProbe = extractCppBlock(intensity, "  if (pearlhash_amd_native_int8(dev)) {");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-capacity-probe-"));
      const source = path.join(fixture, "capacity_probe.cpp");
      const executable = path.join(fixture, "capacity_probe");
      fs.writeFileSync(source, [
        "#include <cstdio>",
        "struct Device {};",
        "struct Profile { unsigned m; };",
        "static bool pearlhash_amd_native_int8(const Device&) { return true; }",
        "static Profile pow_device_profile(const Device&) { throw 7; }",
        "static Profile pearlhash_amd_native_profile_for(const Profile&, bool) { return {524288}; }",
        "int main() {",
        "  Device dev;",
        "  Profile profile{131072};",
        capacityProbe,
        "  if (profile.m != 131072) return 1;",
        "  std::puts(\"PASS capacity fallback\");",
        "}",
        "",
      ].join("\n"));
      try {
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
        assert.equal(result.stdout, "PASS capacity fallback\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("integrated coverage fails unavailable devices and algorithms only when required", async (t) => {
      const previous = process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"];
      process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = "1";
      t.after(() => {
        if (previous === undefined) {
          delete process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"];
        } else {
          process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = previous;
        }
      });

      const results = [
        {
          result: {skipped: /** @type {false} */ (false), devices: [], params: {}},
          reason: /no devices/,
        },
        {
          result: {
            skipped: /** @type {false} */ (false),
            devices: [{dev: "gpu1", description: "Intel integrated test device", integrated: true}],
            params: {},
          },
          reason: /not available/,
        },
      ];
      for (const required of [true, false]) {
        process.env["MOM_REQUIRE_INTEGRATED_GPU_TESTS"] = required ? "1" : "0";
        for (const {result, reason} of results) {
          /** @type {Parameters<typeof addIntegratedCases>[0]} */
          const plan = new Map();
          await addIntegratedCases(plan, "octopus", {
            openclGpu: async () => {throw new Error("unexpected OpenCL discovery");},
            openclCpu: async () => {throw new Error("unexpected CPU discovery");},
            intelIntegrated: async () => result,
          });
          const cases = [...plan.values()].flatMap((lane) => lane.serial);
          assert.equal(cases.length, 1);
          assert.equal(cases[0]?.required, required);
          assert.match(cases[0]?.reason || "", reason);
        }
      }
    });

    test("Pearl V3 policy matrix retains both SYCL fallback backends",
      async (/** @type {import("node:test").TestContext} */ t) => {
        const pearls = hashTests.filter(({job}) => job.algo === "pearlhash");
        assert.equal(pearls.length, 3);
        assert.ok(pearls.every(({job}) => job["pearlhash_cert_version"] === 3));
        t.mock.method(policyModule, "nvidiaComputeCapability", () => 120);
        /** @type {Parameters<typeof addDiscreteCases>[0]} */
        const plan = new Map();
        /** @param {string} algo @param {string} vendor
       * @returns {Promise<{skipped: false, devices: Array<{dev: string, description: string, integrated: boolean}>, params: Record<string, string>}>} */
        const discover = async (algo, vendor) => {
          assert.equal(algo, "pearlhash");
          assert.ok(vendor === "nvidia" || vendor === "amd");
          return {
            skipped: false,
            devices: [{dev: "gpu1", description: `${vendor} test GPU`, integrated: false}],
            params: {pearlhash: "gpu1"},
          };
        };
        await addDiscreteCases(plan, "pearlhash", ["nvidia", "amd"], discover);
        assert.deepEqual([...plan.keys()].sort(), ["native", "sycl", "sycl-native"]);
        assert.equal(plan.get("sycl")?.discrete.length, 2);
        assert.equal(plan.get("sycl-native")?.discrete.length, 2);
        assert.equal(plan.get("native")?.discrete.length, 2);

        const nvidiaSelection = policyModule.selection("pearlhash", "nvidia", process.platform, 120);
        const amdSelection = policyModule.selection("pearlhash", "amd", process.platform);
        assert.ok(nvidiaSelection);
        assert.ok(amdSelection);
        assert.equal(nvidiaSelection.backend, "native");
        assert.ok(nvidiaSelection.pearlhashProfile);
        assert.equal(nvidiaSelection.pearlhashProfile.k, 8192);
        assert.equal(nvidiaSelection.pearlhashProfile.rank, 128);
        assert.equal(amdSelection.backend, "native");

        for (const [laneBackend, lane] of plan) {
          for (const testCase of lane.discrete) {
            assert.deepEqual(testCase.definitions, pearls);
            const select = testCase.select;
            assert.ok(select);
            for (const pearl of pearls) {
              const selected = select(pearl);
              assert.equal(selected.job["backend"], laneBackend);
              assert.equal(selected.job["pearlhash_cert_version"],
                pearl.job["pearlhash_cert_version"]);
              assert.equal(selected.env?.["MOM_GPU_BACKEND"],
                testCase.name.startsWith("nvidia ") ? "nvidia" : "amd");
            }
          }
        }
        assert.ok(plan.get("sycl")?.discrete.every(({name}) => name.includes("(sycl fallback)")));
        assert.ok(plan.get("sycl-native")?.discrete.every(({name}) =>
          name.includes("(sycl-native fallback)")));
      });
  });

  test.describe("Incremental builds, hardware capabilities, and portable GPU kernels", {concurrency: false}, () => {
    test("Linux AdaptiveCpp rejects a symlinked build root before cleanup", {
      skip: process.platform === "win32",
    }, () => {
      const repo = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-adaptive-root-test-"));
      const external = fs.mkdtempSync(path.join(os.tmpdir(), "mom-adaptive-external-test-"));
      try {
        fs.mkdirSync(path.join(temp, "scripts"));
        fs.copyFileSync(path.join(repo, "scripts/adaptivecpp-entrypoint.sh"),
          path.join(temp, "scripts/adaptivecpp-entrypoint.sh"));
        fs.copyFileSync(path.join(repo, "scripts/build-helpers.sh"),
          path.join(temp, "scripts/build-helpers.sh"));
        fs.copyFileSync(path.join(repo, "scripts/build-environment.js"),
          path.join(temp, "scripts/build-environment.js"));
        fs.writeFileSync(path.join(temp, "scripts/cxx-adaptivecpp.sh"), "#!/bin/sh\nexit 0\n");
        fs.symlinkSync(external, path.join(temp, "build"), "dir");
        const result = spawnSync("bash", ["scripts/adaptivecpp-entrypoint.sh", "true"], {
          cwd: temp,
          encoding: "utf8",
          env: {...process.env, MOM_ADAPTIVE_BUILD_DIR: "build/cache/acpp-hip"},
        });
        assert.equal(result.status, 2, result.stdout + result.stderr);
        assert.match(result.stderr, /build root must not be a symbolic link/);
        assert.deepEqual(fs.readdirSync(external), []);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
        fs.rmSync(external, {recursive: true, force: true});
      }
    });

    test("Linux source receipts hash the combined compiler wrapper", () => {
      const {sourceInputs, fileHash} = require("../scripts/build-source-inputs");
      assert.equal(sourceInputs(path.resolve(__dirname, "..")).graph["scripts/cxx-combined.sh"],
        fileHash(path.join(__dirname, "../scripts/cxx-combined.sh")));
    });

    test("Windows Intel NexaPoW uses oneAPI after the cold-JIT fix", () => {
      assert.equal(policy.selection("nexapow", "intel", "win32").key, "oneapi");
    });

    test("asynchronous Xelis and Zel retain events without adding waits", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl");
      let search = fs.readFileSync(path.join(root, "xelishashv3/search.inc"), "utf8");
      // Omit only GPU kernel bodies: preserve the real stage helper, dispatch, owner scopes and waits.
      const submissions = [...search.matchAll(/^([ \t]*return s\.queue\.submit\(\[&\]\(sycl::handler& h\) \{)/gm)];
      assert.equal(submissions.length, 4);
      for (let i = submissions.length - 1; i >= 0; --i) {
        const match = submissions[i];
        assert.ok(match && match.index !== undefined && match[1]);
        const block = extractCppBlock(search.slice(match.index), match[1]);
        const end = match.index + block.length - 1;
        assert.equal(search.slice(end, end + 2), ");");
        search = search.slice(0, match.index) + "    return s.queue.submit();" + search.slice(end + 2);
      }
      const proof = fs.readFileSync(path.join(root, "zelhash/proof.inc"), "utf8");
      const rounds = extractCppBlock(proof, "static std::array<sycl::event, 4> run_collision_rounds(");
      const entry = fs.readFileSync(path.join(root, "zelhash/zelhash.cpp"), "utf8");
      const begin = entry.indexOf("#if defined(MOM_SYCL_HAS_CUDA)", entry.indexOf("uint64_t t_round[4]"));
      const end = entry.indexOf("  const uint64_t t_rounds_end", begin);
      assert.ok(begin >= 0 && end > begin);
      const call = entry.slice(begin, end);
      assert.equal((call.match(/const auto round_events/g) || []).length, 3);
      // Compile-only ownership coverage: this CPU fixture does not qualify a GPU or identify an
      // AMD root cause. HIP keeps the existing portable/nonportable pair; CUDA exercises both
      // compact dispatches, and the Intel-only branch exercises the generic dispatch.
      const variants = [
        {name: "hip", defines: ["-DMOM_SYCL_HAS_HIP"], compacts: [true], portable: [false, true]},
        {name: "cuda", defines: ["-DMOM_SYCL_HAS_CUDA"], compacts: [true, false], portable: [false]},
        {name: "intel", defines: [], compacts: [false], portable: [false]},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-async-events-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/async-event-lifetime.cpp"), path.join(fixture, "fixture.cpp"));
        for (const candidate of [false, true]) {
          // Negative control removes ONLY retention. A pending-handle mock is an ownership test,
          // not proof that the actual AMD driver stall was caused by its modeled pool behavior.
          const xelis = candidate ? search : search.replace("      return event;\n", "")
            .replace(/\[\[maybe_unused\]\] const auto stage[13]_event = /g, "");
          const zel = candidate ? rounds : rounds
            .replace("static std::array<sycl::event, 4>", "static void")
            .replace("  return {r0, r1, r2, r3};\n", "");
          fs.writeFileSync(path.join(fixture, "xelis-search.inc"), xelis);
          fs.writeFileSync(path.join(fixture, "zel-rounds.inc"), zel);
          fs.writeFileSync(path.join(fixture, "zel-call.inc"), candidate ? call :
            call.replace(/\[\[maybe_unused\]\] const auto round_events =\s*/g, ""));
          for (const variant of variants) {
            for (const portable of variant.portable) {
              const executable = path.join(fixture, `events-${variant.name}-${Number(portable)}`);
              const compiled = spawnSync("c++", ["-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
                "-Wno-unused-variable", "-Wno-unused-parameter", "-D_WIN32", "-DMOM_XELISHASHV3_ESIMD",
                ...variant.defines, "-DCANDIDATE=" + Number(candidate),
                ...(portable ? ["-DMOM_SYCL_PORTABLE_OPENCL"] : []),
                path.join(fixture, "fixture.cpp"), "-o", executable], {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.status, 0,
                `${variant.name} portable=${Number(portable)}: ${compiled.stderr}`);
              for (const compact of variant.compacts) {
                // Cover dedicated native ordering and ACPP's conditional strong
                // previous_submission owner, replaced only after the new record.
                for (const emulated of [false, true]) {
                  const run = spawnSync(executable, [compact ? "1" : "0", emulated ? "1" : "0"], {encoding: "utf8"});
                  assert.equal(run.error, undefined);
                  assert.equal(run.status, candidate ? 0 : 17,
                    `${variant.name} compact=${Number(compact)} portable=${Number(portable)} emulated=${Number(emulated)}: ${run.stderr}`);
                  const hazardous = candidate ? 0 : portable ? 1 : 3;
                  assert.equal(run.stdout,
                    `ASYNC_EVENT_LIFETIME candidate=${Number(candidate)} portable=${Number(portable)} emulated=${Number(emulated)} cases=8 hazardous=${hazardous}\n`,
                    `${variant.name} compact=${Number(compact)} emulated=${Number(emulated)}`);
                }
              }
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Pearl CUDA checkpoint ternary logic preserves every accumulator word", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const kernel = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/cuda-kernel.inc"), "utf8");
      const helper = extractCppBlock(kernel, "__device__ __forceinline__ uint32_t pearlhash_xor3(");
      const assembly = /asm volatile\("lop3\.b32 %0, %1, %2, %3, (0x[0-9a-f]+);"\s*: "=r"\(out\) : "r"\(a\), "r"\(b\), "r"\(c\)\);/;
      const match = helper.match(assembly);
      assert.ok(match, "the actual PTX operand mapping must match the emulated truth table");
      // The first checkpoint tile reduction: eight accumulator words folded into one part.
      const reduction = /const uint32_t v0 = uint32_t\(accumulator\(lo\)\);[\s\S]*?uint32_t part = [^;]+;/.exec(kernel);
      assert.ok(reduction);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-checkpoint-xor-"));
      try {
        const source = path.join(fixture, "reduction.cpp"), executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/pearlhash-cuda-reduction.cpp"), source);
        fs.writeFileSync(path.join(fixture, "reduction.inc"), reduction[0] + "\n");
        for (const negative of [false, true]) {
          const truth = Number(match[1]) ^ Number(negative);
          fs.writeFileSync(path.join(fixture, "xor3.inc"),
            helper.replace(assembly, `out = fixture_lop3(a, b, c, ${truth}u);`));
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            source, "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          assert.equal(result.status, Number(negative), result.stdout + result.stderr);
          assert.match(negative ? result.stderr : result.stdout, negative ? /differs from the canonical reduction/ : /^PASS actual-source CUDA checkpoint/);
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Xelis cleanup attempts every allocation and preserves replacement ownership", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const state = fs.readFileSync(path.join(__dirname, "../sycl/xelishashv3/state.inc"), "utf8");
      const internal = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-xelis-ownership-"));
      try {
        const source = path.join(fixture, "fixture.cpp");
        const executable = path.join(fixture, "fixture");
        fs.copyFileSync(path.join(__dirname, "native/xelishashv3-ownership.cpp"), source);
        fs.writeFileSync(path.join(fixture, "ownership.inc"),
          extractCppBlock(state, "  ~State()") + extractCppBlock(state, "  void ensure(unsigned count)"));
        fs.writeFileSync(path.join(fixture, "usm.inc"), extractCppBlock(internal, "class UsmPtr {") + ";\n" +
          extractCppBlock(internal, "UsmPtr<T> mom_usm_alloc("));
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "PASS XelisHashV3 ownership cleanup and growth\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Zel and Beam reserve solver diagnostics for tests or explicit profiling", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-solver-diagnostics-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/solver-diagnostics.cpp"), path.join(fixture, "fixture.cpp"));
        const parts = ["zelhash", "beamhash3"].map((algo) => {
          const source = fs.readFileSync(path.join(__dirname, "../sycl", algo, algo + ".cpp"), "utf8");
          const prof = source.match(/static const bool prof = [^;]+;/)?.[0];
          const log = source.match(/const bool log = [^;]+;/)?.[0];
          const arena = source.match(/state\.ensure_arenas\([^;]+;/)?.[0];
          assert.ok(prof && log && arena);
          assert.ok(prof.includes(algo === "zelhash" ? "MOM_ZELHASH_PROF" : "MOM_BEAMHASH3_PROF"));
          const start = algo === "zelhash"
            ? source.indexOf("if (log) {", source.indexOf("uint32_t cand_count"))
            : source.lastIndexOf("if (log) {");
          assert.ok(start >= 0);
          // Replace reporting calls only: retain the actual guard, copy size, counter walk and wait.
          const diagnostic = extractCppBlock(source.slice(start), "if (log) {")
            .replaceAll("std::fprintf", "report").replaceAll("std::fflush", "flush");
          return {prefix: algo === "zelhash" ? "zel" : "beam", prof, log, arena, diagnostic};
        });
        for (const legacy of [true, false]) {
          for (const part of parts) {
            const log = legacy ? "const bool log = !is_benchmark;" : part.log;
            const arena = legacy ? "state.ensure_arenas(!is_benchmark);" : part.arena;
            fs.writeFileSync(path.join(fixture, part.prefix + "-setup.inc"), part.prof + "\n" + log + "\n" + arena);
            fs.writeFileSync(path.join(fixture, part.prefix + "-diagnostic.inc"), part.diagnostic);
          }
          const executable = path.join(fixture, legacy ? "legacy" : "candidate");
          const compile = spawnSync("c++", ["-std=c++17", "-O0", path.join(fixture, "fixture.cpp"), "-o", executable], {encoding: "utf8"});
          assert.equal(compile.status, 0, compile.stderr);
          for (const part of parts) {
            for (const mode of legacy ? [0] : [0, 1, 2]) {
              for (const prof of legacy ? [0] : [0, 1]) {
                const env = {...process.env};
                delete env["MOM_ZELHASH_PROF"]; delete env["MOM_BEAMHASH3_PROF"];
                if (prof) {
                  // Presence is the production contract, even for the string "0".
                  env["MOM_ZELHASH_PROF"] = "0"; env["MOM_BEAMHASH3_PROF"] = "0";
                }
                const result = spawnSync(executable, [String(mode), String(prof), part.prefix], {encoding: "utf8", env});
                assert.equal(result.status, legacy ? (part.prefix === "zel" ? 17 : 18) : 0, result.stderr);
                if (legacy) {
                  assert.equal(result.stderr, part.prefix === "zel"
                    ? "FAIL Zel diagnostic mode=0 prof=0 copies=1 waits=1\n"
                    : "FAIL Beam diagnostic mode=0 prof=0 reports=1\n");
                } else {
                  assert.equal(result.stdout, "PASS actual-source solver diagnostic gates\n");
                }
              }
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("BeamHash3 profiles its single test solve without changing steady mining profiling", {
      skip: process.platform === "win32" && "requires a host C++ compiler",
    }, t => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/beamhash3/beamhash3.cpp"), "utf8"),
        expression = source.match(/const bool compact_profile = ([^;\n]+);/)?.[1];
      assert.ok(expression, "exercise the actual compact profile selector");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-beam-profile-selector-")),
        file = path.join(fixture, "profile.cpp"), executable = path.join(fixture, "profile");
      t.after(() => fs.rmSync(fixture, {recursive: true, force: true}));
      fs.writeFileSync(file, `bool profile(bool prof, bool is_test, unsigned& compact_profile_call) { return ${expression}; }
int main() {
  unsigned test_calls = 0, mining_calls = 0, disabled_calls = 0;
  if (!profile(true, true, test_calls) || profile(true, true, test_calls) || profile(true, true, test_calls)) return 1;
  if (profile(true, false, mining_calls) || !profile(true, false, mining_calls) || profile(true, false, mining_calls)) return 2;
  if (profile(false, true, disabled_calls) || profile(false, false, disabled_calls) || disabled_calls != 0) return 3;
  return 0;
}\n`);
      const compiled = spawnSync("c++", ["-std=c++17", "-O0", file, "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(compiled.status, 0, compiled.stderr || compiled.error?.message);
      const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, "first test solve, second mining solve, and opt-out must retain their distinct profiling semantics");
    });

    test("BeamHash3 kernel-limit queries stay profiling-only and select the actual round-one specialization", {
      skip: process.platform === "win32" && "requires a host C++ compiler",
    }, t => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/beamhash3/beamhash3.cpp"), "utf8"),
        guard = "    if (compact_profile && sycl_is_level_zero_gpu(state.device))",
        block = extractCppBlock(source, guard), fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-beam-kernel-limit-")),
        file = path.join(fixture, "query.cpp"), executable = path.join(fixture, "query");
      t.after(() => fs.rmSync(fixture, {recursive: true, force: true}));
      fs.writeFileSync(file, `unsigned calls, contexts, observed_wg, observed_round; bool observed_pair, observed_type;
template <unsigned R, bool Pair> struct BeamCompactRoundKernel { static constexpr unsigned round = R; static constexpr bool paired = Pair; };
struct Queue { int get_context() { ++contexts; return 0; } };
struct State { Queue queue; bool device; };
bool sycl_is_level_zero_gpu(bool device) { return device; }
template <class K> void beamhash3_log_kernel_limit(int, bool, bool paired, unsigned wg) {
  ++calls; observed_pair = paired; observed_type = K::paired; observed_round = K::round; observed_wg = wg;
}
void candidate(bool compact_profile, bool level_zero, bool compact_pair_partitions) {
  State state{{}, level_zero}; unsigned compact_pair_round1_wg = 640, compact_round_wg = 768;
  ${block}
}
int main() {
  for (unsigned prof = 0; prof < 2; ++prof) for (unsigned l0 = 0; l0 < 2; ++l0)
    for (unsigned paired = 0; paired < 2; ++paired) {
      calls = contexts = 0; candidate(prof, l0, paired);
      if (calls != (prof && l0) || contexts != calls) return 1;
      if (calls && (observed_round != 1 || observed_pair != bool(paired) || observed_type != bool(paired) ||
          observed_wg != (paired ? 640u : 768u))) return 2;
    }
  return 0;
}\n`);
      const compiled = spawnSync("c++", ["-std=c++17", "-O0", file, "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(compiled.status, 0, compiled.stderr || compiled.error?.message);
      const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, "ordinary/non-Level-Zero jobs must not build or query a diagnostic bundle");
    });

    test("Octopus cooperative polynomial batches preserve every point and ordered FNV", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, async (t) => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/octopus/device.inc"), "utf8");
      const cooperative = extractCppBlock(source, "inline void octopus_hash_cooperative(");
      const begin = cooperative.indexOf("#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)");
      const end = cooperative.indexOf("  uint32_t seed[CACHE_NODE_WORDS];", begin);
      assert.ok(begin >= 0 && end > begin);
      const constants = ["MOD", "POLYNOMIAL_SIZE", "POLYNOMIAL_LANES", "POLYNOMIAL_BATCH", "FNV_PRIME"]
        .map(name => {
          const declaration = source.match(new RegExp("^constexpr uint32_t " + name + " = [^;]+;", "m"));
          assert.ok(declaration, name);
          return declaration[0];
        });
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-octopus-polynomial-"));
      try {
        fs.writeFileSync(path.join(fixture, "field.inc"), constants.join("\n") + "\n" +
          ["inline uint64_t fnv64(", "inline uint32_t mod_field(", "inline uint32_t mad_field("]
            .map(anchor => extractCppBlock(source, anchor)).join("\n"));
        fs.writeFileSync(path.join(fixture, "polynomial.inc"), cooperative.slice(begin, end));
        for (const [name, defines] of Object.entries({
          ordinary: [], "acpp-cuda": ["MOM_SYCL_ADAPTIVECPP", "MOM_SYCL_ADAPTIVECPP_CUDA"],
          "acpp-hip": ["MOM_SYCL_ADAPTIVECPP", "MOM_SYCL_HAS_HIP"],
          portable: ["MOM_SYCL_PORTABLE_OPENCL"],
        })) {
          await t.test(name, () => {
            const executable = path.join(fixture, name);
            const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-pedantic", "-UNDEBUG", ...defines.map(value => "-D" + value), "-I" + fixture,
              path.join(__dirname, "native/octopus-polynomial.cpp"), "-o", executable],
            {encoding: "utf8", timeout: 60000});
            assert.equal(compiled.error, undefined);
            assert.equal(compiled.status, 0, compiled.stderr);
            const result = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout,
              "Octopus polynomial: 512 lanes / 16384 points and ordered FNV passed\n");
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Xelis, HooHash, ESIMD and portable wide products retain exact limb carries", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, async (t) => {
      const shared = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
      const xelis = fs.readFileSync(path.join(__dirname, "../sycl/xelishashv3/device.inc"), "utf8");
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wide-multiply-"));
      t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
      fs.writeFileSync(path.join(directory, "wide-multiply.inc"),
        extractCppBlock(shared, "inline uint64_t mom_mul_wide_u32(") +
        extractCppBlock(shared, "inline uint64_t mom_mul_hi_u64(") +
        "inline uint64_t mul_wide32(uint32_t a, uint32_t b) { return mom_mul_wide_u32(a,b); }\n" +
        extractCppBlock(xelis, "inline uint64_t mul_hi64_portable("));
      const variants = [[], ["MOM_SYCL_PORTABLE_OPENCL"], ["MOM_SYCL_ADAPTIVECPP"], ["MOM_XELISHASHV3_ESIMD"],
        ["MOM_XELISHASHV3_ESIMD", "__INTEL_LLVM_COMPILER=20260000"]];
      for (const defines of variants) {
        const name = defines.join("+") || "ordinary";
        await t.test(name, () => {
          const executable = path.join(directory, name);
          const built = spawnSync("c++", ["-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-UNDEBUG",
            ...defines.map(define => "-D" + define), "-I", directory, path.join(__dirname, "native/wide-multiply.cpp"), "-o", executable],
          {encoding: "utf8", timeout: 30000});
          assert.equal(built.status, 0, built.error?.message || built.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
        });
      }
    });

  });

  test.describe("Equihash geometry, cache policy, and compiler scoping", {concurrency: false}, () => {
    test("BeamHash3 portable worker cannot select uncompiled compact arenas", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, (t) => {
      const layout = fs.readFileSync(path.join(__dirname, "../sycl/beamhash3/layout.inc"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-beam-compact-portable-"));
      t.after(() => fs.rmSync(fixture, {recursive: true, force: true}));
      fs.writeFileSync(path.join(fixture, "beam_compact_enabled.inc"),
        extractCppBlock(layout, "inline bool beam_compact_enabled("));
      // The non-portable control proves the fixture can observe compact selection at all.
      for (const [portable, expected] of [[true, 0], [false, 9]]) {
        const executable = path.join(fixture, portable ? "portable" : "native");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
          ...(portable ? ["-DMOM_SYCL_PORTABLE_OPENCL"] : []), "-I", fixture,
          path.join(__dirname, "native/beam-compact-portable.cpp"), "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
        assert.equal(result.stdout, `beam compact enabled=${expected}\n`);
      }
    });

    test("Combined compiler wrapper fails closed on missing targets or an AOT device without spir64_gen", {
      skip: process.platform === "win32",
    }, () => {
      const env = {...process.env};
      for (const name of ["npm_package_json", "MOM_COMBINED_TARGETS", "MOM_INTEL_AOT_DEVICE"]) {delete env[name];}
      const wrapper = path.join(__dirname, "../scripts/cxx-combined.sh");
      const unset = spawnSync("bash", [wrapper, "-c", "probe.cpp"], {encoding: "utf8", env});
      assert.notEqual(unset.status, 0);
      assert.match(unset.stderr, /MOM_COMBINED_TARGETS must be set by combined-build\.sh/);
      const withoutGen = spawnSync("bash", [wrapper, "-c", "probe.cpp"], {encoding: "utf8",
        env: {...env, MOM_COMBINED_TARGETS: "spir64,nvidia_gpu_sm_80", MOM_INTEL_AOT_DEVICE: "bmg-g21"}});
      assert.equal(withoutGen.status, 2, withoutGen.stderr);
      assert.match(withoutGen.stderr, /MOM_INTEL_AOT_DEVICE requires spir64_gen in MOM_COMBINED_TARGETS/);
    });

  });

  test.describe("NexaPoW and DAG dispatch tuning", {concurrency: false}, () => {
    test("NexaPoW vectors keep the bounded staged-test timeout", () => {
      const vector = require("./vectors").hashTests.find(({job}) => job.algo === "nexapow");
      assert.ok(vector);
      assert.equal(vector.timeoutMs, 25 * 60 * 1000);
    });

    test("Nexa performance tests retain their extended timeout", () => {
      const nexa = require("./vectors").perfTests.find(({algo}) => algo === "nexapow");
      assert.ok(nexa);
      assert.equal(nexa.timeoutMs, 15 * 60 * 1000);
    });

    test("Ethash DAG chunks retire each dispatch and honor overrides before device defaults", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const read = (/** @type {string} */ file) => fs.readFileSync(path.join(root, file), "utf8");
      const kawpow = read("sycl/kawpow/state.inc"), etchash = read("sycl/etchash/state.inc");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ethash-dag-chunks-"));
      try {
        fs.writeFileSync(path.join(fixture, "mom_parse_env_ulong.inc"),
          extractCppBlock(read("sycl/lib-internal.h"), "inline bool mom_parse_env_ulong("));
        fs.writeFileSync(path.join(fixture, "kawpow_dag_chunk.inc"),
          extractCppBlock(kawpow, "  static bool parse_env_u32(") +
          extractCppBlock(kawpow, "  static uint32_t kawpow_dag_chunk_nodes("));
        fs.writeFileSync(path.join(fixture, "etchash_dag_chunk.inc"),
          extractCppBlock(etchash, "  static uint32_t etchash_dag_chunk_nodes("));
        for (const windows of [false, true]) {
          const executable = path.join(fixture, windows ? "windows" : "linux");
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-Wno-unknown-pragmas", ...(windows ? ["-DMOM_FIXTURE_WINDOWS"] : []), "-I", fixture, "-I", root,
            path.join(__dirname, "native/ethash-dag-chunks.cpp"), "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", env: {}});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          assert.equal(result.stdout, "PASS Ethash DAG chunk retirement and override policy\n");
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  });

  /** @param {Map<string, number>} rates */
  async function tuneCnGpu(rates) {
  /** @type {MinerOptions} */
    const opt = {
      algo_params: {"cn/gpu": {backend: "auto", dev: "gpu1*[intensity=1536]", perf: null, tuning: {}}},
      bench_algo_params: 0, default_msrs: {}, gpu_tune: 0,
      job: {algo: null, backend: "auto", blob_hex: "", dev: "", height: 0, seed_hex: ""},
      log_level: 0, pool_ids: {active: 0, primary: 0, donate: null},
      pool_time: {
        close_wait: 0, connect_throttle: 0, donate_interval: 0, donate_length: 0,
        first_job_wait: 0, keepalive: 0, primary_reconnect: 0, stats: 0,
      },
      pools: [], save_config: "",
    };
    const fakeHelper = {
      formatHashrate: String,
      log: () => undefined,
      log_err: (/** @type {string} */ message) => assert.fail(message),
      repeat: (/** @type {(next: () => void) => unknown} */ fn) => { fn(() => fakeHelper.repeat(fn)); },
    };
    const tuner = require("../miner/gpu-autotune")({
      h: fakeHelper,
      opt,
      gpuTuning,
      benchAlgo: (_algo, callback, dev) => {
        const rate = rates.get(dev);
        if (rate === undefined) {throw new Error(`Missing tuning rate for ${dev}`);}
        callback(rate);
      },
    });
    await new Promise((resolve) => tuner.tuneAlgo("cn/gpu", () => resolve(undefined)));
    const algoParams = opt.algo_params["cn/gpu"];
    if (!algoParams) {throw new Error("cn/gpu tuning parameters disappeared");}
    return algoParams.dev;
  }

  test.describe("GPU discovery and backend annotations", {concurrency: false}, () => {
    test("GPU test discovery excludes integrated devices", () => {
    // Device names are deliberately arbitrary: discovery uses only SYCL's integrated marker, never
    // a model-name/PCI-ID list, so new and unlisted GPU generations are covered automatically.
      const output = [
        "gpu1: Unlisted Intel discrete accelerator via Level Zero",
        "gpu2: Unlisted Intel integrated accelerator via Level Zero [integrated]",
      ].join("\n");
      assert.deepEqual(parseDiscreteGpuDevices(output), [{
        dev: "gpu1", description: "Unlisted Intel discrete accelerator via Level Zero",
      }]);

      assert.deepEqual(parseDiscreteGpuDevices([
        "gpu1: Future AMD discrete accelerator via HIP",
        "gpu2: Future AMD integrated accelerator via HIP [integrated]",
      ].join("\n")).map((device) => device.dev), ["gpu1"]);

      assert.deepEqual(parseDiscreteGpuDevices(
        "gpu1: Future NVIDIA discrete accelerator via CUDA"
      ).map((device) => device.dev), ["gpu1"]);

      assert.deepEqual(parseGpuDevices(output, true).map((device) => device.dev), ["gpu2"]);
      assert.deepEqual(parseGpuDevices(output, null).map((device) => device.dev), ["gpu1", "gpu2"]);
    });

    test("reported backend annotations are not copied into GPU device specifications", () => {
      assert.deepEqual(policy.parseReportedAlgoParam("gpu1*[intensity=8]:auto[sycl-native]"),
        {dev: "gpu1*[intensity=8]", backend: "auto"});
      assert.deepEqual(policy.parseReportedAlgoParam("gpu2*[intensity=1]:sycl"),
        {dev: "gpu2*[intensity=1]", backend: "sycl"});
      assert.deepEqual(policy.parseReportedAlgoParam("cpu*8"), {dev: "cpu*8"});
    });
  });

  /** @param {string} compiler @param {string} backend @param {string} profile */
  function policyFixture(compiler = "—", backend = "—", profile = "—") {
    const lines = [
      "| Key | Linux | Windows |",
      "| --- | --- | --- |",
      "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |",
      "",
      "| OS | GPU | Compiler | Backend | PearlHash MxNxK/rank |",
      "| --- | --- | --- | --- | --- |",
      `| Linux | Intel | dpcpp | sycl | ${profile} |`,
      "",
      "| Algorithm | OS | GPU | Compiler | Backend |",
      "| --- | --- | --- | --- | --- |",
    ];
    if (compiler !== "—" || backend !== "—") {
      lines.push(`| foo | Linux | Intel | ${compiler} | ${backend} |`);
    }
    return lines.join("\n");
  }

  test.describe("Compiler policy validation, worker normalization, and completion contracts", {concurrency: false}, () => {
    test("compiler policy cells reject malformed and duplicate entries", () => {
      const parsed = policyModule.parse(policyFixture("dpcpp", "—"));
      const parsedRow = parsed.policies[0];
      assert.ok(parsedRow);
      assert.deepEqual(parsedRow.overrides, {foo: {compiler: "dpcpp"}});
      assert.throws(() => policyModule.parse(policyFixture("dpcpp=extra", "—")),
        /Unknown GPU compiler key: dpcpp=extra/);
      assert.throws(() => policyModule.parse(policyFixture("", "")),
        /GPU override row is empty/);
      assert.throws(() => policyModule.parse(
        `${policyFixture("dpcpp", "—")}\n| foo | Linux | Intel | — | sycl |`
      ), /Duplicate GPU override/);
      const absent = policyModule.parse(policyFixture()).policies[0];
      const dashedCompiler = policyModule.parse(policyFixture("dpcpp", "-")).policies[0];
      const dashedBackend = policyModule.parse(policyFixture("-", "sycl")).policies[0];
      assert.ok(absent && dashedCompiler && dashedBackend);
      assert.deepEqual(absent.overrides, {});
      assert.deepEqual(dashedCompiler.overrides, {foo: {compiler: "dpcpp"}});
      assert.deepEqual(dashedBackend.overrides, {foo: {backend: "sycl"}});

      const duplicateArtifact = policyFixture().replace(
        "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |",
        "| dpcpp | dpcpp/mom.node | dpcpp/mom.node |\n" +
      "| dpcpp | other/mom.node | other/mom.node |"
      );
      assert.throws(() => policyModule.parse(duplicateArtifact), /duplicate GPU artifact key/i);
      const duplicatePolicy = policyFixture().replace(
        "| Linux | Intel | dpcpp | sycl | — |",
        "| Linux | Intel | dpcpp | sycl | — |\n| Linux | Intel | dpcpp | sycl | — |"
      );
      assert.throws(() => policyModule.parse(duplicatePolicy), /Duplicate GPU policy row/);
      assert.throws(() => policyModule.parse(policyFixture("missing", "—")),
        /Unknown GPU compiler key: missing/);
      assert.throws(() => policyModule.parse(policyFixture("—", "invalid")),
        /Invalid GPU backend/);
    });

    test("PearlHash profiles and native shapes enforce safe individual and relational bounds", () => {
      const maxDimension = 1 << 24;
      const maxM = Math.floor(0x7fffffff / 2048 / 32) * 32;
      const maxN = Math.floor(0x7fffffff / 128 / 32) * 32;
      assert.deepEqual(gpuTuning.validatePearlHashShape("128", 128, 2048, 128), {
        m: 128, n: 128, k: 2048, rank: 128,
      });
      assert.deepEqual(gpuTuning.validatePearlHashShape(160, 160, 8192, 128), {
        m: 160, n: 160, k: 8192, rank: 128,
      });
      assert.deepEqual(gpuTuning.validatePearlHashShape(131072, 524288, 8192, 128), {
        m: 131072, n: 524288, k: 8192, rank: 128,
      });
      assert.deepEqual(gpuTuning.validatePearlHashShape(maxM, 128, 2048, 128), {
        m: maxM, n: 128, k: 2048, rank: 128,
      });
      assert.deepEqual(gpuTuning.validatePearlHashShape(128, maxN, 2048, 128), {
        m: 128, n: maxN, k: 2048, rank: 128,
      });
      assert.deepEqual(gpuTuning.validatePearlHashShape(128, 128, 65536, 1024), {
        m: 128, n: 128, k: 65536, rank: 1024,
      });
      /** @type {Array<[unknown, unknown, unknown, unknown]>} */
      const invalidShapes = [
        [96, 128, 2048, 128],
        [161, 128, 2048, 128],
        [maxDimension + 32, 128, 2048, 128],
        [128, 96, 2048, 128],
        [128, 161, 2048, 128],
        [128, 128, 960, 128],
        [128, 128, 1024, 128],
        [128, 128, 1025, 128],
        [128, 128, 65600, 1024],
        [128, 128, 16384, 2048],
        [128, 128, 3072, 192],
        [maxM + 32, 128, 2048, 128],
        [128, maxN + 32, 2048, 128],
        [131072, 4194304, 2048, 128],
        [{toString: () => "128"}, 128, 2048, 128],
      ];
      for (const shape of invalidShapes) {
        assert.throws(() => Reflect.apply(gpuTuning.validatePearlHashShape, null, shape),
          /PearlHash shape/);
      }
      assert.throws(() => policyModule.parse(policyFixture("—", "—", "128x128x16/16")),
        /PearlHash profile/);
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[m=96]", "pearlhash"),
        /at least 128/);
      assert.deepEqual(gpuTuning.parseDeviceEntry(`gpu1*[m=${maxDimension}]`, "pearlhash").tuning,
        {m: maxDimension});
      assert.throws(() => gpuTuning.parseDeviceEntry(`gpu1*[m=${maxDimension + 32}]`, "pearlhash"),
        /at most 16777216/);
      assert.throws(() => gpuTuning.parseDeviceEntry(`gpu1*[n=${maxDimension + 32}]`, "pearlhash"),
        /at most 16777216/);
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[m=161]", "pearlhash"),
        /multiple of 32/);
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[k=960]", "pearlhash"),
        /at least 1024/);
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=1024]", "pearlhash").tuning,
        {k: 1024});
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=65536]", "pearlhash").tuning,
        {k: 65536});
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[k=65600]", "pearlhash"),
        /at most 65536/);
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[rank=1024]", "pearlhash").tuning,
        {rank: 1024});
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=48]", "pearlhash"),
        /between 128 and 1024/);
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=64]", "pearlhash"),
        /between 128 and 1024/);
      assert.throws(() => gpuTuning.parseDeviceEntry("gpu1*[rank=2048]", "pearlhash"),
        /between 128 and 1024/);
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=4160;rank=256]", "pearlhash").tuning,
        {k: 4160, rank: 256});
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=2048;rank=128]", "pearlhash").tuning,
        {k: 2048, rank: 128});
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=8192;rank=128]", "pearlhash").tuning,
        {k: 8192, rank: 128});
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[k=16640;rank=256]", "pearlhash").tuning,
        {k: 16640, rank: 256});
    });

    test("PearlHash Merkle reduction stays canonical, bounded, and leaf-preserving", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const matrix = fs.readFileSync(path.join(root, "sycl/pearlhash/matrix.inc"), "utf8");
      const reduction = extractCppBlock(matrix, "static size_t reduce_tree(");

      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearlhash-merkle-"));
      const source = path.join(fixture, "pearlhash-merkle.cpp");
      const reductionFile = path.join(fixture, "reduce_tree.inc");
      const executable = path.join(fixture, "pearlhash_merkle");
      const sourceFile = path.join(root, "tests/native/pearlhash-merkle.cpp");
      const baseArgs = ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
        "-I", root, source, "-o", executable];
      const sanitizerArgs = process.platform === "win32" ? [] : [
        "-fsanitize=address,undefined", "-fsanitize-address-use-after-scope", "-fno-omit-frame-pointer",
        ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
      ];
      const compile = () => {
        let result = spawnSync("c++", [...baseArgs, ...sanitizerArgs], {encoding: "utf8"});
        // Some host C++ installations expose no sanitizer runtime; retain regular C++17 coverage.
        if (result.status !== 0 && sanitizerArgs.length) {
          result = spawnSync("c++", baseArgs, {encoding: "utf8"});
        }
        assert.equal(result.error, undefined, result.error?.message);
        assert.equal(result.signal, null, result.stderr);
        assert.equal(result.status, 0, result.stderr);
      };
      /** @param {string} body @param {string} needle @param {string} replacement */
      const mutateOnce = (body, needle, replacement) => {
        assert.equal(body.split(needle).length, 2, "mutation anchor must be unique");
        return body.replace(needle, replacement);
      };
      const variants = [
        {name: "candidate", body: reduction, pass: true},
        {name: "ROOT on an odd intermediate level", body: mutateOnce(reduction,
          "const bool is_root = cnt == 2;", "const bool is_root = cnt == 2 || cnt == 3;"), pass: false},
        {name: "wrong odd carry bytes", body: mutateOnce(reduction,
          "out[byte] = left[byte];", "out[byte] = static_cast<uint8_t>(left[byte] ^ 1u);"), pass: false},
        {name: "ping-pong writes over cached leaves", body: mutateOnce(reduction,
          "const size_t first = static_cast<size_t>(nc);", "const size_t first = 0;"), pass: false},
      ];
      try {
        fs.copyFileSync(sourceFile, source);
        for (const variant of variants) {
          fs.writeFileSync(reductionFile, variant.body);
          compile();
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
          assert.equal(result.error, undefined, result.error?.message);
          assert.equal(result.signal, null, result.stderr);
          if (variant.pass) {
            assert.equal(result.status, 0, result.stdout + result.stderr);
            assert.equal(result.stdout,
              "PARALLEL_TREE_SOURCE 1128 canonical-root/leaf-cache/bounds/ROOT/layout/reuse cases PASS\n");
          } else {
            assert.notEqual(result.status, 0,
              variant.name + " unexpectedly passed\n" + result.stdout + result.stderr);
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("GPU backend and compiler policy boundaries reject prefixes and coercive objects", () => {
      assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "nvidia-extra"}), "");
      assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "NVIDIA"}), "nvidia");
      assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "all"}), "");
      assert.equal(policyModule.gpuFromEnv({MOM_GPU_BACKEND: "nvidia"}), "nvidia");
      assert.equal(policyModule.selection("etchash", "future-gpu", "linux"), null);
      assert.throws(() => policyModule.selection({toString: () => "etchash"}, "intel", "linux"),
        /algorithm must be a string/);
      assert.throws(() => policyModule.selection("etchash", {toString: () => "intel"}, "linux"),
        /GPU name must be a string/);
      assert.throws(() => policyModule.selection("etchash", "intel", {toString: () => "linux"}),
        /Platform must be a string/);
      assert.throws(() => Reflect.apply(policyModule.nvidiaComputeCapability, null, [{
        MOM_NVIDIA_COMPUTE_CAPABILITY: {toString: () => "8.0"},
      }]), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);

      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-contract-"));
      try {
        fs.mkdirSync(path.join(root, "oneapi"));
        fs.writeFileSync(path.join(root, "oneapi", "mom.node"), "test");
        fs.mkdirSync(path.join(root, "dpcpp-opencl"));
        fs.writeFileSync(path.join(root, "dpcpp-opencl", "mom.node"), "test");
        assert.throws(() => Reflect.apply(policyModule.workerEnv, null, ["etchash", {
          MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: {toString: () => "1"}, MOM_NATIVE_DIR: root,
        }, "linux"]), /Invalid MOM_GPU_INDEX/);
        assert.throws(() => Reflect.apply(policyModule.workerEnv, null, ["etchash", {
          MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: {toString: () => "gpu"},
          MOM_NATIVE_DIR: root,
        }, "linux"]), /Invalid MOM_OPENCL_DEVICE_TYPE/);
      } finally {
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("PearlHash default M survives CPU worker tuning", () => {
      const api = testJobApi({...process.env, MOM_GPU_BACKEND: "intel"});
      const job = api.prepareBenchmarkJob({
        algo: "pearlhash", dev: "cpu", blob_hex: "00".repeat(76),
      });
      assert.equal(job.intensity, 131072);
      gpuTuning.applyNativeJobTuning(
        job, gpuTuning.parseDeviceEntry("cpu", "pearlhash"), "pearlhash"
      );
      assert.equal(job.intensity, 131072);
      assert.equal(job.pearlhash_n, 131072);
      assert.equal(job.pearlhash_k, 4096);
      assert.equal(job.pearlhash_rank, 256);

      const explicitJob = api.prepareBenchmarkJob({
        algo: "pearlhash", dev: "cpu", blob_hex: "00".repeat(76),
      });
      gpuTuning.applyNativeJobTuning(
        explicitJob, gpuTuning.parseDeviceEntry("cpu*[m=256]", "pearlhash"), "pearlhash"
      );
      assert.equal(explicitJob.intensity, 256);
      assert.equal(explicitJob.pearlhash_n, 256);
      assert.equal(explicitJob.pearlhash_k, 4096);
      assert.equal(explicitJob.pearlhash_rank, 256);

      const gpuJob = api.prepareBenchmarkJob({
        algo: "pearlhash", dev: "gpu1", blob_hex: "00".repeat(76),
      });
      assert.equal(gpuJob.pearlhash_k, 2048);
      assert.equal(gpuJob.pearlhash_rank, 128);

      const mixedJob = api.prepareBenchmarkJob({
        algo: "pearlhash",
        dev: "gpu1*[m=131072;n=524288;k=8192;rank=128],cpu",
        blob_hex: "00".repeat(76),
      });
      assert.equal(mixedJob.pearlhash_k, 4096);
      assert.equal(mixedJob.pearlhash_rank, 256);
      gpuTuning.applyNativeJobTuning(
        mixedJob,
        gpuTuning.parseDeviceEntry("gpu1*[m=131072;n=524288;k=8192;rank=128]", "pearlhash"),
        "pearlhash"
      );
      assert.equal(mixedJob.pearlhash_k, 8192);
      assert.equal(mixedJob.pearlhash_rank, 128);
    });

    test("CPU thread batches do not leak into algorithm tuning", () => {
      const api = testJobApi();
      api.workerRuntimeEnv("ghostrider", "cpu*8");
      assert.equal(api.workerRuntimeEnv("hoohash", "cpu1*8")["MOM_HOOHASH_INTENSITY"], "8");
    });

    test("Intel command-list policy keeps short staged workers off the one-core path", () => {
      const api = testJobApi();
      for (const algo of ["c29", "equihash192_7", "zhash"]) {
        const env = api.workerRuntimeEnv(algo);
        assert.equal(env["SYCL_UR_USE_LEVEL_ZERO_V2"], "0");
        assert.equal(env["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"], "0");
      }
      assert.equal(api.workerRuntimeEnv("hoohash")["SYCL_UR_USE_LEVEL_ZERO_V2"], undefined);
    });

    test("test-only nonce offsets stay optional but explicit invalid offsets fail", () => {
      const api = testJobApi();
      const job = api.prepareTestJob({algo: "rx/0", dev: "cpu", blob_hex: "00"});
      assert.equal(job.nonceoffset, undefined);
      const pearlJob = api.prepareTestJob({
        algo: "pearlhash", dev: "gpu1*[m=256]", blob_hex: "00".repeat(76),
      });
      assert.equal(pearlJob.nonceoffset, undefined);
      assert.equal(pearlJob.noncebytes, 8);
      assert.throws(() => api.prepareTestJob({
        algo: "rx/0", dev: "cpu", blob_hex: "00", nonceoffset: 999,
      }), /Invalid rx\/0 nonce offset/);
      assert.throws(() => api.prepareTestJob({
        algo: "kawpow", dev: "gpu1", blob_hex: "00".repeat(40), noncebytes: 8,
      }), /Invalid kawpow nonce offset/);
    });

    test("C30 benchmark normalization rejects a malformed legacy nonce tail", () => {
      const api = testJobApi();
      assert.throws(() => api.prepareBenchmarkJob({
        algo: "c30", dev: "gpu1", blob_hex: "ab".repeat(32) + "zz".repeat(8),
      }), /Invalid c30 job blob/);
      const job = api.prepareBenchmarkJob({
        algo: "c30", dev: "gpu1", blob_hex: "ab".repeat(32) + "0102030405060708",
      });
      assert.equal(job.blob_hex, "ab".repeat(32));
      assert.equal(job.nonce, "0807060504030201");
    });

    test("FishHash benchmarks use protocol-valid header and nonce layouts", () => {
      const api = testJobApi();
      const defaultJob = api.prepareBenchmarkJob({
        algo: "fishhash", dev: "gpu1", blob_hex: opts.create_default_opts().job.blob_hex,
      });
      assert.equal(defaultJob.blob_hex, "00".repeat(180));
      assert.equal(defaultJob.noncebytes, 8);
      assert.equal(defaultJob.nonceoffset, 172);

      const offlinePrefix = api.prepareBenchmarkJob({
        algo: "fishhash", dev: "gpu1", blob_hex: "ab".repeat(32),
      });
      assert.equal(offlinePrefix.blob_hex, "ab".repeat(32) + "00".repeat(8));
      assert.equal(offlinePrefix.nonceoffset, 32);

      const offlineHeader = api.prepareBenchmarkJob({
        algo: "fishhash", dev: "gpu1", blob_hex: "cd".repeat(40),
      });
      assert.equal(offlineHeader.blob_hex, "cd".repeat(40));
      assert.equal(offlineHeader.nonceoffset, 32);

      const poolHeader = api.prepareBenchmarkJob({
        algo: "fishhash", dev: "gpu1", blob_hex: "ef".repeat(180),
      });
      assert.equal(poolHeader.blob_hex, "ef".repeat(180));
      assert.equal(poolHeader.nonceoffset, 172);
    });

    test("PearlHash emits every captured winner and preserves its seed lifecycle", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const core = fs.readFileSync(path.join(__dirname, "../native/core/execution.inc"), "utf8");
      const branch = extractCppBlock(core,
        "      if (m_dev == DEV::PEARLHASH_GPU) { const uint64_t prev_nonce = m_nonce64;");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-results-"));
      const source = path.join(fixture, "results.cpp");
      const executable = path.join(fixture, "results");
      fs.writeFileSync(source, `
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "native/job-boundary.h"
constexpr unsigned HASH_LEN = 32;
enum class DEV { PEARLHASH_GPU };
using MessageValues = std::map<std::string, std::string>;
static std::string nonce_to_hex(uint64_t nonce, unsigned width) {
  assert(width == 8);
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(nonce));
  return hex;
}
struct Worker {
  DEV m_dev = DEV::PEARLHASH_GPU;
  uint64_t m_nonce64 = 10, m_target = 1, found_seed = 0;
  unsigned m_thread_num = 2, m_pearlhash_seed_stride = 3, claim_calls = 0;
  std::string m_pool_id = "pool", m_worker_id = "worker", m_job_id = "job", m_job_token = "token";
  bool available = true, active = true;
  std::string proof;
  std::vector<std::pair<std::string, MessageValues>> sent;
  // Keep historical gate inputs in this mock so the regression also compiles before the fix.
  uint8_t m_input[76]{};
  unsigned m_input_len = 76, m_batch = 128, m_pearlhash_n = 128, m_pearlhash_k = 2048,
           m_pearlhash_rank = 128, m_pearlhash_cert_version = 3;
  unsigned m_pearlhash_proof_n = 0, m_pearlhash_proof_k = 0, m_pearlhash_proof_rank = 0,
           m_pearlhash_proof_cert_version = 0, m_pearlhash_proof_m = 0;
  std::string m_pearlhash_proof_pool, m_pearlhash_proof_job, m_pearlhash_proof_header;
  std::vector<unsigned> claimed;
  const char* pearlhash_claim(unsigned index, uint8_t* jackpot, uint32_t* factor) {
    ++claim_calls;
    claimed.push_back(index);
    if (!available) return nullptr;
    for (unsigned i = 0; i < HASH_LEN; ++i) jackpot[i] = static_cast<uint8_t>(found_seed);
    *factor = 524288;
    proof = "captured-proof-" + std::to_string(found_seed);
    return proof.c_str();
  }
  std::string bin2hex(const uint8_t* hash, size_t size) {
    std::string hex;
    char digits[3];
    for (size_t i = 0; i < size; ++i) {
      std::snprintf(digits, sizeof(digits), "%02x", hash[i]);
      hex += digits;
    }
    return hex;
  }
  void send_msg(const std::string& name, const MessageValues& values) { sent.emplace_back(name, values); }
  void send_error(const std::string& message) { send_msg("error", {{"message", message}}); }
  void clear_fn() { active = false; }
  void send_last_nonce(uint64_t nonce, unsigned width, const std::string& pool,
                       const std::string& job, const std::string& token) {
    send_msg("last_nonce", {{"nonce", nonce_to_hex(nonce, width)}, {"pool_id", pool},
                            {"job_id", job}, {"job_token", token}});
  }
  void run(int dev_sols, uint64_t dev_nonce) {
    found_seed = dev_nonce;
    for (unsigned once = 0; once < 1; ++once) {
      ${branch}
    }
  }
};
static void expect_result(const Worker& worker, unsigned index, uint64_t seed) {
  const auto& event = worker.sent.at(index);
  assert(event.first == "result" && event.second.size() == 8);
  const auto& result = event.second;
  assert(result.at("nonce") == nonce_to_hex(seed, 8));
  assert(result.at("plain_proof") == "captured-proof-" + std::to_string(seed));
  char byte[3];
  std::snprintf(byte, sizeof(byte), "%02x", static_cast<unsigned>(seed & 255));
  std::string jackpot;
  for (unsigned i = 0; i < HASH_LEN; ++i) jackpot += byte;
  assert(result.at("jackpot") == jackpot && result.at("adjustment_factor") == "524288");
  assert(result.at("pool_id") == worker.m_pool_id && result.at("worker_id") == worker.m_worker_id);
  assert(result.at("job_id") == worker.m_job_id && result.at("job_token") == worker.m_job_token);
}
int main() {
  Worker wins;
  wins.run(1, 10);
  wins.run(1, 16);
  assert(wins.sent.size() == 2 && wins.claim_calls == 2 && wins.m_nonce64 == 22 && wins.active);
  expect_result(wins, 0, 10);
  expect_result(wins, 1, 16);
  Worker retry;
  retry.available = false;
  retry.run(1, 10);
  retry.available = true;
  retry.run(1, 16);
  assert(retry.sent.size() == 2 && retry.claim_calls == 2 && retry.m_nonce64 == 22 && retry.active);
  assert(retry.sent[0].first == "error");
  assert(retry.sent[0].second.at("message") == "PearlHash proof claim unavailable");
  expect_result(retry, 1, 16);
  for (int solutions : {-1, 0}) {
    Worker pending;
    pending.run(solutions, 10);
    assert(pending.sent.empty() && pending.claim_calls == 0 && pending.m_nonce64 == 16 && pending.active);
  }
  // Two winning tiles of one seed are two shares, claimed in capture order.
  Worker pair;
  pair.run(2, 10);
  assert(pair.sent.size() == 2 && pair.claim_calls == 2 && pair.m_nonce64 == 16 && pair.active);
  assert((pair.claimed == std::vector<unsigned>{0, 1}));
  expect_result(pair, 0, 10);
  expect_result(pair, 1, 10);
  Worker no_target;
  no_target.m_target = 0;
  no_target.run(1, 10);
  assert(no_target.sent.empty() && no_target.claim_calls == 0 && no_target.m_nonce64 == 16);
  Worker exhausted;
  exhausted.m_nonce64 = UINT32_MAX - 5;
  exhausted.run(1, exhausted.m_nonce64);
  assert(!exhausted.active && exhausted.sent.size() == 2 && exhausted.claim_calls == 1);
  expect_result(exhausted, 0, UINT32_MAX - 5);
  assert(exhausted.sent[1].first == "last_nonce");
  assert(exhausted.sent[1].second.at("nonce") == nonce_to_hex(UINT32_MAX - 5, 8));
  assert(exhausted.sent[1].second.at("pool_id") == "pool" &&
         exhausted.sent[1].second.at("job_id") == "job" && exhausted.sent[1].second.at("job_token") == "token");
  no_target.m_nonce64 = UINT32_MAX - 5;
  no_target.run(1, no_target.m_nonce64);
  assert(!no_target.active && no_target.sent.empty());
  std::puts("PASS PearlHash captured winners, claims and seed lifecycle");
}
`);
      try {
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I", path.join(__dirname, ".."), source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
        assert.equal(result.stdout, "PASS PearlHash captured winners, claims and seed lifecycle\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("tuning environments normalize validated values and native intensity is numeric", () => {
      assert.deepEqual(gpuTuning.tuningEnvironment("zelhash", {slots: "176"}), {
        MOM_ZELHASH_SLOTS: "176",
      });
      assert.throws(() => gpuTuning.tuningEnvironment("zelhash", {
        slots: {toString: () => "176"},
      }), /base-10 integer/);
      /** @type {MiningJob} */
      const defaultJob = {algo: "kawpow", dev: ""};
      gpuTuning.applyNativeJobTuning(defaultJob, {device: "gpu1", processes: 1, tuning: {}}, "kawpow");
      assert.equal(defaultJob.intensity, 1);
      /** @type {MiningJob} */
      const zeroJob = {algo: "kawpow", dev: ""};
      gpuTuning.applyNativeJobTuning(
        zeroJob, {device: "gpu1", processes: 1, tuning: {intensity: 0}}, "kawpow");
      assert.equal(zeroJob.intensity, 0);
      assert.throws(() => Reflect.apply(gpuTuning.applyNativeJobTuning, null, [
        {algo: "kawpow", dev: ""},
        {device: "gpu1", processes: 1, tuning: {intensity: {toString: () => "8"}}},
        "kawpow",
      ]), /finite number/);
    });

    test("Windows GPU polling pause bounds timer waits and retains low-CPU fallback", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const runtime = fs.readFileSync(path.join(__dirname, "../sycl/runtime.inc"), "utf8");
      const current = extractCppBlock(runtime, "void mom_sycl_poll_pause()");
      const bounded = "WaitForSingleObject(timer.handle, 10)";
      assert.equal(current.split(bounded).length, 2);
      // Mutate only the bound: preserve fallback/result handling across diagnostics/refactors.
      const unbounded = current.replace(bounded, "WaitForSingleObject(timer.handle, INFINITE)");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-poll-pause-"));
      const executable = path.join(fixture, "poll-pause");
      try {
        fs.copyFileSync(path.join(__dirname, "native/windows-poll-pause.cpp"), path.join(fixture, "probe.cpp"));
        for (const baseline of [true, false]) {
          fs.writeFileSync(path.join(fixture, "poll_pause.inc"), baseline ? unbounded : current);
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", "-pthread", path.join(fixture, "probe.cpp"), "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.error, undefined);
          assert.equal(compiled.status, 0, compiled.stderr);
          const scenarios = ["normal", "lost-signal", "create-failure", "arm-failure", "wait-failure"];
          if (!baseline) {
            scenarios.push("timeout-then-signaled", "sequential-workers");
          }
          for (const scenario of scenarios) {
            const result = spawnSync(executable, [scenario], {encoding: "utf8", timeout: 500});
            if (baseline && scenario === "lost-signal") {
              assert.match(result.error?.message ?? "", /ETIMEDOUT$/);
              continue;
            }
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0);
            assert.match(result.stdout, scenario === "sequential-workers" ?
              /^okay=1 creates=2 arms=4 waits=4 sleeps=0 closes=2 / : /^okay=1 creates=1 /);
            if (!baseline && !["create-failure", "arm-failure"].includes(scenario)) {
              assert.match(result.stdout, /limit=10\n$/);
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash Intel SYCL pacing progresses before low-CPU completion polling", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const host = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/pearlhash.cpp"), "utf8");
      const startMarker = "    attempt_start = std::chrono::steady_clock::now();\n";
      assert.equal(host.split(startMarker).length, 2);
      const start = host.indexOf(startMarker) + startMarker.length;
      const end = host.indexOf("    st.have_header = true;", start);
      assert.ok(end > start, "complete PearlHash pacing/wait block is missing");
      const block = host.slice(start, end)
        .replaceAll("std::chrono::", "fixture::chrono::")
        .replaceAll("std::this_thread::", "fixture::this_thread::");
      const internal = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
      const wait = extractCppBlock(internal, "inline void sycl_wait_and_throw(")
        .replaceAll("std::chrono::", "fixture::chrono::")
        .replaceAll("std::this_thread::", "fixture::this_thread::");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-pacing-"));
      const source = path.join(fixture, "pacing.cpp");
      const executable = path.join(fixture, "pacing");
      fs.writeFileSync(source, `
#include <cassert>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>
#if !defined(MOM_FIXTURE_NO_BARRIER)
#define SYCL_EXT_ONEAPI_ENQUEUE_BARRIER 1
#endif
namespace sycl::info::device { struct vendor_id {}; }
namespace sycl::info::event { struct command_execution_status {}; }
namespace sycl::info { enum class event_command_status { submitted, running, complete }; }
struct EventState {
  unsigned pending = 2, queries = 0, waits = 0;
  bool active = false, progressed = false, complete = false, completed_before_wait = false;
  bool status_error = false, sleep_error = false, wait_error = false;
  bool gpu_error = false, opencl_error = false;
  unsigned gpu_queries = 0, opencl_queries = 0;
  unsigned* host_destination = nullptr;
};
static EventState event_state;
static int error_kind;
static std::exception_ptr polling_error;
static void fail_polling(const char* message) {
  try {
    if (error_kind == 1)
      throw std::string(message);
    if (error_kind == 2)
      throw 73;
    throw std::runtime_error(message);
  } catch (...) {
    polling_error = std::current_exception();
    throw;
  }
}
enum class PearlHashSearchBackend { sycl, hip_jit, cuda_jit };
struct Device {
  bool gpu; unsigned vendor;
  bool opencl = false;
  bool is_gpu() const {
    ++event_state.gpu_queries;
    if (event_state.gpu_error) fail_polling("gpu-query");
    return gpu;
  }
  template<class T> unsigned get_info() const { return vendor; }
};
struct Event {
  template<class T> sycl::info::event_command_status get_info() {
    ++event_state.queries;
    if (event_state.status_error)
      fail_polling("status-query");
    event_state.progressed = true;
    if (event_state.pending) {
      --event_state.pending;
      return sycl::info::event_command_status::submitted;
    }
    event_state.complete = true;
    return sycl::info::event_command_status::complete;
  }
  void wait_and_throw() {
    ++event_state.waits;
    event_state.completed_before_wait = event_state.complete;
    event_state.complete = true;
    if (event_state.host_destination)
      *event_state.host_destination = 1;
    if (event_state.wait_error)
      throw std::runtime_error("final-wait");
  }
};
namespace sycl { using event = Event; using device = Device; }
static constexpr bool mom_sycl_portable_opencl =
#if defined(MOM_FIXTURE_PORTABLE_OPENCL)
  true;
#else
  false;
#endif
static bool mom_is_opencl(const Device& device) {
  ++event_state.opencl_queries;
  if (event_state.opencl_error) fail_polling("opencl-query");
  return device.opencl;
}
struct Queue {
  Device device; unsigned waits = 0, barriers = 0;
  Device get_device() const { return device; }
  void wait_and_throw() { ++waits; }
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  Event ext_oneapi_submit_barrier() {
    ++barriers;
    event_state.active = true;
    return {};
  }
#endif
};
struct Jit {
  unsigned waits = 0;
  void wait(double) { ++waits; }
};
struct State { double wait_ema_us; Jit hip{}, cuda{}; bool cuda_tensor = false; };
namespace fixture {
namespace chrono {
using microseconds = std::chrono::microseconds;
using seconds = std::chrono::seconds;
template<class T> using duration = std::chrono::duration<T>;
long elapsed;
struct steady_clock {
  using time_point = std::chrono::steady_clock::time_point;
  static time_point now() { return time_point(microseconds(elapsed)); }
};
template<class T, class U> T duration_cast(U value) { return std::chrono::duration_cast<T>(value); }
}
namespace this_thread {
unsigned calls; long duration;
std::vector<long> durations;
void sleep_for(chrono::microseconds value) {
  ++calls;
  duration = value.count();
  durations.push_back(duration);
  if (event_state.sleep_error)
    fail_polling("sleep");
  if (event_state.active) {
    assert(event_state.progressed && event_state.queries > 0);
    assert(duration == 100);
  }
}
}
}
${wait}
static void run(PearlHashSearchBackend search_backend, bool gpu, unsigned vendor,
                double ema, long elapsed, long sleep, bool opencl = false) {
  fixture::chrono::elapsed = elapsed;
  fixture::this_thread::calls = 0;
  fixture::this_thread::duration = 0;
  fixture::this_thread::durations.clear();
  event_state = {};
  auto attempt_start = fixture::chrono::steady_clock::time_point{};
  Queue q{{gpu, vendor, opencl}};
  State st{ema};
${block}
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  const bool intel_poll = search_backend == PearlHashSearchBackend::sycl && gpu &&
    vendor == 0x8086 && !opencl;
#else
  const bool intel_poll = false;
#endif
  const std::vector<long> expected_sleeps = intel_poll ? std::vector<long>{100, 100} :
    sleep > 0 ? std::vector<long>{sleep} : std::vector<long>{};
  assert(fixture::this_thread::durations == expected_sleeps);
  assert(fixture::this_thread::calls == expected_sleeps.size());
  assert(q.barriers == static_cast<unsigned>(intel_poll));
  assert(q.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::sycl && !intel_poll));
  assert(event_state.queries == (intel_poll ? 3u : 0u));
  assert(event_state.waits == static_cast<unsigned>(intel_poll));
  assert(!intel_poll || event_state.completed_before_wait);
  assert(st.hip.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::hip_jit));
  assert(st.cuda.waits == static_cast<unsigned>(search_backend == PearlHashSearchBackend::cuda_jit));
}
static void check_helper(bool gpu, unsigned pending, const std::string& failure = {},
                         int kind = 0, bool cleanup_error = false, bool opencl = false) {
  event_state = {};
  event_state.active = true;
  event_state.pending = pending;
  event_state.gpu_error = failure == "gpu-query";
  event_state.opencl_error = failure == "opencl-query";
  event_state.status_error = failure == "status-query";
  event_state.sleep_error = failure == "sleep";
  event_state.wait_error = cleanup_error || failure == "final-wait";
  unsigned host_destination = 0;
  event_state.host_destination = &host_destination;
  error_kind = kind;
  polling_error = {};
  fixture::this_thread::calls = 0;
  fixture::this_thread::durations.clear();
  bool committed = false;
  std::exception_ptr caught;
  try {
    sycl_wait_and_throw(Event{}, Device{gpu, 0x8086, opencl});
    committed = true;
  } catch (...) {
    caught = std::current_exception();
  }
  assert(static_cast<bool>(caught) == !failure.empty());
  assert(committed == failure.empty());
  if (polling_error) {
    assert(caught == polling_error);
  } else if (caught) {
    try {
      std::rethrow_exception(caught);
    } catch (const std::runtime_error& error) {
      assert(error.what() == failure);
    }
  }
  const bool polling = gpu && !(mom_sycl_portable_opencl && opencl);
  const bool predicate_error = event_state.gpu_error || event_state.opencl_error;
  const unsigned sleeps = predicate_error || !polling || event_state.status_error ? 0 :
      event_state.sleep_error ? 1 : pending;
  assert(fixture::this_thread::calls == sleeps);
  assert(fixture::this_thread::durations == std::vector<long>(sleeps, 100));
  assert(event_state.queries == (predicate_error ? 0u : polling ? polling_error ? 1u : pending + 1 : 0u));
  assert(event_state.opencl_queries == (mom_sycl_portable_opencl ? 1u : 0u));
  assert(event_state.gpu_queries == (mom_sycl_portable_opencl && (opencl || event_state.opencl_error) ? 0u : 1u));
  assert(event_state.waits == 1 && event_state.complete && host_destination == 1);
  assert(!polling || polling_error || event_state.completed_before_wait);
}
int main() {
  using B = PearlHashSearchBackend;
  const struct { B backend; bool gpu; unsigned vendor; double ema; long elapsed, sleep; } cases[] = {
    {B::sycl, true, 0x8086, 10000, 0, 0},
    {B::sycl, true, 0x8086, 2001, 0, 0},
    {B::sycl, true, 0x8086, 2000, 0, 0},
    {B::sycl, true, 0x8086, 0, 0, 0},
    {B::sycl, false, 0x8086, 10000, 0, 9000},
    {B::sycl, true, 0x1002, 10000, 0, 9000},
    {B::sycl, true, 0x10de, 10000, 0, 9000},
    {B::sycl, true, 0, 10000, 0, 9000},
    {B::hip_jit, true, 0x8086, 10000, 0, 9000},
    {B::cuda_jit, true, 0x8086, 10000, 0, 9000},
    {B::sycl, true, 0x1002, 2000, 0, 0},
    {B::sycl, true, 0x1002, 2001, 0, 1800},
    {B::sycl, true, 0x1002, 10000, 9000, 0},
    {B::sycl, true, 0x1002, 10000, 8999, 1},
  };
  for (const auto& test : cases)
    run(test.backend, test.gpu, test.vendor, test.ema, test.elapsed, test.sleep);
  run(B::sycl, true, 0x8086, 10000, 0, 0, true);
  run(B::sycl, true, 0x1002, 10000, 0, 9000, true);
  run(B::sycl, false, 0x8086, 10000, 0, 9000, true);
  check_helper(true, 0);
  check_helper(true, 3);
  check_helper(false, 3);
  check_helper(true, 0, "final-wait");
  check_helper(false, 3, "final-wait");
  check_helper(true, 3, {}, 0, false, true);
  check_helper(true, 3, "final-wait", 0, false, true);
  for (int kind : {0, 1, 2}) {
    for (bool cleanup_error : {false, true}) {
      check_helper(true, 3, "status-query", kind, cleanup_error);
      check_helper(true, 3, "sleep", kind, cleanup_error);
      check_helper(true, 3, "gpu-query", kind, cleanup_error);
      check_helper(false, 3, "gpu-query", kind, cleanup_error);
      if (mom_sycl_portable_opencl) {
        check_helper(true, 3, "opencl-query", kind, cleanup_error);
        check_helper(false, 3, "opencl-query", kind, cleanup_error);
      }
    }
  }
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
  std::printf("PASS PearlHash actual-source pacing/polling: 17 pacing + %u helper cases; barrier=available; ",
              mom_sycl_portable_opencl ? 43u : 31u);
#else
  std::printf("PASS PearlHash actual-source pacing/polling: 17 pacing + %u helper cases; barrier=absent; ",
              mom_sycl_portable_opencl ? 43u : 31u);
#endif
  std::printf("portable_opencl=%d\\n", mom_sycl_portable_opencl);
}
`);
      try {
        for (const available of [true, false]) {
          for (const portable of [false, true]) {
            const flags = [
              ...(available ? [] : ["-DMOM_FIXTURE_NO_BARRIER=1"]),
              ...(portable ? ["-DMOM_FIXTURE_PORTABLE_OPENCL=1"] : []),
            ];
            const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-pedantic", ...flags, source, "-o", executable], {encoding: "utf8"});
            assert.equal(compiled.error, undefined);
            assert.equal(compiled.signal, null);
            assert.equal(compiled.status, 0, compiled.stderr);
            const result = spawnSync(executable, [], {encoding: "utf8"});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout, "PASS PearlHash actual-source pacing/polling: 17 pacing + " +
            (portable ? 43 : 31) + " helper cases; " +
            "barrier=" + (available ? "available" : "absent") +
            "; portable_opencl=" + Number(portable) + "\n");
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("SYCL host readback faults retain actual storage through unwind", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const nexa = fs.readFileSync(path.join(root, "nexapow/sycl-pipeline.inc"), "utf8");
      const monolithic = readSyclSource(path.join(root, "nexapow/nexapow.cpp"));
      const equi = fs.readFileSync(path.join(root, "equihash192-7/session.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const cut = (source, start, end) => {
        const a = source.indexOf(start), b = source.indexOf(end, a);
        assert.ok(a >= 0 && b > a, start);
        return source.slice(a, b);
      };
      const nexaScope = cut(nexa, "  uint32_t search(sycl::queue&", "    if (!found.count)");
      const nexaOwners = nexaScope.split("\n").filter(line => /SearchResult found|MomSyclHostTransferGuard host_transfers/.test(line)).join("\n") + "\n";
      const monolithicScope = cut(monolithic, "  state.init_portable();", "  if (!found.count)");
      const monolithicOwners = monolithicScope.split("\n").filter(line => /Result found|MomSyclHostTransferGuard host_transfers/.test(line)).join("\n") + "\n";
      const statusScope = cut(equi, "    std::array<std::uint64_t, 10> input{};", "    const std::uint32_t overflow =");
      const statusOwners = statusScope.split("\n").filter(line => /std::array<std::uint64_t, 10> input|std::uint32_t status\[2\]|MomSyclHostTransferGuard input_transfer/.test(line)).join("\n") + "\n";
      const statusCopy = cut(equi, "    sycl_wait_and_throw(queue_.memcpy(status,", "    const std::uint32_t overflow =");
      const routes = [
        {id: 1, name: "Nexa found", owner: "    SearchResult found{};\n", guard: "    MomSyclHostTransferGuard host_transfers(queue, \"nexapow staged host transfers\");\n",
          entry: nexaOwners + cut(nexa, "    sycl_wait_and_throw(queue.memcpy(&found,", "    if (!found.count)")},
        {id: 2, name: "Equihash status", owner: "    std::uint32_t status[2]{};\n", guard: "    MomSyclHostTransferGuard input_transfer(queue_, \"equihash192_7 input upload\");\n",
          entry: statusOwners + statusCopy},
        {id: 3, name: "Equihash overflow counts", entry: statusOwners + statusCopy +
        cut(equi, "    const std::uint32_t overflow =", "    if (root_count > root_capacity)")},
        {id: 4, name: "Nexa monolithic found", owner: "  Result found{};\n", guard: "  MomSyclHostTransferGuard host_transfers(state.queue, \"nexapow monolithic host transfers\");\n",
          entry: monolithicOwners + cut(monolithic, "  sycl_wait_and_throw(state.queue.memcpy(&found,", "  if (!found.count)")},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-readback-wait-"));
      let unexposedMonolithicNegatives = 0;
      try {
        fs.copyFileSync(path.join(__dirname, "native/readback-wait-fault.cpp"), path.join(fixture, "readback.cpp"));
        fs.writeFileSync(path.join(fixture, "wait.inc"), extractCppBlock(library, "inline void sycl_wait_and_throw("));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), ["inline void sycl_log_cleanup_exception(",
          "inline void sycl_cleanup_noexcept(", "class MomSyclHostTransferGuard {"]
          .map(anchor => extractCppBlock(library, anchor)).join("") + ";\n");
        fs.writeFileSync(path.join(fixture, "search-result.inc"), extractCppBlock(nexa, "struct SearchResult {") + ";\n");
        fs.writeFileSync(path.join(fixture, "monolithic-result.inc"), extractCppBlock(monolithic, "struct Result {") + ";\n");
        for (const route of routes) {
          if (route.owner && route.guard) {
            assert.ok(route.entry.includes(route.owner) && route.entry.includes(route.guard), route.name);
          } else {
            assert.match(route.entry, /^\s*MomSyclHostTransferGuard overflow_transfer[^\n]+\n/m, route.name);
          }
          // Revert only the lifetime placement/local guard. The actual copy expression retains
          // the compiler's legal argument order; GCC commonly queries before accepting a copy.
          // Monolithic Nexa instead passes its cached device and faults the accepted copy's wait.
          const baseline = route.owner && route.guard ?
            route.entry.replace(route.owner, "").replace(route.guard, route.guard + route.owner) :
            route.entry.replace(/^\s*MomSyclHostTransferGuard overflow_transfer[^\n]+\n/m, "");
          /** @type {{copies: number, waits: number, drains: number} | null} */
          let baselineHealthy = null;
          /** @type {[string, string][]} */
          const variants = [["baseline", baseline], ["current", route.entry]];
          for (const [variant, entry] of variants) {
            fs.writeFileSync(path.join(fixture, "entry.inc"), entry);
            const executable = path.join(fixture, "readback-" + route.id + "-" + variant);
            const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-g", "-fno-pie", "-no-pie", "-fsanitize=address",
              "-fsanitize-address-use-after-scope", "-DROUTE=" + route.id, path.join(fixture, "readback.cpp"), "-o", executable], {encoding: "utf8"});
            assert.equal(compiled.error, undefined);
            assert.equal(compiled.signal, null);
            assert.equal(compiled.status, 0, compiled.stderr);
            const env = {...process.env, ASAN_OPTIONS: "abort_on_error=0:exitcode=70:detect_leaks=0"};
            const healthy = spawnSync(executable, ["0", "0", "0"], {encoding: "utf8", env});
            assert.equal(healthy.error, undefined);
            assert.equal(healthy.signal, null);
            assert.equal(healthy.status, 0, healthy.stderr);
            const counts = /^PASS original=0 copies=(\d+) waits=(\d+) drains=(\d+) pending=0\n$/.exec(healthy.stdout);
            assert.ok(counts, route.name + " must record the healthy wait trace");
            const trace = {copies: Number(counts[1]), waits: Number(counts[2]), drains: Number(counts[3])};
            if (variant === "baseline") {
              baselineHealthy = trace;
            } else {
              assert.ok(baselineHealthy);
              // Overflow already throws; only its added local lifetime guard may add a drain.
              assert.deepEqual(trace, {
                copies: baselineHealthy.copies,
                waits: baselineHealthy.waits,
                drains: baselineHealthy.drains + Number(route.id === 3),
              });
            }
            for (const kind of [0, 1, 2]) {
              for (const secondary of [0, 1]) {
                const result = spawnSync(executable, ["1", String(kind), String(secondary)], {encoding: "utf8", env});
                assert.equal(result.error, undefined);
                assert.equal(result.signal, null);
                const order = /(?:QUERY|WAIT) order=(copy-first|query-first) copies=([0-9]+)/.exec(result.stdout);
                assert.ok(order, route.name + " must record its fault boundary and accepted copies");
                // GCC may leave a trivial stack object's bytes unpoisoned during exception unwind.
                // Keep that baseline explicitly inconclusive; Clang's ASan exposes the same scope bug.
                const unpoisonedBaseline = route.id === 4 && variant === "baseline" && result.status === 0;
                if (unpoisonedBaseline) {
                  ++unexposedMonolithicNegatives;
                }
                if (variant === "baseline" && order[1] === "copy-first" && !unpoisonedBaseline) {
                  assert.equal(result.status, 70, result.stderr);
                  assert.match(result.stderr, /AddressSanitizer: (stack-use-after-scope|heap-use-after-free)/);
                } else {
                  assert.equal(result.status, 0, result.stderr);
                  assert.match(result.stdout, /PASS original=1 .*pending=0/);
                }
              }
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
      if (unexposedMonolithicNegatives) {
        t.diagnostic("Monolithic baseline lifetime negative not exposed by this compiler's unwind poisoning: " +
        unexposedMonolithicNegatives + " negative controls remain inconclusive; fixed lifetime checks still enforced");
      }
    });

    test("CN/gpu completes kernels before readback without startup or stale-batch pacing", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const cn = path.join(__dirname, "../sycl/cn-gpu");
      const entry = fs.readFileSync(path.join(cn, "cn-gpu.cpp"), "utf8");
      const finalMarker = "  sycl::event final_event =";
      assert.equal(entry.split(finalMarker).length, 2, "one final-kernel submission");
      const finalStart = entry.indexOf(finalMarker);
      const submitEnd = entry.indexOf("\n      });", finalStart);
      const end = entry.lastIndexOf("\n}");
      assert.ok(submitEnd > finalStart && end > submitEnd && entry.slice(end).trim() === "}");
      const tail = entry.slice(submitEnd + "\n      });".length, end)
        .replaceAll("std::chrono::steady_clock::now()", "fixture::clock::now()")
        .replaceAll("std::this_thread::sleep_until", "fixture::sleep_until");
      const finalBodyStart = entry.indexOf("uint64_t* const spad = &d_spads[25 * t];", finalStart);
      const finalBodyEnd = entry.indexOf("out[3] = spad[3];", finalBodyStart);
      assert.ok(finalBodyStart > finalStart && finalBodyEnd > finalBodyStart && finalBodyEnd < submitEnd);
      const finalBody = entry.slice(finalBodyStart, finalBodyEnd + "out[3] = spad[3];".length);
      const starts = [...entry.matchAll(/const auto batch_start\s*=\s*std::chrono::steady_clock::now\(\);[^\n]*/g)];
      assert.ok(starts.length <= 1);
      const startMatch = starts[0];
      const batchStart = startMatch ? startMatch[0]
        .replaceAll("std::chrono::steady_clock::now()", "fixture::clock::now()") : "";
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cn-readback-"));
      const source = path.join(fixture, "readback.cpp");
      const executable = path.join(fixture, "readback");
      try {
        fs.writeFileSync(source, String.raw`
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Integer intrinsics for the production Keccak, not a replacement hash implementation.
namespace sycl {
using uint4 = std::array<uint32_t, 4>;
inline uint64_t rotate(const uint64_t value, const uint64_t amount) {
  return std::rotl(value, static_cast<int>(amount & 63U));
}
inline uint64_t bitselect(const uint64_t a, const uint64_t b, const uint64_t c) {
  return (a & ~c) | (b & c);
}
} // namespace sycl

inline constexpr unsigned cn_gpu_lanes_per_hash = 16;
template <typename T> inline T mom_rotate(const T value, const T amount) {
  return std::rotl(value, static_cast<int>(amount & (sizeof(T) * 8 - 1)));
}
template <typename T> inline T mom_bitselect(const T a, const T b, const T c) {
  return (a & ~c) | (b & c);
}
#include "crypto.inc"

namespace fixture {
inline int64_t now_us = 0;
inline unsigned sleep_calls = 0;
struct clock {
  static std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::time_point{std::chrono::microseconds{now_us}};
  }
};
inline void sleep_until(const std::chrono::steady_clock::time_point deadline) {
  ++sleep_calls;
  now_us = std::max(now_us, std::chrono::duration_cast<std::chrono::microseconds>(
                                deadline.time_since_epoch()).count());
}

constexpr size_t HASH_LEN = 32;
enum class Failure { none, final_pending, final_retired, copy_submit, copy_pending, copy_retired,
                     buffer_read };
enum class Kind { final, copy };
struct Context {
  unsigned batch;
  Failure failure;
  bool kernels_done = false;
  bool copy_done = false;
  bool consumed = false;
  bool kernels_ready_at_copy = true;
  bool submission_pending = true;
  unsigned final_waits = 0;
  unsigned copy_submits = 0;
  unsigned copy_waits = 0;
  unsigned buffer_reads = 0;
  alignas(uint64_t) std::array<uint8_t, 4 * HASH_LEN> device_bytes{};
  std::array<uint64_t, 4 * 25> spads{};
  uint8_t* copy_destination = nullptr;
  const uint8_t* copy_source = nullptr;
  size_t copy_bytes = 0;

  void finish_kernels() {
    if (kernels_done)
      return;
    if (failure == Failure::final_pending)
      throw std::runtime_error("final-pending");
    auto* const d_spads = spads.data();
    auto* const d_outputs = device_bytes.data();
    for (unsigned t = 0; t < batch; ++t) {
${finalBody}
    }
    kernels_done = true;
    now_us += 300;
    if (failure == Failure::final_retired)
      throw std::runtime_error("final-retired");
  }
};
struct Event { Context* context; Kind kind; };
struct Queue {
  Context* context;
  bool khr_empty() {
    context->submission_pending = false;
    return false;
  }
  Event memcpy(uint8_t* destination, const uint8_t* source, const size_t bytes) {
    context->submission_pending = true;
    ++context->copy_submits;
    context->kernels_ready_at_copy &= context->kernels_done;
    if (context->failure == Failure::copy_submit)
      throw std::runtime_error("copy-submit");
    context->copy_destination = destination;
    context->copy_source = source;
    context->copy_bytes = bytes;
    return {context, Kind::copy};
  }
};
struct Buffer {
  Context* context;
  void read(uint8_t* destination, const size_t bytes) {
    ++context->buffer_reads;
    context->kernels_ready_at_copy &= context->kernels_done;
    if (context->failure == Failure::buffer_read)
      throw std::runtime_error("buffer-read");
    std::memcpy(destination, context->device_bytes.data(), bytes);
    context->copy_done = true;
  }
};
struct State {
  int device = 0;
  bool shared_io;
  // Superset lets the same fixture execute historical and fixed source blocks.
  double wait_ema_us = 0;
  bool wait_warmup_done = false;
  Buffer buffered_outputs;
};

inline void sycl_wait_and_throw(const Event event, const int) {
  auto& context = *event.context;
#if defined(MOM_CN_REQUIRE_SUBMISSION)
  if (context.submission_pending)
    throw std::runtime_error("unsubmitted-command");
#endif
  if (event.kind == Kind::final) {
    ++context.final_waits;
    context.finish_kernels();
    return;
  }
  ++context.copy_waits;
  // An in-order queue eventually retires earlier kernels, but this cannot repair a
  // host driver that blocks while submitting the copy, before returning this event.
  context.finish_kernels();
  if (context.failure == Failure::copy_pending)
    throw std::runtime_error("copy-pending");
  std::memcpy(context.copy_destination, context.copy_source, context.copy_bytes);
  context.copy_done = true;
  now_us += 30;
  if (context.failure == Failure::copy_retired)
    throw std::runtime_error("copy-retired");
}

template <bool mom_sycl_portable_opencl>
void readback(State& state, Context& context, uint8_t* output, const size_t output_bytes) {
  Queue q{&context};
  [[maybe_unused]] auto* const d_outputs = context.device_bytes.data();
  const Event final_event{&context, Kind::final};
${batchStart}
${tail}
}

unsigned checks = 0;
unsigned failures = 0;
void require(const bool condition, const char* const route, const char* const label) {
  ++checks;
  if (condition)
    return;
  ++failures;
  std::fprintf(stderr, "FAIL %s: %s\n", route, label);
}

template <bool Portable>
void run(const bool shared, const unsigned batch, const Failure failure,
         const double stale_ema, const bool warmup) {
  const char* const route = Portable ? "portable-buffer" : shared ? "shared-USM" : "device-USM";
  now_us = 0;
  sleep_calls = 0;
  Context context{batch, failure};
  context.device_bytes.fill(0xd7);
  std::array<uint8_t, 4 * HASH_LEN + 16> output;
  output.fill(0x5a);
  const auto untouched = output;
  // Varied nonce-bearing inputs seed existing Keccak state; CN's preceding GPU
  // arithmetic is deliberately not modeled or claimed validated by this test.
  std::array<uint8_t, 4 * 80> inputs{};
  for (size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = static_cast<uint8_t>((i * 37 + batch * 11) & 255);
  const auto original_inputs = inputs;
  for (size_t i = 0; i < context.spads.size(); ++i)
    context.spads[i] = uint64_t{inputs[i % inputs.size()]} * 0x0101010101010101ULL + i;
  auto oracle = context.spads;
  std::array<uint8_t, 4 * HASH_LEN> expected{};
  for (unsigned t = 0; t < batch; ++t) {
    keccak(oracle.data() + 25 * t);
    std::memcpy(expected.data() + HASH_LEN * t, oracle.data() + 25 * t, HASH_LEN);
  }
  State state{0, shared, stale_ema, warmup, Buffer{&context}};
  std::string error;
  try {
    readback<Portable>(state, context, output.data(), HASH_LEN * batch);
    context.consumed = true;
  } catch (const std::exception& e) {
    error = e.what();
  }
  require(context.final_waits == 1, route, "exactly one final-kernel wait");
  require(context.kernels_ready_at_copy, route, "kernels complete before any readback submission");
  require(sleep_calls == 0, route, "no stale-EMA pre-read sleep");
  require(state.wait_ema_us == stale_ema && state.wait_warmup_done == warmup,
          route, "readback has no EMA/warmup state mutation");
  require(inputs == original_inputs, route, "nonce-bearing input bytes unchanged");
  require(std::equal(output.begin() + HASH_LEN * batch, output.end(),
                     untouched.begin() + HASH_LEN * batch), route, "padding and output guard unchanged");
  if (failure == Failure::none) {
    require(error.empty() && context.consumed, route, "successful result consumed");
    require(context.kernels_done, route, "final kernel finished");
    require(std::equal(output.begin(), output.begin() + HASH_LEN * batch, expected.begin()),
            route, "production final-Keccak/hash output bytes unchanged");
  } else {
    const char* const expected_error = failure == Failure::final_pending ? "final-pending" :
        failure == Failure::final_retired ? "final-retired" :
        failure == Failure::copy_submit ? "copy-submit" :
        failure == Failure::copy_pending ? "copy-pending" :
        failure == Failure::copy_retired ? "copy-retired" : "buffer-read";
    require(error == expected_error && !context.consumed, route, "original error prevents consumption");
    if (failure != Failure::copy_retired)
      require(output == untouched, route, "failure before readback leaves host output untouched");
    if (failure == Failure::final_pending || failure == Failure::final_retired)
      require(context.copy_submits == 0 && context.buffer_reads == 0,
              route, "final-kernel error prevents all readback");
  }
  if constexpr (Portable) {
    if (failure == Failure::none || failure == Failure::buffer_read)
      require(context.buffer_reads == 1, route, "one buffered host read");
    require(context.copy_submits == 0 && context.copy_waits == 0, route, "no USM copy for buffer route");
  } else if (shared) {
    require(context.copy_submits == 0 && context.copy_waits == 0 && context.buffer_reads == 0,
            route, "shared route uses host memcpy only");
  } else {
    if (failure == Failure::none || failure == Failure::copy_pending || failure == Failure::copy_retired)
      require(context.copy_submits == 1 && context.copy_waits == 1,
              route, "deferred device copy waited exactly once");
    if (failure == Failure::copy_submit)
      require(context.copy_submits == 1 && context.copy_waits == 0,
              route, "submission failure cannot wait a nonexistent event");
    if (failure == Failure::none)
      require(context.copy_done, route, "device copy completed before consumption");
  }
}
} // namespace fixture

int main() {
  using namespace fixture;
  for (const unsigned batch : {1U, 2U, 3U, 4U}) {
    for (const double ema : {0.0, 8000000.0}) {
      for (const bool warmup : {false, true}) {
        run<false>(false, batch, Failure::none, ema, warmup);
        run<false>(true, batch, Failure::none, ema, warmup);
        run<true>(false, batch, Failure::none, ema, warmup);
        run<true>(true, batch, Failure::none, ema, warmup);
      }
    }
  }
  for (const Failure failure : {Failure::final_pending, Failure::final_retired}) {
    run<false>(false, 3, failure, 0, false);
    run<false>(true, 3, failure, 0, false);
    run<true>(false, 3, failure, 0, false);
    run<true>(true, 3, failure, 0, false);
  }
  for (const Failure failure : {Failure::copy_submit, Failure::copy_pending, Failure::copy_retired})
    run<false>(false, 3, failure, 0, false);
  run<true>(false, 3, Failure::buffer_read, 0, false);
  std::printf("CN/GPU source-derived readback host fixture: %u checks, %u failures\n", checks, failures);
  return failures ? 1 : 0;
}

`);
        const before = path.join(fixture, "without-submission.cpp");
        // Remove only the progress kick from the same source-derived readback scope.
        assert.ok(tail.includes("(void)q.khr_empty();"), "readback must kick queue progress");
        fs.writeFileSync(before, fs.readFileSync(source, "utf8").replaceAll("(void)q.khr_empty();", ""));
        for (const mode of ["ordinary", "windows-hip", "without-submission"]) {
          const flags = mode === "ordinary" ? [] : ["-D_WIN32", "-DMOM_SYCL_ADAPTIVECPP", "-DMOM_SYCL_HAS_HIP", "-DMOM_CN_REQUIRE_SUBMISSION"];
          const compiled = spawnSync("c++", ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", ...flags, "-I", cn, mode === "without-submission" ? before : source, "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.error, undefined);
          assert.equal(compiled.signal, null);
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8"});
          assert.equal(result.error, undefined);
          assert.equal(result.signal, null);
          if (mode === "without-submission") {
            assert.notEqual(result.status, 0, "unsubmitted work must fail the same readback assertions");
          } else {
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout, "CN/GPU source-derived readback host fixture: 802 checks, 0 failures\n");
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  });

  test.describe("Source-JIT fallback boundaries, parameter probes, and GPU/Pearl tuning", {concurrency: false}, () => {
    test("NVRTC compiler API header compiles without CUDA driver headers", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const headerPath = path.resolve(__dirname, "../sycl/nvrtc-api.h");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-header-"));
      try {
        fs.writeFileSync(path.join(fixture, "nvrtc.h"), `
#pragma once
#include <stddef.h>
using nvrtcProgram = void*;
using nvrtcResult = int;
inline constexpr nvrtcResult NVRTC_SUCCESS = 0;
nvrtcResult nvrtcVersion(int*, int*);
nvrtcResult nvrtcCreateProgram(nvrtcProgram*, const char*, const char*, int,
                               const char* const*, const char* const*);
nvrtcResult nvrtcCompileProgram(nvrtcProgram, int, const char* const*);
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram, size_t*);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram, char*);
nvrtcResult nvrtcGetPTXSize(nvrtcProgram, size_t*);
nvrtcResult nvrtcGetPTX(nvrtcProgram, char*);
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram, size_t*);
nvrtcResult nvrtcGetCUBIN(nvrtcProgram, char*);
nvrtcResult nvrtcGetNumSupportedArchs(int*);
nvrtcResult nvrtcGetSupportedArchs(int*);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram*);
const char* nvrtcGetErrorString(nvrtcResult);
`);
        const source = path.join(fixture, "nvrtc_api_header.cpp");
        fs.writeFileSync(source, `#include "${headerPath}"
static_assert(sizeof(mom::NvrtcApi) > 0);
static_assert(sizeof(mom::NvrtcProgram) > 0);
`);
        assert.equal(fs.existsSync(path.join(fixture, "cuda.h")), false);
        for (const mode of ["dynamic", "static"]) {
          const compiled = spawnSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-fsyntax-only", ...(mode === "static" ? ["-DMOM_STATIC_NVRTC"] : []), "-I", fixture, source],
          {encoding: "utf8", timeout: 30000});
          assert.equal(compiled.error, undefined, mode);
          assert.equal(compiled.signal, null, mode);
          assert.equal(compiled.status, 0, `${mode}: ${compiled.stderr}`);
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("All native algorithms share the managed NVRTC compiler with legacy fallbacks", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const header = fs.readFileSync(path.join(__dirname, "../sycl/nvrtc-api.h"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-shared-"));
      try {
        fs.writeFileSync(path.join(fixture, "loader.inc"), extractCppBlock(header, "class NvrtcApi {") + ";\n");
        const source = path.join(__dirname, "native/nvrtc-loader.cpp");
        const executable = path.join(fixture, "loader");
        const build = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, source, "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(build.status, 0, build.error?.message || build.stderr);
        for (const mode of ["managed", "unversioned", "alias13", "legacy", "none",
          "missing-version", "missing-cubin", "missing-required", "missing-builtins"]) {
          await t.test(mode, () => {
            const run = spawnSync(executable, [mode], {encoding: "utf8", timeout: 10000});
            assert.equal(run.status, 0, run.error?.message || run.stderr);
            assert.equal(run.stdout, "SHARED_NVRTC_LOADER_PASS mode=" + mode + " GPU_calls=0\n");
          });
        }
        await t.test("static archive binding performs no dynamic loader calls", () => {
          const native = path.join(fixture, "static-loader");
          const build = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", "-DMOM_STATIC_NVRTC", "-I" + fixture, source, "-o", native], {encoding: "utf8", timeout: 30000});
          assert.equal(build.status, 0, build.error?.message || build.stderr);
          const run = spawnSync(native, ["static"], {encoding: "utf8", timeout: 10000});
          assert.equal(run.status, 0, run.error?.message || run.stderr);
          assert.equal(run.stdout, "SHARED_NVRTC_LOADER_PASS mode=static GPU_calls=0\n");
        });
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("NVRTC temporary programs release once across source-JIT failures", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const cudaHeader = fs.readFileSync(path.join(root, "cuda-api.h"), "utf8");
      const nvrtcHeader = fs.readFileSync(path.join(root, "nvrtc-api.h"), "utf8");
      const helpers = cudaHeader.indexOf("inline void cuda_check(");
      const cudaNamespaceEnd = cudaHeader.lastIndexOf("} // namespace mom");
      const rtcErrorStart = nvrtcHeader.indexOf("inline std::string rtc_error(");
      const nvrtcNamespaceEnd = nvrtcHeader.lastIndexOf("} // namespace mom");
      assert.ok(helpers >= 0 && cudaNamespaceEnd > helpers);
      assert.ok(rtcErrorStart >= 0 && nvrtcNamespaceEnd > rtcErrorStart);
      const owner = extractCppBlock(nvrtcHeader, "class NvrtcProgram {") + ";\n";
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-program-"));
      const source = path.join(fixture, "nvrtc-program.cpp");
      const executable = path.join(fixture, "nvrtc_program");
      try {
        fs.copyFileSync(path.join(__dirname, "native/nvrtc-program.cpp"), source);
        fs.writeFileSync(path.join(fixture, "api.inc"), "namespace mom {\n" + owner +
        cudaHeader.slice(helpers, cudaNamespaceEnd) + nvrtcHeader.slice(rtcErrorStart, nvrtcNamespaceEnd) + "\n}\n");
        const cn = fs.readFileSync(path.join(root, "cn-gpu/cuda-jit.inc"), "utf8");
        const cnEnd = cn.indexOf("  void launch(");
        const cnDestructor = cn.indexOf("  ~CnGpuCudaExpand()");
        const cnClassEnd = cn.indexOf("\n};", cnDestructor);
        assert.ok(cnEnd > 0 && cnDestructor > cnEnd && cnClassEnd > cnDestructor);
        fs.writeFileSync(path.join(fixture, "cn.inc"), cn.slice(0, cnEnd) +
        cn.slice(cnDestructor, cnClassEnd) + "\n};\n#endif\n");
        const pearl = fs.readFileSync(path.join(root, "pearlhash/cuda-search.inc"), "utf8");
        const pearlEnd = pearl.indexOf("  void launch(");
        const pearlDestructor = pearl.indexOf("  ~PearlHashCudaSearch()");
        const pearlClassEnd = pearl.indexOf("\n};", pearlDestructor);
        assert.ok(pearlEnd > 0 && pearlDestructor > pearlEnd && pearlClassEnd > pearlDestructor);
        fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) +
        pearl.slice(pearlDestructor, pearlClassEnd) + "\n};\n");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_FIXTURE_HAS_OWNER=1", "-I", path.join(root, "cn-gpu"),
          "-I", path.join(root, "pearlhash"), source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const cases = [
          {id: 0, name: "success releases before module loading"},
          {id: 1, name: "unavailable API creates no program"},
          {id: 2, name: "failed creation with a null handle"},
          {id: 3, name: "failed creation with a returned handle"},
          {id: 4, name: "compile failure retains diagnostic fallback"},
          {id: 5, name: "log-size failure retains diagnostic fallback"},
          {id: 6, name: "log retrieval failure retains diagnostic fallback"},
          {id: 7, name: "code-size failure releases the program"},
          {id: 8, name: "code retrieval failure releases the program"},
          {id: 9, name: "log allocation failure releases the program"},
          {id: 10, name: "code allocation failure releases the program"},
          {id: 11, name: "unsupported capability creates no program"},
          {id: 12, name: "null program is not destroyed"},
          {id: 15, name: "cache hit loads without creating a program"},
          {id: 16, name: "non-CUDA device creates no program"},
        ];
        for (const [variant, name] of ["CN native CUBIN", "Pearl CUBIN", "Pearl forward PTX",
          "CN forward PTX", "CN PTX-only API"].entries()) {
          await t.test(name, {concurrency: false}, async (variantTest) => {
            for (const {id, name: description} of cases) {
              await variantTest.test(name + ": " + description, () => {
                const result = spawnSync(executable, [String(variant + 1), String(id)], {encoding: "utf8"});
                assert.equal(result.status, 0, result.error?.message || result.stderr);
                assert.equal(result.signal, null);
                assert.equal(result.stdout, "PASS actual-source NVRTC lifetime\n");
              });
            }

          });
        }
        for (const [id, description] of [
          [21, "architecture-count query failure"],
          [22, "empty architecture list"],
          [23, "unbounded architecture list"],
          [24, "architecture query failure"],
          [25, "empty compiler output"],
          [26, "oversized compiler output"],
        ]) {
          await t.test(`CN native CUBIN: ${description}`, () => {
            const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
            assert.equal(result.status, 0, result.error?.message || result.stderr);
            assert.equal(result.signal, null);
            assert.equal(result.stdout, "PASS actual-source NVRTC lifetime\n");
          });
        }
        for (const [id, description] of [
          [17, "null A allocation falls back before NVRTC"],
          [18, "null B allocation falls back before NVRTC"],
          [19, "misaligned A allocation falls back before NVRTC"],
          [20, "misaligned B allocation falls back before NVRTC"],
        ]) {
          for (const variant of [2, 3]) {
            await t.test(`${variant === 2 ? "Pearl CUBIN" : "Pearl forward PTX"}: ${description}`, () => {
              const result = spawnSync(executable, [String(variant), String(id)], {encoding: "utf8"});
              assert.equal(result.status, 0, result.error?.message || result.stderr);
              assert.equal(result.signal, null);
              assert.equal(result.stdout, "PASS actual-source NVRTC lifetime\n");
            });
          }
        }
        for (const {id, name, output} of [
          {id: 13, name: "empty owner/reset is a no-op", output: "PASS empty owner\n"},
          {id: 14, name: "reset and destructor do not repeat destruction", output: "PASS explicit reset\n"},
        ]) {
          await t.test(name, () => {
            const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
            assert.equal(result.status, 0, result.error?.message || result.stderr);
            assert.equal(result.signal, null);
            assert.equal(result.stdout, output);
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Nexa monolithic result clear stays ordered without pre-search wait", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl");
      const host = fs.readFileSync(path.join(root, "nexapow/nexapow.cpp"), "utf8");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const section = (source, start, end) => {
        const begin = source.indexOf(start), stop = source.indexOf(end, begin + start.length);
        assert.ok(begin >= 0 && stop > begin, start);
        return source.slice(begin, stop);
      };
      const entry = section(host, "  state.init_portable();", "  return static_cast<int>(hits);\n}") +
        "  return static_cast<int>(hits);\n";
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nexapow-submission-"));
      try {
        const source = path.join(fixture, "submission.cpp"), executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/nexapow-submission.cpp"), source);
        fs.writeFileSync(path.join(fixture, "result.inc"), extractCppBlock(host, "struct Result {") + ";\n");
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), ["inline void sycl_log_cleanup_exception(",
          "inline void sycl_cleanup_noexcept(", "class MomSyclHostTransferGuard {"]
          .map(anchor => extractCppBlock(library, anchor)).join("") + ";\n");
        for (const fixed of [false, true]) {
          const clear = "state.queue.memset(state.result, 0, sizeof(Result));";
          assert.ok(entry.includes(clear));
          fs.writeFileSync(path.join(fixture, "entry.inc"), fixed ? entry :
            entry.replace(clear, "sycl_wait_and_throw(state.queue.memset(state.result, 0, sizeof(Result)), state.device);"));
          const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-fsanitize=address", "-fsanitize-address-use-after-scope", "-fno-omit-frame-pointer",
            ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []), source, "-o", executable],
          {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          assert.equal(result.signal, null);
          assert.equal(result.status, fixed ? 0 : 1, result.stdout + result.stderr);
          if (fixed) {
            assert.equal(result.stdout, "PASS actual-source NexaPoW monolithic submission order\n");
          } else {
            assert.match(result.stderr, /NexaPoW command order changed/);
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("CN/gpu CUDA expansion falls back only before native submission", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const directory = path.join(__dirname, "../sycl/cn-gpu");
      const cudaHeader = fs.readFileSync(path.join(directory, "../cuda-api.h"), "utf8");
      const nvrtcHeader = fs.readFileSync(path.join(directory, "../nvrtc-api.h"), "utf8");
      const owner = extractCppBlock(nvrtcHeader, "class NvrtcProgram {") + ";\n";
      const entry = fs.readFileSync(path.join(directory, "cn-gpu.cpp"), "utf8");
      const start = entry.indexOf("  bool native_expansion = false;");
      const end = entry.indexOf("  if (!native_expansion) {", start);
      assert.ok(start >= 0 && end > start);
      const dispatch = entry.slice(start, end);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cn-cuda-"));
      const source = path.join(fixture, "cuda.cpp");
      const executable = path.join(fixture, "cuda");
      fs.writeFileSync(source, `
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using CUresult = int;
using CUdevice = int;
using CUmodule = void*;
using CUfunction = void*;
using CUevent = void*;
using CUcontext = void*;
using nvrtcProgram = void*;
using nvrtcResult = int;
constexpr int CUDA_SUCCESS = 0;
constexpr int CUDA_ERROR_NOT_READY = 600;
constexpr int NVRTC_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 1;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 2;
constexpr int CU_EVENT_DEFAULT = 0;
static int fault;
static bool driver_available = true;
struct Counts {
  unsigned retain = 0;
  unsigned unload = 0;
  unsigned release = 0;
  unsigned destroy = 0;
  unsigned launch = 0;
  unsigned record = 0;
  unsigned query = 0;
  unsigned synchronize = 0;
  unsigned create_program = 0;
  unsigned destroy_program = 0;
  unsigned initial_wait = 0;
  unsigned generic = 0;
} counts;
static int handles[4];
namespace sycl {
enum class backend { ext_oneapi_cuda };
struct device { bool cuda = true; };
struct queue { device get_device() const { return {}; } };
template<backend> CUdevice get_native(const device&) { return 0; }
}
static bool mom_is_cuda(const sycl::device& device) { return device.cuda; }
namespace mom {
struct CudaDriverApi {
  std::string error = "driver-unavailable";
  static CudaDriverApi& instance() {
    static CudaDriverApi api;
    return api;
  }
  bool basic_available() const { return driver_available; }
  int (*init)(unsigned) = +[](unsigned) { return 0; };
  int (*device_attribute)(int*, int, int) = +[](int* out, int attribute, int) {
    *out = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 9 : 0;
    return 0;
  };
  int (*retain_primary)(void**, int) = +[](void** out, int) {
    ++counts.retain;
    *out = &handles[0];
    return 0;
  };
  int (*set_current)(void*) = +[](void*) { return 0; };
  int (*synchronize)() = +[]() {
    ++counts.synchronize;
    return 999; // Secondary cleanup errors must not replace the original native failure.
  };
  int (*module_load)(void**, const void*) = +[](void** out, const void*) {
    *out = &handles[1];
    return 0;
  };
  int (*module_function)(void**, void*, const char*) = +[](void** out, void*, const char*) {
    if (fault == 5) return 999;
    *out = &handles[2];
    return 0;
  };
  int (*event_create)(void**, unsigned) = +[](void** out, unsigned) {
    *out = &handles[3];
    return 0;
  };
  int (*event_destroy)(void*) = +[](void*) {
    ++counts.destroy;
    return 0;
  };
  int (*module_unload)(void*) = +[](void*) {
    ++counts.unload;
    return 0;
  };
  int (*release_primary)(int) = +[](int) {
    ++counts.release;
    return 0;
  };
  int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, void*, void**, void**) =
      +[](void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
          unsigned, void*, void**, void**) {
        ++counts.launch;
        return fault == 2 ? 999 : 0;
      };
  int (*event_record)(void*, void*) = +[](void*, void*) {
    ++counts.record;
    return fault == 4 ? 999 : 0;
  };
  int (*event_query)(void*) = +[](void*) {
    ++counts.query;
    if (fault == 3) return 999;
    return counts.query == 1 ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
  };
};
struct NvrtcApi {
  std::string error = "rtc-unavailable";
  static NvrtcApi& instance() {
    static NvrtcApi api;
    return api;
  }
  bool ptx_available() const { return true; }
  bool cubin_available() const { return false; }
  int (*get_num_supported_archs)(int*) = nullptr;
  int (*get_supported_archs)(int*) = nullptr;
  int (*get_cubin_size)(void*, size_t*) = nullptr;
  int (*get_cubin)(void*, char*) = nullptr;
  int (*create_program)(void**, const char*, const char*, int, const char* const*, const char* const*) =
      +[](void** out, const char*, const char*, int, const char* const*, const char* const*) {
        ++counts.create_program;
        *out = &handles[2];
        return 0;
      };
  int (*compile_program)(void*, int, const char* const*) =
      +[](void*, int, const char* const*) { return 0; };
  int (*get_ptx_size)(void*, size_t*) = +[](void*, size_t* out) {
    *out = 1;
    return 0;
  };
  int (*get_ptx)(void*, char*) = +[](void*, char* out) {
    *out = 0;
    return 0;
  };
  int (*destroy_program)(void**) = +[](void** program) {
    ++counts.destroy_program;
    *program = nullptr;
    return 0;
  };
};
${owner}
void cuda_check(const CudaDriverApi&, int status, const char* message) {
  if (status != CUDA_SUCCESS) throw std::string(message);
}
${extractCppBlock(cudaHeader, "inline void cuda_check_submitted(")}
std::string rtc_error(const NvrtcApi&, int, void*, const char* message) { return message; }
namespace jit_cache {
uint64_t hash(const std::string&) { return 1; }
std::filesystem::path directory() { return {}; }
std::vector<char> read(const std::filesystem::path&) { return {}; }
void write(const std::filesystem::path&, const std::vector<char>&) {}
}
}
#include "cuda-jit.inc"
struct State {
  sycl::device device;
  CnGpuCudaExpand cuda_expand;
  bool native_notice = false;
  uint64_t storage[8]{};
  uint64_t* spads = storage;
  uint64_t* lpads = storage;
};
static constexpr bool mom_sycl_portable_opencl = false;
static void sycl_wait_and_throw(int, const sycl::device&) {
  ++counts.initial_wait;
  if (fault == 1) throw std::runtime_error("initial-event");
}
static bool dispatch(State& state) {
  sycl::queue q;
  const std::string backend = "auto";
  const int initial_keccak = 0;
  const unsigned batch_eff = 4;
${dispatch}
  if (!native_expansion) ++counts.generic;
  return native_expansion;
}
static void check(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("case argument");
    const int id = std::atoi(argv[1]);
#if defined(MOM_SYCL_HAS_CUDA)
    if (id == 9)
      mom::CudaDriverApi::instance().synchronize = nullptr;
    if (id == 6) {
      fault = 5;
      {
        CnGpuCudaExpand expand;
        sycl::queue queue;
        check(!expand.ensure(queue, nullptr), "nullable startup error did not return false");
        check(expand.checked && !expand.enabled && !expand.context && !expand.module &&
              !expand.kernel && !expand.done && !expand.retained_primary,
              "partial startup resources survived");
        check(counts.retain == 1 && counts.release == 1 && counts.unload == 1 &&
              counts.destroy == 0 && counts.create_program == counts.destroy_program,
              "partial startup cleanup changed");
      }
      check(counts.release == 1 && counts.unload == 1, "startup cleanup ran twice");
    } else {
      driver_available = id != 1;
      fault = id == 3 ? 1 : id == 4 ? 2 : id == 5 ? 3 : id == 7 ? 4 : 0;
      const bool expected_fault = id == 3 || id == 4 || id == 5 || id == 7;
      {
        State state;
        bool native = false;
        std::string caught;
        try {
          native = dispatch(state);
        } catch (const std::string& error) {
          caught = error;
        } catch (const std::exception& error) {
          caught = error.what();
        }
        std::printf("OBS case=%d native=%d generic=%u caught=%s retain=%u release=%u unload=%u destroy=%u launch=%u record=%u query=%u\\n",
                    id, native, counts.generic, caught.c_str(), counts.retain, counts.release,
                    counts.unload, counts.destroy, counts.launch, counts.record, counts.query);
        if (expected_fault) {
          check(!caught.empty(), "post-submit fault was swallowed");
          check(counts.generic == 0, "post-submit fault retried generic expansion");
          check(state.cuda_expand.enabled && counts.release == 0 && counts.unload == 0 &&
                counts.destroy == 0, "post-submit resources released before owner cleanup");
          const std::string expected = id == 3 ? "initial-event" :
              id == 4 ? "cuLaunchKernel(cn/gpu)" :
              id == 5 ? "cuEventQuery(cn/gpu)" : "cuEventRecord(cn/gpu)";
          check(caught == expected, "fault identity changed");
          check(counts.synchronize == (id == 3 ? 0u : 1u),
                "native failure did not retire work before returning to SYCL");
        } else if (id == 1 || id == 9) {
          check(!native && counts.generic == 1 && caught.empty() && counts.launch == 0 &&
                state.native_notice, "startup unavailability did not fall back");
          check(counts.create_program == 0, "unsafe native backend compiled before fallback");
        } else {
          check(id == 2 && native && counts.generic == 0 && caught.empty() &&
                counts.launch == 1 && counts.record == 1 && counts.query == 2 &&
                counts.synchronize == 0,
                "native success changed");
        }
      }
      check(counts.release == (id == 1 || id == 9 ? 0u : 1u) &&
            counts.unload == counts.release && counts.destroy == counts.release,
            "owner cleanup did not release resources once");
      check(counts.create_program == counts.destroy_program, "NVRTC program leaked");
    }
#else
    check(id == 8, "unsupported-worker case argument");
    {
      State state;
      check(!dispatch(state) && counts.generic == 1 && counts.initial_wait == 0,
            "unsupported worker did not fall back");
      sycl::queue queue;
      check(!state.cuda_expand.ensure(queue, nullptr), "stub nullable reason changed");
    }
#endif
    std::printf("PASS case=%d\\n", id);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\\n", error.what());
    return 1;
  }
}
`);
      const cases = [
        "startup unavailability retains generic expansion",
        "native success retains deferred owner cleanup",
        "initial SYCL event fault propagates without generic retry",
        "native launch fault propagates without generic retry",
        "native event-query fault propagates without generic retry",
        "nullable startup reason cleans partial resources once",
        "post-launch event-record fault propagates without generic retry",
        "unsupported worker retains nullable-reason fallback",
        "missing native retirement API falls back before submission",
      ];
      try {
        for (const enabled of [true, false]) {
          const target = executable + (enabled ? "-enabled" : "-stub");
          const flags = enabled ? ["-DMOM_SYCL_HAS_CUDA=1"] : [];
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", ...flags, "-I", directory, source, "-o", target], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const indexes = enabled ? [0, 1, 2, 3, 4, 5, 6, 8] : [7];
          for (const index of indexes) {
            const name = cases[index];
            assert.ok(name);
            await t.test(name, () => {
              const result = spawnSync(target, [String(index + 1)], {encoding: "utf8"});
              assert.equal(result.status, 0, result.error?.message || result.stderr);
              assert.equal(result.signal, null);
              assert.ok(result.stdout.endsWith("PASS case=" + (index + 1) + "\n"));
            });
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("HIPRTC temporary programs release once across source-JIT failures", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const header = fs.readFileSync(path.join(root, "hiprtc-api.h"), "utf8");
      const owner = extractCppBlock(header, "class HiprtcProgram {") + ";\n";
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hiprtc-program-"));
      const source = path.join(fixture, "hiprtc-program.cpp");
      const executable = path.join(fixture, "hiprtc_program");
      try {
        fs.copyFileSync(path.join(__dirname, "native/hiprtc-program.cpp"), source);
        fs.writeFileSync(path.join(fixture, "owner.inc"), "namespace mom {\n" + owner + "}\n");
        fs.writeFileSync(path.join(fixture, "capability.inc"), "namespace mom {\n" +
          extractCppBlock(header, "inline bool hip_gfx12_int8_wmma_supported(") + "}\n");
        for (const algorithm of ["octopus", "walahash"]) {
          const caller = fs.readFileSync(path.join(root, algorithm, "amd-wmma.inc"), "utf8");
          const start = caller.indexOf("struct ");
          const end = caller.indexOf("  bool launch(");
          assert.ok(start >= 0 && end > start);
          fs.writeFileSync(path.join(fixture, algorithm + ".inc"), caller.slice(start, end) + "};\n");
        }
        const pearl = fs.readFileSync(path.join(root, "pearlhash/hip-jit.inc"), "utf8");
        const pearlEnd = pearl.indexOf("  void launch(");
        assert.ok(pearlEnd > 0);
        fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) + "#endif\n};\n");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_HIP=1", "-DMOM_FIXTURE_HAS_OWNER=1", "-I", path.join(root, "pearlhash"),
          source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const cases = [
          "successful compilation releases before module loading",
          "unavailable API creates no program",
          "failed creation with a null handle",
          "failed creation with a returned handle",
          "compile failure preserves diagnostic fallback",
          "log-size failure preserves diagnostic fallback",
          "log retrieval failure preserves diagnostic fallback",
          "code-size failure releases the program",
          "code retrieval failure releases the program",
          "log allocation failure releases the program",
          "code allocation failure releases the program",
          "unsupported architecture creates no program",
          "null program is not destroyed",
        ];
        for (const [algorithm, name] of ["Octopus", "WalaHash", "PearlHash"].entries()) {
          await t.test(name, {concurrency: false}, async (algorithmTest) => {
            for (const [id, description] of cases.entries()) {
              await algorithmTest.test(name + ": " + description, () => {
                const result = spawnSync(executable, [String(algorithm + 1), String(id)], {encoding: "utf8"});
                assert.equal(result.status, 0, result.error?.message || result.stderr);
                assert.equal(result.signal, null);
                assert.equal(result.stdout, "PASS actual-source RTC lifetime\n");
              });
            }

          });
        }
        for (const {id, name, output} of [
          {id: 13, name: "empty owner/reset is a no-op", output: "PASS empty owner\n"},
          {id: 14, name: "reset and destructor do not repeat destruction", output: "PASS explicit reset\n"},
        ]) {
          await t.test(name, () => {
            const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
            assert.equal(result.status, 0, result.error?.message || result.stderr);
            assert.equal(result.signal, null);
            assert.equal(result.stdout, output);
          });
        }
        for (const [id, output] of [
          [15, "PASS capability property failure\n"],
          [16, "PASS capability interop propagation\n"],
        ]) {
          await t.test("optional AMD capability probe failure " + id, () => {
            const result = spawnSync(executable, ["1", String(id)], {encoding: "utf8"});
            assert.equal(result.status, 0, result.error?.message || result.stderr);
            assert.equal(result.signal, null);
            assert.equal(result.stdout, output);
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("HIP interop and Octopus submitted faults propagate without fallback", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const header = fs.readFileSync(path.join(root, "hiprtc-api.h"), "utf8");
      const octopus = fs.readFileSync(path.join(root, "octopus/nvidia-tensor.inc"), "utf8");
      const classStart = octopus.indexOf("class OctopusSyclNativeSearch {");
      const wmma = fs.readFileSync(path.join(root, "octopus/amd-wmma.inc"), "utf8");
      const wmmaStart = wmma.indexOf("  bool ensure(");
      const wmmaEnd = wmma.indexOf("    hipDeviceProp_t properties{}", wmmaStart);
      assert.ok(classStart > 0 && wmmaStart > 0 && wmmaEnd > wmmaStart);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-interop-"));
      const source = path.join(fixture, "hip-queue-identity.cpp");
      const executable = path.join(fixture, "hip_queue_identity");
      try {
        fs.copyFileSync(path.join(__dirname, "native/hip-queue-identity.cpp"), source);
        fs.writeFileSync(path.join(fixture, "identity.inc"), "namespace mom {\n" +
          extractCppBlock(header, "inline void hip_queue_identity(") +
          extractCppBlock(header, "inline void hip_queue_device(") + "}\n");
        fs.writeFileSync(path.join(fixture, "octopus.inc"), octopus.slice(classStart));
        fs.writeFileSync(path.join(fixture, "gemm-context.inc"), extractCppBlock(octopus, "struct NativeGemmContext {") + ";\n");
        fs.writeFileSync(path.join(fixture, "wmma-init.inc"), wmma.slice(wmmaStart, wmmaEnd));
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_HIP=1", source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        /** @type {[number, string][]} */
        const hipErrors = [[1, "InvalidValue"], [3, "NotInitialized"], [4, "Deinitialized"],
          [201, "InvalidContext"], [400, "InvalidHandle"], [709, "ContextIsDestroyed"], [999, "Unknown"]];
        /** @type {[number, string][]} */
        const interopOperations = [[1, "submit"], [2, "native queue"], [3, "interop wait"]];
        /** @type {[number, string][]} */
        const octopusOperations = [[4, "B preparation"], [5, "B preparation wait"],
          [6, "A preparation"], [7, "GEMM launch"], [8, "finish launch"], [9, "completion"],
          [10, "compiler exception"]];
        /** @type {[number, string][]} */
        const octopusFallbacks = [[11, "queue allocation"], [12, "buffer allocation"],
          [13, "compiler/capability absence"], [15, "point-version cache invalidation"],
          [16, "non-HIP device fallback"]];
        const cases = [];
        for (const [target, name] of ["identity", "device alias", "Octopus search"].entries()) {
          cases.push({target, stage: 0, kind: 0, name: name + " success/cache/pipeline"});
          for (const [kind, error] of hipErrors) {
            cases.push({target, stage: 14, kind, name: name + " rejects HIP " + error});
          }
          for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
            for (const [stage, operation] of interopOperations) {
              cases.push({target, stage, kind, name: name + " " + operation + " propagates " + error});
            }
            if (target === 2) {
              for (const [stage, operation] of octopusOperations) {
                cases.push({target, stage, kind, name: name + " " + operation + " propagates " + error});
              }
            }
          }
        }
        for (const [stage, name] of octopusFallbacks) {
          cases.push({target: 2, stage, kind: 0, name: "Octopus retains " + name + " behavior"});
        }
        for (const target of [3, 4]) {
          for (const kind of [0, 1, 2]) {
            for (const [stage, operation] of octopusOperations.filter(([stage]) => stage < 10)) {
              cases.push({target, stage, kind,
                name: operation + " retires pipeline before retry" + (target === 4 ? " despite cleanup faults" : "")});
            }
          }
        }
        await runGroupedCases(
          t, cases,
          ({target, stage, kind}) => {
            if (target >= 3) {
              return "Octopus live-owner retirement: " + ["std::exception", "std::string", "unknown"][kind] +
                (target === 4 ? " with cleanup faults" : "");
            }
            if (target < 2) {return target ? "Device alias" : "HIP identity";}
            if (stage === 0 || stage === 14) {return "Octopus HIP compatibility";}
            if (stage <= 10) {
              return "Octopus submitted faults: " + ["std::exception", "std::string", "unknown"][kind];
            }
            return "Octopus allocation, cache, and non-HIP fallback";
          },
          async (caseTest, {target, stage, kind, name}) => {
            await caseTest.test(name, () => {
              const result = spawnSync(executable, [String(target), String(stage), String(kind)],
                {encoding: "utf8"});
              assert.equal(result.error, undefined);
              assert.equal(result.signal, null);
              assert.equal(result.status, 0, result.stderr);
              assert.equal(result.stdout, "PASS actual-source HIP interop/Octopus fault boundary\n");
            });
          });
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("algo params probes share identical effective environments only", async () => {
      let probeCount = 0;
      const cache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
        ++probeCount;
        return Promise.resolve({params: {}, stdout: "", stderr: ""});
      });
      const dpcppKawpow = {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NATIVE_PATH: "runtime/dpcpp/mom.node",
        MOM_SYCL_COMPILER: "dpcpp",
        ONEAPI_DEVICE_SELECTOR: "cuda:gpu",
      };
      const dpcppEtchash = {
        ONEAPI_DEVICE_SELECTOR: "cuda:gpu",
        MOM_SYCL_COMPILER: "dpcpp",
        MOM_NATIVE_PATH: "runtime/dpcpp/mom.node",
        MOM_GPU_BACKEND: "nvidia",
      };
      const acppFishhash = {
        ...dpcppKawpow,
        MOM_NATIVE_PATH: "runtime/acpp-cuda/mom.node",
        MOM_SYCL_COMPILER: "acpp-cuda",
        ACPP_VISIBILITY_MASK: "cuda",
      };

      const [first, second] = await Promise.all([
        cache(dpcppKawpow), cache(dpcppEtchash),
      ]);
      assert.strictEqual(first, second);
      assert.equal(probeCount, 1);
      await cache(acppFishhash);
      assert.equal(probeCount, 2);

      let envProbeCount = 0;
      const envCache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
        ++envProbeCount;
        return Promise.resolve({params: {}, stdout: "", stderr: ""});
      });
      const envWithSentinel = {
        ...dpcppKawpow,
        MOM_ALGO_PARAMS_CACHE_SENTINEL: "inherited",
      };
      const deletedEnvValue = await envCache({
        ...envWithSentinel,
        MOM_ALGO_PARAMS_CACHE_SENTINEL: undefined,
      });
      const emptyEnvValue = await envCache({
        ...envWithSentinel,
        MOM_ALGO_PARAMS_CACHE_SENTINEL: "",
      });
      assert.notStrictEqual(deletedEnvValue, emptyEnvValue);
      assert.equal(envProbeCount, 2);
    });

    test("algo params probe cache retries rejected probes without duplicating concurrent work", async () => {
      let probeCount = 0;
      let shouldReject = true;
      const cache = createAlgoParamsReportCache((/** @type {TestEnv} */ _env) => {
        ++probeCount;
        return shouldReject
          ? Promise.reject(new Error("probe failed"))
          : Promise.resolve({params: {}, stdout: "", stderr: ""});
      });
      const env = {MOM_ALGO_PARAMS_CACHE_RETRY: "1"};
      const [first, concurrent] = [cache(env), cache({...env})];
      assert.strictEqual(first, concurrent);
      assert.equal(probeCount, 1);
      await assert.rejects(first, /probe failed/);

      shouldReject = false;
      const retry = cache(env);
      assert.notStrictEqual(retry, first);
      assert.equal(probeCount, 2);
      await retry;
    });

    test("GPU tuning syntax preserves per-device workers and partial overrides", () => {
      const entries = gpuTuning.parseDeviceList(
        "gpu1*[intensity=39612672;workgroup=256]^2,gpu2*[workgroup=128]", "kawpow");
      assert.equal(gpuTuning.formatDeviceList(entries),
        "gpu1*[intensity=39612672;workgroup=256]^2,gpu2*[workgroup=128]");
      assert.equal(gpuTuning.formatDeviceList(
        gpuTuning.parseDeviceList("gpu1*39612672", "kawpow")),
      "gpu1*[intensity=39612672]");
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*128", "c29")),
        "gpu1*[seed_workgroup=128]");
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*8192", "pearlhash")),
        "gpu1*[m=8192]");
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*176", "zelhash")),
        "gpu1*[slots=176]");
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*1", "zhash")),
        "gpu1*[intensity=1]");
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "zhash"), /intensity must be 1/);
      assert.equal(gpuTuning.formatDeviceList(
        gpuTuning.parseDeviceList("gpu1*1", "equihash192_7")), "gpu1*[intensity=1]");
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "equihash192_7"),
        /intensity must be 1/);
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*1", "c30")),
        "gpu1*[intensity=1]");
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*2", "c30"), /intensity must be 1/);
      assert.deepEqual(gpuTuning.tuningEnvironment("zelhash", {slots: 176}), {
        MOM_ZELHASH_SLOTS: "176",
      });
      assert.equal(gpuTuning.formatDeviceList(gpuTuning.parseDeviceList("gpu1*256", "beamhash3")),
        "gpu1*[workgroup=256]");
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*[]", "kawpow"), /must not be empty/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1[intensity=2]", "kawpow"),
        /invalid device entry/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*[intensity=2]", "beamhash3"),
        /intensity/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*[workgroup=63]", "kawpow"),
        /must be one of/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1*[intensity=1e3]", "kawpow"),
        /base-10 integer/);
      assert.throws(() => gpuTuning.validateTuning("fishhash", {intensity: true}),
        /base-10 integer/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1024", "kawpow"), /index must be at most 1023/);
      assert.throws(() => gpuTuning.parseDeviceList("gpu1^9007199254740992", "kawpow"),
        /process count must be at most 1024/);
      const dagEntry = gpuTuning.parseDeviceList("gpu1*[dag_chunk=0]", "kawpow")[0];
      const cacheEntry = gpuTuning.parseDeviceList("gpu1*[cache_block=0]", "pearlhash")[0];
      assert.ok(dagEntry && cacheEntry);
      assert.deepEqual(dagEntry.tuning, {dag_chunk: 0});
      assert.deepEqual(cacheEntry.tuning, {cache_block: 0});
    });

    test("empirical GPU tuning candidates stay bounded around portable heuristics", () => {
    /** @param {string} algo @param {string} dev */
      const formats = (algo, dev) => gpuTuning.autotuneCandidates(
        algo, gpuTuning.parseDeviceEntry(dev, algo)
      ).map(gpuTuning.formatDeviceEntry);
      assert.deepEqual(formats("cn/gpu", "gpu1*[intensity=1536]"), [
        "gpu1*[intensity=1536]",
        "gpu1*[intensity=768]",
        "gpu1*[intensity=1152]",
      ]);
      const autolykos = formats("autolykos2", "gpu2*[intensity=26843520;workgroup=64]^2");
      assert(autolykos.includes("gpu2*[intensity=33554176;workgroup=64]^2"));
      assert(autolykos.includes("gpu2*[intensity=26843520;workgroup=256]^2"));
      assert.deepEqual(formats("autolykos2", "gpu1*[intensity=4294967295;workgroup=64]"), [
        "gpu1*[intensity=4294967295;workgroup=64]",
        "gpu1*[intensity=2147483392;workgroup=64]",
        "gpu1*[intensity=3221225216;workgroup=64]",
        "gpu1*[intensity=4294967040;workgroup=64]",
        "gpu1*[intensity=4294967295;workgroup=32]",
        "gpu1*[intensity=4294967295;workgroup=128]",
        "gpu1*[intensity=4294967295;workgroup=256]",
      ]);
      assert.equal(formats("zelhash", "gpu1*[slots=4480]").length, 1);
      for (const algo of ["c30", "equihash192_7", "zhash"]) {
        assert.deepEqual(formats(algo, "gpu1*[intensity=1]"), ["gpu1*[intensity=1]"]);
      }
      assert(formats("beamhash3", "gpu1*[workgroup=640]")
        .every((dev) => !dev.includes("workgroup=768") && !dev.includes("workgroup=1024")));
      assert.equal(formats("kawpow", "cpu1*8").length, 1);
    });

    test("PearlHash tuning candidates never scale m below its minimum", () => {
      /** @param {string} dev */
      const formats = (dev) => gpuTuning.autotuneCandidates(
        "pearlhash", gpuTuning.parseDeviceEntry(dev, "pearlhash")
      ).map(gpuTuning.formatDeviceEntry);
      assert.deepEqual(formats("gpu0*384"), ["gpu0*[m=384]", "gpu0*[m=192]"]);
      assert.deepEqual(formats("gpu0*128"), ["gpu0*[m=128]"]);
    });

    test("empirical tuner selects the fastest candidate after requiring a material baseline gain", async () => {
      const rates = new Map([
        ["gpu1*[intensity=1536]", 100],
        ["gpu1*[intensity=768]", 105],
        ["gpu1*[intensity=1152]", 106],
      ]);
      assert.equal(await tuneCnGpu(rates), "gpu1*[intensity=1152]");
    });

    test("empirical tuner keeps the heuristic across benchmark noise", async () => {
      const rates = new Map([
        ["gpu1*[intensity=1536]", 100],
        ["gpu1*[intensity=768]", 101],
        ["gpu1*[intensity=1152]", 101.9],
      ]);
      assert.equal(await tuneCnGpu(rates), "gpu1*[intensity=1536]");
    });

    test("Pearl tuning is applied independently to each native worker job", () => {
      const first = {
        algo: "pearlhash",
        dev: "gpu1",
        pearlhash_base_target: "1",
        pearlhash_n: 131072,
        pearlhash_k: 4096,
        pearlhash_rank: 256,
      };
      gpuTuning.applyNativeJobTuning(
        first, gpuTuning.parseDeviceEntry("gpu1*[m=8192;k=2048;rank=128]", "pearlhash"), "pearlhash");
      assert.deepEqual(first, {
        algo: "pearlhash",
        dev: "gpu1",
        intensity: 8192,
        pearlhash_base_target: "1",
        pearlhash_n: 8192,
        pearlhash_k: 2048,
        pearlhash_rank: 128,
        target: "0".repeat(59) + "80000",
      });
      const second = {
        algo: "pearlhash",
        dev: "gpu1",
        pearlhash_base_target: "1",
        pearlhash_n: 131072,
        pearlhash_k: 4096,
        pearlhash_rank: 256,
      };
      gpuTuning.applyNativeJobTuning(
        second,
        gpuTuning.parseDeviceEntry("gpu1*[m=16384;n=32768;rank=256]", "pearlhash"),
        "pearlhash",
      );
      assert.deepEqual(second, {
        algo: "pearlhash",
        dev: "gpu1",
        intensity: 16384,
        pearlhash_base_target: "1",
        pearlhash_n: 32768,
        pearlhash_k: 4096,
        pearlhash_rank: 256,
        target: "0".repeat(59) + "80000",
      });
    });

    test("Pearl tuning preserves V3 targets after worker handoff", () => {
    /** @returns {MiningJob} */
      const makeJob = () => ({
        algo: "pearlhash",
        dev: "gpu1",
        intensity: 131072,
        pearlhash_base_target: "1",
        pearlhash_n: 131072,
        pearlhash_k: 4096,
        pearlhash_rank: 256,
        pearlhash_cert_version: 3,
      });
      const v3Target = helper.pearlhashTarget("1", 4096, 256, 3);

      const v3 = makeJob();
      gpuTuning.applyNativeJobTuning(
        v3, gpuTuning.parseDeviceEntry("gpu1*[m=8192]", "pearlhash"), "pearlhash");
      assert.equal(v3.target, v3Target);

      for (const rank of [128, 256]) {
        const k = rank * 16;
        const overridden = makeJob();
        gpuTuning.applyNativeJobTuning(
          overridden,
          gpuTuning.parseDeviceEntry(`gpu1*[m=8192;k=${k};rank=${rank}]`, "pearlhash"),
          "pearlhash",
        );
        assert.equal(overridden.pearlhash_k, k);
        assert.equal(overridden.pearlhash_rank, rank);
        assert.equal(overridden.target, helper.pearlhashTarget("1", k, rank, 3));
      }

      const explicit = makeJob();
      explicit.target = "ab".repeat(32);
      delete explicit.pearlhash_base_target;
      gpuTuning.applyNativeJobTuning(
        explicit,
        gpuTuning.parseDeviceEntry("gpu1*[m=8192;k=4096;rank=256]", "pearlhash"),
        "pearlhash",
      );
      assert.equal(explicit.target, "ab".repeat(32));

      for (const certVersion of [1, 2, 4]) {
        const unsupported = makeJob();
        Reflect.set(unsupported, "pearlhash_cert_version", certVersion);
        assert.throws(() => gpuTuning.applyNativeJobTuning(
          unsupported, gpuTuning.parseDeviceEntry("gpu1*[m=8192]", "pearlhash"), "pearlhash"
        ), /certificate version/);
      }
    });

    test("thread selection preserves algorithm-specific *B shorthand until worker resolution", () => {
      assert.equal(helper.get_dev_threads("gpu1*128^2,gpu2*[workgroup=256]"), 3);
      assert.equal(helper.get_thread_dev(0, "gpu1*128^2,gpu2*[workgroup=256]"), "gpu1*128");
      assert.equal(helper.get_thread_dev(1, "gpu1*128^2,gpu2*[workgroup=256]"), "gpu1*128");
      assert.equal(helper.get_thread_dev(2, "gpu1*128^2,gpu2*[workgroup=256]"),
        "gpu2*[workgroup=256]");

      const c29 = gpuTuning.parseDeviceEntry(helper.get_thread_dev(0, "gpu1*128^2"), "c29");
      assert.deepEqual(c29.tuning, {seed_workgroup: 128});
      const job = {algo: "c29", dev: "gpu1"};
      gpuTuning.applyNativeJobTuning(job, c29, "c29");
      assert.deepEqual(job, {algo: "c29", dev: "gpu1", intensity: 1});
      assert.deepEqual(gpuTuning.tuningEnvironment("c29", c29.tuning),
        {MOM_C29_SEED_LOCAL_SIZE: "128"});
      assert.deepEqual(gpuTuning.parseDeviceEntry("gpu1*[intensity=1]", "c30").tuning,
        {intensity: 1});
      assert.deepEqual(gpuTuning.tuningEnvironment("beamhash3", {workgroup: 256}), {
        MOM_BEAMHASH3_WORKGROUP: "256",
        MOM_BEAMHASH3_COMPACT_WG: "256",
      });
    });

    test("portable Pearl tuning maps generic controls onto relevant vendor kernels", () => {
      assert.deepEqual(gpuTuning.tuningEnvironment("pearlhash", {
        cache_block: 32, tile: "4x2",
      }), {
        MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK: "32",
        MOM_PEARLHASH_CU_BLK: "32",
        MOM_PEARLHASH_AMD_DP4A_TILE: "4x2",
      });
    });
  });

  test.describe("Compiler selection, runtime isolation, kernel caching, and launcher policy", {concurrency: false}, () => {
    test("GPU compiler Markdown selects platform defaults and overrides", () => {
      assert.equal(policy.selection("etchash", "intel", "linux").key, "oneapi");
      assert.equal(policy.selection("fishhash", "intel", "linux").key, "oneapi");
      assert.equal(policy.selection("karlsenhashv2", "intel", "linux").key, "oneapi");
      assert.equal(policy.selection("zhash", "intel", "linux").key, "dpcpp");
      assert.equal(policy.selection("zhash", "intel", "linux").backend, "sycl");
      assert.equal(policy.selection("zhash", "intel", "win32").key, "oneapi");
      assert.equal(policy.selection("xelishashv3", "intel", "linux").backend, "sycl-native");
      assert.equal(policy.selection("xelishashv3", "intel", "win32").backend, "sycl-native");
      assert.equal(policy.selection("autolykos2", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("c30", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("beamhash3", "nvidia", "linux").key, "dpcpp");
      assert.equal(policy.selection("equihash192_7", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("fishhash", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("karlsenhashv2", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("zelhash", "nvidia", "linux").key, "dpcpp");
      assert.equal(policy.selection("zhash", "nvidia", "linux").key, "acpp-cuda");
      assert.equal(policy.selection("zhash", "nvidia", "linux").backend, "sycl");
      assert.equal(policy.selection("pearlhash", "nvidia", "linux").backend, "native");
      assert.equal(policy.selection("octopus", "nvidia", "linux").backend, "sycl-native");
      assert.equal(policy.selection("xelishashv3", "nvidia", "linux").backend, "sycl-native");
      assert.equal(policy.selection("nexapow", "nvidia", "linux").backend, "sycl-native");
      assert.equal(policy.selection("walahash", "nvidia", "linux").backend, "sycl-native");
      assert.deepEqual(policy.selection("pearlhash", "nvidia", "linux").pearlhashProfile,
        {m: 131072, n: 524288, k: 8192, rank: 128});
      assert.equal(policy.selection("autolykos2", "nvidia", "win32").key, "acpp-cuda");
      assert.equal(policy.selection("c30", "nvidia", "win32").key, "acpp-cuda");
      assert.equal(policy.selection("beamhash3", "nvidia", "win32").key, "dpcpp");
      assert.equal(policy.selection("cn/gpu", "nvidia", "win32").key, "dpcpp");
      assert.equal(policy.selection("cn/gpu", "nvidia", "win32").backend, "native");
      assert.equal(policy.selection("fishhash", "nvidia", "win32").key, "acpp-cuda");
      assert.equal(policy.selection("karlsenhashv2", "nvidia", "win32").key, "dpcpp");
      assert.equal(policy.selection("etchash", "nvidia", "win32").key, "dpcpp");
      assert.equal(policy.selection("pearlhash", "nvidia", "win32").backend, "native");
      assert.equal(policy.selection("octopus", "nvidia", "win32").backend, "sycl-native");
      assert.equal(policy.selection("walahash", "nvidia", "win32").backend, "sycl-native");
      assert.deepEqual(policy.selection("pearlhash", "nvidia", "win32").pearlhashProfile,
        {m: 131072, n: 524288, k: 8192, rank: 128});
      assert.equal(policy.selection("autolykos2", "amd", "linux").key, "acpp-hip");
      assert.equal(policy.selection("beamhash3", "amd", "linux").key, "acpp-hip");
      assert.equal(policy.selection("karlsenhashv2", "amd", "linux").key, "acpp-hip");
      assert.equal(policy.selection("pearlhash", "amd", "linux").key, "acpp-hip");
      assert.equal(policy.selection("pearlhash", "amd", "linux").backend, "native");
      assert.deepEqual(policy.selection("pearlhash", "amd", "linux").pearlhashProfile,
        {m: 131072, n: 131072, k: 2048, rank: 128});
      assert.equal(policy.selection("etchash", "amd", "linux").backend, "sycl");
      assert.equal(policy.selection("autolykos2", "amd", "linux").backend, "sycl-native");
      assert.deepEqual(policy.selection("pearlhash", "intel", "linux").pearlhashProfile,
        {m: 131072, n: 131072, k: 2048, rank: 128});
      assert.deepEqual(policy.selection("pearlhash", "intel", "win32").pearlhashProfile,
        {m: 131072, n: 131072, k: 2048, rank: 128});
      assert.equal(policy.selection("etchash", "amd", "win32").key, "acpp-hip");
      assert.equal(policy.selection("pearlhash", "amd", "win32").key, "acpp-hip");
      assert.equal(policy.selection("pearlhash", "amd", "win32").backend, "native");
      assert.deepEqual(policy.selection("pearlhash", "amd", "win32").pearlhashProfile,
        {m: 131072, n: 131072, k: 2048, rank: 128});
      assert.equal(policy.selection("cn/gpu", "intel", "linux").backend, "sycl-opencl");
      assert.equal(policy.selection("etchash", "intel", "linux").backend, "sycl");
      assert.equal(policy.selection("pearlhash", "intel", "linux").backend, "sycl-native");
      assert.equal(policy.selection("walahash", "intel", "linux").backend, "sycl-native");
      assert.equal(policy.selection("walahash", "amd", "linux").backend, "sycl-native");
      assert.equal(policy.selection("walahash", "amd", "win32").backend, "sycl-native");
      assert.equal(policy.selection("octopus", "intel", "linux").backend, "sycl-native");
      assert.equal(policy.selection("fishhash", "nvidia", "linux").backend, "sycl-native");
      assert.equal(policy.selection("c29", "nvidia", "linux").backend, "sycl");
      assert.equal(policy.selection("etchash", "opencl", "linux").backend, "sycl-opencl");
      assert.equal(policy.selection("etchash", "opencl", "linux").key, "dpcpp-opencl");
      assert.equal(policy.selection("etchash", "opencl", "win32").key, "dpcpp-opencl");
    });

    test("Windows Karlsen compiler policy selects DPC++ only for modern NVIDIA", () => {
      for (const sm of [80, 120]) {
        const selected = policy.selection("karlsenhashv2", "nvidia", "win32", sm);
        assert.equal(selected.key, "dpcpp", `SM${sm} compiler`);
        assert.equal(selected.backend, "sycl-native", `SM${sm} backend`);
      }
      for (const sm of [0, 61, 70, 75]) {
        assert.equal(policy.selection("karlsenhashv2", "nvidia", "win32", sm).key,
          "acpp-cuda", `unknown/legacy SM${sm} compiler`);
      }
      assert.equal(policy.selection("karlsenhashv2", "nvidia", "linux", 120).key, "acpp-cuda");
      assert.equal(policy.selection("fishhash", "nvidia", "win32", 120).key, "acpp-cuda");
      assert.equal(policy.selection("karlsenhashv2", "amd", "win32").key, "acpp-hip");
    });

    test("NVIDIA capability override selects compatibility paths", () => {
      assert.equal(policy.selection("pearlhash", "nvidia", "linux", 70).key, "acpp-cuda");
      assert.equal(policy.selection("pearlhash", "nvidia", "linux", 70).backend, "sycl-native");
      assert.equal(policy.selection("cn/gpu", "nvidia", "win32", 70).key, "acpp-cuda");
      assert.equal(policy.selection("cn/gpu", "nvidia", "win32", 70).backend, "sycl");
      assert.equal(policy.selection("etchash", "nvidia", "linux", 75).key, "acpp-cuda");
      assert.equal(policy.selection("etchash", "nvidia", "linux", 80).key, "dpcpp");
      assert.equal(policy.selection("octopus", "nvidia", "linux", 75).key, "acpp-cuda");
      assert.equal(policy.selection("octopus", "nvidia", "linux", 75).backend, "sycl");
      assert.equal(policy.selection("octopus", "nvidia", "linux", 80).key, "dpcpp");
      assert.equal(policy.selection("octopus", "nvidia", "linux", 80).backend, "sycl-native");
      assert.equal(policy.selection("xelishashv3", "nvidia", "linux", 75).backend, "sycl-native");
      assert.equal(policy.selection("xelishashv3", "nvidia", "linux", 80).backend, "sycl-native");
      assert.equal(policy.selection("nexapow", "nvidia", "linux", 70).backend, "sycl");
      assert.equal(policy.selection("nexapow", "nvidia", "linux", 75).backend, "sycl");
      assert.equal(policy.selection("nexapow", "nvidia", "linux", 80).backend, "sycl-native");
      assert.equal(policy.selection("walahash", "nvidia", "win32", 60).backend, "sycl");
      assert.equal(policy.selection("walahash", "nvidia", "win32", 61).backend, "sycl-native");
      assert.equal(policy.selection("walahash", "nvidia", "win32", 80).key, "dpcpp");
      assert.equal(policy.selection("walahash", "nvidia", "win32", 80).backend, "sycl-native");
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
        fs.mkdirSync(path.join(root, key));
        fs.writeFileSync(path.join(root, key, "mom.node"), "test");
      }
      const volta = policy.workerEnv("pearlhash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "7.0"
      }, "linux");
      assert.equal(volta.MOM_SYCL_COMPILER, "acpp-cuda");
      assert.equal(volta.ACPP_VISIBILITY_MASK, "cuda");
      const ampere = policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux");
      assert.equal(ampere.MOM_SYCL_COMPILER, "dpcpp");
      const turingNexa = policy.workerEnv("nexapow", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "7.5"
      }, "linux");
      assert.equal(turingNexa.MOM_SYCL_COMPILER, "acpp-cuda");
      assert.equal(turingNexa.MOM_NEXAPOW_NATIVE, undefined);
      const legacyNexa = policy.workerEnv("nexapow", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "7.0"
      }, "linux");
      assert.equal(legacyNexa.MOM_NEXAPOW_NATIVE, undefined);
      const ampereNexa = policy.workerEnv("nexapow", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux");
      assert.equal(ampereNexa.MOM_SYCL_COMPILER, "dpcpp");
      assert.equal(ampereNexa.MOM_NEXAPOW_NATIVE, undefined);
      const portableNexa = policy.workerEnv("nexapow", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux", "sycl");
      assert.equal(portableNexa.MOM_SYCL_COMPILER, "acpp-cuda");
      assert.equal(portableNexa.MOM_NEXAPOW_NATIVE, undefined);
      const syclNativeOctopus = policy.workerEnv("octopus", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux");
      assert.equal(syclNativeOctopus.MOM_OCTOPUS_SYCL_NATIVE, "1");
      assert.equal(policy.workerEnv("octopus", {
        MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
      }, "linux", "sycl-native").MOM_OCTOPUS_SYCL_NATIVE, "1");
      assert.equal(policy.workerEnv("octopus", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux", "sycl").MOM_OCTOPUS_SYCL_NATIVE, "0");
      const nativeXelis = policy.workerEnv("xelishashv3", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux");
      assert.equal(nativeXelis.MOM_XELISHASHV3_SYCL_NATIVE, "1");
      assert.equal(policy.workerEnv("xelishashv3", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux").MOM_XELISHASHV3_SYCL_NATIVE, "1");
      assert.equal(policy.workerEnv("xelishashv3", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "linux", "sycl").MOM_XELISHASHV3_SYCL_NATIVE, "0");
      const nativeWala = policy.workerEnv("walahash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux");
      assert.equal(nativeWala.MOM_WALAHASH_SYCL_NATIVE, "1");
      const portableWala = policy.workerEnv("walahash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux", "sycl");
      assert.equal(portableWala.MOM_SYCL_COMPILER, "dpcpp-opencl");
      assert.equal(portableWala.MOM_WALAHASH_SYCL_NATIVE, "0");
      assert.equal(policy.workerEnv("walahash", {
        MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
      }, "linux").MOM_WALAHASH_SYCL_NATIVE, "1");
      const nativeNvidiaWala = policy.workerEnv("walahash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "win32");
      assert.equal(nativeNvidiaWala.MOM_SYCL_COMPILER, "dpcpp");
      assert.equal(nativeNvidiaWala.MOM_WALAHASH_SYCL_NATIVE, "1");
      assert.equal(policy.workerEnv("walahash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0"
      }, "win32", "sycl").MOM_WALAHASH_SYCL_NATIVE, "0");
      fs.rmSync(root, {recursive: true, force: true});
    });

    test("Windows compiler addons live in isolated runtime directories", () => {
      assert.equal(policy.selection("etchash", "intel", "win32").addon, "oneapi/mom.node");
      assert.equal(policy.selection("etchash", "nvidia", "win32").addon, "dpcpp/mom.node");
      assert.equal(policy.selection("autolykos2", "nvidia", "win32").addon, "acpp-cuda/mom.node");
    });

    test("Linux compiler addons also live in isolated runtime directories", () => {
      assert.equal(policy.selection("etchash", "intel", "linux").addon, "oneapi/mom.node");
      assert.equal(policy.selection("etchash", "nvidia", "linux").addon, "dpcpp/mom.node");
      assert.equal(policy.selection("autolykos2", "amd", "linux").addon, "acpp-hip/mom.node");
      assert.equal(policy.selection("etchash", "opencl", "linux").addon, "dpcpp-opencl/mom.node");
      assert.equal(policy.selection("etchash", "opencl", "win32").addon, "dpcpp-opencl/mom.node");
    });

    test("Linux worker environment isolates the selected compiler runtime", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      fs.mkdirSync(path.join(root, "acpp-cuda"));
      fs.writeFileSync(path.join(root, "acpp-cuda", "mom.node"), "test");
      const env = policy.workerEnv("autolykos2", {
        MOM_GPU_BACKEND: "nvidia", MOM_GPU_INDEX: "2", MOM_NATIVE_DIR: root,
        MOM_NVIDIA_COMPUTE_CAPABILITY: "120",
        LD_LIBRARY_PATH: "/system/lib"
      }, "linux");
      assert.equal(env.MOM_SYCL_COMPILER, "acpp-cuda");
      assert.equal(env.ACPP_VISIBILITY_MASK, "cuda");
      assert.equal(env.CUDA_VISIBLE_DEVICES, "2");
      assert.equal(env["MOM_GPU_INDEX"], "0");
      assert.equal(env.MOM_NATIVE_PATH, path.join(root, "acpp-cuda", "mom.node"));
      assert.equal(env.MOM_RUNTIME_DIR, path.join(root, "acpp-cuda"));
      assert.equal(env.LD_LIBRARY_PATH, [path.join(root, "acpp-cuda"),
        path.join(root, "acpp-cuda", "hipSYCL"), "/system/lib"].join(path.delimiter));
      fs.rmSync(root, {recursive: true, force: true});
    });

    test("coinstalled ROCm versions cannot reproduce the v0.8.0 mixed-runtime crash", {
      skip: process.platform !== "linux",
    }, () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      try {
        fs.mkdirSync(path.join(root, "native", "acpp-hip"), {recursive: true});
        fs.writeFileSync(path.join(root, "native", "acpp-hip", "mom.node"), "test");

        const rocm = path.join(root, "rocm", "core-7.14");
        const rocmBin = path.join(rocm, "bin");
        const rocmLib = path.join(rocm, "lib");
        const distroRocmLib = path.join(root, "usr-lib-rocm-7.1");
        const pathBin = path.join(root, "path-bin");
        fs.mkdirSync(rocmBin, {recursive: true});
        fs.mkdirSync(rocmLib);
        fs.mkdirSync(distroRocmLib);
        fs.mkdirSync(pathBin);
        fs.writeFileSync(path.join(rocmBin, "hipconfig"), "#!/bin/sh\n", {mode: 0o755});
        fs.symlinkSync(path.join(rocmBin, "hipconfig"), path.join(pathBin, "hipconfig"));
        for (const library of [
          "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
          "libhiprtc.so.7", "libhiprtc-builtins.so.7",
        ]) {
          fs.writeFileSync(path.join(rocmLib, library), "test");
        }
        const amd = policy.workerEnv("kawpow", {
          MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: path.join(root, "native"),
          PATH: pathBin, LD_LIBRARY_PATH: distroRocmLib,
        }, "linux");
        assert.equal(amd.MOM_SYCL_COMPILER, "acpp-hip");
        assert.equal(amd.LD_LIBRARY_PATH, [path.join(root, "native", "acpp-hip"),
          path.join(root, "native", "acpp-hip", "hipSYCL"), rocmLib, distroRocmLib]
          .join(path.delimiter));

        const incompleteRocm = path.join(root, "incomplete-rocm");
        fs.mkdirSync(path.join(incompleteRocm, "lib"), {recursive: true});
        for (const library of [
          "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1", "libhiprtc.so.7",
        ]) {
          fs.writeFileSync(path.join(incompleteRocm, "lib", library), "test");
        }
        const fallback = policy.workerEnv("kawpow", {
          MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: path.join(root, "native"),
          ROCM_PATH: incompleteRocm, PATH: pathBin,
        }, "linux");
        assert.ok(fallback.LD_LIBRARY_PATH?.split(path.delimiter).includes(rocmLib));
        assert.ok(!fallback.LD_LIBRARY_PATH?.split(path.delimiter)
          .includes(path.join(incompleteRocm, "lib")));
      } finally {
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("vendor toolkit paths remain scoped to their matching Linux worker", {
      skip: process.platform !== "linux",
    }, () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      try {
        for (const key of ["acpp-cuda", "oneapi", "dpcpp-opencl"]) {
          fs.mkdirSync(path.join(root, "native", key), {recursive: true});
          fs.writeFileSync(path.join(root, "native", key, "mom.node"), "test");
        }
        const cuda = path.join(root, "cuda");
        fs.mkdirSync(path.join(cuda, "lib64"), {recursive: true});
        fs.writeFileSync(path.join(cuda, "lib64", "libnvrtc.so.12"), "test");
        const rocm = path.join(root, "rocm");
        fs.mkdirSync(path.join(rocm, "lib"), {recursive: true});
        for (const library of [
          "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
        ]) {
          fs.writeFileSync(path.join(rocm, "lib", library), "test");
        }
        const nvidia = policy.workerEnv("autolykos2", {
          MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: path.join(root, "native"), CUDA_PATH: cuda,
        }, "linux");
        assert.equal(nvidia.MOM_SYCL_COMPILER, "acpp-cuda");
        assert.ok(nvidia.LD_LIBRARY_PATH?.split(path.delimiter).includes(path.join(cuda, "lib64")));
        assert.ok(policy.workerEnv("autolykos2", {
          MOM_GPU_BACKEND: "nvidia", MOM_NATIVE_DIR: path.join(root, "native"), CUDA_HOME: cuda,
        }, "linux").LD_LIBRARY_PATH?.split(path.delimiter).includes(path.join(cuda, "lib64")));

        const intel = policy.workerEnv("etchash", {
          MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: path.join(root, "native"),
          ROCM_PATH: rocm, CUDA_PATH: cuda,
        }, "linux");
        const opencl = policy.workerEnv("etchash", {
          MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: path.join(root, "native"),
          ROCM_PATH: rocm, CUDA_PATH: cuda,
        }, "linux");
        assert.ok(!intel.LD_LIBRARY_PATH?.includes(path.join(rocm, "lib")));
        assert.ok(!intel.LD_LIBRARY_PATH?.includes(path.join(cuda, "lib64")));
        assert.ok(!opencl.LD_LIBRARY_PATH?.includes(path.join(rocm, "lib")));
        assert.ok(!opencl.LD_LIBRARY_PATH?.includes(path.join(cuda, "lib64")));
      } finally {
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("SYCL workers persist kernels and preserve cache controls", (t) => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-cache-"));
      const localBuild = path.resolve(__dirname, "..", "build", "win");
      const exists = fs.existsSync;
      let developmentBuild = false;
      t.mock.method(fs, "existsSync", (/** @type {import("node:fs").PathLike} */ file) =>
        String(file) === localBuild
          ? developmentBuild : exists(file));
      try {
        for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
          fs.mkdirSync(path.join(root, key));
          fs.writeFileSync(path.join(root, key, "mom.node"), "test");
        }
        /** @type {NodeJS.Platform[]} */
        const platforms = ["linux", "win32"];
        for (const platform of platforms) {
          for (const gpu of ["intel", "nvidia", "opencl"]) {
            const env = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: gpu,
              MOM_NVIDIA_COMPUTE_CAPABILITY: "100", LOCALAPPDATA: "C:\\Users\\test\\AppData\\Local"};
            const defaults = policy.workerEnv("etchash", env, platform);
            assert.equal(defaults["SYCL_CACHE_PERSISTENT"], "1");
            assert.equal(defaults["SYCL_CACHE_DIR"], platform === "win32" && gpu === "intel"
              ? path.join(env.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "etchash")
              : undefined);
            const custom = {...env, SYCL_CACHE_PERSISTENT: "0", SYCL_CACHE_DIR: "custom-cache"};
            const overridden = {...custom, ...policy.workerEnv("etchash", custom, platform)};
            assert.equal(overridden["SYCL_CACHE_PERSISTENT"], "0");
            assert.equal(overridden["SYCL_CACHE_DIR"], platform === "win32" && gpu === "intel"
              ? path.join("custom-cache", "oneapi", "etchash")
              : "custom-cache");
          }
        }
        const windowsIntel = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "intel",
          LOCALAPPDATA: "C:\\Users\\test\\AppData\\Local"};
        assert.equal(policy.workerEnv("walahash", windowsIntel, "win32")["SYCL_CACHE_DIR"],
          path.join(windowsIntel.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "walahash"));
        assert.equal(policy.workerEnv("xelishashv3", windowsIntel, "win32")["SYCL_CACHE_DIR"],
          path.join(windowsIntel.LOCALAPPDATA, "mom-sycl-cache", "oneapi", "xelishashv3"));
        developmentBuild = true;
        const env = {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "intel"};
        assert.equal(policy.workerEnv("nexapow", env, "win32")["SYCL_CACHE_DIR"],
          path.join(localBuild, ".sycl-cache", "oneapi", "nexapow"));
        const custom = {...env, SYCL_CACHE_PERSISTENT: "0", SYCL_CACHE_DIR: "retained-cache"};
        const overridden = {...custom, ...policy.workerEnv("nexapow", custom, "win32")};
        assert.equal(overridden["SYCL_CACHE_PERSISTENT"], "0");
        assert.equal(overridden["SYCL_CACHE_DIR"],
          path.join("retained-cache", "oneapi", "nexapow"));
        const amd = policy.workerEnv("etchash", {...env, MOM_GPU_BACKEND: "amd"}, "win32");
        assert.equal(amd["SYCL_CACHE_PERSISTENT"], undefined);

      } finally {
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("Windows worker environment puts only the selected compiler runtime first", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      fs.mkdirSync(path.join(root, "acpp-hip"));
      fs.writeFileSync(path.join(root, "acpp-hip", "mom.node"), "test");
      const env = policy.workerEnv("pearlhash", {
        MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "3", MOM_NATIVE_DIR: root,
        Path: "C:\\Windows\\System32", ROCM_PATH: "C:\\ROCm", CUDA_PATH: "C:\\CUDA",
      }, "win32");
      assert.equal(env.MOM_SYCL_COMPILER, "acpp-hip");
      assert.equal(env.ACPP_VISIBILITY_MASK, "hip");
      assert.equal(env.HIP_VISIBLE_DEVICES, "3");
      assert.equal(env["MOM_GPU_INDEX"], "0");
      assert.equal(env.MOM_RUNTIME_DIR, path.join(root, "acpp-hip"));
      assert.equal(env["Path"], [path.join(root, "acpp-hip"),
        path.join(root, "acpp-hip", "hipSYCL"), "C:\\Windows\\System32"].join(path.delimiter));
      assert.equal(Object.hasOwn(env, "PATH"), false);
      fs.rmSync(root, {recursive: true, force: true});
    });

    test("explicit native addon path overrides compiler policy", () => {
      assert.deepEqual(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "amd",
        MOM_NATIVE_PATH: "C:\\custom\\mom.node",
      }, "win32"), {});
    });

    test("Windows portable workers share oneAPI JIT without crossing into nightly runtimes", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-runtime-cohort-"));
      try {
        for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
          fs.mkdirSync(path.join(root, key));
          fs.writeFileSync(path.join(root, key, "mom.node"), "fixture");
        }
        fs.writeFileSync(path.join(root, "oneapi", "sycl-jit.dll"), "matching-oneapi");
        fs.writeFileSync(path.join(root, "dpcpp", "sycl-jit.dll"), "poisoned-nightly");
        const inheritedPath = path.join(root, "dpcpp");
        const base = {MOM_NATIVE_DIR: root, Path: inheritedPath, UR_ADAPTERS_FORCE_LOAD: "caller-adapter",
          OCL_ICD_FILENAMES: "caller-icd", SYCL_CACHE_PERSISTENT: "0"};
        const portable = {...base, ...policy.workerEnv("etchash", {
          ...base, MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu",
        }, "win32")};
        const portablePath = portable["Path"];
        assert.ok(portablePath);
        const jit = portablePath.split(path.delimiter).map(dir => path.join(dir, "sycl-jit.dll"))
          .find(fs.existsSync);
        assert.equal(jit, path.join(root, "oneapi", "sycl-jit.dll"));
        assert.ok(jit);
        assert.equal(fs.readFileSync(jit, "utf8"), "matching-oneapi");
        assert.equal(portablePath.split(path.delimiter).at(-1), inheritedPath);
        assert.equal(portable.UR_ADAPTERS_FORCE_LOAD, base.UR_ADAPTERS_FORCE_LOAD);
        assert.equal(portable.OCL_ICD_FILENAMES, base.OCL_ICD_FILENAMES);
        assert.equal(portable.SYCL_CACHE_PERSISTENT, "0");
        assert.deepEqual(policy.workerEnv("etchash", {...base, MOM_GPU_BACKEND: "opencl",
          MOM_NATIVE_PATH: "caller-addon"}, "win32"), {});
        /** @type {Array<[string, string]>} */
        const vendors = [["nvidia", "dpcpp"], ["amd", "acpp-hip"]];
        for (const [gpu, key] of vendors) {
          const vendor = policy.workerEnv("etchash", {...base, MOM_GPU_BACKEND: gpu,
            MOM_NVIDIA_COMPUTE_CAPABILITY: "90", MOM_GPU_INDEX: "2"}, "win32");
          assert.equal(vendor.MOM_RUNTIME_DIR, path.join(root, key));
          assert.equal(vendor["Path"], [path.join(root, key), path.join(root, key, "hipSYCL"),
            inheritedPath].join(path.delimiter));
        }
        const linux = policy.workerEnv("etchash", {MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: "opencl",
          LD_LIBRARY_PATH: "caller-linux"}, "linux");
        assert.equal(linux.LD_LIBRARY_PATH, [path.join(root, "dpcpp-opencl"),
          path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "dpcpp"), "caller-linux"]
          .join(path.delimiter));
      } finally {
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("OpenCL helper selects the portable addon despite an inherited native override", (
    /** @type {import("node:test").TestContext} */ t,
    ) => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-opencl-policy-"));
      const candidate = path.join(root, "acpp-cuda", "mom.node");
      const portable = path.join(root, "dpcpp-opencl", "mom.node");
      fs.mkdirSync(path.dirname(candidate), {recursive: true});
      fs.mkdirSync(path.dirname(portable), {recursive: true});
      fs.writeFileSync(candidate, "candidate");
      fs.writeFileSync(portable, "portable");
      const previousNativeDir = process.env["MOM_NATIVE_DIR"];
      const previousNativePath = process.env["MOM_NATIVE_PATH"];
      process.env["MOM_NATIVE_DIR"] = root;
      process.env["MOM_NATIVE_PATH"] = candidate;
      try {
        const env = openclSyclEnv("cpu");
        assert.equal(env["MOM_SYCL_COMPILER"], "dpcpp-opencl");
        assert.equal(env["MOM_NATIVE_PATH"], portable);
        assert.notEqual(env["MOM_NATIVE_PATH"], candidate);
        assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "opencl:cpu");

        fs.rmSync(path.dirname(portable), {recursive: true});
        t.mock.method(fs, "existsSync", () => false);
        assert.throws(() => openclSyclEnv("cpu"), /Missing dpcpp-opencl\/mom\.node/);
      } finally {
        if (previousNativeDir === undefined) {
          delete process.env["MOM_NATIVE_DIR"];
        } else {
          process.env["MOM_NATIVE_DIR"] = previousNativeDir;
        }
        if (previousNativePath === undefined) {
          delete process.env["MOM_NATIVE_PATH"];
        } else {
          process.env["MOM_NATIVE_PATH"] = previousNativePath;
        }
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("launcher default native path does not disable per-algorithm policy", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      fs.mkdirSync(path.join(root, "acpp-cuda"));
      fs.writeFileSync(path.join(root, "acpp-cuda", "mom.node"), "test");
      const env = policy.workerEnv("autolykos2", {
        MOM_GPU_BACKEND: "nvidia",
        MOM_NVIDIA_COMPUTE_CAPABILITY: "120",
        MOM_NATIVE_DIR: root,
        MOM_NATIVE_PATH: path.join(root, "dpcpp", "mom.node"),
        MOM_NATIVE_PATH_LAUNCHER_DEFAULT: path.join(root, "dpcpp", "mom.node"),
      }, "linux");
      assert.equal(env.MOM_SYCL_COMPILER, "acpp-cuda");
      assert.equal(env.MOM_NATIVE_PATH, path.join(root, "acpp-cuda", "mom.node"));
      fs.rmSync(root, {recursive: true, force: true});
    });

    test("compiler workers select the backend matching their artifact", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-policy-"));
      for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-hip"]) {
        fs.mkdirSync(path.join(root, key));
        fs.writeFileSync(path.join(root, key, "mom.node"), "test");
      }
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0", MOM_NATIVE_DIR: root
      }, "linux")
        .ONEAPI_DEVICE_SELECTOR, "cuda:gpu");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "8.0",
        MOM_GPU_INDEX: "2", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "cuda:0");
      assert.equal(policy.workerEnv("etchash", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root}, "linux")
        .ACPP_VISIBILITY_MASK, "hip");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "1", MOM_NATIVE_DIR: root
      }, "linux").HIP_VISIBLE_DEVICES, "1");
      assert.equal(policy.workerEnv("autolykos2", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root}, "linux")
        .ACPP_VISIBILITY_MASK, "hip");
      const intelOneapi = policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "4", MOM_NATIVE_DIR: root
      }, "linux");
      assert.equal(intelOneapi.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(intelOneapi.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(policy.workerEnv("cn/gpu", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
      assert.equal(policy.workerEnv("__control__", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(policy.workerEnv("cn/gpu", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux", "sycl-l0").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      const intelKarlsen = policy.workerEnv("karlsenhashv2", {
        MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "4", MOM_NATIVE_DIR: root
      }, "linux");
      assert.equal(intelKarlsen.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(intelKarlsen.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS: "0"
      }, "linux").UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "0");
      const windowsIntel = policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, Path: "C:\\Windows\\System32"
      }, "win32");
      assert.equal(windowsIntel.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(windowsIntel.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, "1");
      assert.equal(policy.workerEnv("__control__", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root, Path: "C:\\Windows\\System32"
      }, "win32").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      const windowsPortable = policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu", MOM_NATIVE_DIR: root,
        Path: "C:\\Windows\\System32"
      }, "win32");
      assert.equal(windowsPortable.ONEAPI_DEVICE_SELECTOR, "opencl:cpu");
      assert.equal(windowsPortable["Path"], [path.join(root, "dpcpp-opencl"),
        path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "oneapi"),
        "C:\\Windows\\System32"].join(path.delimiter));
      assert.equal(Object.hasOwn(windowsPortable, "PATH"), false);
      const opencl = policy.workerEnv("etchash", {MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: root}, "linux");
      assert.equal(opencl.MOM_SYCL_COMPILER, "dpcpp-opencl");
      assert.equal(opencl.ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
      assert.equal(opencl.LD_LIBRARY_PATH, [path.join(root, "dpcpp-opencl"),
        path.join(root, "dpcpp-opencl", "hipSYCL"), path.join(root, "dpcpp")].join(path.delimiter));
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "opencl", MOM_GPU_INDEX: "6", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "cpu", MOM_NATIVE_DIR: root
      }, "linux").ONEAPI_DEVICE_SELECTOR, "opencl:cpu");
      const portableIntel = policy.workerEnv("pearlhash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux", "sycl-l0");
      assert.equal(portableIntel.MOM_SYCL_COMPILER, "dpcpp-opencl");
      assert.equal(portableIntel.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      const portableIntelOpencl = policy.workerEnv("pearlhash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux", "sycl-opencl");
      assert.equal(portableIntelOpencl.MOM_SYCL_COMPILER, "dpcpp-opencl");
      assert.equal(portableIntelOpencl.ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
      assert.equal(policy.workerEnv("pearlhash", {
        MOM_GPU_BACKEND: "intel", MOM_NATIVE_DIR: root
      }, "linux", "sycl").ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
      assert.equal(policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "opencl", MOM_NATIVE_DIR: root
      }, "linux", "sycl-opencl").ONEAPI_DEVICE_SELECTOR, "opencl:gpu");
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "amd", MOM_NATIVE_DIR: root
      }, "linux", "sycl-l0"), /incompatible/);
      assert.throws(() => policy.validateBackend("unknown"), /Invalid GPU backend/);
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "opencl", MOM_OPENCL_DEVICE_TYPE: "accelerator", MOM_NATIVE_DIR: root
      }, "linux"), /Invalid MOM_OPENCL_DEVICE_TYPE/);
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "not-a-number", MOM_NATIVE_DIR: root
      }, "linux"), /Invalid MOM_GPU_INDEX/);
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "intel", MOM_GPU_INDEX: "1024", MOM_NATIVE_DIR: root
      }, "linux"), /Invalid MOM_GPU_INDEX/);
      assert.throws(() => policy.workerEnv("etchash", {
        MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "9".repeat(400),
        MOM_NATIVE_DIR: root
      }, "linux"), /Invalid MOM_NVIDIA_COMPUTE_CAPABILITY/);
      fs.rmSync(root, {recursive: true, force: true});
    });
  });

  test.describe("Intel discrete and integrated compiler policy", {concurrency: false}, () => {
    /** @param {import("node:test").TestContext} t */
    function portableFixture(t) {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-igpu-policy-"));
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      fs.mkdirSync(path.join(root, "dpcpp-opencl"));
      fs.writeFileSync(path.join(root, "dpcpp-opencl", "mom.node"), "test");
      return {MOM_GPU_BACKEND: "intel_igpu", MOM_NATIVE_DIR: root};
    }

    test("Intel cohorts are exact case-insensitive environment values", () => {
      assert.equal(policy.gpuFromEnv({MOM_GPU_BACKEND: "InTeL_iGpU"}), "intel_igpu");
      assert.equal(policy.gpuFromEnv({MOM_GPU_BACKEND: "INTEL"}), "intel");
      for (const value of ["intel_igpu-extra", "intel-extra", "igpu", "intel_igpu "]) {
        assert.equal(policy.gpuFromEnv({MOM_GPU_BACKEND: value}), "");
        assert.equal(policyModule.selection("etchash", value, "linux"), null);
      }
    });

    test("integrated jobs resolve auto to portable SYCL without discrete Pearl tuning", () => {
      const api = testJobApi({MOM_GPU_BACKEND: "INTEL_IGPU"});
      assert.equal(api.jobBackend("pearlhash"), "sycl");
      assert.equal(api.jobBackend("walahash"), "sycl");
      assert.equal(policy.selection("pearlhash", "intel_igpu").pearlhashProfile, null);
    });

    test("integrated discovery and clones replace an explicitly pinned discrete worker", async (t) => {
      const fixture = portableFixture(t);
      const pinned = path.join(fixture.MOM_NATIVE_DIR, "oneapi", "mom.node");
      const previous = {...process.env};
      Object.assign(process.env, fixture, {MOM_GPU_BACKEND: "intel", MOM_NATIVE_PATH: pinned});
      t.after(() => {
        for (const key of Object.keys(process.env)) {
          if (!Object.hasOwn(previous, key)) {delete process.env[key];}
        }
        Object.assign(process.env, previous);
      });
      const minerCommand = require("./common/miner-command");
      let probes = 0;
      t.mock.method(minerCommand, "getGpuDevices", async (
        /** @type {string} */ vendor,
        /** @type {Parameters<typeof minerCommand.getGpuDevices>[1]} */ options = {},
      ) => {
        ++probes;
        assert.equal(vendor, "intel_igpu");
        assert.equal(options.integrated, true);
        const key = JSON.parse(minerCommand.gpuDeviceDiscoveryKey(vendor, options));
        const env = Object.fromEntries(JSON.parse(key[0]));
        assert.equal(env["MOM_NATIVE_PATH"], path.join(fixture.MOM_NATIVE_DIR, "dpcpp-opencl", "mom.node"));
        assert.equal(env["MOM_GPU_BACKEND"], "intel_igpu");
        assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu");
        return {skipped: false, devices: [], params: {}};
      });
      // Reload only the test helper so its destructured probe uses the stub, while its actual
      // discovery/environment policy remains intact and cannot launch a native miner.
      const matrixPath = require.resolve("./common/gpu-test-matrix");
      const cachedMatrix = require.cache[matrixPath];
      delete require.cache[matrixPath];
      t.after(() => {
        if (cachedMatrix) {
          require.cache[matrixPath] = cachedMatrix;
        } else {
          delete require.cache[matrixPath];
        }
      });
      const discovery = require("./common/gpu-test-matrix").sharedDiscovery();
      await discovery.intelIntegrated("walahash", "sycl");
      assert.equal(probes, 1);
      const vector = hashTests.find(({job}) => job.algo === "walahash");
      assert.ok(vector);
      const clone = cloneForIntelIntegrated(vector, "gpu2", "sycl");
      const base = {...process.env, ...clone.env};
      const env = policy.workerEnv("walahash", base);
      assert.equal(env.MOM_NATIVE_PATH, path.join(fixture.MOM_NATIVE_DIR, "dpcpp-opencl", "mom.node"));
      assert.equal(env.MOM_SYCL_COMPILER, "dpcpp-opencl");
      assert.equal(env.MOM_WALAHASH_SYCL_NATIVE, "0");
    });

    for (const platform of /** @type {const} */ (["linux", "win32"])) {
      test(`${platform} integrated algorithms and control use standards-only policy`, () => {
        for (const algo of ["__control__", "cn/gpu", "pearlhash", "walahash", "octopus",
          "xelishashv3", "kawpow", "zhash"]) {
          assert.deepEqual(policy.selection(algo, "INTEL_IGPU", platform), {
            key: "dpcpp-opencl", addon: "dpcpp-opencl/mom.node", backend: "sycl",
            pearlhashProfile: null,
          });
        }
        assert.equal(policy.selection("walahash", "intel", platform).backend, "sycl-native");
        assert.equal(policy.selection("__control__", "intel", platform).key, "oneapi");
      });

      test(`${platform} integrated control selects Level Zero without remapping native cohort`, (t) => {
        const base = {...portableFixture(t), MOM_GPU_BACKEND: "InTeL_iGpU", MOM_GPU_INDEX: "2"};
        const env = {...base, ...policy.workerEnv("__control__", base, platform)};
        assert.equal(env.MOM_SYCL_COMPILER, "dpcpp-opencl");
        assert.equal(env.MOM_NATIVE_PATH, path.join(base.MOM_NATIVE_DIR, "dpcpp-opencl", "mom.node"));
        assert.equal(env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
        assert.equal(env.MOM_GPU_BACKEND, "InTeL_iGpU");
        assert.equal(env.MOM_GPU_INDEX, "2");
        assert.equal(env.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, undefined);
        const worker = policy.workerEnv("walahash", {
          ...env, MOM_NATIVE_PATH_LAUNCHER_DEFAULT: env.MOM_NATIVE_PATH,
        }, platform);
        assert.equal(worker.MOM_NATIVE_PATH, env.MOM_NATIVE_PATH);
        assert.equal(worker.MOM_SYCL_COMPILER, "dpcpp-opencl");
        assert.equal(worker.MOM_WALAHASH_SYCL_NATIVE, "0");
      });

      test(`${platform} integrated workers preserve explicit portable transports`, (t) => {
        const base = portableFixture(t);
        for (const backend of ["sycl", "sycl-l0", "sycl-opencl"]) {
          const env = policy.workerEnv("pearlhash", base, platform, backend);
          assert.equal(env.MOM_SYCL_COMPILER, "dpcpp-opencl");
          assert.equal(env.ONEAPI_DEVICE_SELECTOR,
            backend === "sycl-opencl" ? "opencl:gpu" : "level_zero:gpu");
        }
      });

      test(`${platform} integrated workers disable inherited tuned kernel switches`, (t) => {
        const base = portableFixture(t);
        for (const [algo, variable] of [["octopus", "MOM_OCTOPUS_SYCL_NATIVE"],
          ["walahash", "MOM_WALAHASH_SYCL_NATIVE"],
          ["xelishashv3", "MOM_XELISHASHV3_SYCL_NATIVE"]]) {
          assert.ok(algo && variable);
          /** @type {NodeJS.ProcessEnv} */
          const env = {...base, [variable]: "1", ...policy.workerEnv(algo, base, platform)};
          assert.equal(env[variable], "0");
        }
      });

      test(`${platform} integrated workers keep compiler runtime and cache isolation`, (t) => {
        const base = {...portableFixture(t), Path: "system-path", LD_LIBRARY_PATH: "system-libs",
          SYCL_CACHE_DIR: "cache-root"};
        const env = policy.workerEnv("walahash", base, platform);
        const directory = path.join(base.MOM_NATIVE_DIR, "dpcpp-opencl");
        const runtime = [directory, path.join(directory, "hipSYCL"),
          path.join(base.MOM_NATIVE_DIR, platform === "win32" ? "oneapi" : "dpcpp")];
        assert.equal(env["SYCL_CACHE_PERSISTENT"], "1");
        if (platform === "win32") {
          assert.equal(env["Path"], [...runtime, "system-path"].join(path.delimiter));
          assert.equal(Object.hasOwn(env, "PATH"), false);
          assert.equal(env["SYCL_CACHE_DIR"], path.join("cache-root", "dpcpp-opencl", "walahash"));
          assert.notEqual(env["SYCL_CACHE_DIR"],
            policy.workerEnv("xelishashv3", base, platform)["SYCL_CACHE_DIR"]);
        } else {
          assert.equal(env.LD_LIBRARY_PATH, [...runtime, "system-libs"].join(path.delimiter));
          assert.equal(env["SYCL_CACHE_DIR"], undefined);
        }
      });

      test(`${platform} integrated policy never falls back to a missing portable artifact`, (t) => {
        t.mock.method(fs, "existsSync", () => false);
        assert.throws(() => policy.workerEnv("__control__", {
          MOM_GPU_BACKEND: "intel_igpu", MOM_COMPILER_POLICY_STRICT: "1",
        }, platform), /Missing dpcpp-opencl\/mom.node for intel_igpu\/__control__/);
        assert.throws(() => policy.workerEnv("walahash", {
          MOM_GPU_BACKEND: "intel_igpu",
        }, platform), /Missing dpcpp-opencl\/mom.node for intel_igpu\/walahash/);
        assert.throws(() => policy.workerEnv("etchash", {
          MOM_GPU_BACKEND: "intel_igpu", MOM_GPU_INDEX: "1024",
        }, platform), /Invalid MOM_GPU_INDEX/);
      });
    }
  });

  test.describe("Submitted faults, upload ownership, and cache readiness", {concurrency: false}, () => {
    test("PearlHash compact inputs preserve legacy bytes and immutable cached proofs", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const matrix = fs.readFileSync(path.join(root, "matrix.inc"), "utf8");
      const host = fs.readFileSync(path.join(root, "pearlhash.cpp"), "utf8");
      const proofHost = fs.readFileSync(path.join(root, "host.inc"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-inputs-"));
      const generator = extractCppBlock(matrix, "static inline int8_t gv(");
      const assignment = "  b.seed_prefix_inputs = compact_mode && b.positive_inputs;\n";
      assert.equal(host.split(assignment).length, 2);
      try {
        fs.writeFileSync(path.join(fixture, "bounds.inc"),
          extractCppBlock(matrix, "static size_t checked_elements(") +
          extractCppBlock(matrix, "static size_t checked_chunk_count("));
        const run = (/** @type {string} */ body, /** @type {string} */ select,
          /** @type {RegExp | undefined} */ failure = undefined,
          /** @type {string} */ proofBody = proofHost) => {
          fs.writeFileSync(path.join(fixture, "gv.inc"), body);
          fs.writeFileSync(path.join(fixture, "select.inc"), select);
          fs.writeFileSync(path.join(fixture, "host.inc"), proofBody);
          const executable = path.join(fixture, "test");
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-Wno-unused-function", "-I" + fixture, path.join(__dirname, "native/pearlhash-input-modes.cpp"),
            "-o", executable], {encoding: "utf8", timeout: 30000});
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 30000,
            env: {...process.env, MOM_TEST_EXPORT_PROOFS: ""}});
          assert.equal(result.error, undefined); assert.equal(result.signal, null);
          assert.equal(result.status, failure ? 1 : 0, result.stdout + result.stderr);
          if (failure) {assert.match(result.stderr, failure);} else {assert.match(result.stdout, /PEARL_INPUT_MODES_PASS/u);}
        };
        await t.test("actual generator and proof cache", () => run(generator, assignment));
        const leaked = generator.replace("positive_inputs && seed_prefix_inputs", "positive_inputs && (seed_prefix_inputs || true)");
        assert.notEqual(leaked, generator);
        await t.test("prefix cannot replace legacy input streams", () =>
          run(leaked, assignment, /legacy generator changed/u));
        const unguarded = generator.replace("positive_inputs && seed_prefix_inputs", "positive_inputs || seed_prefix_inputs");
        assert.notEqual(unguarded, generator);
        await t.test("prefix cannot escape the positive-input gate", () =>
          run(unguarded, assignment, /prefix escaped positive gate/u));
        const stale = assignment.replace("compact_mode && b.positive_inputs", "compact_mode ? b.positive_inputs : b.seed_prefix_inputs");
        assert.notEqual(stale, assignment);
        await t.test("fallback must clear a stale prefix mode", () =>
          run(generator, stale, /stale seed-prefix mode/u));
        const bound = "(uint64_t(m) + uint64_t(n)) * uint64_t(k) <= (uint64_t{1} << 32)";
        assert.equal(proofHost.split(bound).length, 2);
        await t.test("constant leaf fill must retain wrapped raw indices", () =>
          run(generator, assignment, /compact leaf bytes/u, proofHost.replace(bound, "true")));
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash positive-input gate admits bounded native routes and pair mode rejects signed bytes", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const production = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/pearlhash.cpp"), "utf8");
      const gate = extractCppBlock(production, "static bool pearlhash_positive_inputs(");
      const pairGuard = production.match(/if \(pair_dist8 && !b\.positive_inputs\)\s*throw [^;]+;/u)?.[0];
      assert.ok(pairGuard, "unsigned CUDA MMA must reject incompatible producer bytes");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearlhash-positive-gate-"));
      try {
        const source = [
          "#include <cstdint>",
          "#include <cstdio>",
          "#include <limits>",
          "#include <string>",
          "namespace sycl {",
          "class device {",
          " public:",
          "  device(std::uint32_t metadata, bool gpu) : metadata_(metadata), gpu_(gpu) {}",
          "  bool is_gpu() const { return gpu_; }",
          " private:",
          "  std::uint32_t metadata_;",
          "  bool gpu_;",
          "};",
          "}",
          gate,
          "static bool rejects_pair(bool pair_dist8, bool positive) {",
          "  struct { bool positive_inputs; } b{positive};",
          "  try {", pairGuard, "  } catch (const std::string&) { return true; }",
          "  return false;",
          "}",
          "struct Case { const char* name; std::uint32_t vendor; bool gpu; bool native; int k; bool expected; };",
          "int main() {",
          "  const Case cases[] = {",
          "    {\"AMD AdaptiveCpp vendor metadata 0x1022\", 0x1022u, true, true, 2048, true},",
          "    {\"AMD AdaptiveCpp vendor metadata decimal 1022\", 1022u, true, true, 2048, true},",
          "    {\"NVIDIA native\", 0x10deu, true, true, 8192, true},",
          "    {\"F16/portable native=false\", 0x10deu, true, false, 8192, false},",
          "    {\"CPU\", 0x1002u, false, true, 2048, false},",
          "    {\"Intel route not admitted\", 0x8086u, true, false, 2048, false},",
          "    {\"unknown trusted native route\", 0x1u, true, true, 2048, true},",
          "    {\"route not admitted\", 0x10deu, true, false, 2048, false},",
          "    {\"zero K\", 0x10deu, true, true, 0, false},",
          "    {\"aligned signed-bound control\", 0x10deu, true, true, 133120, true},",
          "    {\"exact signed-bound edge\", 0x10deu, true, true, 133144, true},",
          "    {\"exact signed-bound overflow\", 0x10deu, true, true, 133145, false},",
          "    {\"aligned signed-bound overflow control\", 0x10deu, true, true, 133184, false},",
          "    {\"negative K\", 0x10deu, true, true, -1, false},",
          "  };",
          "  for (const Case& c : cases) {",
          "    const sycl::device device(c.vendor, c.gpu);",
          "    if (pearlhash_positive_inputs(device, c.native, c.k) != c.expected) {",
          "      std::fprintf(stderr, \"gate mismatch: %s\\n\", c.name);",
          "      return 1;",
          "    }",
          "  }",
          "  for (const bool pair : {false, true})",
          "    for (const bool positive : {false, true})",
          "      if (rejects_pair(pair, positive) != (pair && !positive)) return 1;",
          "  std::puts(\"PASS actual-source PearlHash positive gate\");",
          "}",
        ].join("\n");
        const sourceFile = path.join(fixture, "positive_gate.cpp");
        const executable = path.join(fixture, "positive_gate");
        fs.writeFileSync(sourceFile, source);
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", sourceFile, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined, compiled.error?.message);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined, result.error?.message);
        assert.equal(result.signal, null, result.stderr);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.equal(result.stdout, "PASS actual-source PearlHash positive gate\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash HIP shape admission supports bounded row slices", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/hip-jit.inc"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-hip-shape-"));
      try {
        fs.writeFileSync(path.join(fixture, "hip-shape.inc"),
          extractCppBlock(source, "  static bool supports_shape(const int"));
        const executable = path.join(fixture, "hip-shape");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, path.join(__dirname, "native/pearlhash-hip-shape.cpp"),
          "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.match(result.stdout, /supports bounded row slices/);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash CUDA cache swizzles cover each global CTA exactly once", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/cuda-kernel.inc"), "utf8");
      const starts = [...source.matchAll(/^[ \t]*int block_m = int\(blockIdx\.x\);$/gmu)]
        .map(({index}) => index);
      assert.equal(starts.length, 2, "both staged and packed CUDA branches must retain swizzle");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-cuda-swizzle-"));
      try {
        for (const [branch, start] of starts.entries()) {
          const ctaEnd = source.indexOf("auto cta_coord = make_coord(block_m, block_n, _);", start);
          const warpEnd = source.indexOf("const int warp = int(threadIdx.x) >> 5;", start);
          const end = ctaEnd > start ? ctaEnd : warpEnd;
          assert.ok(end > start, `swizzle branch ${branch} is incomplete`);
          fs.writeFileSync(path.join(fixture, "swizzle.inc"), source.slice(start, end));
          const executable = path.join(fixture, `swizzle-${branch}`);
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", "-I" + fixture, path.join(__dirname, "native/pearlhash-cuda-swizzle.cpp"),
            "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8"});
          assert.equal(result.status, 0, result.stdout + result.stderr);
          assert.match(result.stdout, /CUDA swizzle coordinate checks passed/u);
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash CUDA vector128 producers are CUTE-bijective past 4 GiB", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const matrix = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/matrix.inc"), "utf8");
      const helper = extractCppBlock(matrix, "static inline size_t nvidia_vector128_offset(");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-cuda-offset-"));
      const executable = path.join(fixture, "offset");
      const source = path.join(__dirname, "native/pearlhash-cuda-offset.cpp");
      const compile = () => spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-I" + fixture, source, "-o", executable], {encoding: "utf8"});
      try {
        fs.writeFileSync(path.join(fixture, "offset.inc"), helper);
        let compiled = compile();
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        let result = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout, /SCATTER_GATHER is_a=1 values=2097152 PASS/u);
        assert.match(result.stdout, /SCATTER_GATHER is_a=0 values=2097152 PASS/u);
        assert.match(result.stdout, /BOUNDARIES is_a=1 .*2G=0x80000000 4G=0x100000000 PASS/u);
        assert.match(result.stdout, /BOUNDARIES is_a=0 .*2G=0x80000000 4G=0x100000000 PASS/u);
        assert.match(result.stdout, /CAST32_NEGATIVE_CONTROL rejected=PASS/u);
        assert.match(result.stdout, /ROLLING_PREFETCH_SEQUENCE packets=256 B_end=0xffffffff PASS/u);
        assert.match(result.stdout, /PRODUCER_OFFSET_TEST PASS/u);

        // Negative control: a 32-bit truncation of the actual offset expression must be caught.
        assert.equal(helper.match(/\breturn\b/gu)?.length, 1, "offset helper must have one return");
        const truncated = helper.replace(/\breturn ([^;]+);/u, "return static_cast<uint32_t>($1);");
        assert.notEqual(truncated, helper);
        fs.writeFileSync(path.join(fixture, "offset.inc"), truncated);
        compiled = compile();
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        result = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.notEqual(result.status, 0, "32-bit offset mutation unexpectedly passed");
        assert.match(result.stderr, /exact 4 GiB producer offset changed/u);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash CUDA rank-128 store preserves the dense output ABI", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const kernel = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/cuda-kernel.inc"), "utf8");
      // The dense store runs from its index declaration to the end of the enclosing kernel.
      const storeStart = kernel.indexOf("const uint64_t dense_proof_index =");
      const storeEnd = kernel.indexOf("\n}\n", storeStart);
      assert.ok(storeStart >= 0 && storeEnd > storeStart,
        "the actual rank-128 dense store body must remain extractable");
      const store = kernel.slice(storeStart, storeEnd);

      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-cuda-store-"));
      const source = path.join(__dirname, "native/pearlhash-cuda-store.cpp");
      const executable = path.join(fixture, "store");
      /** @param {string} body @param {string} output */
      const compile = (body, output) => {
        fs.writeFileSync(path.join(fixture, "store.inc"), body);
        const result = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, source, "-o", output], {encoding: "utf8"});
        assert.equal(result.error, undefined, result.error?.message);
        assert.equal(result.status, 0, result.stderr);
      };
      try {
        compile(store, executable);
        let result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined, result.error?.message);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout,
          /CUDA_STORE_FIXTURE PASS source=actual rank128 words=256 canaries=PASS warp_coords=PASS endpoint_B=0xffffffff/u);

        result = spawnSync(executable, ["--duplicate"], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined, result.error?.message);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout, /CUDA_STORE_NEGATIVE duplicate_write=REJECTED/u);

        // Mutate only the actual dense row coordinate. A valid source body must reject this
        // permutation through the independent canonical map or bounded sink, regardless of
        // whether the implementation obtains the paired word with a shuffle or writes it
        // directly from each lane.
        assert.equal(store.split("lane >> 3").length, 2,
          "dense row coordinate must have one source anchor");
        const wrongLane = store.replace("lane >> 3", "lane >> 2");
        const wrongExecutable = path.join(fixture, "store-wrong-lane");
        compile(wrongLane, wrongExecutable);
        result = spawnSync(wrongExecutable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined, result.error?.message);
        assert.notEqual(result.status, 0, "wrong dense lane mutation unexpectedly passed");
        assert.match(result.stderr, /CUDA_STORE_FIXTURE FAIL/u);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash CUDA pair proofs preserve physical coordinates", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const kernel = fs.readFileSync(path.join(root, "cuda-kernel.inc"), "utf8");
      const inverseStart = kernel.indexOf("const uint32_t warp_tiles_n =");
      const inverseEnd = kernel.indexOf("\n#else", inverseStart);
      const storeStart = kernel.indexOf("const uint64_t dense_proof_index =");
      const storeEnd = kernel.indexOf(";\n", storeStart) + 2;
      assert.ok(inverseStart > 0 && inverseEnd > inverseStart && storeStart > inverseEnd && storeEnd > storeStart);
      // The rank-128 checkpoint recurrence follows its transcript-word selection; it is either the
      // even/odd split chain or the single-chain control, and both must reproduce the oracle.
      const checkpointAnchor = kernel.indexOf("const int transcript_word = (rank_chunk - 1) & 15;");
      assert.ok(checkpointAnchor > 0, "rank-128 checkpoint anchor is required");
      const split = kernel.indexOf("uint32_t lane_part_even = 0;", checkpointAnchor) >= 0;
      const checkpointKind = split ? "xor2-even-odd" : "single-chain-control";
      const checkpointStart = kernel.indexOf(split ? "uint32_t lane_part_even = 0;" : "uint32_t lane_part = 0;",
        checkpointAnchor);
      const checkpointEndNeedle = split ? "const uint32_t lane_part = lane_part_even ^ lane_part_odd;" :
        "// Every lane participates";
      assert.ok(checkpointStart > checkpointAnchor,
        "rank-128 recurrence must follow the exact transcript anchor");
      const checkpointEnd = kernel.indexOf(checkpointEndNeedle, checkpointStart);
      assert.ok(checkpointEnd > checkpointStart,
        "rank-128 recurrence end anchor is required");
      const checkpointBody = kernel.slice(checkpointStart, checkpointEnd + checkpointEndNeedle.length);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-cuda-pair-"));
      try {
        for (const fixed of [false, true]) {
          // The failed private prototype decoded this warp-major index as
          // row-major 16x16 tiles. Keep that real defect as the negative control.
          const inverse = fixed ? kernel.slice(inverseStart, inverseEnd) : [
            "const uint32_t tile_row = tile / (n / 16u), tile_col = tile % (n / 16u);",
            "const uint32_t row = row_offset + (tile_row / 4u) * 64u + (tile_row % 4u) * 2u;",
            "const uint32_t col = (tile_col / 4u) * 64u + (tile_col % 4u) * 2u;",
          ].join("\n");
          fs.writeFileSync(path.join(fixture, "pair_dense_mapping_generated.h"), [
            "#include <array>", "#include <cstdint>", "#include <cstdlib>",
            "#ifndef CUTE_UNROLL", "#define CUTE_UNROLL", "#endif",
            "namespace pair_mapping_actual {", "using std::uint8_t;",
            "using std::uint32_t;", "using std::uint64_t;",
            "struct Point { uint64_t row, col; };",
            "inline Point actual_hash_inverse(uint64_t tile, uint32_t n, uint32_t row_offset) {",
            inverse, "return {row,col}; }",
            "inline uint64_t actual_dense_store_index(uint32_t block_m, uint32_t warp_m,",
            "uint32_t block_n, uint32_t warp_n, uint32_t n, uint32_t lane) {",
            kernel.slice(storeStart, storeEnd), "return dense_proof_index; }",
            "inline uint32_t pearlhash_xor3(uint32_t a, uint32_t b, uint32_t c) {",
            "return a ^ b ^ c; }",
            "struct CheckpointAccumulator {",
            "const std::array<uint32_t, 128>* values;",
            "std::array<uint8_t, 128>* touched;",
            "uint32_t operator()(int index) const {",
            "if (index < 0 || index >= 128) std::abort();",
            "++(*touched)[static_cast<unsigned>(index)];",
            "return (*values)[static_cast<unsigned>(index)];",
            "}", "};",
            "inline uint32_t actual_checkpoint(const std::array<uint32_t, 128>& values,",
            "std::array<uint8_t, 128>& touched) {",
            "CheckpointAccumulator accumulator{&values, &touched};",
            checkpointBody, "return lane_part; }",
            `// extracted_source_kind=${checkpointKind}`, "}",
          ].join("\n"));
          const executable = path.join(fixture, "pair");
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", "-I" + fixture, path.join(__dirname, "native/pearlhash-cuda-pair.cpp"),
            "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          assert.equal(result.signal, null);
          assert.equal(result.status, fixed ? 0 : 1, result.stdout + result.stderr);
          if (fixed) {
            assert.match(result.stdout, /PASS pair_dense_mapping/u);
          } else {
            assert.match(result.stderr, /hash inverse disagrees with dense producer coordinate/u);
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash compact fixed-K dispatch preserves its ABI and generic fallback", () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/cuda-compact-kernel.inc"), "utf8");
      const wrapper = extractCppBlock(source, 'extern "C" __global__ __launch_bounds__(256)');
      const helper = extractCppBlock(source, "__device__ __forceinline__ void compact_gemm_impl(");
      // The public kernel parameters are the launch ABI; the dispatch must forward all of them.
      const parameters = wrapper.match(/void compact_gemm\(([^)]*)\)/u)?.[1];
      assert.ok(parameters);
      const names = parameters.split(",").map(part => part.trim().match(/\w+$/u)?.[0] ?? "");
      const kIndex = names.indexOf("k");
      assert.ok(kIndex >= 0);
      const start = wrapper.search(/\bif\s*\(\s*k\s*==/u);
      assert.ok(start >= 0);
      const dispatch = wrapper.slice(start, wrapper.lastIndexOf("}"));
      const binding = helper.match(/const int k = ([^;\n]+);/u)?.[1];
      assert.ok(binding);
      // Execute the actual scalar dispatch/argument list, not a second hand-written selector.
      // This CPU-only fixture deliberately does not emulate the CUDA numerical body.
      const exercise = (/** @type {string} */ body) => {
        const calls = [...body.matchAll(/compact_gemm_impl<(\d+)>\(([^;]+)\);/gu)];
        assert.equal(calls.length, 2);
        const js = body.replace(/compact_gemm_impl<(\d+)>\(([^;]+)\);/gu,
          "return {fixed:$1,args:[$2]};");
        const invoke = runInNewContext(`((${names.join(",")}) => { const storage={fixture:true}; ${js} })`);
        const effective = runInNewContext(`((FixedK,input_k) => (${binding}))`);
        for (const k of [-2147483648, -1, 0, 1, 127, 128, 256, 2048, 4096, 8192,
          65535, 65536, 65537, 2147483520, 2147483647]) {
          const args = names.map((_, index) => index === kIndex ? k : index + 100);
          const result = invoke(...args);
          assert.equal(result.fixed, k === 65536 ? 65536 : 0);
          assert.equal(result.args[0].fixture, true);
          assert.deepEqual(Array.from(result.args).slice(1), args);
          assert.equal(effective(result.fixed, k), k);
        }
      };
      exercise(dispatch);
      const wrongGuard = dispatch.replace("k == 65536", "k == 8192");
      const noFallback = dispatch.replace("compact_gemm_impl<0>", "compact_gemm_impl<65536>");
      assert.notEqual(wrongGuard, dispatch);
      assert.notEqual(noFallback, dispatch);
      assert.throws(() => exercise(wrongGuard));
      assert.throws(() => exercise(noFallback));
    });

    test("PearlHash initial selector scores and records each winning rank exactly", () => {
      const source = fs.readFileSync(path.join(__dirname,
        "../sycl/pearlhash/cuda-compact-kernel.inc"), "utf8");
      const exercise = (/** @type {string} */ code) => {
        const prepare = extractCppBlock(code, "void selector_prepare(");
        // Preserve C++ integer division while exercising its actual score/update in JS.
        const scoreText = prepare.match(/score\+=([^;]+);/u)?.[1]?.replaceAll("lane/4", "(lane>>2)");
        assert.ok(scoreText);
        const score = runInNewContext(`((word,groups,counts,lane) => {
          let score=0; for(let s=0;s<4;++s) {
            const rank=(word>>>(8*s))&127; score+=${scoreText};
          } return score;
        })`);
        const updateText = extractCppBlock(prepare, "      if(lane<4) {")
          .replace("if(lane<4)", "")
          .replace(/const unsigned /gu, "const ")
          .replaceAll("winner/4", "(winner>>2)");
        const update = runInNewContext(`((word,groups,counts,winner,lane) => ${updateText})`);
        for (let rank = 0; rank < 120; ++rank) {
          for (let winner = 0; winner < 16; ++winner) {
            const groups = Array.from({length: 16}, () =>
              Array.from({length: 4}, () => ({ranks: new Uint8Array(128)})));
            const counts = Array.from({length: 16}, () =>
              Array.from({length: 4}, () => new Uint8Array(8)));
            const word = rank * 0x01010101;
            assert.equal(score(word, groups, counts, winner), 26);
            for (let repeat = 0; repeat < 3; ++repeat) {
              for (let lane = 0; lane < 4; ++lane) {update(word, groups, counts, winner, lane);}
              assert.equal(score(word, groups, counts, winner), 0);
              for (let lane = 0; lane < 4; ++lane) {
                assert.equal(counts[winner]?.[lane]?.[rank & 7], 1);
              }
            }
            assert.equal(score((rank + 8) * 0x01010101, groups, counts, winner), 78);
          }
        }
      };
      exercise(source);
      for (const [from, to] of [["ranks[rank]?0:", "ranks[rank]?1:"],
        ["if(!groups[winner][lane].ranks[rank])", "if(true)"]]) {
        assert.ok(typeof from === "string" && typeof to === "string");
        assert.ok(source.includes(from));
        assert.throws(() => exercise(source.replace(from, to)));
      }
    });

    test("PearlHash selector swaps and score keys remain exact and bounded", () => {
      const source = fs.readFileSync(path.join(__dirname,
        "../sycl/pearlhash/cuda-compact-kernel.inc"), "utf8");
      const exercise = (/** @type {string} */ code) => {
        const body = (/** @type {string} */ signature) => {
          const block = extractCppBlock(code, signature);
          return block.slice(block.indexOf("{") + 1, block.lastIndexOf("}"))
            .replace(/#pragma unroll/gu, "")
            .replace(/for\(unsigned /gu, "for(let ")
            .replace(/\bconst unsigned\b/gu, "const")
            .replace(/\bunsigned\s+(?=[a-z])/gu, "let ")
            .replace(/\bunsigned\(/gu, "Number(");
        };
        // These execute the actual scalar helpers, not the warp intrinsics.
        // Every source group contains eight ranks, so removing an absent rank
        // is deliberately outside this contract.
        const context = {max: Math.max, Number};
        const refresh = runInNewContext(`((g) => {${body("__device__ __forceinline__ void refresh(")}})`, context);
        const swap = runInNewContext(`((g,outgoing,incoming) => {${body("__device__ __forceinline__ unsigned swap_max(")}})`, context);
        const exchange = runInNewContext(`((g,outgoing,incoming) => {${body("__device__ __forceinline__ void exchange(")}})`,
          {...context, refresh});
        const group = (/** @type {number[]} */ values) => {
          const ranks = new Uint8Array(128), banks = new Uint8Array(8);
          for (const rank of values) {ranks[rank] = (ranks[rank] ?? 0) + 1;}
          for (let rank = 0; rank < 128; ++rank) {
            if (ranks[rank]) {banks[rank & 7] = (banks[rank & 7] ?? 0) + 1;}
          }
          const maximum = Math.max(...banks);
          return {ranks, banks, maximum,
            at_maximum: Array.from(banks).filter(value => value === maximum).length};
        };
        const fixtures = [Array(8).fill(0), Array.from({length: 8}, (_, i) => i),
          Array.from({length: 8}, (_, i) => 8 * i), [0, 0, 8, 8, 16, 1, 1, 9]];
        for (let seed = 0; seed < 12; ++seed) {
          fixtures.push(Array.from({length: 8}, (_, i) =>
            (Math.imul(seed + 17, i * 41 + 13) ^ (i << (seed & 3))) & 127));
        }
        for (const values of fixtures) {
          for (let out = 0; out < values.length; ++out) {
            for (let incoming = 0; incoming < 128; ++incoming) {
              const original = group(values), next = values.slice();
              next[out] = incoming;
              const expected = group(next);
              assert.equal(swap(original, values[out], incoming), expected.maximum);
              exchange(original, values[out], incoming);
              assert.deepEqual(original, expected);
            }
          }
        }
        const swaps = extractCppBlock(code, "__device__ __forceinline__ uint32_t swap_checkpoint(");
        const prepare = extractCppBlock(code, "void selector_prepare(");
        const weightTexts = [...code.matchAll(/\(((?:s|stream)<2\?\d+:\d+)\)/gu)].map(match => match[1]);
        assert.equal(weightTexts.length, 2);
        const weights = weightTexts.map(expression =>
          runInNewContext(`((s,stream=s) => (${expression}))`));
        const terms = Array.from({length: 4}, (_, s) => Number(weights[0](s)));
        assert.deepEqual(terms, Array.from({length: 4}, (_, s) => Number(weights[1](s))));
        assert.ok(terms.every(value => Number.isSafeInteger(value) && value > 0));
        // One replacement changes each group's maximum by at most one.
        const bound = 2 * terms.reduce((sum, value) => sum + value, 0);
        assert.ok(bound < 256);
        const keyText = swaps.match(/const unsigned local=__reduce_min_sync\(0xffffffffu,([^;\n]+)\);/u)?.[1];
        const acceptText = swaps.match(/if\((\(key>>\d+\)<\d+)\) \{/u)?.[1];
        assert.ok(keyText && acceptText);
        const key = runInNewContext(`((delta,thread) => (${keyText.replace(/unsigned\(/gu, "Number(")}))`, {Number});
        const accept = runInNewContext(`((key) => (${acceptText}))`);
        for (let delta = -bound; delta <= bound; ++delta) {
          for (let lane = 0; lane < 128; ++lane) {
            const actual = Number(key(delta, lane));
            assert.ok(Number.isSafeInteger(actual) && actual >= 0 && actual <= 0xffffffff);
            assert.equal(Boolean(accept(actual)), delta < 0);
            if (lane) {assert.ok(actual > key(delta, lane - 1));}
          }
        }
        const greedyText = prepare.match(/key=(\([^;\n]*score[^;\n]+);/u)?.[1];
        assert.ok(greedyText);
        const greedy = runInNewContext(`((score,sizes,lane) => (${greedyText}))`);
        const dimensions = prepare.match(/if\(lane<(\d+)&&sizes\[lane\]<(\d+)\)/u);
        assert.ok(dimensions);
        const bins = Number(dimensions[1]), capacity = Number(dimensions[2]);
        assert.equal(bins * capacity, 128);
        const maxScore = (2 * capacity + 1) * terms.reduce((sum, value) => sum + value, 0);
        for (const score of [0, 1, maxScore]) {
          for (let size = 0; size < capacity; ++size) {
            for (let lane = 0; lane < bins; ++lane) {
              const actual = Number(greedy(score, Array(bins).fill(size), lane));
              assert.ok(Number.isSafeInteger(actual) && actual >= 0 && actual <= 0xffffffff);
              assert.ok(actual < greedy(score + 1, Array(bins).fill(0), 0));
              if (lane) {assert.ok(actual > greedy(score, Array(bins).fill(size), lane - 1));}
              if (size) {assert.ok(actual > greedy(score, Array(bins).fill(size - 1), bins - 1));}
            }
          }
        }
      };
      exercise(source);
      for (const [from, to] of [["g.at_maximum==1", "g.at_maximum==0"],
        ["b=incoming&7", "b=incoming&3"],
        ["unsigned(delta+256)*128+thread", "unsigned(delta+256)*64+thread"]]) {
        assert.ok(typeof from === "string" && typeof to === "string");
        assert.ok(source.includes(from));
        assert.throws(() => exercise(source.replace(from, to)));
      }
    });

    require("./logic/pearlhash-compact");

    test("PearlHash compact admission commits geometry only after native admission", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const host = fs.readFileSync(path.join(root, "pearlhash.cpp"), "utf8");
      const policy = fs.readFileSync(path.join(root, "compact-policy.inc"), "utf8");
      const begin = host.indexOf("bool native_int8_route = false;");
      const end = host.indexOf("if (st.have_header &&", begin);
      assert.ok(begin >= 0 && end > begin);
      const admission = host.slice(begin, end);
      const selected = extractCppBlock(host, "static bool selected(int m,int n,int k,int rank) {");
      const shape = extractCppBlock(policy, "  static bool supports(int m, int n, int requested_k, int requested_rank) {");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-compact-admission-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/pearlhash-compact-admission.cpp"), path.join(fixture, "test.cpp"));
        const run = (/** @type {string} */ mode, /** @type {string} */ body,
          /** @type {string} */ select, /** @type {RegExp | undefined} */ fail = undefined) => {
          fs.writeFileSync(path.join(fixture, "admission.inc"), body);
          fs.writeFileSync(path.join(fixture, "selected.inc"), select);
          fs.writeFileSync(path.join(fixture, "shape.inc"), shape);
          const executable = path.join(fixture, mode);
          const built = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_SYCL_HAS_HIP=1", "-I" + fixture, "-I" + root,
            "-I" + path.join(__dirname, "../native"), path.join(fixture, "test.cpp"), "-o", executable], {encoding: "utf8"});
          assert.equal(built.status, 0, built.error?.message || built.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8"});
          assert.equal(result.status, fail ? 1 : 0, result.stdout + result.stderr);
          if (fail) {assert.match(result.stderr, fail);} else {assert.match(result.stdout, /COMPACT_ADMISSION_PASS cases=32/u);}
        };
        await t.test("source-derived admission, stale-state reset and target scaling", () => run("healthy", admission, selected));
        const flag = selected.replace("return CompactProfile::supports(m,n,k,rank);",
          'return std::getenv("MOM_PRIVATE_COMPACT") && CompactProfile::supports(m,n,k,rank);');
        assert.notEqual(flag, selected);
        await t.test("old private-flag requirement really fails", () =>
          run("private-flag", admission, flag, /private flag still required/u));
        const eager = admission.replace("cuda_native_int8_route && st.cuda.compact", "compact_private::selected(m,n,k,rank)");
        assert.notEqual(eager, admission);
        await t.test("pre-admission geometry really fails", () =>
          run("eager-geometry", eager, selected, /wrong compact admission/u));
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash shared matrices survive optional allocation faults", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const state = fs.readFileSync(path.join(root, "state.inc"), "utf8");
      const hip = fs.readFileSync(path.join(root, "hip-jit.inc"), "utf8");
      const guardStart = hip.indexOf("if (!b.Ap || !b.Bp)");
      const guardEnd = hip.indexOf("if (!mom_is_hip(", guardStart);
      assert.ok(guardStart >= 0 && guardEnd > guardStart);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-allocation-"));
      try {
        fs.writeFileSync(path.join(fixture, "pearlhash_allocation_policy.inc"),
          extractCppBlock(state, "static bool pearlhash_required_layout_admitted("));
        fs.writeFileSync(path.join(fixture, "pearlhash_native_allocation.inc"), [
          "template <typename T> static T* alloc_buffer(",
          "template <typename T> static T* alloc_optional(",
          "static void free_allocations(",
        ].map(anchor => extractCppBlock(state, anchor)).join("\n"));
        fs.writeFileSync(path.join(fixture, "hip-layout-admission.inc"),
          hip.slice(guardStart, guardEnd) + "    return true;\n");
        const executable = path.join(fixture, "allocation");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, "-I" + path.join(__dirname, "../sycl"),
          path.join(__dirname, "native/pearlhash-allocation.cpp"), "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout, /allocation admission and native fallback passed/u);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash HIP buffer loads preserve bounded packed input and gfx12 descriptor flags", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/hip-kernel.inc"), "utf8");
      const helper = extractCppBlock(source, "DEV __amdgpu_buffer_rsrc_t pearlhash_input_resource(");
      const matrix = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/matrix.inc"), "utf8");
      const loads = source.match(/[ab]\[[ij]\] = __builtin_amdgcn_raw_buffer_load_b64\([\s\S]*?\);/gu);
      assert.equal(loads?.length, 2);
      const [loadA, loadB] = loads || [];
      assert.ok(loadA && loadB);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-hip-buffer-"));
      try {
        fs.writeFileSync(path.join(fixture, "load-a.inc"), loadA);
        fs.writeFileSync(path.join(fixture, "load-b.inc"), loadB);
        fs.writeFileSync(path.join(fixture, "pack-offset.inc"),
          extractCppBlock(matrix, "static inline size_t amd_wmma_offset("));
        const zeroFlags = helper.replace("0x31004000u", "0u");
        const divergentCount = helper.replace("const int count = __builtin_amdgcn_readfirstlane(",
          "const int count = (");
        assert.notEqual(zeroFlags, helper);
        assert.notEqual(divergentCount, helper);
        for (const k of [2048, 8192, 65536]) {
          for (const [variant, expected] of /** @type {Array<[string, RegExp]>} */ ([
            [helper, /PASS actual-source HIP resource/u],
            [zeroFlags, /invalid gfx12 resource/u],
            [divergentCount, /must both be explicitly wave-uniform/u],
          ])) {
            fs.writeFileSync(path.join(fixture, "input-resource.inc"), variant);
            const executable = path.join(fixture, "hip-buffer");
            const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-pedantic", `-DPEARLHASH_K=${k}`, "-I" + fixture,
              path.join(__dirname, "native/pearlhash-hip-buffer.cpp"), "-o", executable], {encoding: "utf8"});
            assert.equal(compiled.status, 0, compiled.stderr);
            const result = spawnSync(executable, [], {encoding: "utf8"});
            assert.equal(result.status, variant === helper ? 0 : 1, result.stdout + result.stderr);
            assert.match(result.stdout + result.stderr, expected);
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash HIP lane transcripts match their canonical openings, tails and coordinates", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const source = fs.readFileSync(path.join(root, "hip-kernel.inc"), "utf8");
      // Slice the fused kernel's fold, lane LDS row, coordinates and tail guard into a CPU oracle
      // that recomputes each opening from the host axes; the mutations prove each slice matters.
      const fold = extractCppBlock(source, "    if ((kk + 32) % PEARLHASH_RANK == 0) {");
      const rowStart = source.indexOf("  uint32_t* const transcript =");
      const rowEnd = source.indexOf("\n", rowStart);
      const quarterStart = source.indexOf("  const uint32_t proof_quarter =");
      const quarterEnd = source.indexOf(";\n", quarterStart) + 2;
      const coordinatesStart = source.indexOf("  const uint32_t proof_row =");
      const coordinatesEnd = source.indexOf("  if (active_wave &&", coordinatesStart);
      const guardStart = source.indexOf("active_wave &&", coordinatesEnd);
      const guardEnd = source.indexOf(") {\n    uint32_t tr_hash[16];", guardStart);
      assert.ok(rowStart >= 0 && rowEnd > rowStart && quarterStart >= 0 && quarterEnd > quarterStart &&
        coordinatesStart >= 0 && coordinatesEnd > coordinatesStart && guardStart > coordinatesEnd &&
        guardEnd > guardStart);
      const laneRow = source.slice(rowStart, rowEnd) + "\n";
      const coordinates = source.slice(quarterStart, quarterEnd) +
        source.slice(coordinatesStart, coordinatesEnd);
      const guard = source.slice(guardStart, guardEnd) + "\n";
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-hip-transcript-"));
      try {
        fs.copyFileSync(path.join(root, "proof-axis.h"), path.join(fixture, "proof-axis.h"));
        /** @typedef {"fold" | "laneRow" | "coordinates" | "guard"} TranscriptPart */
        /** @type {Record<string, Array<[TranscriptPart, string, string]>>} */
        const mutations = {
          valid: [],
          reduction: [["fold", "(uint32_t)acc[i][3][2] ^ ", ""]],
          slot: [["fold", "checkpoint & 15", "checkpoint & 7"]],
          wrap: [["fold", "checkpoint < 16", "checkpoint < 8"]],
          "lane-row": [["laneRow", "threadIdx.x * 17u", "threadIdx.x * 8u"]],
          coordinate: [["coordinates", "proof_quarter * 4u", "proof_quarter * 8u"]],
          quarter: [["coordinates", "* 2u + (uint32_t)(lane >> 5)", "+ (uint32_t)(lane >> 5) * 2u"]],
          tail: [["guard", "tile_row + PEARLHASH_ER <= m / 16", "tile_row < m / 16"]],
        };
        for (const [variant, edits] of Object.entries(mutations)) {
          /** @type {Record<TranscriptPart, string>} */
          const parts = {fold, laneRow, coordinates, guard};
          for (const [part, from, to] of edits) {
            assert.ok(parts[part].includes(from), `${variant}: ${from}`);
            parts[part] = parts[part].replace(from, to);
          }
          fs.writeFileSync(path.join(fixture, "checkpoint-fold.inc"), parts.fold + "\n");
          fs.writeFileSync(path.join(fixture, "lane-row.inc"), parts.laneRow);
          fs.writeFileSync(path.join(fixture, "coordinates.inc"), parts.coordinates);
          fs.writeFileSync(path.join(fixture, "tail-guard.inc"), parts.guard);
          /** @type {Array<[number, number]>} */
          const shapes = [[2048, 128], [4096, 128], [4096, 256], [8192, 128]];
          for (const [k, rank] of shapes) {
            const executable = path.join(fixture, `hip-transcript-${variant}-${k}-${rank}`);
            const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-Wno-unknown-pragmas", "-pedantic", `-DPEARLHASH_K=${k}`, `-DPEARLHASH_RANK=${rank}`,
              "-I" + fixture, path.join(__dirname, "native/pearlhash-hip-transcript.cpp"),
              "-o", executable], {encoding: "utf8"});
            assert.equal(compiled.status, 0, `${variant}/${k}/${rank}: ${compiled.stderr}`);
            const result = spawnSync(executable, [], {encoding: "utf8"});
            if (variant === "valid") {
              assert.equal(result.status, 0, `${k}/${rank}: ${result.stdout}${result.stderr}`);
              assert.match(result.stdout, new RegExp(`CPU lane transcript oracle passed: K=${k} ` +
                `rank=${rank} checkpoints=${k / rank} openings=128`, "u"));
            } else {
              assert.equal(result.status, 1, `${variant}/${k}/${rank} must fail: ${result.stdout}`);
              assert.match(result.stderr, /lane transcript differs|tail group published|not canonical|cover complete groups/u);
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash HIP reuses bounded row slices without changing global proof coordinates", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const source = fs.readFileSync(path.join(root, "hip-jit.inc"), "utf8");
      const kernel = fs.readFileSync(path.join(root, "hip-kernel.inc"), "utf8");
      const state = fs.readFileSync(path.join(root, "state.inc"), "utf8");
      const matrix = fs.readFileSync(path.join(root, "matrix.inc"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-hip-slices-"));
      try {
        fs.writeFileSync(path.join(fixture, "hip-launch.inc"), extractCppBlock(source, "  void launch("));
        fs.writeFileSync(path.join(fixture, "hip-wait.inc"), extractCppBlock(source, "  void wait(double learned_us)"));
        fs.writeFileSync(path.join(fixture, "hip-wave-groups.inc"), extractCppBlock(kernel, "DEV void wave_groups("));
        fs.writeFileSync(path.join(fixture, "matrix-offset.inc"),
          extractCppBlock(matrix, "static inline size_t amd_wmma_offset("));
        const allocationStart = state.indexOf("constexpr uint64_t transcript_limit =");
        const allocationEnd = state.indexOf("if (transcript_bytes > 0 &&", allocationStart);
        assert.ok(allocationStart >= 0 && allocationEnd > allocationStart);
        fs.writeFileSync(path.join(fixture, "transcript-allocation.inc"), state.slice(allocationStart, allocationEnd));
        const executable = path.join(fixture, "slices");
        const er = 13; // kAmdLaneRows: thirteen 16-row tiles per wave
        for (const k of [1024, 2048, 3072, 4096, 5632, 8192, 16384]) {
          const compile = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-Wno-unused-parameter", "-Wno-range-loop-construct", "-pedantic", "-I" + fixture,
            `-DPEARLHASH_ER=${er}`, "-DPEARLHASH_K=" + k,
            path.join(__dirname, "native/pearlhash-hip-slices.cpp"), "-o", executable],
          {encoding: "utf8"});
          assert.equal(compile.status, 0, `ER${er}/K${k}: ${compile.stderr}`);
          const result = spawnSync(executable, [], {encoding: "utf8"});
          assert.equal(result.status, 0, `ER${er}/K${k}: ${result.stderr}`);
          const cacheRows = Math.max(16, Math.min(64, Math.floor(131072 / k)));
          const chunkRows = cacheRows * er * 16;
          assert.match(result.stdout,
            new RegExp(`CPU fused launch oracle passed: K=${k} cache_rows=${cacheRows} chunk_rows=${chunkRows}`));
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash portable OpenCL launches bound work without changing other backends", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl-search.inc"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-launch-budget-"));
      try {
        fs.writeFileSync(path.join(fixture, "launch-budget.inc"),
          extractCppBlock(source, "static size_t pearlhash_portable_launch_groups("));
        const executable = path.join(fixture, "launch-budget");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, path.join(__dirname, "native/pearlhash-launch-budget.cpp"),
          "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.match(result.stdout, /bounded with complete tile coverage/);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash split queues retire independently before retry", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const production = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/pearlhash.cpp"), "utf8");
      const attempt = production.indexOf("attempt_start =");
      const start = production.indexOf("\n  } catch (...)", attempt);
      assert.ok(attempt >= 0 && start > attempt);
      // Retire only the catch block, independent of whichever success-path code follows it.
      const catchBody = production.slice(production.indexOf("catch (...)", start));
      const block = "  } " + extractCppBlock(catchBody, "catch (...)");
      const prep = 'sycl_cleanup_noexcept("pearlhash preparation", [&] { st.prep_queue.wait_and_throw(); });';
      assert.ok(block.includes(prep));
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-split-cleanup-"));
      try {
        const source = path.join(fixture, "fixture.cpp");
        const executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/pearlhash-split-cleanup.cpp"), source);
        for (const fixed of [false, true]) {
          fs.writeFileSync(path.join(fixture, "catch.inc"), fixed ? block : block.replace(prep, ""));
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", source, "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          for (const [targetFault, prepFault] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            const result = spawnSync(executable, [String(targetFault), String(prepFault)],
              {encoding: "utf8", timeout: 2000});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, fixed ? 0 : 1, result.stderr);
            if (fixed) {
              assert.equal(result.stdout, "PASS PearlHash split preparation cleanup\n");
            } else {
              assert.match(result.stderr, /both queue cleanups/);
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("PearlHash submitted legacy native faults propagate without fallback", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const dispatch = fs.readFileSync(path.join(root, "dispatch.inc"), "utf8");
      const host = fs.readFileSync(path.join(root, "pearlhash.cpp"), "utf8");
      const invalidateStart = host.indexOf("  g_pf.count = 0;");
      // Keep only the entry invalidation block. It intentionally precedes argument validation so
      // a rejected call cannot expose a previous share; the validation body is not self-contained
      // in the native fault fixture and must not be copied into it.
      const invalidateEnd = host.indexOf("  if (!input ||", invalidateStart);
      const waitStart = host.indexOf("  if (search_backend == PearlHashSearchBackend::hip_jit ||");
      const uploadEnd = host.indexOf("  if (is_test && q.get_device().is_gpu())", waitStart);
      const drainStart = host.indexOf("\n  } catch (...)", waitStart);
      const waitEnd = drainStart < 0 ? uploadEnd : drainStart;
      const stateStart = host.indexOf("  const bool prior_fixed_b_valid = st.fixed_b_valid;");
      const stateEnd = host.indexOf("\n\n  const bool native_requested", stateStart);
      const uploadStart = Math.min(host.indexOf("  if (!st.have_header ||"),
        host.indexOf("  uint8_t tgtLE[32];"));
      const readbackEnd = host.indexOf("  // search() retains", uploadEnd);
      const commitStart = host.indexOf("  if (!b.result->found)", waitEnd);
      const commitEnd = host.indexOf("\n}\n// Builds", commitStart);
      const commitFixedStart = host.indexOf("  const auto commit_fixed_b = [&] {", waitEnd);
      const commitFixedEnd = host.indexOf("\n  if (is_test &&", commitFixedStart);
      const pairModeStart = host.indexOf(
        "  if (st.have_header && st.header_pair_dist8 != pair_dist8)", stateEnd);
      const pairModeEnd = host.indexOf("  bool reuse_fixed_b = false;", pairModeStart);
      const pairReuseDeclEnd = host.indexOf("\n", pairModeEnd) + 1;
      const pairHeaderStart = host.indexOf("    if (!st.have_header ||", pairModeEnd);
      const pairReuseStart = host.indexOf("    reuse_fixed_b = pair_dist8", pairHeaderStart);
      const pairReuseEnd = host.indexOf(";\n", pairReuseStart) + 2;
      const fixedSeedStart = host.indexOf(
        "    // The producer needs the selected B seed before it submits roots/noise/packing.",
        pairReuseEnd);
      const fixedSeedAssignment = "    b.fixed_b_seed = reuse_fixed_b ? prior_fixed_b_seed : attempt_seed;";
      const fixedSeedEnd = host.indexOf(fixedSeedAssignment, fixedSeedStart) + fixedSeedAssignment.length;
      const pairCatchStart = host.indexOf("catch (...)", waitEnd);
      const pairCatch = extractCppBlock(host.slice(pairCatchStart), "catch (...)");
      assert.ok(invalidateStart > 0 && invalidateEnd > invalidateStart &&
      waitStart > invalidateEnd && waitEnd > waitStart && stateStart > 0 && stateEnd > stateStart &&
      commitStart > waitEnd && commitEnd > commitStart && commitFixedStart > waitEnd &&
      commitFixedEnd > commitFixedStart &&
      uploadStart > invalidateEnd && uploadEnd >= waitEnd && readbackEnd > uploadEnd &&
      pairModeStart > stateEnd && pairModeEnd > pairModeStart && pairHeaderStart > pairModeEnd &&
      pairReuseStart > pairHeaderStart && pairReuseEnd > pairReuseStart &&
      fixedSeedStart > pairReuseEnd && fixedSeedEnd > fixedSeedStart && pairCatchStart > waitEnd &&
      pairCatch.length > 0);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-native-fault-"));
      const source = path.join(fixture, "pearlhash-fault.cpp");
      const executable = path.join(fixture, "pearlhash_fault");
      const mutantExecutable = path.join(fixture, "pearlhash_fault_mutant");
      try {
        fs.copyFileSync(path.join(__dirname, "native/pearlhash-fault.cpp"), source);
        fs.copyFileSync(path.join(root, "claim.h"), path.join(fixture, "claim.h"));
        fs.copyFileSync(path.join(root, "proof-axis.h"), path.join(fixture, "proof-axis.h"));
        // This fixture exercises the legacy pair and HIP/CUDA lifecycle, not compact root caching.
        // Specialize only the distinct compact blocks away; retain actual legacy operations,
        // target conversion, exception boundaries, and the compiled stale-header negative.
        const compactRoots = extractCppBlock(host, "    if (compact_mode && reuse_fixed_b) {");
        const compactCapture = extractCppBlock(host, "  if (compact_mode && !reuse_fixed_b) {");
        const compactWinner = extractCppBlock(host, "  if (compact_mode) {").trimEnd() + " else ";
        assert.equal(host.split(compactWinner).length, 2);
        const legacyCommit = host.slice(commitStart, commitEnd).replace(compactWinner, "  ");
        assert.doesNotMatch(legacyCommit, /compact_mode|compact_roots/u);
        const compactKey = "if (compact_mode)\n        CompactProfile::derive_key(input,st.key);\n      else\n";
        assert.equal(host.split(compactKey).length, 2);
        const legacyUpload = host.slice(uploadStart, uploadEnd)
          .replace("    sycl::event compact_roots_ready;\n", "")
          .replace(compactRoots, "")
          .replace(compactCapture, "")
          .replace(compactKey, "")
          .replace("compact_mode && reuse_fixed_b ? &compact_roots_ready : nullptr", "nullptr");
        assert.doesNotMatch(legacyUpload, /compact_mode|CompactProfile|compact_roots_ready/u);
        const legacyCommitFixed = host.slice(commitFixedStart, commitFixedEnd)
          .replace(compactCapture, "");
        assert.doesNotMatch(legacyCommitFixed, /compact_mode|compact_roots/u);
        const compactDispatch = extractCppBlock(dispatch, "    if (cuda->compact) {").trimEnd() + " else ";
        assert.equal(dispatch.split(compactDispatch).length, 2);
        fs.writeFileSync(path.join(fixture, "dispatch.inc"), dispatch.replace(compactDispatch, "    "));
        fs.writeFileSync(path.join(fixture, "invalidate.inc"), host.slice(invalidateStart, invalidateEnd));
        fs.writeFileSync(path.join(fixture, "state-invalidation.inc"), host.slice(stateStart, stateEnd));
        fs.writeFileSync(path.join(fixture, "wait.inc"), host.slice(waitStart, waitEnd));
        fs.writeFileSync(path.join(fixture, "upload.inc"), legacyUpload);
        fs.writeFileSync(path.join(fixture, "readback.inc"), host.slice(uploadEnd, readbackEnd));
        fs.writeFileSync(path.join(fixture, "commit-fixed-b.inc"),
          legacyCommitFixed);
        fs.writeFileSync(path.join(fixture, "commit.inc"), legacyCommit);
        const pairHeader = host.slice(pairModeStart, pairModeEnd) +
          host.slice(pairModeEnd, pairReuseDeclEnd) +
          host.slice(pairHeaderStart, pairReuseEnd).replace(compactKey, "if (pair_dist8)");
        fs.writeFileSync(path.join(fixture, "pair-header.inc"), pairHeader);
        fs.writeFileSync(path.join(fixture, "fixed-b-seed.inc"),
          host.slice(fixedSeedStart, fixedSeedEnd));
        fs.writeFileSync(path.join(fixture, "pair-catch.inc"), pairCatch);
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_SYCL_HAS_HIP=1", source, "-o", executable],
        {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        await t.test("CUDA pair decline after key admission cannot fall back", () => {
          const result = spawnSync(executable, ["pair", "0", "0", "0"], {encoding: "utf8"});
          assert.equal(result.error, undefined);
          assert.equal(result.status, 0, result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, "PASS actual-source Pearl pair decline fails closed\n");
        });
        await t.test("CUDA pair fixed-B miss, hit, invalidation, and fault stay source-consistent", () => {
          const result = spawnSync(executable, ["pair-lifecycle", "0", "0", "0"],
            {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          assert.equal(result.status, 0, result.stdout + result.stderr);
          assert.equal(result.signal, null);
          assert.equal(result.stdout, "PASS actual-source Pearl pair fixed-B lifecycle\n");
        });
        await t.test("CUDA pair header-invalidation mutation is rejected", () => {
          const headerInvalidation = "      st.have_header = false;\n";
          assert.equal(pairHeader.split(headerInvalidation).length, 2,
            "fixture must contain one extracted header invalidation assignment");
          try {
            fs.writeFileSync(path.join(fixture, "pair-header.inc"),
              pairHeader.replace(headerInvalidation, "      /* mutated: stale header state */\n"));
            const mutant = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_SYCL_HAS_HIP=1", source, "-o", mutantExecutable],
            {encoding: "utf8"});
            assert.equal(mutant.error, undefined);
            assert.equal(mutant.signal, null);
            assert.equal(mutant.status, 0, mutant.stderr);
            const result = spawnSync(mutantExecutable, ["pair-lifecycle", "0", "0", "0"],
              {encoding: "utf8", timeout: 5000});
            assert.equal(result.error, undefined);
            assert.notEqual(result.status, 0, "stale-header mutant unexpectedly passed");
            assert.equal(result.signal, null);
            assert.match(result.stderr, /header change reused fixed B/u);
          } finally {
            fs.writeFileSync(path.join(fixture, "pair-header.inc"), pairHeader);
          }
        });
        const cases = [];
        for (const [hip, name] of ["CUDA", "HIP"].entries()) {
          cases.push({dispatch: 0, hip, stage: 0, stringError: 0, name: name + " wait succeeds"});
          cases.push({dispatch: 1, hip, stage: 0, stringError: 0, name: name + " native attempt succeeds"});
          cases.push({dispatch: 1, hip, stage: 1, stringError: 0, name: name + " unavailable JIT retains SYCL"});
          for (const [stringError, error] of ["std::exception", "std::string"].entries()) {
            cases.push({dispatch: 0, hip, stage: 7, stringError, name: name + " wait propagates " + error});
            const operations = hip ? [[3, "preparation"], [6, "launch"]] :
              [[2, "queue wait"], [3, "preparation"], [6, "launch"]];
            for (const [stage, operation] of operations) {
              cases.push({dispatch: 1, hip, stage, stringError,
                name: name + " " + operation + " propagates " + error});
            }
            if (hip) {
              for (const [stage, operation] of [[4, "A event wait"], [5, "B event wait"],
                [8, "A event query"], [9, "B event query"]]) {
                cases.push({dispatch: 1, hip, stage, stringError,
                  name: name + " " + operation + " propagates " + error});
              }
            }
          }
        }
        await runGroupedCases(
          t, cases,
          ({hip}) => hip ? "HIP native dispatch and completion" : "CUDA native dispatch and completion",
          async (caseTest, {dispatch: dispatchCase, hip, stage, stringError, name}) => {
            await caseTest.test(name, () => {
              const result = spawnSync(executable,
                [String(dispatchCase), String(hip), String(stage), String(stringError)], {encoding: "utf8"});
              assert.equal(result.error, undefined);
              assert.equal(result.status, 0, result.stderr);
              assert.equal(result.signal, null);
              assert.equal(result.stdout, "PASS actual-source Pearl native fault boundary\n");
            });
          });
        const captureCases = [];
        for (const [hip, name] of ["CUDA", "HIP"].entries()) {
          captureCases.push({hip, stage: 0, error: 0, name: name + " retains exact winning leaves"});
          for (const [stage, operation] of [[14, "A readback"], [15, "B readback"], [2, "completion"]]) {
            for (let error = 0; error < 4; ++error) {
              captureCases.push({hip, stage, error,
                name: name + " " + operation + " fails closed with " + (error % 2 ? "string" : "exception") +
                  (error >= 2 ? " despite cleanup failure" : "")});
            }
          }
        }
        await runGroupedCases(t, captureCases,
          ({hip}) => hip ? "HIP winning proof readbacks" : "CUDA winning proof readbacks",
          async (caseTest, {hip, stage, error, name}) => {
            await caseTest.test(name, () => {
              const result = spawnSync(executable,
                ["capture", String(hip), String(stage), String(error)], {encoding: "utf8"});
              assert.equal(result.error, undefined);
              assert.equal(result.status, 0, result.stderr);
              assert.equal(result.signal, null);
              assert.equal(result.stdout, "PASS actual-source Pearl native proof capture\n");
            });
          });
        const copyCases = [
          {readback: 0, stage: 0, error: 0, name: "healthy deferred target upload"},
          {readback: 1, stage: 0, error: 0, name: "healthy deferred commitment readbacks"},
        ];
        /** @type {[number, string][]} */
        const copyFaults = [[2, "upload completion"], [7, "native completion"],
          [10, "roots submission"], [11, "noise submission"],
          [12, "key upload submission"], [13, "target upload submission"],
          [14, "first commitment readback"], [15, "second commitment readback"]];
        for (const [stage, operation] of copyFaults) {
          for (let error = 0; error < 4; ++error) {
            copyCases.push({readback: stage >= 14 ? 1 : 0, stage, error,
              name: operation + " retains owners and original " + (error % 2 ? "string" : "exception") +
              (error >= 2 ? " despite cleanup failure" : "")});
          }
        }
        await runGroupedCases(
          t, copyCases,
          ({stage}) => stage === 0 ? "Healthy deferred copies" : stage <= 7 ? "Upload and native completion faults" : stage <= 13 ? "Queued upload submission faults" : "Commitment readback faults",
          async (caseTest, {readback, stage, error, name}) => {
            await caseTest.test(name, () => {
              const result = spawnSync(executable,
                ["copies", String(readback), String(stage), String(error)], {encoding: "utf8"});
              assert.equal(result.error, undefined);
              assert.equal(result.status, 0, result.stderr);
              assert.equal(result.signal, null);
              assert.equal(result.stdout, "PASS actual-source Pearl queued-copy lifetime\n");
            });
          });
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Autolykos CUDA pipeline selection preserves batches, overrides, and noise margins", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, () => {
      const state = readSyclSource(path.join(__dirname, "../sycl/autolykos2/state.inc"));
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-autolykos-pipeline-"));
      try {
        fs.writeFileSync(path.join(fixture, "autolykos_pipeline_selector.inc"),
          extractCppBlock(state, "struct AutolykosPipelineSelection {") + ";\n");
        const executable = path.join(fixture, "selector");
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I" + fixture, path.join(__dirname, "native/autolykos-pipeline-selection.cpp"),
          "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 2000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("ZHash CUDA scheduling selection preserves canonical proofs, retries, and timing margins", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const state = readSyclSource(path.join(root, "sycl/zhash/zhash.cpp"));
      const session = readSyclSource(path.join(root, "sycl/zhash/session.h"));
      const layout = readSyclSource(path.join(root, "sycl/zhash/session-layout.h"));
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-zhash-scheduling-"));
      try {
        fs.writeFileSync(path.join(fixture, "reports.inc"),
          extractCppBlock(layout, "enum class SessionStatus : std::uint8_t {") + ";\n" +
          extractCppBlock(session, "struct Solution {") + ";\n" +
          extractCppBlock(session, "struct RunReport {") + ";\n");
        for (const [name, anchor] of Object.entries({same_solutions: "  static bool same_solutions(",
          run: "  RunReport run(", select_cuda_schedule: "  RunReport select_cuda_schedule(",
          create_fast_session: "  void create_fast_session("})) {
          fs.writeFileSync(path.join(fixture, `${name}.inc`),
            extractCppBlock(state, anchor).replaceAll("std::chrono::", "fixture_clock::"));
        }
        fs.writeFileSync(path.join(fixture, "set_cuda_partition.inc"),
          extractCppBlock(session, "  void set_cuda_partition("));
        for (const adaptiveCuda of [false, true]) {
          const executable = path.join(fixture, adaptiveCuda ? "acpp-cuda" : "other-backend");
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", ...(adaptiveCuda ? ["-DMOM_SYCL_ADAPTIVECPP_CUDA=1"] : []), "-I" + fixture,
            path.join(__dirname, "native/zhash-schedule-selection.cpp"), "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.error, undefined);
          assert.equal(compiled.signal, null);
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 2000});
          assert.equal(result.error, undefined);
          assert.equal(result.signal, null);
          assert.equal(result.status, 0, result.stderr);
          assert.equal(result.stdout, "PASS actual-source ZHash scheduling selection\n");
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Equihash192 CUDA arena selection compares full proofs and preserves fault boundaries", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, () => {
      const session = readSyclSource(path.join(__dirname,
        "../sycl/equihash192-7/session.h"));
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-e192-arena-selection-"));
      try {
        const source = path.join(fixture, "fixture.cpp");
        fs.copyFileSync(path.join(__dirname, "native/equihash192-arena-selection.cpp"), source);
        for (const name of ["run_with_cache", "select_cuda_arena"]) {
          const member = extractCppBlock(session, `  std::vector<Solution> ${name}(`);
          fs.writeFileSync(path.join(fixture, `${name}.inc`), member.replaceAll("std::chrono::", "fixture_clock::"));
        }
        for (const adaptiveCuda of [false, true]) {
          const executable = path.join(fixture, adaptiveCuda ? "acpp-cuda" : "other-backend");
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", ...(adaptiveCuda ? ["-DMOM_SYCL_ADAPTIVECPP_CUDA=1"] : []),
            source, "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.error, undefined);
          assert.equal(compiled.status, 0, compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 2000});
          assert.equal(result.error, undefined);
          assert.equal(result.signal, null);
          assert.equal(result.status, 0, result.stderr);
          assert.equal(result.stdout, "PASS actual-source Equihash192 arena selection\n");
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Octopus DAG faults propagate without light-search fallback", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "..");
      const octopus = readSyclSource(path.join(root, "sycl/octopus/octopus.cpp"));
      const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
      const library = fs.readFileSync(path.join(root, "sycl/lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const catchStart = execution.indexOf("      } catch (const std::string& err) {",
        execution.indexOf("switch (m_dev)"));
      const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
      const callerStart = octopus.indexOf("int octopus(unsigned,");
      assert.ok(catchStart > 0 && catchEnd > catchStart && callerStart > 0);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-octopus-dag-fault-"));
      const source = path.join(fixture, "octopus-dag-fault.cpp");
      const executable = path.join(fixture, "octopus_dag_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/octopus-dag-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "state.inc"),
          extractCppBlock(octopus, "  template <typename T> void free_ptr") + "\n" +
        extractCppBlock(octopus, "  void ensure_dag(") + "\n" +
        extractCppBlock(octopus, "  void ensure_epoch(") + "\n");
        const hits = fs.readFileSync(path.join(root, "sycl/pow-hits.h"), "utf8");
        fs.writeFileSync(path.join(fixture, "result.inc"),
          hits.slice(hits.indexOf("template <typename Word = uint8_t>"), hits.indexOf("// Claim the next hit slot")) +
          hits.slice(hits.indexOf("// Copy completed hits")) + "using Result = PowHits<>;\n");
        fs.writeFileSync(path.join(fixture, "caller.inc"), octopus.slice(callerStart));
        fs.copyFileSync(path.join(root, "sycl/octopus/validation.h"), path.join(fixture, "validation.inc"));
        fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_PORTABLE_OPENCL=1", source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const modes = ["full DAG/cache/release", "unsupported memory", "null allocation",
          "allocation std::exception", "allocation std::string", "allocation unknown",
          "first build", "second build", "first completion", "second completion",
          "unresolved memory query", "completion fault plus cleanup wait fault"];
        for (const scalar of [0, 1]) {
          await runGroupedCases(
            t, Array.from(modes.entries()),
            ([mode]) => (scalar ? "Scalar " : "Batched ") + (mode < 6 ? "DAG allocation and availability" : "DAG build, completion, and memory query"),
            async (caseTest, [mode, name]) => {
              const kinds = mode >= 6 ? [0, 1, 2] : [mode >= 3 ? mode - 3 : 0];
              for (const kind of kinds) {
                const error = mode >= 6 ? ": " + ["std::exception", "std::string", "unknown"][kind] : "";
                await caseTest.test((scalar ? "scalar: " : "batched: ") + name + error, () => {
                  const result = spawnSync(executable, [String(mode), String(kind), String(scalar)],
                    {encoding: "utf8", timeout: 2000});
                  assert.equal(result.error, undefined);
                  assert.equal(result.signal, null);
                  assert.equal(result.status, 0, result.stderr);
                  assert.equal(result.stdout, "PASS actual-source Octopus DAG fault boundary\n");
                });
              }
            });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Etchash submitted probe faults do not select a fallback search", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const state = fs.readFileSync(path.join(root, "etchash/state.inc"), "utf8");
      const entry = fs.readFileSync(path.join(root, "etchash/etchash.cpp"), "utf8");
      const device = fs.readFileSync(path.join(root, "etchash/device.inc"), "utf8");
      const hits = fs.readFileSync(path.join(root, "pow-hits.h"), "utf8");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-etchash-probe-fault-"));
      const source = path.join(fixture, "etchash-probe-fault.cpp");
      const executable = path.join(fixture, "etchash_probe_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/etchash-probe-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "probe.inc"),
          extract(state, "  bool use_inline_pair()", "  explicit State("));
        fs.writeFileSync(path.join(fixture, "result.inc"),
          extract(hits, "template <typename Word = uint8_t>", "// Claim the next hit slot") +
          hits.slice(hits.indexOf("// Copy completed hits")) +
          extract(device, "struct EtchashResult : PowHits<> {", "struct Uint2 {"));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "selection.inc"),
          extract(entry, "    sycl_wait_and_throw(\n        state.use_inline_pair()", "\n  } else {"));
        fs.writeFileSync(path.join(fixture, "commit.inc"),
          extract(entry, "  const int hits = mom_take_pow_hits", "\n}"));
        const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
          "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
          source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const cases = ["numeric match", "output mismatch", "mix mismatch", "count mismatch"].map(
          (name, scenario) => ({name, scenario, stage: 0, kind: 0}));
        cases.push({name: "forced noinline", scenario: 13, stage: 0, kind: 0},
          {name: "forced inline", scenario: 14, stage: 0, kind: 0});
        const errors = ["std::exception", "std::string", "unknown"];
        for (let stage = 1; stage <= 5; ++stage) {
          cases.push({name: "allocation " + stage + " null", scenario: 4, stage, kind: 0});
          for (const [kind, error] of errors.entries()) {
            cases.push({name: "allocation " + stage + ": " + error, scenario: 5, stage, kind});
          }
        }
        const operations = ["copy", "copy completion", "inline submit", "inline completion",
          "reference submit", "reference completion", "reference fault plus cleanup fault"];
        for (const [offset, operation] of operations.entries()) {
          const scenario = offset + 6;
          for (const [kind, error] of errors.entries()) {
            for (const stage of scenario === 6 ? [1, 2, 3] : [0]) {
              cases.push({name: operation + (stage ? " " + stage : "") + ": " + error,
                scenario, stage, kind});
            }
          }
        }
        for (const [kind, error] of errors.entries()) {
          cases.push({name: "successful probe cleanup: " + error, scenario: 15, stage: 0, kind});
        }
        await runGroupedCases(
          t, cases,
          ({scenario}) => {
            if (scenario <= 3 || scenario === 13 || scenario === 14) {return "Healthy probe selection";}
            if (scenario <= 5) {return "Allocation availability and exceptions";}
            if (scenario <= 7) {return "Host copies and transfer completion";}
            if (scenario <= 11) {return "Inline and reference kernel submission/completion";}
            return "Fault preservation across cleanup";
          },
          async (caseTest, {name, scenario, stage, kind}) => {
            await caseTest.test(name, () => {
              const result = spawnSync(executable, [String(scenario), String(stage), String(kind)],
                {encoding: "utf8", timeout: 2000});
              assert.equal(result.error, undefined);
              assert.equal(result.signal, null);
              assert.equal(result.status, 0, result.stderr);
              assert.equal(result.stdout, "PASS actual-source Etchash probe fault boundary\n");
            });
          });
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Nexa staged fields preserve fallback arithmetic and reflected target admission", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl/nexapow");
      const pipeline = fs.readFileSync(path.join(root, "sycl-pipeline.inc"), "utf8");
      const boundary = "// Precompute SHA blocks";
      assert.equal(pipeline.split(boundary).length, 2);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nexa-fields-"));
      try {
        fs.writeFileSync(path.join(fixture, "nexapow_fields.inc"),
          pipeline.slice(0, pipeline.indexOf(boundary)) + "} // namespace sycl_pipeline\n");
        fs.writeFileSync(path.join(fixture, "nexapow_field_gate.inc"), extractCppBlock(pipeline,
          "inline __attribute__((always_inline)) bool field_uses_acpp_ptx()"));
        for (const [name, defines] of Object.entries({
          ordinary: [], hip: ["MOM_SYCL_HAS_HIP"],
          "acpp-host": ["TEST_NEXAPOW_ACPP_HOST"],
          "acpp-forced-portable": ["TEST_NEXAPOW_ACPP_HOST", "MOM_NEXAPOW_FORCE_PORTABLE"],
          "dpcpp-host": ["MOM_NEXAPOW_SYCL_NATIVE_FIELD"],
        })) {
          await t.test(name, () => {
            const executable = path.join(fixture, name);
            const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra",
              "-Werror", "-pedantic", "-UNDEBUG", ...defines.map(value => "-D" + value),
              "-I" + fixture, "-I" + root, path.join(__dirname, "native/nexapow-field-arithmetic.cpp"),
              "-o", executable], {encoding: "utf8", timeout: 60000});
            assert.equal(compiled.error, undefined);
            assert.equal(compiled.status, 0, compiled.stderr);
            const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
            assert.equal(result.error, undefined);
            assert.equal(result.signal, null);
            assert.equal(result.status, 0, result.stderr);
            assert.equal(result.stdout,
              "Nexa field fallback: 52 arithmetic cases; 960 reflected admission cases passed\n");
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("NexaPoW submitted faults abort without fallback and do not latch failed setup", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/nexapow");
      const host = readSyclSource(path.join(root, "nexapow.cpp"));
      const pipeline = fs.readFileSync(path.join(root, "sycl-pipeline.inc"), "utf8");
      const library = fs.readFileSync(path.join(root, "../lib-internal.h"), "utf8");
      const section = (/** @type {string} */ source, /** @type {string} */ start,
      /** @type {string} */ end) => {
        const begin = source.indexOf(start);
        assert.ok(begin >= 0 && source.indexOf(start, begin + start.length) === -1, start);
        const stop = source.indexOf(end, begin + start.length);
        assert.ok(stop > begin, end);
        return source.slice(begin, stop);
      };
      const prefix = section(pipeline, "class NexaPowSyclSearch {", "  uint32_t search(");
      const probe = fs.readFileSync(path.join(__dirname, "../sycl/nexapow/validation.cpp"), "utf8");
      // The full parameter list selects the SYCL overload among the preprocessor alternatives.
      const caller = extractCppBlock(host, "int nexapow(unsigned, uint32_t, const uint8_t* input, " +
        "unsigned input_size, uint8_t* output, uint8_t*, uint64_t* pnonce, const uint8_t* target, " +
        "const uint8_t*, unsigned intensity, bool is_test, bool is_benchmark, const std::string& dev)");
      const support = section(host, "struct Result {", "\nstatic constexpr uint8_t") + "\n" +
      section(host, "inline bool np_equal(", "\n#ifndef MOM_NEXAPOW_HOST_TEST");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nexapow-fault-"));
      const source = path.join(fixture, "nexapow-fault.cpp");
      const executable = path.join(fixture, "nexapow_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/nexapow-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "search-prefix.inc"), prefix);
        fs.writeFileSync(path.join(fixture, "probe.inc"),
          section(probe, "void nexapow_test_sha256d_49(", "\n} // namespace mom_nexapow"));
        fs.writeFileSync(path.join(fixture, "caller.inc"), caller);
        fs.writeFileSync(path.join(fixture, "caller-support.inc"), support);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          section(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
          "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
          source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.match(result.stdout, /^# tests 211$/m);
        assert.match(result.stdout, /^# pass 211$/m);
        assert.match(result.stdout, /^# fail 0$/m);
        assert.match(result.stdout, /^# skipped 0$/m);
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("C30 asynchronous completion faults stop solver stages and Core commits", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "..");
      const c30 = fs.readFileSync(path.join(root, "sycl/c30/c30.cpp"), "utf8");
      const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
      const library = fs.readFileSync(path.join(root, "sycl/lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const catchStart = execution.indexOf("      } catch (const std::string& err) {",
        execution.indexOf("switch (m_dev)"));
      const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
      assert.ok(catchStart > 0 && catchEnd > catchStart);
      const entryStart = c30.indexOf("int c30(unsigned,");
      assert.ok(entryStart > 0);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c30-async-fault-"));
      const source = path.join(fixture, "c30-async-fault.cpp");
      const executable = path.join(fixture, "c30_async_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/c30-async-fault.cpp"), source);
        fs.copyFileSync(path.join(root, "sycl/c30/types.h"), path.join(fixture, "c30-types.inc"));
        fs.copyFileSync(path.join(root, "native/job-boundary.h"), path.join(fixture, "job-boundary.inc"));
        fs.writeFileSync(path.join(fixture, "solver.inc"),
          extract(c30, "struct Solver {", "\n} // namespace mom_c30"));
        fs.writeFileSync(path.join(fixture, "entry.inc"), c30.slice(entryStart));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "active-guard.inc"),
          extract(execution, "    if (!m_has_fn) {", "\n    { // A compute function is active"));
        fs.writeFileSync(path.join(fixture, "dispatch.inc"),
          extract(execution, "          case DEV::C30_GPU:", "          case DEV::KAWPOW_GPU:"));
        fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
        fs.writeFileSync(path.join(fixture, "accounting.inc"),
          extract(execution, "      uint64_t completed = m_batch;",
            "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU ||"));
        fs.writeFileSync(path.join(fixture, "nonce-commit.inc"),
          extract(execution, "      if (m_dev == DEV::C30_GPU) {", "      if (m_nonce_bytes == 4) {"));
        const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
          "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
          source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const stages = ["seed", "trim", "count readback", "compact", "compact readback",
          "edge readback", "recovery", "nonce readback"];
        for (const layout of [0, 1]) {
          for (const owner of [0, 1]) {
            const cases = [{kind: 0, stage: 0, name: "empty asynchronous error lists"}];
            for (const [stage, name] of stages.entries()) {
              for (const [kind, errors] of ["one SYCL", "multiple SYCL", "non-SYCL"].entries()) {
                cases.push({kind: kind + 1, stage: stage + 1, name: name + ": " + errors});
              }
            }
            await runGroupedCases(
              t, cases,
              ({stage}) => (owner ? "Core " : "Entry ") + (layout ? "packed: " : "wide: ") + (stage <= 4 ? "seed, trim, count, and compact" : "readbacks and recovery"),
              async (caseTest, {kind, stage, name}) => {
                await caseTest.test((owner ? "Core" : "entry") + (layout ? " packed: " : " wide: ") + name, () => {
                  const result = spawnSync(executable,
                    [String(kind), String(stage), String(layout), String(owner)], {encoding: "utf8"});
                  assert.equal(result.error, undefined);
                  assert.equal(result.signal, null);
                  assert.equal(result.status, 0, result.stderr);
                  assert.equal(result.stdout, "PASS actual-source C30 asynchronous completion boundary\n");
                });
              });
          }
          await t.test((layout ? "Packed" : "Wide") + " deferred-copy lifetime", {concurrency: false}, async (copyTest) => {
            for (const copy of [2, 6]) {
              for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                for (const cleanupError of [0, 1]) {
                  await copyTest.test((layout ? "packed" : "wide") + " deferred copy " + copy + ": " + error +
                (cleanupError ? " with secondary cleanup fault" : ""), () => {
                    const result = spawnSync(executable,
                      ["copies", String(copy), String(kind), String(cleanupError), String(layout)],
                      {encoding: "utf8"});
                    assert.equal(result.error, undefined);
                    assert.equal(result.signal, null);
                    assert.equal(result.status, 0, result.stderr);
                    assert.equal(result.stdout, "PASS actual-source C30 asynchronous completion boundary\n");
                  });
                }
              }
            }
          });
          await t.test((layout ? "Packed" : "Wide") + " accepted readback faults", {concurrency: false}, async (copyTest) => {
            for (const copy of [3, 4, 7]) {
              for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                for (const cleanupError of [0, 1]) {
                  await copyTest.test("copy " + copy + ": " + error + (cleanupError ? " with cleanup fault" : ""), () => {
                    const result = spawnSync(executable,
                      ["copies", String(copy), String(kind), String(cleanupError), String(layout), "after-enqueue"],
                      {encoding: "utf8", timeout: 5000});
                    assert.equal(result.error, undefined);
                    assert.equal(result.signal, null);
                    assert.equal(result.status, 0, result.stderr);
                    assert.equal(result.stdout, "PASS actual-source C30 asynchronous completion boundary\n");
                  });
                }
              }
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("C29 loop telemetry distinguishes graph starts from empty-input polls", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const execution = fs.readFileSync(path.join(__dirname, "../native/core/execution.inc"), "utf8"),
        loopStats = extractCppBlock(execution, "struct LoopStats {");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-loop-stats-")),
        source = path.join(fixture, "loop_stats.cpp"), executable = path.join(fixture, "loop_stats");
      try {
        fs.writeFileSync(source, `#include <cstdint>\nusing std::uint64_t;\n${loopStats};
int main() {
  LoopStats stats;
  stats.record_c29(64, 0);
  stats.record_c29(0, 0);
  stats.record_c29(64, 1);
  stats.record_c29(0, -1);
  stats.record_c29(0, 2);
  if (stats.c29_input_graphs != 1 || stats.c29_empty_polls != 1 || stats.c29_proofs != 1 ||
      stats.c29_drained != 1 || stats.c29_other != 1 || !stats.has_c29_activity()) return 1;
  LoopStats reset{};
  if (reset.has_c29_activity() || reset.c29_input_graphs || reset.c29_empty_polls ||
      reset.c29_proofs || reset.c29_drained || reset.c29_other) return 2;
  return 0;
}\n`);
        const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
          "-pedantic", source, "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
      } finally {fs.rmSync(fixture, {recursive: true, force: true});}
    });

    test("C29 profile mode samples once and keeps host timing separate from device event profiling", {
      skip: process.platform === "win32" ? "requires a POSIX host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, ".."), state = fs.readFileSync(path.join(root, "sycl/c29/state.inc"), "utf8"),
        modeEnum = extractCppBlock(state, "enum class C29ProfileMode {"),
        modeFunction = extractCppBlock(state, "static C29ProfileMode c29_profile_mode()"),
        profileEvent = extractCppBlock(state, "struct C29ProfileEvent {"),
        profile = extractCppBlock(state, "struct C29Profile {");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-profile-mode-")),
        source = path.join(fixture, "profile_mode.cpp"), executable = path.join(fixture, "profile_mode");
      try {
        // Stub only the SYCL device/event surface; count timestamp queries to observe device profiling.
        fs.writeFileSync(source, `#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace sycl {
struct exception { const char* what() const { return "unsupported"; } };
namespace info {
namespace device { struct name {}; }
namespace platform { struct name {}; }
namespace event_profiling { struct command_start {}; struct command_end {}; }
}
struct platform { template <typename T> std::string get_info() const { return "platform"; } };
struct device {
  template <typename T> std::string get_info() const { return "device"; }
  platform get_platform() const { return {}; }
};
struct event {
  static int queries;
  template <typename T> uint64_t get_profiling_info() const { ++queries; return 0; }
};
int event::queries = 0;
}
constexpr unsigned COMPUTE_THREADS = 1;
static uint64_t c29_now_us() { return 0; }
static uint32_t c29_profile_limit() { return UINT32_MAX; }
static uint32_t c29_trim_rounds(const sycl::device&) { return 1; }
static uint32_t c29_seed_local_size() { return 64; }
static uint32_t c29_seed_blocks_per_item() { return 4; }
${modeEnum};
${modeFunction}
${profileEvent};
${profile};
int main() {
  const C29ProfileMode mode = c29_profile_mode();
  setenv("MOM_C29_PROFILE", mode == C29ProfileMode::device ? "0" : "1", 1);
  if (c29_profile_mode() != mode) return 3;
  C29Profile profile(1, 2, 32, sycl::device{});
  profile.mark("probe");
  profile.add_event("seed", sycl::event{});
  profile.print_events();
  std::fprintf(stderr, "mode=%s enabled=%d events=%zu queries=%d\\n",
               mode == C29ProfileMode::disabled ? "disabled" : mode == C29ProfileMode::host ? "host" : "device",
               profile.enabled ? 1 : 0, profile.events.size(), sycl::event::queries);
  return 0;
}
`);
        const compiled = spawnSync("c++", ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
          "-pedantic", source, "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);

        const disabled = "mode=disabled enabled=0 events=0 queries=0\n";
        const host = "mode=host enabled=1 events=0 queries=0\n";
        const device = "mode=device enabled=1 events=1 queries=2\n";
        const cases = [
          {name: "unset", value: undefined, expected: disabled},
          {name: "empty", value: "", expected: disabled},
          {name: "zero", value: "0", expected: disabled},
          {name: "leading zero", value: "0legacy", expected: disabled},
          {name: "host", value: "host", expected: host},
          {name: "device numeric", value: "1", expected: device},
          {name: "legacy normal", value: "normal", expected: device},
          {name: "legacy flag", value: "legacy", expected: device},
          {name: "other nonzero", value: "other", expected: device},
        ];
        for (const fixtureCase of cases) {
          const env = {...process.env};
          if (fixtureCase.value === undefined) {delete env["MOM_C29_PROFILE"];} else {env["MOM_C29_PROFILE"] = fixtureCase.value;}
          const result = spawnSync(executable, [], {encoding: "utf8", env, timeout: 5000});
          assert.equal(result.error, undefined, `${fixtureCase.name}: ${result.error?.message}`);
          assert.equal(result.signal, null, `${fixtureCase.name}: ${result.stderr}`);
          assert.equal(result.status, 0, `${fixtureCase.name}: ${result.stderr}`);
          assert.equal(result.stderr, fixtureCase.expected, `${fixtureCase.name}: mode selection`);
          // Host and device modes both emit bounded host timing marks; disabled mode prints nothing.
          assert.equal(result.stdout.includes("[c29-profile] host probe"), fixtureCase.expected !== disabled,
            `${fixtureCase.name}: host timing`);
          assert.equal(result.stdout.includes("[c29-profile] device seed"), fixtureCase.expected === device,
            `${fixtureCase.name}: device timestamps`);
        }
      } finally {fs.rmSync(fixture, {recursive: true, force: true});}
    });

    test("C29 submitted graph faults propagate before cycle search or accounting", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "..");
      const search = fs.readFileSync(path.join(root, "sycl/c29/search.inc"), "utf8");
      const entry = fs.readFileSync(path.join(root, "sycl/c29/c29.cpp"), "utf8");
      const state = fs.readFileSync(path.join(root, "sycl/c29/state.inc"), "utf8");
      const trim = fs.readFileSync(path.join(root, "sycl/c29/trim.inc"), "utf8");
      const execution = fs.readFileSync(path.join(root, "native/core/execution.inc"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const completeSearch = search.slice(search.indexOf("struct C29SearchCompletionGuard {"));
      assert.ok(completeSearch.startsWith("struct C29SearchCompletionGuard {"));
      assert.equal(completeSearch.split("std::async(").length, 2);
      const catchStart = execution.indexOf("      } catch (const std::string& err) {",
        execution.indexOf("switch (m_dev)"));
      const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
      assert.ok(catchStart > 0 && catchEnd > catchStart);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-submitted-fault-"));
      const source = path.join(fixture, "c29-submitted-fault.cpp");
      const executable = path.join(fixture, "c29_submitted_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/c29-submitted-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "search.inc"),
          completeSearch.replace("std::async(", "TestAsync(") + "\n#include \"entry.inc\"\n");
        fs.writeFileSync(path.join(fixture, "entry.inc"), extractCppBlock(entry, "int c29("));
        fs.writeFileSync(path.join(fixture, "result-state.inc"),
          extract(state, "struct C29QueuedSolution {", "static sycl::async_handler c29_exception_handler()"));
        fs.writeFileSync(path.join(fixture, "activate.inc"), state.slice(state.indexOf("static void c29_activate_job(")));
        fs.copyFileSync(path.join(root, "sycl/c29/cycle.h"), path.join(fixture, "cycle.inc"));
        fs.copyFileSync(path.join(root, "native/job-boundary.h"), path.join(fixture, "job-boundary.inc"));
        fs.writeFileSync(path.join(fixture, "seed.inc"), "(void)c29_buffers;\n++seeds;\nif (mode == 2) inject();\n");
        fs.writeFileSync(path.join(fixture, "trim.inc"),
          "++trims;\nif (mode == 3) inject();\n" +
        "const sycl::event tail_event{};\nuint32_t trimmed_edge_count = 0;\n" +
        "sycl::buffer<uint32_t, 1> buffer_trimmed_edge_count{sycl::range<1>{1}};\n" +
        "sycl::buffer<sycl::uint2, 1> buffer_trimmed_edges_u2{sycl::range<1>{64}};\n" +
        extract(trim, "profile.mark(\"before tail wait\");", "profile.finish(trimmed_edge_count);")
          .replace("std::vector<sycl::uint2> trimmed_edges", "GraphEdges trimmed_edges") +
        "profile.finish(trimmed_edge_count);\n");
        fs.writeFileSync(path.join(fixture, "active-guard.inc"),
          extract(execution, "    if (!m_has_fn) {", "\n    { // A compute function is active"));
        // Keep this fault-only fixture independent of the separately compiled telemetry hook.
        const dispatch = extract(execution, "          case DEV::C29_GPU:", "          case DEV::C30_GPU:")
          .replace(/\n\s*if \(loop_stats\) \{\s*ls\.record_c29\(m_input_len, dev_sols\);\s*\}/, "");
        fs.writeFileSync(path.join(fixture, "dispatch.inc"), dispatch);
        fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
        fs.writeFileSync(path.join(fixture, "accounting.inc"),
          extract(execution, "      uint64_t completed = m_batch;",
            "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU ||"));
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const modes = ["valid graph", "device lookup", "seed submission", "trim submission",
          "tail completion", "count readback", "edge readback", "legacy thread-start failure"];
        for (const proof of [32, 42]) {
          await t.test("Proof size " + proof, {concurrency: false}, async (proofTest) => {
            for (const owner of [0, 1]) {
              for (const [mode, name] of modes.entries()) {
                await proofTest.test((owner ? "Core" : "entry") + " proof" + proof + ": " + name, () => {
                  const result = spawnSync(executable, [String(mode), String(proof), String(owner)],
                    {encoding: "utf8"});
                  assert.equal(result.error, undefined);
                  assert.equal(result.signal, null);
                  assert.equal(result.status, 0, result.stderr);
                  assert.match(result.stdout, /PASS actual-source C29 submitted fault boundary\n$/);
                });
              }
            }

          });
        }
        await t.test("Proof recovery completion", {concurrency: false}, async (recoveryTest) => {
          for (const proof of [32, 42]) {
            for (const [mode, name] of [[8, " waits before readback"], [9, " fault preserves output"],
              [10, " uses the cached device"]]) {
              await recoveryTest.test("proof" + proof + name, () => {
                const result = spawnSync(executable, [String(mode), String(proof), "0"], {encoding: "utf8"});
                assert.equal(result.status, 0, result.stderr);
                assert.match(result.stdout, /PASS actual-source C29 submitted fault boundary\n$/);
              });
            }
          }
          const wait = "sycl_wait_and_throw(recovery_event, state.device);";
          assert.ok(entry.includes(wait));
          fs.writeFileSync(path.join(fixture, "entry.inc"),
            extractCppBlock(entry, "int c29(").replace(wait, "(void)recovery_event;"));
          const negative = spawnSync("c++", ["-std=c++17", "-O2", source, "-o", executable], {encoding: "utf8"});
          assert.equal(negative.status, 0, negative.stderr);
          const result = spawnSync(executable, ["8", "32", "0"], {encoding: "utf8"});
          assert.equal(result.status, 1);
          assert.match(result.stderr, /recovery readback bypassed the low-CPU completion boundary/);
          fs.writeFileSync(path.join(fixture, "entry.inc"),
            extractCppBlock(entry, "int c29(").replace(wait,
              "sycl_wait_and_throw(recovery_event, compute_queue.get_device());"));
          const queryControl = spawnSync("c++", ["-std=c++17", "-O2", source, "-o", executable], {encoding: "utf8"});
          assert.equal(queryControl.status, 0, queryControl.stderr);
          const queried = spawnSync(executable, ["10", "32", "0"], {encoding: "utf8"});
          assert.equal(queried.status, 1);
          assert.match(queried.stderr, /recovery fault was swallowed/);
          fs.writeFileSync(path.join(fixture, "entry.inc"), extractCppBlock(entry, "int c29("));
          const move = "trimmed_edges = std::move(trimmed_edges),";
          assert.ok(completeSearch.includes(move));
          fs.writeFileSync(path.join(fixture, "search.inc"),
            completeSearch.replace("std::async(", "TestAsync(").replace(move, "") + "\n#include \"entry.inc\"\n");
          const copyControl = spawnSync("c++", ["-std=c++17", "-O2", source, "-o", executable], {encoding: "utf8"});
          assert.equal(copyControl.status, 0, copyControl.stderr);
          const copied = spawnSync(executable, ["0", "32", "0"], {encoding: "utf8"});
          assert.equal(copied.status, 1);
          assert.match(copied.stderr, /cycle thread copied the retired graph/);
        });
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("HooHash and Beam nested readbacks retain local owners through faults", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl");
      const hoo = fs.readFileSync(path.join(root, "hoohash/hoohash.cpp"), "utf8");
      const beam = fs.readFileSync(path.join(root, "beamhash3/beamhash3.cpp"), "utf8");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const results = [...hoo.matchAll(/Result result;\s+MomSyclHostTransferGuard result_transfer[^\n]+\n\s*sycl_wait_and_throw\(state\.queue\.memcpy\(&result\.offset,[^\n]+/g)];
      assert.equal(results.length, 2, "both test and mining readbacks need local ownership");
      const start = beam.indexOf("std::vector<uint32_t> ns(NBUCKETS);");
      const end = beam.indexOf("for (uint32_t v : ns)", start);
      assert.ok(start >= 0 && end > start);
      const routes = [
        ...results.map(match => ({id: 1, entry: match[0]})),
        {id: 2, entry: beam.slice(start, end).replace("std::vector", "TrackedVector")},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nested-local-readback-"));
      try {
        const source = path.join(fixture, "readback.cpp"), executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/nested-readback-lifetime.cpp"), source);
        const cleanupStart = library.indexOf("inline void sycl_log_cleanup_exception(");
        const cleanupEnd = library.indexOf("sycl::device get_dev(", cleanupStart);
        assert.ok(cleanupStart >= 0 && cleanupEnd > cleanupStart);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), library.slice(cleanupStart, cleanupEnd));
        for (const route of routes) {
          const guard = /\s*MomSyclHostTransferGuard (?:result_transfer|bucket_count_transfer)[^\n]+\n/;
          assert.match(route.entry, guard);
          for (const fixed of [false, true]) {
            fs.writeFileSync(path.join(fixture, "entry.inc"), fixed ? route.entry : route.entry.replace(guard, "\n"));
            const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
              "-DMOM_NESTED_ROUTE=" + route.id, source, "-o", executable], {encoding: "utf8"});
            assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
            for (const [fault, secondary] of [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]) {
              const result = spawnSync(executable, [String(fault), String(secondary)],
                {encoding: "utf8", timeout: 5000});
              assert.equal(result.error, undefined);
              assert.equal(result.status, !fixed && fault ? 1 : 0, result.stdout + result.stderr);
              if (!fixed && fault) {
                assert.match(result.stderr, /nested readback left an accepted copy pending/);
              } else {
                assert.match(result.stdout, /^PASS nested readback route=/);
              }
            }
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("GPU cache uploads retain local sources through faults", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cache-upload-lifetime-"));
      const source = path.join(fixture, "cache-upload-lifetime.cpp");
      const executable = path.join(fixture, "probe");
      const cleanupStart = library.indexOf("inline void sycl_log_cleanup_exception(");
      const cleanupEnd = library.indexOf("sycl::device get_dev(", cleanupStart);
      assert.ok(cleanupStart >= 0 && cleanupEnd > cleanupStart);
      const removeGuards = /^ *MomSyclHostTransferGuard[^\n]+\n/gm;
      const routes = [
        {name: "Wala matrix", file: "walahash/walahash.cpp", owner: "    uint8_t matrix[4096];",
          start: "    MomSyclHostTransferGuard matrix_transfers(",
          end: "    std::memcpy(state.matrix_seed, input, 32);", shared: false},
        {name: "Wala tiled matrix", file: "walahash/walahash.cpp", owner: "    uint8_t matrix[4096];",
          start: "    MomSyclHostTransferGuard matrix_transfers(",
          end: "    std::memcpy(state.matrix_seed, input, 32);", shared: false, tiled: true},
        {name: "Etchash", file: "etchash/state.inc", owner: "    std::vector<uint32_t> host_cache;",
          start: "    MomSyclHostTransferGuard host_cache_transfers(",
          end: "    const uint32_t light_nodes", shared: true},
        {name: "KawPow", file: "kawpow/state.inc", owner: "    std::vector<uint32_t> host_cache;",
          start: "    MomSyclHostTransferGuard host_cache_transfers(",
          end: "    const uint32_t light_nodes", shared: true},
        {name: "FishHash", file: "fishhash/state.inc", owner: "      std::vector<uint64_t> hcache;",
          start: "      MomSyclHostTransferGuard host_cache_transfers(",
          end: "      cache_built = true;", shared: true},
        {name: "Octopus points", file: "octopus/state.inc",
          owner: "    std::array<uint32_t, POLYNOMIAL_SIZE> host_points;",
          start: "    MomSyclHostTransferGuard host_points_transfers(",
          end: "    std::memcpy(points_header.data(), next_header, HASH_LEN);", shared: true},
        {name: "Octopus cache", file: "octopus/state.inc",
          owner: "    std::vector<uint32_t> host_cache(static_cast<size_t>(bytes / sizeof(uint32_t)));",
          start: "    MomSyclHostTransferGuard host_cache_transfers(",
          end: "    cache_nodes = nodes;", shared: true},
        {name: "Verthash", file: "verthash/verthash.cpp", owner: "    std::vector<Uint2> host;",
          start: "    MomSyclHostTransferGuard host_data_transfers(",
          end: "\n  }\n};", shared: true},
      ];
      try {
        fs.copyFileSync(path.join(__dirname, "native/cache-upload-lifetime.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), library.slice(cleanupStart, cleanupEnd));
        for (const route of routes) {
          await t.test(route.name, {concurrency: false}, () => {
            const entrySource = fs.readFileSync(path.join(root, route.file), "utf8");
            const owner = entrySource.indexOf(route.owner);
            const guard = entrySource.indexOf(route.start, owner);
            assert.ok(owner >= 0 && guard > owner, route.file + ": source owner must precede guard");
            const end = entrySource.indexOf(route.end, guard);
            assert.ok(end > guard, route.file + ": upload span end");
            const entry = entrySource.slice(guard, end);
            assert.ok(entry.includes("queue.memcpy"), route.name + ": extracted copy");
            assert.ok(entry.includes("MomSyclHostTransferGuard"), route.name + ": extracted guard");
            const baseline = entry.replace(removeGuards, "");
            assert.notEqual(baseline, entry, route.name + ": negative control removed no guard");
            for (const fixed of [false, true]) {
              fs.writeFileSync(path.join(fixture, "cache_scope.inc"), fixed ? entry : baseline);
              const args = ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-DMOM_CACHE_GUARDED=" + (fixed ? 1 : 0),
                "-DMOM_CACHE_HAS_SHARED=" + (route.shared ? 1 : 0),
                "-DMOM_CACHE_WALA_TILED=" + (route.tiled ? 1 : 0)];
              if (route.tiled) {
                args.push("-DWALAHASH_ESIMD=1");
              }
              args.push(source, "-o", executable);
              const compiled = spawnSync("c++", args, {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.signal, null);
              assert.equal(compiled.status, 0, compiled.stderr);
              const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
              assert.equal(result.error, undefined);
              assert.equal(result.signal, null);
              assert.equal(result.status, fixed ? 0 : 1, result.stdout + result.stderr);
              if (fixed) {
                assert.equal(result.stdout, "PASS actual-source cache upload lifetime and exception preservation\n");
              } else {
                assert.equal(result.stderr, "FAIL accepted cache upload outlived its local source\n");
              }
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("C29 readback retires copies accepted before a submission throws", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl");
      const state = fs.readFileSync(path.join(root, "c29/state.inc"), "utf8");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-readback-"));
      try {
        const source = path.join(fixture, "readback.cpp"), executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/c29-buffer-readback.cpp"), source);
        const start = library.indexOf("inline void sycl_log_cleanup_exception(");
        const end = library.indexOf("sycl::device get_dev(", start);
        assert.ok(start >= 0 && end > start);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), library.slice(start, end));
        const readback = extractCppBlock(state, "static void c29_read_buffer(");
        const guard = '  MomSyclHostTransferGuard host_transfers(queue, "c29 buffer readback");';
        assert.ok(readback.includes(guard));
        for (const fixed of [false, true]) {
          fs.writeFileSync(path.join(fixture, "readback.inc"), fixed ? readback : readback.replace(guard, ""));
          const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
            source, "-o", executable], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          assert.equal(result.status, fixed ? 0 : 1, result.stdout + result.stderr);
          if (fixed) {
            assert.equal(result.stdout, "PASS actual-source C29 accepted readback lifetime\n");
          } else {
            assert.match(result.stderr, /C29 accepted readback outlived its caller/);
          }
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("C29 cleanup joins owned host searches past their result publication", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl/c29");
      const state = fs.readFileSync(path.join(root, "state.inc"), "utf8");
      const search = fs.readFileSync(path.join(root, "search.inc"), "utf8");
      const destructor = extractCppBlock(state, "  ~C29State()");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-lifetime-"));
      try {
        const source = path.join(fixture, "lifetime.cpp"), executable = path.join(fixture, "probe");
        fs.copyFileSync(path.join(__dirname, "native/c29-search-lifetime.cpp"), source);
        fs.writeFileSync(path.join(fixture, "result-state.inc"),
          extractCppBlock(state, "struct C29QueuedSolution {") + ";\n" +
          extractCppBlock(state, "struct C29ResultState {") + ";\n");
        fs.writeFileSync(path.join(fixture, "guard.inc"),
          extractCppBlock(search, "struct C29SearchCompletionGuard {") + ";\n");
        const local = search.match(/^\s*C29SearchCompletionGuard active_completion_guard[^\n]+/m);
        assert.ok(local);
        fs.writeFileSync(path.join(fixture, "local-guard.inc"), local[0] + "\n");
        // Run the actual destructor body after the fixture marks cleanup as started.
        fs.writeFileSync(path.join(fixture, "drain.inc"),
          destructor.slice(destructor.indexOf("{") + 1, destructor.lastIndexOf("}")));
        const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
          "-pthread", "-fsanitize=address", "-fno-omit-frame-pointer",
          ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []), source, "-o", executable],
        {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.equal(result.stdout, "PASS actual-source C29 search lifetime\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Host upload guards retire accepted copies before caller-owned buffers", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const cleanup = extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev(");
      const routes = [
        {name: "cn/gpu", file: "cn-gpu/cn-gpu.cpp", end: "  // Initial Keccak", uploads: 1},
        {name: "Etchash", file: "etchash/etchash.cpp", end: "  sycl::queue& q = state.queue;", uploads: 2},
        {name: "KawPow", file: "kawpow/kawpow.cpp", end: "  sycl::queue& q = state.queue;", uploads: 1},
        {name: "FishHash", file: "fishhash/fishhash.cpp", end: "  sycl::event search;", uploads: 2},
        {name: "KarlsenHashV2", file: "fishhash/fishhash.cpp", scope: "int karlsenhashv2(",
          end: "#if !defined(MOM_SYCL_PORTABLE_OPENCL)\n  const bool cooperative =",
          uploads: 2, focused: true, shared: true},
        {name: "Nexa monolithic", file: "nexapow/nexapow.cpp", scope: "  state.init_portable();",
          end: "  state.queue.memset(state.result,", uploads: 3, focused: true},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-host-transfer-fault-"));
      const source = path.join(fixture, "host-transfer-fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host-transfer-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), cleanup);
        for (const route of routes) {
          await t.test(route.name, async (routeTest) => {
            const entry = fs.readFileSync(path.join(root, route.file), "utf8");
            const scope = route.scope ? entry.slice(entry.indexOf(route.scope)) : entry;
            if (route.scope) {
              assert.ok(entry.includes(route.scope), route.scope);
            }
            const upload = extract(scope, "  MomSyclHostTransferGuard ", route.end);
            fs.writeFileSync(path.join(fixture, "upload.after.inc"), upload);
            // The negative removes only the fix from the otherwise identical real upload scope.
            fs.writeFileSync(path.join(fixture, "upload.before.inc"),
              upload.replace(/^\s*MomSyclHostTransferGuard[^\n]+\n/, ""));
            for (const guarded of [0, 1]) {
              const executable = path.join(fixture, "host_transfer_fault_" + guarded);
              const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
                "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
                "-DMOM_HOST_TRANSFER_WITH_GUARD=" + guarded,
                "-DMOM_HOST_TRANSFER_UPLOADS=" + route.uploads, source, "-o", executable],
              {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.signal, null);
              assert.equal(compiled.status, 0, compiled.stderr);
              const cases = [];
              if (route.focused) {
                for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                  cases.push({stage: 4, kind, secondary: kind ? 1 : 0, mode: 0,
                    name: "second upload: " + error + (kind ? " and cleanup fault" : "")});
                }
                cases.push({stage: 1, kind: 0, secondary: 1, mode: 0,
                  name: "first submit and cleanup fault"});
              } else {
                for (const [stage, operation] of ["first submit", "later submit", "device query"].entries()) {
                  for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                    for (const secondary of [0, 1]) {
                      cases.push({stage: stage + 1, kind, secondary, mode: 0,
                        name: operation + ": " + error + (secondary ? " and cleanup fault" : "")});
                    }
                  }
                }
              }
              cases.push({stage: 0, kind: 0, secondary: 0, mode: 0, name: "device success wait trace"});
              if (!route.focused || route.shared) {
                cases.push({stage: 0, kind: 0, secondary: 0, mode: 1, name: "shared success wait trace"});
              }
              if (route.name === "cn/gpu") {
                cases.push({stage: 0, kind: 0, secondary: 0, mode: 2, name: "portable success wait trace"});
              }
              await runGroupedCases(
                routeTest, cases,
                ({stage}) => (guarded ? "Fixed " : "Baseline ") + (stage ? "upload submission and device query faults" : "success wait traces"),
                async (caseTest, {stage, kind, secondary, mode, name}) => {
                  await caseTest.test((guarded ? "fixed: " : "baseline: ") + name, () => {
                    const result = spawnSync(executable,
                      [String(stage), String(kind), String(secondary), String(mode)],
                      {encoding: "utf8", timeout: 2000});
                    assert.equal(result.error, undefined);
                    assert.equal(result.signal, null);
                    if (!guarded && stage) {
                      assert.equal(result.status, 1, result.stdout + result.stderr);
                      assert.equal(result.stderr, "FAIL accepted transfer outlived its host owner\n");
                    } else {
                      assert.equal(result.status, 0, result.stderr);
                      assert.equal(result.stdout, "PASS actual-source host-transfer lifetime\n");
                    }
                  });
                });
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Changed HooHash and WalaHash matrices invalidate readiness before fallible preparation", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-matrix-readiness-fault-"));
      const source = path.join(fixture, "host-transfer-fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host-transfer-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        for (const [hoo, name] of ["walahash", "hoohash"].entries()) {
          await t.test(name, async (matrixTest) => {
            const host = readSyclSource(path.join(root, name, name + ".cpp"));
            const matrix = hoo
              ? extract(host, "  std::unique_ptr<FloatPair[]> normal;", "  state.queue.memcpy(state.input")
              : extract(host, "  MomSyclHostTransferGuard ", "\n") + "\n" +
              extract(host, "  if (!state.matrix_ready || std::memcmp(state.matrix_seed, input, 32)) {",
                "  state.queue.memset(state.result");
            for (const fixed of [0, 1]) {
              fs.writeFileSync(path.join(fixture, "matrix.inc"), fixed ? matrix :
                matrix.replace("    state.matrix_ready = false;\n", ""));
              const executable = path.join(fixture, "matrix_fault_" + fixed);
              const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
                "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
                "-DMOM_MATRIX_FIXTURE=1", "-DMOM_MATRIX_HOO=" + hoo, source, "-o", executable],
              {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.signal, null);
              assert.equal(compiled.status, 0, compiled.stderr);
              const cases = [{stage: 0, kind: 0, secondary: 0, name: "success, old-header change and cache hit"}];
              for (const [stage, operation] of ["preparation", "copy submission", "copy completion"].entries()) {
                for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                  for (const secondary of [0, 1]) {
                    cases.push({stage: stage + 1, kind, secondary,
                      name: operation + ": " + error + (secondary ? " and cleanup fault" : "")});
                  }
                }
              }
              await runGroupedCases(
                matrixTest, cases,
                () => fixed ? "Fixed readiness invalidation" : "Baseline stale readiness",
                async (caseTest, {stage, kind, secondary, name}) => {
                  await caseTest.test((fixed ? "fixed: " : "baseline: ") + name, () => {
                    const result = spawnSync(executable, [String(stage), String(kind), String(secondary)],
                      {encoding: "utf8", timeout: 2000});
                    assert.equal(result.error, undefined);
                    assert.equal(result.signal, null);
                    if (!fixed && stage) {
                      assert.equal(result.status, 1, result.stdout + result.stderr);
                      assert.equal(result.stderr, "FAIL failed changed header kept the previous matrix ready\n");
                    } else {
                      assert.equal(result.status, 0, result.stderr);
                      assert.equal(result.stdout, "PASS actual-source matrix readiness and retry\n");
                    }
                  });
                });
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("ZHash nested readbacks retire before catches convert device errors", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const session = fs.readFileSync(path.join(root, "zhash/session.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const fastReadback = extract(session, "      FastReadback fallback_fast_readback{};",
        "      if (options_.collect_stage_times)\n        for (");
      const tailReadback = extract(session, "        std::vector<std::uint32_t> leaves(",
        "        recover_roots(report, leaves.data(), valid.data());");
      const routes = [
        {name: "fast", tail: false, guard: "fast_readback_guard", copies: 2, failCopy: 2, body: fastReadback},
        {name: "tail", tail: true, guard: "tail_readback", copies: 4, failCopy: 4,
          body: fastReadback + tailReadback},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nested-readback-fault-"));
      const source = path.join(fixture, "host-transfer-fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host-transfer-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "nested-fast-readback.inc"),
          extractCppBlock(session, "  struct FastReadback {") + ";\n");
        fs.writeFileSync(path.join(fixture, "converted-catch.inc"),
          extract(session, "    } catch (const sycl::exception& error) {", "\nprivate:"));
        fs.writeFileSync(path.join(fixture, "error-report.inc"),
          extract(session, "  RunReport& device_error_report(", "  void recover_roots("));
        for (const route of routes) {
          await t.test(route.name, async (readbackTest) => {
            for (const fixed of [0, 1]) {
              fs.writeFileSync(path.join(fixture, "nested.inc"), fixed ? route.body :
                route.body.replace(new RegExp(`^\\s*MomSyclHostTransferGuard ${route.guard}[^\\n]+\\n`, "m"), ""));
              const executable = path.join(fixture, "nested_fault_" + fixed);
              const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
                "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
                "-DMOM_NESTED_FIXTURE=1", "-DMOM_NESTED_FAIL_COPY=" + route.failCopy,
                "-DMOM_NESTED_COPIES=" + route.copies, "-DMOM_NESTED_TAIL=" + Number(route.tail),
                source, "-o", executable], {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.signal, null);
              assert.equal(compiled.status, 0, compiled.stderr);
              const cases = [{stage: 0, kind: 0, secondary: 0, name: "success wait trace"}];
              for (const [kind, error] of ["std::exception conversion", "std::string", "unknown"].entries()) {
                for (const secondary of [0, 1]) {
                  cases.push({stage: 1, kind, secondary,
                    name: error + (secondary ? " and cleanup fault" : "")});
                }
              }
              await runGroupedCases(
                readbackTest, cases,
                () => fixed ? "Fixed nested readback lifetime" : "Baseline nested readback lifetime",
                async (caseTest, {stage, kind, secondary, name}) => {
                  await caseTest.test((fixed ? "fixed: " : "baseline: ") + name, () => {
                    const result = spawnSync(executable, [String(stage), String(kind), String(secondary)],
                      {encoding: "utf8", timeout: 2000});
                    assert.equal(result.error, undefined);
                    if (!fixed && stage) {
                      if (result.signal || /AddressSanitizer/.test(result.stderr)) {
                        assert.match(result.stderr, /AddressSanitizer: heap-use-after-free/);
                      } else {
                        assert.equal(result.status, 1, result.stdout + result.stderr);
                        assert.equal(result.stderr, "FAIL nested readback left its try-local destination pending\n");
                      }
                    } else {
                      assert.equal(result.signal, null);
                      assert.equal(result.status, 0, result.stderr);
                      assert.equal(result.stdout, "PASS actual-source nested readback and converted status\n");
                    }
                  });
                });
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("ZHash host-USM session buffers zero, fall back independently, and retire exactly once", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "../sycl"),
        session = fs.readFileSync(path.join(root, "zhash/session.h"), "utf8"),
        library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8"),
        fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-zhash-host-usm-")),
        source = path.join(fixture, "host-transfer-fault.cpp");
      /** @param {string} text @param {string} start @param {string} end */
      const extract = (text, start, end) => {
        const first = text.indexOf(start), last = text.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return text.slice(first, last);
      };
      try {
        fs.copyFileSync(path.join(__dirname, "native/host-transfer-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "zhash-fast-readback.inc"),
          extractCppBlock(session, "  struct FastReadback {") + ";\n");
        const fieldsStart = session.indexOf("  FastReadback* fast_readback_ = nullptr;"),
          fieldsEnd = session.indexOf("\n#ifdef MOM_SYCL_ADAPTIVECPP_CUDA", fieldsStart);
        assert.ok(fieldsStart >= 0 && fieldsEnd > fieldsStart, "per-session host-USM members");
        fs.writeFileSync(path.join(fixture, "zhash-host-fields.inc"), session.slice(fieldsStart, fieldsEnd));
        const allocateStart = session.indexOf("      if (device.has(sycl::aspect::usm_host_allocations)) {"),
          allocateEnd = session.indexOf("      input_storage_ = allocate_objects", allocateStart);
        assert.ok(allocateStart >= 0 && allocateEnd > allocateStart, "host-USM allocation block");
        fs.writeFileSync(path.join(fixture, "zhash-host-alloc.inc"), session.slice(allocateStart, allocateEnd));
        fs.writeFileSync(path.join(fixture, "zhash-alloc-failure.inc"),
          extract(session, "      free_all();\n      throw;", "    }\n  }"));
        fs.writeFileSync(path.join(fixture, "zhash-free-ptr.inc"),
          extractCppBlock(session, "  void free_ptr("));
        fs.writeFileSync(path.join(fixture, "zhash-free-all.inc"),
          extractCppBlock(session, "  void free_all() noexcept {"));
        fs.writeFileSync(path.join(fixture, "zhash-destructor.inc"),
          extractCppBlock(session, "  ~Session() {").replace("~Session()", "~ZHashUsmSession()"));
        const inputStart = session.indexOf("      std::uint64_t midstate[8];"),
          inputEnd = session.indexOf("      std::array<sycl::event, 6> stages;", inputStart);
        assert.ok(inputStart >= 0 && inputEnd > inputStart, "host input staging block");
        const fallback = "      alignas(std::uint64_t) std::uint8_t fallback_input[input_storage_bytes];";
        const input = session.slice(inputStart, inputEnd + "      std::array<sycl::event, 6> stages;".length)
          .replace(fallback, `${fallback}\n      std::memset(fallback_input, 0xa5, sizeof(fallback_input));`);
        assert.notEqual(input, session.slice(inputStart, inputEnd + "      std::array<sycl::event, 6> stages;".length),
          "seed stack fallback with nonzero bytes before the actual production clear");
        const fallbackReadback = "      FastReadback fallback_fast_readback{};",
          readbackSource = extract(session, fallbackReadback,
            "      if (options_.collect_stage_times)\n        for ("),
          readback = readbackSource.replace(fallbackReadback,
            `${fallbackReadback}\n      std::memset(&fallback_fast_readback, 0xa5, sizeof(fallback_fast_readback));`);
        assert.notEqual(readback, readbackSource, "poison stack readback before the actual full-packet copy");
        fs.writeFileSync(path.join(fixture, "zhash-input.inc"), input);
        fs.writeFileSync(path.join(fixture, "zhash-readback.inc"), readback);
        fs.writeFileSync(path.join(fixture, "zhash-catch.inc"),
          extract(session, "    } catch (const sycl::exception& error) {", "\nprivate:"));
        fs.writeFileSync(path.join(fixture, "zhash-error-report.inc"),
          extractCppBlock(session, "  RunReport& device_error_report("));

        /** @param {string} name */
        const compile = (name) => {
          const executable = path.join(fixture, name);
          const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-fsanitize=address", "-fsanitize-address-use-after-scope", "-fno-omit-frame-pointer",
            ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
            "-DMOM_ZHASH_USM_FIXTURE=1", source, "-o", executable], {encoding: "utf8", timeout: 30000});
          assert.equal(compiled.error, undefined);
          assert.equal(compiled.signal, null);
          assert.equal(compiled.status, 0, compiled.stderr);
          return executable;
        };
        /** @param {string} executable @param {string} mode */
        const run = (executable, mode) => {
          const result = spawnSync(executable, [mode], {encoding: "utf8", timeout: 5000});
          assert.equal(result.error, undefined);
          return result;
        };
        const fixed = compile("zhash_host_usm_fixed");
        for (const mode of ["buffers", "fallback-readback", "fallback-input", "fallback-both",
          "partial-free", "input-fault", "readback-fault"]) {
          const result = run(fixed, mode);
          assert.equal(result.signal, null, `${mode}: ${result.stderr}`);
          assert.equal(result.status, 0, `${mode}: ${result.stdout}${result.stderr}`);
          assert.equal(result.stdout, "PASS actual-source ZHash host-USM lifetime, fallback and retirement\n");
        }

        const inputGuard = "      MomSyclHostTransferGuard input_transfer(queue_, \"zhash input upload\");";
        assert.ok(input.includes(inputGuard), "input fault case must exercise the actual input guard");
        fs.writeFileSync(path.join(fixture, "zhash-input.inc"), input.replace(inputGuard + "\n", ""));
        const noInputGuard = compile("zhash_host_usm_no_input_guard"), inputFailure = run(noInputGuard, "input-fault");
        assert.notEqual(inputFailure.status, 0, "the removed input guard must fail");
        assert.match(inputFailure.stderr, /AddressSanitizer: stack-use-after-return/);

        const readbackGuard = "      MomSyclHostTransferGuard fast_readback_guard(queue_, \"zhash fast readback\");";
        assert.ok(readback.includes(readbackGuard), "readback fault case must exercise the actual readback guard");
        fs.writeFileSync(path.join(fixture, "zhash-input.inc"), input);
        fs.writeFileSync(path.join(fixture, "zhash-readback.inc"), readback.replace(readbackGuard + "\n", ""));
        const noReadbackGuard = compile("zhash_host_usm_no_readback_guard"),
          readbackFailure = run(noReadbackGuard, "readback-fault");
        assert.equal(readbackFailure.signal, null);
        assert.equal(readbackFailure.status, 1, readbackFailure.stdout + readbackFailure.stderr);
        assert.equal(readbackFailure.stderr,
          "FAIL readback guard did not retire the fallback packet before unwinding\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("Nexa staged and Octopus entry faults retire uploads and invalidate retry caches", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const octopus = readSyclSource(path.join(root, "octopus/octopus.cpp"));
      const nexa = fs.readFileSync(path.join(root, "nexapow/sycl-pipeline.inc"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const catchStart = octopus.lastIndexOf("  } catch (...) {");
      const catchEnd = octopus.lastIndexOf("\n}");
      assert.ok(catchStart > 0 && catchEnd > catchStart);
      const routes = [
        {name: "Nexa staged", octopus: 0, entry: extract(nexa,
          "    MomSyclHostTransferGuard host_transfers(queue, \"nexapow staged host transfers\");",
          "    U256* private_keys = private_keys_;")},
        {name: "Octopus", octopus: 1, entry: extract(octopus,
          "  const bool header_changed = state.ensure_points(input);", "    const FastModData cache_mod =")},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-entry-transfer-fault-"));
      const source = path.join(fixture, "host-transfer-fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host-transfer-fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "search-result.inc"),
          extract(nexa, "struct SearchResult {", "struct HeaderContext {"));
        fs.writeFileSync(path.join(fixture, "points-cache-hit.inc"),
          extract(octopus, "    if (points_ready && std::memcmp(points_header.data()",
            "    const uint64_t v0 ="));
        fs.writeFileSync(path.join(fixture, "points-cache-commit.inc"),
          extract(octopus, "    std::memcpy(points_header.data(), next_header, HASH_LEN);",
            "    points_version ="));
        for (const route of routes) {
          await t.test(route.name, async (entryTest) => {
            const variants = [{name: "baseline lifetime", guard: 0, ready: 1},
              ...(route.octopus ? [{name: "baseline readiness", guard: 1, ready: 0}] : []),
              {name: "fixed", guard: 1, ready: 1}];
            for (const variant of variants) {
              fs.writeFileSync(path.join(fixture, "entry.inc"), variant.guard ? route.entry :
                route.entry.replace(/^\s*MomSyclHostTransferGuard[^\n]+\n/m, ""));
              const caught = octopus.slice(catchStart, catchEnd);
              fs.writeFileSync(path.join(fixture, "entry-catch.inc"), variant.ready ? caught :
                caught.replace("    state.points_ready = state.target_ready = false;\n", ""));
              const executable = path.join(fixture, "entry_fault_" + route.octopus + "_" + variant.name);
              const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
                "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
                "-DMOM_ENTRY_FIXTURE=1", "-DMOM_ENTRY_OCTOPUS=" + route.octopus,
                "-DMOM_SYCL_PORTABLE_OPENCL=1", source, "-o", executable], {encoding: "utf8"});
              assert.equal(compiled.error, undefined);
              assert.equal(compiled.signal, null);
              assert.equal(compiled.status, 0, compiled.stderr);
              await entryTest.test(variant.name, {concurrency: false}, async (variantTest) => {
                const cases = [{stage: 0, kind: 0, secondary: 0, shared: 0, name: "device success wait trace"}];
                if (route.octopus) {
                  cases.push({stage: 0, kind: 0, secondary: 0, shared: 1, name: "shared success wait trace"});
                }
                const operations = route.octopus
                  ? ["first submit", "later submit", "device query", "second upload"]
                  : ["result memset submit", "later submit", "device query"];
                for (const [offset, operation] of operations.entries()) {
                  const stage = offset + 1;
                  for (const [kind, error] of ["std::exception", "std::string", "unknown"].entries()) {
                    for (const secondary of [0, 1]) {
                      const modes = route.octopus && variant.guard && stage !== 4 ? [0, 1] : [0];
                      for (const shared of modes) {
                        cases.push({stage, kind, secondary, shared,
                          name: (shared ? "shared " : "device ") + operation + ": " + error +
                        (secondary ? " and cleanup fault" : "")});
                      }
                    }
                  }
                }
                await runGroupedCases(
                  variantTest, cases,
                  ({stage}) => variant.name + ": " + (stage ? operations[stage - 1] : "success wait traces"),
                  async (caseTest, {stage, kind, secondary, shared, name}) => {
                    await caseTest.test(variant.name + ": " + name, () => {
                      const result = spawnSync(executable,
                        [String(stage), String(kind), String(secondary), String(shared)],
                        {encoding: "utf8", timeout: 2000});
                      assert.equal(result.error, undefined);
                      assert.equal(result.signal, null);
                      if (stage && !variant.guard) {
                        assert.equal(result.status, 1, result.stdout + result.stderr);
                        assert.equal(result.stderr, "FAIL accepted entry transfer outlived its host owner\n");
                      } else if (stage && !variant.ready) {
                        assert.equal(result.status, 1, result.stdout + result.stderr);
                        assert.equal(result.stderr, "FAIL Octopus fault kept header or target cache ready\n");
                      } else {
                        assert.equal(result.status, 0, result.stderr);
                        assert.equal(result.stdout, "PASS actual-source entry transfers and cache readiness\n");
                      }
                    });
                  });
              });
            }
          });
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  });
});
