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
const {addDiscreteCases, addGenericCases, addIntegratedCases} = require("./common/gpu_test_matrix");
const {
  cloneForDiscreteGpu, cloneForIntelIntegrated, cloneForOpenclSycl, openclSyclEnv, requestedVendors,
} = require("./common/gpu_test_modes");
const {
  createAlgoParamsReportCache,
  isMissingGpuOutput,
  maxCapturedOutputBytes,
  parseDiscreteGpuDevices, parseGpuDevices,
  runNode, wrapWindowsCmd,
} = require("./common/miner_command");

const {readSyclSource, extractCppBlock} = require("./common/cpp_source");

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
    test("npm tooling stays lockfile-free", () => {
      const root = path.join(__dirname, "..");
      const packageJson = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
      const npmrc = fs.readFileSync(path.join(root, ".npmrc"), "utf8");
      assert.equal(fs.existsSync(path.join(root, "package-lock.json")), false);
      assert.match(npmrc, /^package-lock=false$/m);
      assert.match(npmrc, /^save-exact=true$/m);
      assert.match(fs.readFileSync(path.join(root, ".gitignore"), "utf8"), /^package-lock\.json$/m);
      for (const version of Object.values(packageJson.devDependencies)) {
        assert.match(version, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
      }
      const excludedDirectories = new Set([
        ".git", "node_modules", "build", "dist", "out", "release", "release-build",
        "release-combined", "release-combined-build", "coverage", "bin", "obj", "target", "vendor",
      ]);
      const files = ["package.json", "binding.gyp"].map((file) => path.join(root, file));
      for (const directory of ["scripts", ".github"]) {
        const base = path.join(root, directory);
        for (const entry of fs.readdirSync(base, {recursive: true, withFileTypes: true})) {
          if (!entry.isFile()) {continue;}
          const file = path.join(entry.parentPath, entry.name);
          const relative = path.relative(root, file);
          if (relative.split(path.sep).some((part) => excludedDirectories.has(part))) {continue;}
          files.push(file);
        }
      }
      const forbidden = /\bnpm(?:\.cmd)?\s+(?:ci|update)\b/;
      for (const file of files) {
        const contents = fs.readFileSync(file);
        if (contents.includes(0)) {continue;}
        assert.doesNotMatch(contents.toString("utf8"), forbidden, file);
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
        "--require", path.join(root, "tests/common/test_output_buffer.js"), "-e", source,
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

    test("release packagers never recursively remove a caller-selected archive", () => {
      const scripts = path.join(__dirname, "../.github/workflows/scripts");
      const linux = fs.readFileSync(path.join(scripts, "package-linux-combined.sh"), "utf8");
      assert.match(linux, /\[ -d "\$archive" \]/);
      assert.doesNotMatch(linux, /rm -f -- "\$archive"/);
      assert.doesNotMatch(linux, /rm -rf[^\n]*"\$archive"/);

      const windows = fs.readFileSync(path.join(scripts, "package-windows.ps1"), "utf8");
      assert.match(windows, /Test-Path -LiteralPath \$Archive -PathType Container/);
      assert.doesNotMatch(windows, /Remove-Item -Force -LiteralPath \$Archive/);
      assert.doesNotMatch(windows, /Remove-Item[^\n]*-Recurse[^\n]*\$Archive/);
      assert.match(windows, /\[IO\.Path\]::GetFullPath\(\$Archive\)/);
      assert.match(windows, /\[StringComparison\]::OrdinalIgnoreCase/);
    });

    test("Windows package launcher keeps the selected runtime ahead of shared oneAPI", () => {
      const script = fs.readFileSync(
        path.join(__dirname, "../.github/workflows/scripts/package-windows.ps1"), "utf8");
      const runtimePathLine = String.raw`if defined MOM_RUNTIME_DIR set "PATH=%MOM_RUNTIME_DIR%;%MOM_RUNTIME_DIR%\hipSYCL;%PATH%"`;
      const sharedDpcppPathLine = String.raw`if /I "%MOM_GPU_BACKEND%"=="opencl" set "PATH=%MOM_LIBS%\oneapi;%PATH%"`;
      const integratedPathLine = String.raw`if /I "%MOM_GPU_BACKEND%"=="intel_igpu" set "PATH=%MOM_LIBS%\oneapi;%PATH%"`;
      assert.equal(script.split(runtimePathLine).length - 1, 1);
      assert.deepEqual(script.split(/\r?\n/).filter((line) =>
        line.includes('set "PATH=%MOM_LIBS%\\oneapi;')), [sharedDpcppPathLine, integratedPathLine]);
      assert.ok(script.indexOf(sharedDpcppPathLine) < script.indexOf(runtimePathLine),
        "OpenCL fallback dependencies must be prepended before the selected runtime");
      assert.ok(script.indexOf(integratedPathLine) < script.indexOf(runtimePathLine),
        "Integrated worker dependencies must be prepended before the selected runtime");
    });
  });

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
      const script = path.join(root, ".github/workflows/scripts/package-linux-combined.sh");
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

    test("Linux native packager checks every compiler worker before collecting runtimes", {
      skip: process.platform === "win32",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const compilers = source.match(/^compilers=\([^\n]+\)$/m)?.[0];
      // eslint-disable-next-line no-template-curly-in-string -- Match the Bash array expansion.
      const start = source.indexOf("for key in \"${compilers[@]}\"; do", source.indexOf("cp sycl/kawpow/keccak.inc"));
      const end = source.indexOf("\ndone", start);
      assert.ok(compilers && start >= 0 && end > start);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-workers-"));
      const keys = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
      const guard = ["set -euo pipefail", compilers, "compiler_build_dir=build/lin/Release", source.slice(start, end + 5)].join("\n");
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
      const source = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const start = source.indexOf("runtime_matches() (");
      const end = source.indexOf("copy_matches() {", start);
      assert.ok(start >= 0 && end > start);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-patterns-"));
      const driver = source.slice(start, end) + "\nruntime_matches \"$1\"";
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
      const source = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const patterns = source.slice(source.indexOf("runtime_matches() ("), source.indexOf("copy_dependency_closure() {"));
      const start = source.indexOf("copy_oneapi_opencl_runtime() {");
      const end = source.indexOf("\nbegin_runtime oneapi", start);
      assert.ok(start >= 0 && end > start);
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
          patterns,
          source.slice(start, end).replaceAll("/opt/intel/oneapi", sdk),
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
        const scripts = path.join(__dirname, "../.github/workflows/scripts");
        const packager = fs.readFileSync(path.join(scripts, "package-linux-combined.sh"), "utf8");
        const validator = fs.readFileSync(path.join(scripts, "test-release-linux.sh"), "utf8");
        const functionStart = packager.indexOf("copy_acpp_runtime() {");
        const setupEnd = packager.indexOf("  begin_runtime", functionStart);
        const stagingStart = packager.indexOf('  redist_bin="$runtime_dest/hipSYCL/ext/llvm/bin"', setupEnd);
        const stagingEnd = packager.indexOf('  if [ "$key" = acpp-hip ]; then', stagingStart);
        const validationStart = validator.indexOf("for tool in opt llc lld ld.lld; do");
        const validationEnd = validator.indexOf("for bitcode in", validationStart);
        assert.ok(functionStart >= 0 && setupEnd > functionStart && stagingStart > setupEnd && stagingEnd > stagingStart);
        assert.ok(validationStart >= 0 && validationEnd > validationStart);
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-package-llvm-test-"));
        const image = path.join(temp, "image");
        const libs = path.join(temp, "libs");
        const tools = ["opt", "llc", "lld", "ld.lld"];
        const env = {...process.env, MOM_FIXTURE_IMAGE: image, MOM_FIXTURE_LIBS: libs};
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
            fs.mkdirSync(prefix);
            fs.writeFileSync(path.join(prefix, ".mom-build-identity"), `${"a".repeat(40)}:${version}:${backend}:7.1.1:1:none\n`);
          }
          // Execute the real metadata selection and staging statements against only fixture SDKs.
          const staging = [
            "set -euo pipefail",
            "container=fixture; closure_queue=(); libs_dir=$MOM_FIXTURE_LIBS",
            packager.slice(functionStart, setupEnd).replaceAll("/usr/lib/llvm-", "$MOM_FIXTURE_IMAGE/usr/lib/llvm-") +
          packager.slice(stagingStart, stagingEnd) + "}",
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
      const packager = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
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
      const end = installer.indexOf("\nNODE_VERSION=");
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
      const start = installer.indexOf("install_dpcpp() {");
      const end = installer.indexOf("\n}\n", start);
      assert.ok(start >= 0 && end > start);
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
          installer.slice(start, end + 2).replaceAll("/opt/", `${temp}/`),
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
      const start = installer.indexOf("ensure_llvm() {");
      const end = installer.indexOf("\n}\n", start);
      assert.ok(start >= 0 && end > start);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-llvm-mode-"));
      try {
      // Redirect both readiness probes: this fixture never inspects a real host SDK.
        const body = installer.slice(start, end + 2)
          .replaceAll("/usr/bin/", `${temp}/bin/`).replaceAll("/usr/lib/", `${temp}/lib/`);
        const invoke = 'bash "$WORKSPACE/llvm.sh" "$LLVM_VERSION"';
        assert.ok(body.includes(invoke));
        const legacy = body.replace(invoke, () => `${invoke} all`);
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
          const before = run(legacy, version);
          assert.equal(before.status, 0);
          assert.deepEqual(before.calls[3], [...installerArgs, "all"]);
          assert.throws(() => assert.deepEqual(before.calls, expected), {code: "ERR_ASSERTION"},
            "The old all mode must fail the same argument policy");
          const unverified = run(body, version, "0".repeat(64));
          assert.notEqual(unverified.status, 0);
          assert.deepEqual(unverified.calls, expected.slice(0, 3)
            .map(call => call[0] === "verify" ? ["verify", `${temp}/llvm.sh`, "0".repeat(64)] : call));
          fs.mkdirSync(path.join(temp, "bin"), {recursive: true});
          fs.mkdirSync(path.join(temp, `lib/llvm-${version}/lib/cmake/llvm`), {recursive: true});
          fs.writeFileSync(path.join(temp, `bin/clang-${version}`), "fixture\n", {mode: 0o755});
          const ready = run(body, version);
          assert.equal(ready.status, 0);
          assert.deepEqual(ready.calls, [], "An already-ready fake SDK must not invoke network or apt");
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("Native Node installer rejects a same-version binary without SEA or matching build headers", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const extract = (/** @type {string} */ name) => {
        const start = installer.indexOf(`${name}() {`);
        return start < 0 ? "" : installer.slice(start, installer.indexOf("\n}\n", start) + 2);
      };
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-node-readiness-"));
      try {
        const bin = path.join(temp, "bin");
        const headers = path.join(temp, "include/node");
        fs.mkdirSync(bin); fs.mkdirSync(headers, {recursive: true});
        fs.writeFileSync(path.join(bin, "node-gyp"), "#!/bin/sh\nexit 0\n", {mode: 0o755});
        const cases = ["ready", "missing-fuse", "missing-header", "wrong-headers"];
        for (const name of cases) {
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

    test("Native base validation requires both Linux packaging tools", {
      skip: process.platform === "win32",
    }, () => {
      const installer = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
      const start = installer.indexOf("validate_component() {");
      const end = installer.indexOf("\n}\n", start);
      assert.ok(start >= 0 && end > start);
      const bash = spawnSync("bash", ["-c", "command -v bash"], {encoding: "utf8"});
      assert.equal(bash.status, 0);
      const program = `${installer.slice(start, end + 2)}\nvalidate_component base`;
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
        ".github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const buildStart = installer.indexOf("build_adaptivecpp() {");
      const optionsStart = installer.indexOf("  local -a options=(", buildStart);
      const optionsEnd = installer.indexOf("\n  cmake -S", optionsStart);
      const functionStart = packager.indexOf("copy_acpp_runtime() {");
      const setupEnd = packager.indexOf("  begin_runtime", functionStart);
      const stagingStart = packager.indexOf('  redist_bin="$runtime_dest/hipSYCL/ext/llvm/bin"', setupEnd);
      const stagingEnd = packager.indexOf('  if [ "$key" = acpp-hip ]; then', stagingStart);
      assert.ok(buildStart >= 0 && optionsStart > buildStart && optionsEnd > optionsStart);
      assert.ok(functionStart >= 0 && setupEnd > functionStart && stagingStart > setupEnd && stagingEnd > stagingStart);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-hip-linker-"));
      const sdk = path.join(temp, "sdk");
      const llvm = path.join(sdk, "usr/lib/llvm-21/bin");
      const runtime = path.join(temp, "runtime");
      const env = {...process.env, LLVM_VERSION: "21", MOM_LLVM_VERSION: "21", JOBS: "1",
        MOM_FIXTURE_SDK: sdk, MOM_FIXTURE_RUNTIME: runtime};
      try {
        fs.mkdirSync(llvm, {recursive: true});
        const prefix = path.join(sdk, "adaptivecpp-hip");
        fs.mkdirSync(prefix);
        fs.writeFileSync(path.join(prefix, ".mom-build-identity"), `${"a".repeat(40)}:21:hip:7.1.1:1-hip-lld1:none\n`);
        // An unrelated legacy SDK must never choose the toolchain for this installed compiler.
        fs.mkdirSync(path.join(sdk, "opt/llvm20-ubuntu24/bin"), {recursive: true});
        for (const tool of ["opt", "llc", "lld", "ld.lld"]) {
          fs.writeFileSync(path.join(llvm, tool), "#!/bin/sh\nprintf '%s\\n' fixture-llvm21\n", {mode: 0o755});
        }
        const configured = spawnSync("bash", ["-c", [
          "set -euo pipefail",
          "capture_options() { local backend=hip dest=/unused cuda_dir=/unused rocm_dir=/unused;",
          installer.slice(optionsStart, optionsEnd),
          // eslint-disable-next-line no-template-curly-in-string -- This expands the real Bash options array.
          'printf "%s\\n" "${options[@]}"; }',
          "capture_options",
        ].join("\n")], {encoding: "utf8", env});
        assert.equal(configured.status, 0, "Real installer options must evaluate without SDK access");
        const linkerOption = configured.stdout.split("\n").find(line => line.startsWith("-DACPP_LLD_PATH="));
        assert.ok(linkerOption, "The installer must configure the HIP linker");
        const staging = (packager.slice(functionStart, setupEnd) +
        packager.slice(stagingStart, stagingEnd) + "}")
          .replaceAll("/opt/llvm", "$MOM_FIXTURE_SDK/opt/llvm")
          .replaceAll("/usr/lib/llvm-", "$MOM_FIXTURE_SDK/usr/lib/llvm-");
        const staged = spawnSync("bash", ["-c", [
          "set -euo pipefail",
          'closure_queue=(); runtime_dest="$MOM_FIXTURE_RUNTIME"',
          staging,
          'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_SDK/adaptivecpp-hip"',
        ].join("\n")], {encoding: "utf8", env});
        assert.equal(staged.status, 0, "Real packaging statements must stage only the fixture tools");
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
      const packager = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const start = packager.indexOf("copy_acpp_runtime() {");
      const end = packager.indexOf("  begin_runtime", start);
      assert.ok(start >= 0 && end > start);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-llvm-identity-"));
      const marker = path.join(temp, ".mom-build-identity");
      try {
        for (const identity of [null, "", `${"a".repeat(40)}:21:cuda:12-6:1:none`,
          `${"a".repeat(40)}:21;exit 0:hip:7.1.1:1:none`, `${"a".repeat(40)}:21:hip:7.1.1:1:none\nextra`]) {
          if (identity !== null) {fs.writeFileSync(marker, identity);}
          const result = spawnSync("bash", ["-c", ["set -euo pipefail",
            packager.slice(start, end) + "}", 'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_PREFIX"',
          ].join("\n")], {encoding: "utf8", env: {...process.env, MOM_FIXTURE_PREFIX: temp}});
          assert.equal(result.status, 1, "Invalid SDK identity must fail before staging any tools");
          assert.match(result.stderr, /valid matching \.mom-build-identity/);
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("AdaptiveCpp HIP bitcode stays with the recorded SDK cohort", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const matchesStart = packager.indexOf("runtime_matches() (");
      const matchesEnd = packager.indexOf("\n)\n", matchesStart);
      const setupStart = packager.indexOf("copy_acpp_runtime() {");
      const setupEnd = packager.indexOf("  begin_runtime", setupStart);
      const bitcodeStart = packager.indexOf("    local device_bitcode", setupEnd);
      const bitcodeEnd = packager.indexOf('  elif [ "$key" = acpp-cuda ]; then', bitcodeStart);
      assert.ok(matchesStart >= 0 && matchesEnd > matchesStart);
      assert.ok(setupStart >= 0 && setupEnd > setupStart && bitcodeStart > setupEnd && bitcodeEnd > bitcodeStart);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-cohort-"));
      try {
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
          const program = ["set -euo pipefail",
            packager.slice(matchesStart, matchesEnd + 2),
            packager.slice(setupStart, setupEnd),
            packager.slice(bitcodeStart, bitcodeEnd), "}",
            'copy_acpp_runtime acpp-hip "$MOM_FIXTURE_PREFIX"',
          ].join("\n").replaceAll("/opt/rocm", `${opt}/rocm`);
          const result = spawnSync("bash", ["-c", program], {encoding: "utf8", env: {...process.env,
            MOM_FIXTURE_PREFIX: prefix, MOM_ROCM_VERSION: fixture.requested, runtime_dest: runtime}});
          assert.equal(result.status, fixture.expected ? 0 : 1, `Wrong bitcode selection for ${fixture.name}`);
          if (fixture.expected) {
            assert.equal(fs.readFileSync(path.join(destination, "ockl.bc"), "utf8"), fixture.expected);
          } else {
            assert.deepEqual(fs.readdirSync(destination), [], "No mismatched bitcode may be packaged");
          }
        }
      } finally { fs.rmSync(temp, {recursive: true, force: true}); }
    });

    test("AdaptiveCpp CUDA bitcode stays with the recorded SDK cohort", {
      skip: process.platform === "win32",
    }, () => {
      const packager = fs.readFileSync(path.join(__dirname, "../.github/workflows/scripts/package-linux-combined.sh"), "utf8");
      const setupStart = packager.indexOf("copy_acpp_runtime() {");
      const setupEnd = packager.indexOf("  begin_runtime", setupStart);
      const branchStart = packager.indexOf('  elif [ "$key" = acpp-cuda ]; then\n', setupEnd);
      const branchEnd = packager.indexOf("\n  fi\n  while IFS=", branchStart);
      assert.ok(setupStart >= 0 && setupEnd > setupStart && branchStart > setupEnd && branchEnd > branchStart);
      const cudaBody = packager.slice(branchStart + '  elif [ "$key" = acpp-cuda ]; then\n'.length, branchEnd);
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-cohort-"));
      try {
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
          fs.mkdirSync(prefix);
          fs.writeFileSync(path.join(prefix, ".mom-build-identity"),
            `${"a".repeat(40)}:21:cuda:${fixture.version}:1:none\n`);
          const program = ["set -euo pipefail", packager.slice(setupStart, setupEnd), cudaBody, "}",
            'copy_acpp_runtime acpp-cuda "$MOM_FIXTURE_PREFIX"',
          ].join("\n").replaceAll("/usr/local/", `${sdkRoot}/`);
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
      const start = installer.indexOf("adaptivecpp_identity() {");
      const end = installer.indexOf("\n}\n", start);
      assert.ok(start >= 0 && end > start);
      const patchHash = createHash("sha256").update(fs.readFileSync(path.join(root,
        "scripts/patches/adaptivecpp-cuda-unloading.patch"))).digest("hex");
      const identity = spawnSync("bash", ["-c", [
        "set -euo pipefail",
        installer.slice(start, end + 2),
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
      const script = path.join(root, ".github/workflows/scripts/package-linux-combined.sh");
      const workers = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
      const contents = "worker fixture\n";
      const sha256 = createHash("sha256").update(contents).digest("hex");
      /** @typedef {{name: string, target: string, kind: string, expected: RegExp}} ProfileCase */
      /** @type {ProfileCase[]} */
      const profileCases = [
        ...workers.map((worker) => ({
          name: `${worker} portable=0`, target: worker, kind: "portable-zero",
          expected: /is not a portable release worker/,
        })),
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
          expected: /is not a portable release worker/},
        {name: "cpu=x86-64-v4", target: "oneapi", kind: "cpu-v4",
          expected: /is not a portable release worker/},
        {name: "missing patchelf", target: "oneapi", kind: "missing-patchelf",
          expected: /Linux release packaging requires patchelf/},
        {name: "missing readelf", target: "oneapi", kind: "missing-readelf",
          expected: /Linux release packaging requires readelf/},
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
        path.join(root, ".github/workflows/scripts/package-linux-combined.sh"), traversalVersion,
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
        path.join(root, ".github/workflows/scripts/package-windows.ps1"),
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
      assert.match(deploySource, /apt-get install[^\n]*\bpython3\b/);
      assert.match(deploySource, /MOM_VERTHASH_DATA:\/verthash\.dat:ro/);
      assert.match(deploySource, /win-mom-dev-base\.qcow2/);
      assert.doesNotMatch(deploySource, /win-mom-dev\.qcow2/);
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
      const multiRelease = deploySource.match(/^test_windows_multi_release\(\) \{\n[\s\S]*?\n\}\n/m)?.[0];
      assert.ok(multiRelease, "mixed-vendor release function must exist");
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
      const functionSource = source.match(/^test_linux_release\(\) \{\n[\s\S]*?\n\}\n/m)?.[0];
      assert.ok(functionSource);
      assert.match(functionSource, /\[ -c "\/dev\/dri\/\$name" \]/);
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

    test("Windows AdaptiveCpp exported source silences only its expected Git probe failure", () => {
      const build = fs.readFileSync(path.join(__dirname,
        "../scripts/build-windows-adaptivecpp-amd.ps1"), "utf8");
      const oldStart = build.indexOf("$rootGitStatusOld =");
      const newStart = build.indexOf("$rootGitStatusNew =", oldStart);
      const replaceStart = build.indexOf(
        "Replace-RequiredText $rootCmakeText $rootGitStatusOld $rootGitStatusNew", newStart);
      assert.ok(oldStart >= 0 && newStart > oldStart && replaceStart > newStart,
        "the checked exported-source workaround must remain intact");
      assert.match(build.slice(oldStart, newStart), /RESULT_VARIABLE GIT_STATUS/);
      assert.doesNotMatch(build.slice(oldStart, newStart), /ERROR_QUIET/);
      assert.match(build.slice(newStart, replaceStart), /RESULT_VARIABLE GIT_STATUS[\s\S]*ERROR_QUIET/);
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

    test("Octopus NVIDIA DAG loads retain the measured cache policy", () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/octopus/nvidia_tensor.inc"), "utf8");
      assert.match(source, /ld\.global\.L1::evict_first\.v4\.u32/);
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

    test("Verthash bounded CPU fixture checks its route and independent scalar gold", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const root = path.join(__dirname, "..");
      const source = fs.readFileSync(path.join(root, "sycl/verthash/verthash.cpp"), "utf8");
      const begin = source.indexOf("static State& get_state(");
      const end = source.indexOf("\nvoid verthash_cleanup_states()", begin);
      assert.ok(begin >= 0 && end > begin);
      assert.match(source, /get_state\(dev_str, is_test && !is_benchmark\)/);
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-verthash-small-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/verthash_test_route.cpp"),
          path.join(directory, "route.cpp"));
        fs.writeFileSync(path.join(directory, "verthash_get_state.inc"), source.slice(begin, end));
        /** @type {[string, string][]} */
        const cases = [
          [path.join(directory, "route.cpp"), "verthash-test-route:passed"],
          [path.join(__dirname, "reference/verthash_small.cpp"), "verthash-small-reference:passed"],
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
          path.join(__dirname, "reference/etchash_small.cpp"),
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
      const source = fs.readFileSync(path.join(__dirname, "native_addon.js"), "utf8");
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

    test("PearlHash routes integrated Intel GPUs away from ESIMD prefetch", () => {
      const dispatch = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dispatch.inc"), "utf8");
      assert.match(dispatch,
        /const bool discrete_gpu = device\.is_gpu\(\) && !is_integrated_gpu\(device\);/);
      assert.match(dispatch,
        /pearlhash_esimd_route\(tuned_sycl, supported_backend, discrete_gpu, intel_matrix, width\)/);
      assert.match(dispatch, /use_portable = !cuda_sycl && esimd_width == 0u;/);
      assert(dispatch.indexOf("pearlhash_esimd_route(") < dispatch.indexOf("compute_ab("),
        "ESIMD capability gating must precede matrix layout selection");
    });

    test("PearlHash discovery emits complete measured profiles with a low-memory fallback", () => {
      const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
      const params = fs.readFileSync(path.join(__dirname, "../sycl/algo_params.inc"), "utf8");
      assert.ok(intensity.includes("global_mem_size>() >= 8 * GiB"));
      assert.ok(intensity.includes("pearlhash_profile_t{131072, 524288, 8192, 128}"));
      assert.ok(intensity.includes("pearlhash_profile_t{65536, 65536, 4096, 256}"));
      assert.ok(params.includes("\"*[m=\" + std::to_string(profile.m)"));
      for (const field of ["n", "k", "rank"]) {
        assert.ok(params.includes(`";${field}=" + std::to_string(profile.${field})`));
      }
    });

    test("required integrated coverage requires a device and skips unsupported algorithms", async (t) => {
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
          required: true,
          reason: /no devices/,
        },
        {
          result: {
            skipped: /** @type {false} */ (false),
            devices: [{dev: "gpu1", description: "Intel integrated test device", integrated: true}],
            params: {},
          },
          required: false,
          reason: /not available/,
        },
      ];
      for (const {result, required, reason} of results) {
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
    test("Linux oneAPI developer builds preserve incremental objects", () => {
      const build = fs.readFileSync(
        path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
      const start = build.indexOf("build_oneapi() {");
      const end = build.indexOf("\n}\n\nbuild_dpcpp()", start);
      assert.ok(start >= 0 && end > start, "oneAPI build function must exist");
      const oneapi = build.slice(start, end);
      assert.match(oneapi, /\[ ! -d "\$out" \] \|\| mv "\$out" build/);
      assert.match(oneapi, /node scripts\/linux-build-state\.js prepare build oneapi icx icpx/);
      assert.match(oneapi, /if \[ ! -s build\/Makefile \] \|\|/);
      assert.match(oneapi, /CXX\.target \?= \$PWD\/scripts\/cxx-oneapi\.sh/);
      assert.doesNotMatch(oneapi, /\.node-version| -nt /);
      assert.match(oneapi, /node-gyp build/);
      assert.match(oneapi, /mv build "\$out"/);
      assert.doesNotMatch(oneapi, /find binding\.gyp native sycl/);
      assert.doesNotMatch(oneapi, /rm -rf build "\$out"/);
    });

    test("AMD release workers stay architecture-neutral on Linux and Windows", () => {
      const linux = fs.readFileSync(
        path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
      const windows = fs.readFileSync(
        path.join(__dirname, "../.github/workflows/scripts/build-sycl-adaptivecpp-win.ps1"), "utf8");
      assert.match(linux, /ACPP_TARGETS=generic ACPP_VISIBILITY_MASK="\$target"/);
      assert.match(windows, /--acpp-targets=generic/);
      assert.doesNotMatch(`${linux}\n${windows}`, /(?:--offload-arch=|--acpp-targets=hip:|ACPP_TARGETS=hip:)/);
    });

    test("Intel CN tuning selects its width8 recurrence only from reported hardware capability", () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/algo_params.inc"), "utf8");
      // The compiled intel_geometry tests cover capability failures and excluded worker profiles.
      assert.match(source, /all_intel_native_width8 = all_intel_native_width8 &&\s+mom_intel_eu_simd_width\(dev\) == 8/);
      assert.match(source, /backend_hint = cn_gpu_intel_backend\(needs_intel_opencl, all_intel_native_width8\)/);
      const recurrence = fs.readFileSync(path.join(__dirname, "../sycl/cn_gpu/cn_gpu.cpp"), "utf8");
      assert.match(recurrence, /use_native_width8 = mom_intel_eu_simd_width\(state\.device\) == 8/);
      assert.doesNotMatch(`${source}\n${recurrence}`, /get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>/);
    });

    test("AMD matrix paths share a conservative selected-device instruction gate", () => {
      for (const file of ["octopus/amd_wmma.inc", "walahash/amd_wmma.inc", "pearlhash/hip_jit.inc"]) {
        const source = fs.readFileSync(path.join(__dirname, "../sycl", file), "utf8");
        assert.match(source, /hipGetDeviceProperties\(/);
        assert.match(source, /mom::amd::has_gfx12_int8_wmma\(/);
        assert.doesNotMatch(source, /strncmp\([^\n]*gcnArchName/);
        assert.match(source, /"--offload-arch=" \+ arch/);
      }
      const octopus = readSyclSource(path.join(__dirname, "../sycl/octopus/octopus.cpp"));
      assert.match(octopus, /subgroup_32 = mom_is_cuda\(state\.device\) \|\|\s+std::find\(subgroup_sizes/);
      assert.match(octopus, /!std::getenv\("MOM_OCTOPUS_BATCHED"\) && subgroup_32/);
    });

    test("Linux AdaptiveCpp developer builds preserve incremental objects", () => {
      const entrypoint = fs.readFileSync(
        path.join(__dirname, "../scripts/adaptivecpp-entrypoint.sh"), "utf8");
      assert.match(entrypoint, /mv "\$cache_dir" build/);
      assert.match(entrypoint,
        /grep -Fqx "CXX\.target \?= \$compiler" build\/Makefile; then\s+mom_run_quiet "\[adaptivecpp-\$backend\] node-gyp configure"/);
      assert.match(entrypoint, /mv "\$source_dir\/build" "\$cache_dir"/);
      assert.match(entrypoint, /preserve_partial_build/);
      assert.match(entrypoint, /build\/build-output\.log/);
      assert.doesNotMatch(entrypoint, /rm -rf \/tmp\/mom-adaptive-build "\$build_dir"/);
    });

    test("Linux content receipts own invalidation before configure/build and retain path admission", () => {
      for (const file of ["multicompiler-entrypoint.sh", "combined-build.sh", "adaptivecpp-entrypoint.sh"]) {
        const source = fs.readFileSync(path.join(__dirname, "../scripts", file), "utf8");
        const body = file === "multicompiler-entrypoint.sh"
          ? source.slice(source.indexOf("build_oneapi() {"), source.indexOf("\n}\n\nbuild_dpcpp()"))
          : source;
        const prepare = body.indexOf("node scripts/linux-build-state.js prepare build");
        const configure = body.indexOf("node-gyp configure", prepare);
        const build = body.indexOf("node-gyp build", configure);
        const finish = body.indexOf("node scripts/linux-build-state.js finish build", build);
        assert.ok(prepare >= 0 && configure > prepare && build > configure && finish > build, file);
        assert.doesNotMatch(body, /\.node-version|compiler_first_line| -nt /);
        assert.match(body, /\[ ! -s build\/Makefile \]/);
        if (file === "combined-build.sh") {
          assert.match(body, /grep -Fq -- "\$ROOT" build\/Makefile/);
        } else if (file === "adaptivecpp-entrypoint.sh") {
          assert.match(body, /grep -Fqx "CXX\.target \?= \$compiler" build\/Makefile/);
        }
      }
    });

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

    test("combined builds preserve an unchanged linked CUDA image", () => {
      const build = fs.readFileSync(
        path.join(__dirname, "../scripts/combined-build.sh"), "utf8");
      const prepare = build.indexOf('node scripts/linux-build-state.js prepare build "$MOM_DPCPP_IMPL"');
      const compile = build.indexOf("node-gyp build", prepare);
      const finish = build.indexOf('node scripts/linux-build-state.js finish build "$MOM_DPCPP_IMPL"', compile);
      assert.ok(prepare >= 0 && compile > prepare && finish > compile,
        "content receipts must decide reuse before Make and admit the completed link afterward");
      const state = fs.readFileSync(path.join(__dirname, "../scripts/linux-build-state.js"), "utf8");
      assert.match(state, /"scripts\/cxx-combined\.sh"/);
      assert.doesNotMatch(build, /link_target|\$WRAP -nt/);
      assert.doesNotMatch(build, /octopus_archive/);
      assert.doesNotMatch(build, /\nrm -f build\/Release\/mom\.node build\/Release\/obj\.target\/mom\.node/);
    });

    test("Linux runner locks mutable build state without advisory GPU locks", () => {
      const runner = fs.readFileSync(path.join(__dirname, "../scripts/dev-container.sh"), "utf8");
      const entrypoint = fs.readFileSync(
        path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");

      assert.match(runner, /name="mom-\$backend"/);
      assert.doesNotMatch(runner, /--privileged/);
      assert.match(runner, /--cap-drop ALL/);
      assert.match(runner, /--security-opt no-new-privileges:true/);
      assert.match(runner, /add_drm_vendor 0x8086/);
      assert.match(runner, /add_drm_vendor 0x1002/);
      assert.match(runner,
        /sycl_cache_volume=\$\{MOM_SYCL_CACHE_VOLUME:-mom-sycl-cache-\$backend\}/);
      const reuseStart = runner.indexOf('if [ "$reuse_built_worker" = 1 ]; then');
      const reuseEnd = runner.indexOf("\nelse\n  flock 9", reuseStart);
      assert.ok(reuseStart >= 0 && reuseEnd > reuseStart, "reuse lock branch must exist");
      const reuseLocks = runner.slice(reuseStart, reuseEnd);
      assert.match(reuseLocks, /flock -s 9/);
      assert.match(runner, /exec 9>"\$build_lock"/);
      assert.doesNotMatch(runner, /MOM_GPU_LOCK_KEY|gpu_lock_key|exec 8>|\.gpu-/);
      assert.equal((runner.match(/\bflock\b/g) || []).length, 2);
      assert.match(runner,
        /if \[ "\$reuse_built_worker" != 1 \]; then[\s\S]*docker_flags\+=\(-it\)[\s\S]*docker_flags\+=\(-i\)[\s\S]*fi/);
      assert.match(runner,
        /elif \[ "\$container_stdin" = 1 \]; then[\s\S]*docker_flags\+=\(-i\)/);
      assert.doesNotMatch(runner, /docker rm -f "\$name"/);
      assert.match(runner,
        /if container_running=\$\(docker container inspect[\s\S]*if \[ "\$container_running" = true \][\s\S]*exit 2/);

      const reuseBranchStart = entrypoint.search(/case "\$\{MOM_REUSE_BUILT_WORKER:-0\}" in/);
      const buildTreeMutation = entrypoint.indexOf("platforms_hold=build-platforms-hold");
      assert.ok(reuseBranchStart >= 0 && buildTreeMutation > reuseBranchStart,
        "worker reuse branch must precede build-tree mutation");
      assert.match(entrypoint, /if \[\[ -e "\$platforms_hold" \|\| -L "\$platforms_hold" \]\]/);
      assert.match(entrypoint,
        /if \[\[ -e build \|\| -L build \]\] && \[\[ ! -d build \|\| -L build \]\]; then/);
      const reuseBranch = entrypoint.slice(reuseBranchStart, buildTreeMutation);
      assert.match(reuseBranch, /reused_worker="\$PWD\/build\/lin\/Release\/\$default\/mom\.node"/);
      assert.match(reuseBranch, /\[ ! -s "\$reused_worker" \]/);
      assert.match(entrypoint,
        /source_fingerprint\(\)[\s\S]*binding\.gyp[\s\S]*native[\s\S]*sycl[\s\S]*sha256sum/);
      assert.match(reuseBranch, /fingerprint_file="\$reused_worker\.sources\.sha256"/);
      assert.match(reuseBranch,
        /recorded_fingerprint[\s\S]*current_fingerprint[\s\S]*source fingerprint differs/);
      assert.doesNotMatch(reuseBranch, /-newer/);
      assert.match(reuseBranch, /exec "\$@"/);
    });

    test("Windows Intel NexaPoW uses oneAPI after the cold-JIT fix", () => {
      assert.equal(policy.selection("nexapow", "intel", "win32").key, "oneapi");
      const build = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
      assert.match(build, /PortableOpencl[\s\S]*MOM_NEXAPOW_PORTABLE_FIELD32/);
    });

    test("Windows CUDA device flags enable SYCL-native Octopus", () => {
      const build = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
      const withCudaBlock = build.match(
        /if \(\$withCuda\) \{\n\s{2}\$targetList \+= \$CudaArch\n\s{2}\$F \+= @\(([\s\S]*?)\)\n\}/);
      assert.ok(withCudaBlock, "Windows CUDA device flag block must exist");
      const withCudaFlags = withCudaBlock?.[1] ?? "";
      assert.match(withCudaFlags, /"-DMOM_SYCL_HAS_CUDA"/);
      assert.match(withCudaFlags, /"-DMOM_OCTOPUS_HAS_SYCL_NATIVE"/);
      assert.match(withCudaFlags, /"-DMOM_NEXAPOW_SYCL_NATIVE_FIELD"/);
    });

    test("Windows oneAPI NexaPoW build enables the portable field", () => {
      const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      assert.match(binding,
        /msbuild_toolset": "Intel\(R\) oneAPI DPC\+\+ Compiler 2026"[\s\S]*?"AdditionalOptions": \[[\s\S]*?"\/DMOM_NEXAPOW_PORTABLE_FIELD32"/);
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
      const rounds = "template <bool GPU_COMPACT>\n" +
        extractCppBlock(proof, "static std::array<sycl::event, 4> run_collision_rounds(");
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
        fs.copyFileSync(path.join(__dirname, "native/async_event_lifetime.cpp"), path.join(fixture, "fixture.cpp"));
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

    test("Xelis orders setup transfers and Windows ESIMD stages", () => {
      const source = readSyclSource(path.join(__dirname, "../sycl/xelishashv3/xelishashv3.cpp"));
      const entry = source.slice(source.indexOf("int xelishashv3("));
      assert.match(entry,
        /sycl_wait_and_throw\(s\.queue\.memcpy\(s\.input,[\s\S]*?\n\s*sycl_wait_and_throw\(s\.queue\.memcpy\(s\.target,[\s\S]*?\n\s*sycl_wait_and_throw\(s\.queue\.memset\(s\.result, 0,[\s\S]*?\n\s*search<(?:true|false)>\(/);
      assert.match(source,
        /#ifdef _WIN32\s+required = required \|\| use_esimd;\s+#endif/);
      assert.match(source,
        /fence<esimd::memory_kind::global, esimd::fence_flush_op::clean,\s+esimd::fence_scope::group>/);
    });

    test("Zel and Beam reserve solver diagnostics for tests or explicit profiling", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-solver-diagnostics-"));
      try {
        fs.copyFileSync(path.join(__dirname, "native/solver_diagnostics.cpp"), path.join(fixture, "fixture.cpp"));
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

    test("ZelHash discovery leaves allocation-sensitive slots to the workload process", () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/algo_params.inc"), "utf8");
      assert.match(source,
        /mom_zelhash::zelhash_slot_capacity, "slots",\s+GpuUsm::both, 1024, 0, true, false/);
    });

    test("Octopus portable OpenCL uses exact field multiply-add reduction", () => {
      const source = readSyclSource(path.join(__dirname, "../sycl/octopus/octopus.cpp"));
      assert.match(source,
        /if constexpr \(mom_sycl_portable_opencl\) \{[\s\S]*?return mod_field\(static_cast<uint64_t>\(a\) \* b \+ c\);/);
      assert.match(source,
        /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?static_cast<uint64_t>\(static_cast<uint32_t>\(value\)\) \* RECIPROCAL/);
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
              path.join(__dirname, "native/octopus_polynomial.cpp"), "-o", executable],
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

    test("Verthash keeps the large portable CPU dataset in shared USM", () => {
      const source = fs.readFileSync(path.join(__dirname, "../sycl/verthash/verthash.cpp"), "utf8");
      assert.match(source,
        /const bool cpu_data = device\.is_cpu\(\);[\s\S]*?cpu_data \? sycl::malloc_shared<Uint2>[\s\S]*?: sycl::malloc_device<Uint2>/);
      assert.match(source,
        /const size_t data_bytes = host\.size\(\) \* sizeof\(Uint2\);/);
      assert.match(source,
        /if \(cpu_data\)\s+std::memcpy\(data, host\.data\(\), data_bytes\);\s+else\s+sycl_wait_and_throw\(queue\.memcpy\(data, host\.data\(\), data_bytes\)/);
    });

    test("Xelis portable OpenCL avoids vendor-specific 64-bit mul_hi", () => {
      const source = readSyclSource(path.join(__dirname, "../sycl/xelishashv3/xelishashv3.cpp"));
      assert.match(source,
        /#ifdef MOM_XELISHASHV3_HOST_TEST\s+return static_cast<uint64_t>\(a\) \* b;\s+#else\s+return mo_mul_wide_u32\(a, b\);/);
      assert.match(source,
        /const uint32_t a0 = static_cast<uint32_t>\(a\), a1 = a >> 32,[\s\S]*?return mul_wide32\(a1, b1\)/);
      assert.match(source,
        /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?return mul_hi64_portable\(a, b\);[\s\S]*?#else\s+return sycl::mul_hi\(a, b\);/);
    });

    test("HooHash portable OpenCL avoids unsupported 64-bit mul_hi", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/hoohash/hoohash.cpp"), "utf8");
      const shared = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
      assert.match(source, /return mo_mul_hi_u64\(a, b\);/);
      assert.match(shared,
        /#if defined\(MOM_SYCL_PORTABLE_OPENCL\)\s+const uint32_t a0 = static_cast<uint16_t>\(a\), a1 = a >> 16,[\s\S]*?return p0 \+ \(\(p1 \+ p2\) << 16\) \+ \(p3 << 32\);/);
      assert.match(shared,
        /#if defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?mo_mul_wide_u32\(a1, b1\)/);
    });

    test("HooHash verifies GPU filter candidates with the canonical host hash", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/hoohash/hoohash.cpp"), "utf8");
      const hostMath = fs.readFileSync(
        path.join(__dirname, "../sycl/hoohash/host_math.c"), "utf8");
      const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      const searchStart = source.indexOf("static sycl::event search(");
      const testStart = source.indexOf("static sycl::event test_hash(");
      const testEnd = source.indexOf("} // namespace mom_hoohash", testStart);
      assert.ok(searchStart >= 0 && testStart > searchStart && testEnd > testStart,
        "HooHash search/test functions must be inside the expected namespace");
      const testKernel = source.slice(testStart, testEnd);
      const searchKernel = source.slice(searchStart, testStart);
      assert.match(testKernel, /single_task<TestKernel>[\s\S]*?filter_hash\(/);
      assert.match(searchKernel, /filter_hash\([\s\S]*?meets_target\(/);
      assert.doesNotMatch(searchKernel, /canonical_hash\(/);
      assert.doesNotMatch(searchKernel, /\bis_test\b/);
      assert.match(source,
        /mom_hoohash_canonical_hash\(input, candidate_nonce, output\);[\s\S]*?if \(meets_target\(output, target\)\)/);
      assert.match(source,
        /fetch_min\(static_cast<uint32_t>\(id\)\)[\s\S]*?remaining -= consumed;[\s\S]*?first_nonce = candidate_nonce \+ 1;/);
      assert.match(source,
        /\+\+rejected_candidates == MAX_FILTER_RETRIES[\s\S]*?return 0;/);
      assert.match(hostMath,
        /return mom_hoohash_exp\(mom_hoohash_sin\(y\) \+ mom_hoohash_cos\(y\)\);[\s\S]*?return 1 \/ mom_hoohash_sqrt\(mom_hoohash_fabs\(y\) \+ 1\);/);
      assert.match(hostMath,
        /dlvsym\(RTLD_NEXT, #name, "GLIBC_2\.2\.5"\)[\s\S]*?pthread_once\(&math_once, resolve_math\);[\s\S]*?MOM_HOST_MATH_WRAPPER\(sin\)[\s\S]*?MOM_HOST_MATH_WRAPPER\(fabs\)/);
      assert.match(source,
        /if \(const char\* error = mom_hoohash_math_error\(\)\)[\s\S]*?throw std::string\(error\);/);
      assert.equal(binding.match(/"sycl\/hoohash\/host_math\.c"/g)?.length, 1);
      assert.match(binding,
        /"target_name": "hoohash_host"[\s\S]*?"-fno-fast-math"[\s\S]*?"\/fp:strict"[\s\S]*?"target_name": "sycl"[\s\S]*?"dependencies": \[ "hoohash_host" \]/);
      assert.match(hostMath,
        /if \(y == pi \/ 2 \|\| y == 3 \* pi \/ 2\)[\s\S]*?return 0;/);
    });

    test("Windows unified GPU workers link HooHash's strict host verifier", () => {
      const cuda = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
      const acpp = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-sycl-adaptivecpp-win.ps1"), "utf8");
      for (const script of [cuda, acpp]) {
        assert.match(script, /-fno-fast-math[\s\S]*?-ffp-contract=off[\s\S]*?-fno-builtin/);
        assert.match(script, /sycl\\hoohash\\host_math\.c/);
        assert.match(script, /hoohash_host\.obj/);
      }
      assert.match(cuda, /\$objs \+= \$hoo[\s\S]*?Invoke-MomIncrementalLink \$clang \([^\n]*'-shared'[^\n]*\+ \$objs[^\n]*\) \$objs \$out/);
      assert.match(acpp, /\$objects \+= \$hoohashObject[\s\S]*?Invoke-MomIncrementalLink \$python \([^\n]*'-shared'[^\n]*\+ \$objects[^\n]*\) \$objects \$out/);
    });
  });

  test.describe("Equihash geometry, cache policy, and compiler scoping", {concurrency: false}, () => {
    test("BeamHash3 paired 32-bit arithmetic stays scoped to Intel oneAPI", () => {
      const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      const source = fs.readFileSync(path.join(__dirname, "../sycl/beamhash3/common.inc"), "utf8");
      assert.equal(binding.match(/MOM_BEAMHASH3_INTEL_PAIR32/g)?.length, 2);
      assert.match(binding, /"\/DMOM_BEAMHASH3_INTEL_PAIR32"/);
      assert.match(binding, /mom_sycl_impl=='dpcpp'[\s\S]*?-DMOM_BEAMHASH3_INTEL_PAIR32/);
      assert.equal(source.match(/defined\(MOM_BEAMHASH3_INTEL_PAIR32\)/g)?.length, 2);
    });

    test("BeamHash3 compact kernels request subgroup 16 through the portable compiler gate", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/beamhash3/compact_solver.inc"), "utf8");
      assert.match(source,
        /parallel_for<BeamCompactSeedKernel>[\s\S]*?MOM_REQD_SG_16/);
      assert.match(source,
        /parallel_for<BeamCompactRoundKernel<R, PairPartitions>>[\s\S]*?MOM_REQD_SG_16/);
    });

    test("BeamHash3 portable worker cannot select uncompiled compact arenas", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/beamhash3/layout.inc"), "utf8");
      assert.match(source,
        /beam_compact_enabled[\s\S]*?#if defined\(MOM_SYCL_PORTABLE_OPENCL\)[\s\S]*?return false;[\s\S]*?#else/);
    });

    test("Equihash 192,7 scopes wider first-round partitioning to HIP", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      assert.match(source,
        /defined\(MOM_SYCL_HAS_HIP\)[\s\S]*?default_round0_partitions = 8;[\s\S]*?#else[\s\S]*?default_round0_partitions = 2;/);
      assert.match(source, /unsigned Round0Partitions = default_round0_partitions/);
    });

    test("Equihash 192,7 derives first split partitions from the active bucket width", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      assert.match(source,
        /constexpr unsigned partitions = 1u << \(15 - ActiveRound1::bucket_bits\);/);
      assert.match(source,
        /typename ActiveArena::Round2Record, partitions>/);
    });

    test("Equihash 192,7 scopes split-record partitioning to Level Zero", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      const splitOutput = source.indexOf("submit_collision_round_bucketed_split_output_partitioned");
      const start = source.lastIndexOf("if (level_zero_)", splitOutput);
      const end = source.indexOf("if (level_zero_)", splitOutput);
      const splitRounds = source.slice(start, end);
      assert.equal((splitRounds.match(/submit_collision_round_bucketed_split_output_partitioned</g) || []).length, 1);
      assert.equal((splitRounds.match(/submit_split_collision_round_bucketed_partitioned</g) || []).length, 1);
      assert.equal((splitRounds.match(/submit_collision_round_bucketed_split_output</g) || []).length, 1);
      assert.equal((splitRounds.match(/submit_split_collision_round_bucketed</g) || []).length, 1);
      assert.match(splitRounds,
        /constexpr unsigned partitions =\s*std::is_same_v<ActiveArena, HybridArena> \? 2 : 4;/);
      assert.match(splitRounds, /typename ActiveArena::Round3Record, partitions>/);
      assert.match(splitRounds, /typename ActiveArena::Round4Record, 4>/);
    });

    test("ZHash bucket tuning stays scoped to Intel oneAPI", () => {
      const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      const core = fs.readFileSync(
        path.join(__dirname, "../sycl/zhash/equihash_core.hpp"), "utf8");
      const layout = fs.readFileSync(
        path.join(__dirname, "../sycl/zhash/zhash_session_layout.hpp"), "utf8");
      assert.equal((binding.match(/MOM_ZHASH_INTEL_LATE_BUCKETS/g) || []).length, 2);
      assert.match(binding, /"\/DMOM_ZHASH_INTEL_LATE_BUCKETS"/);
      assert.match(binding,
        /mom_sycl_impl=='dpcpp'[\s\S]*?-DMOM_ZHASH_INTEL_LATE_BUCKETS/);
      assert.match(layout,
        /#ifdef MOM_ZHASH_INTEL_LATE_BUCKETS[\s\S]*?ZHashArenaLayout<4480, 5632, 13, 13>[\s\S]*?ZHashArenaLayout<4672, 5632, 13, 13>/);
      assert.match(core,
        /#elif defined\(MOM_ZHASH_INTEL_LATE_BUCKETS\)[\s\S]*?Spec::n == 144 && Spec::k == 5 && LocalBits == 11;/);
    });

    test("BMG combined AOT defaults and runtime wiring stay fail-closed", () => {
      const cxx = fs.readFileSync(path.join(__dirname, "../scripts/cxx-combined.sh"), "utf8");
      assert.match(cxx,
        /MOM_COMBINED_TARGETS:\?MOM_COMBINED_TARGETS must be set by combined-build\.sh/);
      assert.doesNotMatch(cxx, /MOM_COMBINED_TARGETS:-spir64/);
      assert.match(cxx,
        /if \[ -n "\$intel_aot_device" \] && \[\[ "\$src" == \*sycl\/zhash\/zhash\.cpp \]\]; then/);
      assert.match(cxx, /MOM_INTEL_AOT_DEVICE requires spir64_gen in MOM_COMBINED_TARGETS/);
      assert.match(cxx, /MOM_INTEL_AOT_DEVICE requires the pinned \/usr\/bin\/ocloc/);
      assert.match(cxx, /\[ -x \/usr\/bin\/ocloc \]/);
      assert.match(cxx, /intel_aot_args=\(-Xsycl-target-backend=spir64_gen "-device \$intel_aot_device"\)/);

      const entrypoint = fs.readFileSync(
        path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
      const start = entrypoint.indexOf("build_dpcpp() {");
      const end = entrypoint.indexOf("\n}\n\nbuild_dpcpp_opencl()", start);
      assert.ok(start >= 0 && end > start, "DPC++ build function must exist");
      const dpcpp = entrypoint.slice(start, end);
      assert.match(dpcpp, /bash scripts\/combined-build\.sh/);
      assert.doesNotMatch(dpcpp, /MOM_COMBINED_TARGETS=/);
      assert.doesNotMatch(dpcpp, /MOM_INTEL_AOT_DEVICE=/);
      const openclStart = entrypoint.indexOf("build_dpcpp_opencl() {");
      const openclEnd = entrypoint.indexOf("\n}\n\nbuild_acpp()", openclStart);
      assert.ok(openclStart >= 0 && openclEnd > openclStart,
        "DPC++ OpenCL build function must exist");
      const dpcppOpencl = entrypoint.slice(openclStart, openclEnd);
      assert.match(dpcppOpencl,
        /MOM_DPCPP_IMPL=dpcpp-opencl MOM_COMBINED_TARGETS=spir64 MOM_INTEL_AOT_DEVICE=/);

      const combined = fs.readFileSync(
        path.join(__dirname, "../scripts/combined-build.sh"), "utf8");
      assert.match(combined,
        /MOM_COMBINED_TARGETS="\$\{MOM_COMBINED_TARGETS:-spir64,spir64_gen,nvidia_gpu_sm_80\}"/);
      assert.match(combined,
        /\[ -z "\$\{MOM_INTEL_AOT_DEVICE:-\}" \] && \[\[ ",\$MOM_COMBINED_TARGETS," == \*,spir64_gen,\* \]\]/);
      assert.match(combined, /export MOM_INTEL_AOT_DEVICE=bmg-g21/);

      const dockerfile = fs.readFileSync(
        path.join(__dirname, "../scripts/build-combined.dockerfile"), "utf8");
      assert.match(dockerfile, /intel_aot_device="\$\{MOM_INTEL_AOT_DEVICE:-\}"/);
      assert.match(dockerfile, /export MOM_COMBINED_TARGETS="\$combined_targets"/);
      assert.match(dockerfile, /export MOM_INTEL_AOT_DEVICE="\$intel_aot_device"/);

      const runtime = fs.readFileSync(
        path.join(__dirname, "../scripts/install-intel-compute-runtime.sh"), "utf8");
      assert.match(runtime, /\n {2}intel-ocloc\n/);
      assert.match(runtime, /intel-ocloc_\$\{COMPUTE_RUNTIME_VERSION\}-0_amd64\.deb/);
      assert.match(runtime, /12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e/);
    });

    test("Equihash 192,7 scopes late-round partitioning to Level Zero", () => {
      const source = fs.readFileSync(
        path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      const start = source.indexOf("submit_split_collision_round_bucketed<");
      const end = source.indexOf("#endif", start);
      const lateRounds = source.slice(start, end);
      assert.match(lateRounds, /if \(level_zero_\)/);
      assert.equal((lateRounds.match(/submit_collision_round_bucketed_partitioned</g) || []).length, 2);
      assert.equal((lateRounds.match(/submit_collision_round_bucketed</g) || []).length, 2);
    });

    test("CN/GPU keeps Windows HIP input and output in device memory", () => {
      const state = fs.readFileSync(path.join(__dirname, "../sycl/cn_gpu/state.inc"), "utf8");
      assert.doesNotMatch(state, /mom_is_hip\(device\)/);
      assert.match(state, /outputs = allocate<uint8_t>\([^;]*shared_io\)/);
    });

    test("Equihash 192,7 gates its HIP input cache on local memory", () => {
      const session = fs.readFileSync(
        path.join(__dirname, "../sycl/equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      const shared = fs.readFileSync(
        path.join(__dirname, "../sycl/zhash/equihash_sycl.hpp"), "utf8");
      assert.match(shared, /bool CacheInput = false/);
      assert.match(shared, /partitioned_collision_local_bytes/);
      assert.match(session,
        /local_mem_size>\(\) >= required_cache_local_bytes/);
      assert.match(session,
        /if \(cache_partition_inputs_\) return run_with_cache<true>\(header\);/);
    });

    test("ZHash gates its HIP collision caches on local memory", () => {
      const session = fs.readFileSync(
        path.join(__dirname, "../sycl/zhash/zhash_session.hpp"), "utf8");
      const shared = fs.readFileSync(
        path.join(__dirname, "../sycl/zhash/equihash_sycl.hpp"), "utf8");
      assert.match(shared, /bool CacheHead = false/);
      assert.match(shared, /partitioned_split_collision_local_bytes/);
      assert.match(session,
        /cache_partition_inputs_ = [\s\S]*?local_mem_size>\(\) >=[\s\S]*?required_cache_local_bytes/);
      assert.match(session, /Round1Record, 8, true/);
      assert.match(session, /Round2Record, 4, true/);
      assert.match(session, /Round3Record, Round4Record, 4, true/);
    });
  });

  test.describe("NexaPoW and DAG dispatch tuning", {concurrency: false}, () => {
    test("NexaPoW vectors keep the bounded default and expose a fail-closed staged gate", () => {
      const source = readSyclSource(path.join(__dirname, "../sycl/nexapow/nexapow.cpp"));
      const probe = fs.readFileSync(path.join(__dirname, "../sycl/nexapow/validation.cpp"), "utf8");
      assert.match(probe, /single_task<TestShaKernel>/);
      assert.match(probe, /MOM_NEXAPOW_SHA_ONLY/);
      assert.match(source, /MOM_NEXAPOW_STAGED_TEST/);
      assert.match(source, /MOM_NEXAPOW_STAGED_REQUIRE/);
      assert.match(source, /NexaPoW staged SYCL active \(field=%s, table=%s, points=%u, bytes=%llu, count=%u\)/);
      assert.equal((source.match(/const bool recorded_test = is_test && input_size == 48u;/g) || []).length, 2);
      assert.match(source, /if \(recorded_test && !staged_test && !portable_cpu_test\) \{\s*test_recorded_vector\(/);
      assert.match(source, /NexaPoW staged SYCL recorded vector mismatch/);
      const vector = require("./vectors").hashTests.find(({job}) => job.algo === "nexapow");
      assert.ok(vector);
      assert.equal(vector.timeoutMs, 25 * 60 * 1000);
    });

    test("NexaPoW staged tables use the Windows HIP small table and capacity elsewhere", () => {
      const binding = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      const windowsBuild = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-sycl-cuda-win.ps1"), "utf8");
      const pipeline = fs.readFileSync(path.join(__dirname, "../sycl/nexapow/sycl_pipeline.inc"), "utf8");
      const source = readSyclSource(path.join(__dirname, "../sycl/nexapow/nexapow.cpp"));
      assert.doesNotMatch(binding, /MOM_NEXAPOW_LARGE_TABLE/);
      assert.doesNotMatch(windowsBuild, /MOM_NEXAPOW_LARGE_TABLE/);
      assert.match(pipeline, /kSmallTable\s*\{22u, 4u, 21u, 12u\}/);
      assert.match(pipeline, /kLargeTable\s*\{24u, 3u, 23u, 11u\}/);
      assert.match(pipeline,
        /#if defined\(_WIN32\) && defined\(MOM_SYCL_HAS_HIP\)[\s\S]*?preferred = sycl_pipeline::kSmallTable;[\s\S]*?#else[\s\S]*?preferred =\s*memory >= \(uint64_t\{10\} << 30\)\s*\? sycl_pipeline::kLargeTable\s*:\s*sycl_pipeline::kSmallTable;[\s\S]*?#endif/);
      assert.match(pipeline, /memory >= \(uint64_t\{10\} << 30\)/);
      assert.match(pipeline, /try_setup\(sycl_pipeline::kSmallTable, small_failure\)/);
      assert.match(pipeline, /large table setup failed:[\s\S]*small table setup failed:/);
      assert.match(source, /state\.portable\.table_name\(\)/);
      assert.match(source, /state\.portable\.table_points\(\)/);
      assert.match(source, /state\.portable\.table_bytes\(\)/);
      assert.match(pipeline, /const char\* table_name\(\)/);
      assert.match(pipeline, /uint32_t table_points\(\)/);
      assert.match(pipeline, /uint64_t table_bytes\(\)/);
    });

    test("NexaPoW uses the measured Level Zero batch and combined hash default", () => {
      const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
      const pipeline = fs.readFileSync(path.join(__dirname, "../sycl/nexapow/sycl_pipeline.inc"), "utf8");
      assert.match(intensity,
        /sycl_is_level_zero_gpu\(dev\) && memory >= 10 \* GiB\)\s*return 1u << 21/);
      assert.match(pipeline,
        /split_hash_value \? std::strcmp\(split_hash_value, "0"\)\s*:\s*!sycl_is_level_zero_gpu/);
    });

    test("DPC++ final addon and Windows SYCL links split device code per kernel", () => {
      const build = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
      const momStart = build.indexOf('"target_name": "mom"');
      const syclStart = build.indexOf('"target_name": "sycl"', momStart);
      const syclEnd = build.indexOf('"target_name":', syclStart + 1);
      assert(momStart >= 0 && syclStart > momStart && syclEnd > syclStart);
      const finalAddon = build.slice(momStart, syclStart);
      const syclDll = build.slice(syclStart, syclEnd);
      assert.match(finalAddon,
        /mom_sycl_impl=='dpcpp' or mom_sycl_impl=='dpcpp-combined'[\s\S]*?"ldflags\+":\s*\[ "-fsycl-device-code-split=per_kernel" \]/);
      const linker = syclDll.indexOf('"VCLinkerTool"');
      assert(linker >= 0);
      const linkerOptions = syclDll.slice(linker).match(/"AdditionalOptions":\s*\[([^\]]*)\]/)?.[1];
      assert(linkerOptions);
      assert.match(linkerOptions, /"\/DLL"/);
      assert.doesNotMatch(linkerOptions, /fsycl|device-code-split/);
      const producer = fs.readFileSync(path.join(__dirname,
        "../.github/workflows/scripts/build-windows.ps1"), "utf8");
      const driverProperty = producer.match(/"\/p:DPCPPLINKOptions=([^"\r\n]+)"/);
      assert(driverProperty, "Windows SYCL link must forward the driver-prefix property");
      assert.equal(driverProperty[1], "/clang:-fsycl-device-code-split=per_kernel");
    });

    test("Nexa performance tests retain their extended timeout", () => {
      const nexa = require("./vectors").perfTests.find(({algo}) => algo === "nexapow");
      assert.ok(nexa);
      assert.equal(nexa.timeoutMs, 15 * 60 * 1000);
    });

    test("Verthash defaults use safe Level Zero batches", () => {
      const intensity = fs.readFileSync(path.join(__dirname, "../sycl/intensity.inc"), "utf8");
      const start = intensity.indexOf("static unsigned verthash_intensity");
      const end = intensity.indexOf("\n}\n", start);
      assert.ok(start >= 0 && end > start, "Verthash intensity function must exist");
      const block = intensity.slice(start, end);
      assert.match(block, /sycl_is_level_zero_gpu\(dev\) && !is_integrated_gpu\(dev\)/);
      assert.match(block, /#ifdef _WIN32[\s\S]*return 1u << 18;[\s\S]*#else[\s\S]*return 1u << 19;/);
      assert.match(block, /return 1u << 16;/);
    });

    test("KawPow DAG chunks honor overrides and bound integrated-GPU dispatches", () => {
      const state = fs.readFileSync(path.join(__dirname, "../sycl/kawpow/state.inc"), "utf8");
      const internal = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
      assert.match(internal, /inline bool is_integrated_gpu\(const sycl::device& device\)/);
      assert.doesNotMatch(state, /^bool is_integrated_gpu/m,
        "KawPow must use the shared helper instead of declaring a namespace-local Windows symbol");
      const start = state.indexOf("static uint32_t kawpow_dag_chunk_nodes(");
      const end = state.indexOf("\n  }", start);
      assert.ok(start >= 0 && end > start, "KawPow DAG chunk policy must exist");
      const block = state.slice(start, end);
      const override = block.indexOf('parse_env_u32("MOM_KAWPOW_DAG_CHUNK_NODES", parsed)');
      const integrated = block.indexOf("if (is_integrated_gpu(dev))");
      const windows = block.indexOf("#if defined(_WIN32)");
      assert.ok(override >= 0 && integrated > override && windows > integrated,
        "KawPow must apply the explicit override before automatic dispatch bounds");
      assert.match(block, /if \(is_integrated_gpu\(dev\)\)\s*return 1u << 12;/);
      assert.match(block,
        /if \(mom_is_cuda\(dev\) \|\| mom_is_hip\(dev\)\)\s*return 1u << 18;/);
      assert.match(block, /return 0;/);
      const chunkGuard = state.indexOf("if (!chunk_nodes)", end);
      const chunkWait = state.indexOf("sycl_wait_and_throw(dag_event, device);", chunkGuard);
      const chunkAdvance = state.indexOf("start_node += current_nodes;", chunkGuard);
      assert.ok(chunkGuard > end && chunkWait > chunkGuard && chunkAdvance > chunkWait,
        "chunked DAG dispatches must retire before the next chunk is recorded");
    });

    test("ProgPoW benchmarks do not prefetch an unused period bundle", () => {
      const entry = fs.readFileSync(path.join(__dirname, "../sycl/kawpow/kawpow.cpp"), "utf8");
      const state = fs.readFileSync(path.join(__dirname, "../sycl/kawpow/state.inc"), "utf8");
      assert.match(entry,
        /ensure_period_bundle\(period, epoch, dag_mod, !is_benchmark\)/);

      const start = state.indexOf("void ensure_period_bundle(");
      const end = state.indexOf("\n  static uint64_t now_ms", start);
      assert.ok(start >= 0 && end > start, "ProgPoW period-bundle policy must exist");
      const block = state.slice(start, end);
      assert.match(block, /const bool prefetch_next/);
      const reset = block.indexOf("next_bundle.reset();");
      const skip = block.indexOf("if (!prefetch_next)");
      const next = block.indexOf("next_bundle_period = new_period + 1;");
      assert.ok(reset >= 0 && skip > reset && next > skip,
        "benchmark prefetch must stop before next-period metadata or work is created");
    });

    test("Etchash DAG chunks retire integrated-GPU dispatches", () => {
      const state = fs.readFileSync(path.join(__dirname, "../sycl/etchash/state.inc"), "utf8");
      const start = state.indexOf("static uint32_t etchash_dag_chunk_nodes(");
      const end = state.indexOf("\n  }", start);
      assert.ok(start >= 0 && end > start, "Etchash DAG chunk policy must exist");
      const block = state.slice(start, end);
      const override = block.indexOf('mom_parse_env_ulong("MOM_ETCHASH_DAG_CHUNK_NODES", parsed)');
      const integrated = block.indexOf("if (is_integrated_gpu(dev))");
      assert.ok(override >= 0 && integrated > override,
        "Etchash must apply the explicit override before its integrated-GPU default");
      assert.match(block, /if \(is_integrated_gpu\(dev\)\)\s*return 1u << 12;/);
      assert.match(block, /return 0;/);
      const chunkGuard = state.indexOf("if (!chunk_nodes)", end);
      const chunkWait = state.indexOf("sycl_wait_and_throw(dag_event, device);", chunkGuard);
      const chunkAdvance = state.indexOf("start_node += current_nodes;", chunkGuard);
      assert.ok(chunkGuard > end && chunkWait > chunkGuard && chunkAdvance > chunkWait,
        "chunked Etchash DAG dispatches must retire before the next chunk is recorded");
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
    const tuner = require("../miner/gpu_autotune")({
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

    test("PearlHash emits subsequent winning proofs and preserves its seed lifecycle", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const core = fs.readFileSync(path.join(__dirname, "../native/core/execution.inc"), "utf8");
      const branch = core.match(
        /if\s*\(m_dev\s*==\s*DEV::PEARLHASH_GPU\)\s*\{\s*const\s+uint64_t\s+prev_nonce\s*=\s*m_nonce64;[\s\S]*?(?=\s*if\s*\(m_dev\s*==\s*DEV::C30_GPU\))/
      )?.[0];
      assert.ok(branch, "native PearlHash result branch is missing");
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
  const char* pearlhash_claim(uint8_t* jackpot, uint32_t* factor) {
    ++claim_calls;
    if (!available) return nullptr;
    for (unsigned i = 0; i < HASH_LEN; ++i) jackpot[i] = static_cast<uint8_t>(found_seed);
    *factor = 524288;
    proof = "captured-proof-" + std::to_string(found_seed);
    return proof.c_str();
  }
  char* hash_bin2hex(const uint8_t* hash, char* hex, unsigned batch) {
    assert(batch == 0);
    for (unsigned i = 0; i < HASH_LEN; ++i) std::snprintf(hex + 2 * i, 3, "%02x", hash[i]);
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
  for (int solutions : {-1, 0, 2}) {
    Worker pending;
    pending.run(solutions, 10);
    assert(pending.sent.empty() && pending.claim_calls == 0 && pending.m_nonce64 == 16 && pending.active);
  }
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
  std::puts("PASS PearlHash subsequent proofs, claims and seed lifecycle");
}
`);
      try {
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I", path.join(__dirname, ".."), source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
        assert.equal(result.stdout, "PASS PearlHash subsequent proofs, claims and seed lifecycle\n");
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
        fs.copyFileSync(path.join(__dirname, "native/windows_poll_pause.cpp"), path.join(fixture, "probe.cpp"));
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
      const waitStartMarker = "inline void sycl_wait_and_throw(";
      assert.equal(internal.split(waitStartMarker).length, 2);
      const waitStart = internal.indexOf(waitStartMarker);
      const waitEnd = internal.indexOf("\ninline void sycl_log_cleanup_exception", waitStart);
      assert.ok(waitEnd > waitStart, "complete shared SYCL wait helper is missing");
      const wait = internal.slice(waitStart, waitEnd)
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
      const nexa = fs.readFileSync(path.join(root, "nexapow/sycl_pipeline.inc"), "utf8");
      const monolithic = readSyclSource(path.join(root, "nexapow/nexapow.cpp"));
      const equi = fs.readFileSync(path.join(root, "equihash192_7/equihash192_7_direct_session.hpp"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const cut = (source, start, end) => {
        const a = source.indexOf(start), b = source.indexOf(end, a);
        assert.ok(a >= 0 && b > a, start);
        return source.slice(a, b);
      };
      const nexaScope = cut(nexa, "  bool search(sycl::queue&", "    if (!found.count)");
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
        fs.copyFileSync(path.join(__dirname, "native/readback_wait_fault.cpp"), path.join(fixture, "readback.cpp"));
        fs.writeFileSync(path.join(fixture, "wait.inc"), cut(library, "inline void sycl_wait_and_throw(", "\ninline void sycl_log_cleanup_exception("));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"), cut(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "search-result.inc"), cut(nexa, "struct SearchResult {", "struct HeaderContext {"));
        fs.writeFileSync(path.join(fixture, "monolithic-result.inc"), cut(monolithic, "struct Result {", "static constexpr uint8_t kRecordedHeader"));
        for (const route of routes) {
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
      const cn = path.join(__dirname, "../sycl/cn_gpu");
      const entry = fs.readFileSync(path.join(cn, "cn_gpu.cpp"), "utf8");
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
template <typename T> inline T mo_rotate(const T value, const T amount) {
  return std::rotl(value, static_cast<int>(amount & (sizeof(T) * 8 - 1)));
}
template <typename T> inline T mo_bitselect(const T a, const T b, const T c) {
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
  Event memcpy(uint8_t* destination, const uint8_t* source, const size_t bytes) {
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
        const compiled = spawnSync("c++", ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-I", cn, source, "-o", executable], {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8"});
        assert.equal(result.error, undefined);
        assert.equal(result.signal, null);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, "CN/GPU source-derived readback host fixture: 802 checks, 0 failures\n");
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });
  });

  test.describe("Source-JIT fallback boundaries, parameter probes, and GPU/Pearl tuning", {concurrency: false}, () => {
    test("NVRTC temporary programs release once across source-JIT failures", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const header = fs.readFileSync(path.join(root, "cuda-api.h"), "utf8");
      const helpers = header.indexOf("inline void cuda_check(");
      const namespaceEnd = header.lastIndexOf("} // namespace mom");
      assert.ok(helpers >= 0 && namespaceEnd > helpers);
      const ownerStart = header.indexOf("class NvrtcProgram {");
      const owner = ownerStart < 0 ? "" : header.slice(ownerStart, helpers);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-program-"));
      const source = path.join(fixture, "nvrtc_program.cpp");
      const executable = path.join(fixture, "nvrtc_program");
      try {
        fs.copyFileSync(path.join(__dirname, "native/nvrtc_program.cpp"), source);
        fs.writeFileSync(path.join(fixture, "api.inc"), "namespace mom {\n" + owner +
        header.slice(helpers, namespaceEnd) + "\n}\n");
        const cn = fs.readFileSync(path.join(root, "cn_gpu/cuda_jit.inc"), "utf8");
        const cnEnd = cn.indexOf("  void launch(");
        const cnDestructor = cn.indexOf("  ~CnGpuCudaExpand()");
        const cnClassEnd = cn.indexOf("\n};", cnDestructor);
        assert.ok(cnEnd > 0 && cnDestructor > cnEnd && cnClassEnd > cnDestructor);
        fs.writeFileSync(path.join(fixture, "cn.inc"), cn.slice(0, cnEnd) +
        cn.slice(cnDestructor, cnClassEnd) + "\n};\n#endif\n");
        const pearl = fs.readFileSync(path.join(root, "pearlhash/cuda_search.inc"), "utf8");
        const pearlEnd = pearl.indexOf("  void launch(");
        const pearlDestructor = pearl.indexOf("  ~PearlHashCudaSearch()");
        const pearlClassEnd = pearl.indexOf("\n};", pearlDestructor);
        assert.ok(pearlEnd > 0 && pearlDestructor > pearlEnd && pearlClassEnd > pearlDestructor);
        fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) +
        pearl.slice(pearlDestructor, pearlClassEnd) + "\n};\n");
        const flags = owner ? ["-DMOM_FIXTURE_HAS_OWNER=1"] : [];
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", ...flags, "-I", path.join(root, "cn_gpu"),
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
        for (const [variant, name] of ["CN PTX", "Pearl CUBIN", "Pearl forward PTX"].entries()) {
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
        if (owner) {
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
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
    });

    test("CN/gpu CUDA expansion falls back only before native submission", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const directory = path.join(__dirname, "../sycl/cn_gpu");
      const header = fs.readFileSync(path.join(directory, "../cuda-api.h"), "utf8");
      const ownerStart = header.indexOf("class NvrtcProgram {");
      const helpers = header.indexOf("inline void cuda_check(");
      assert.ok(helpers >= 0);
      const owner = ownerStart < 0 ? "" : header.slice(ownerStart, helpers);
      const entry = fs.readFileSync(path.join(directory, "cn_gpu.cpp"), "utf8");
      const start = entry.indexOf("  bool native_expansion = false;");
      const end = entry.indexOf("  if (!native_expansion) {", start);
      assert.ok(start >= 0 && end > start);
      const dispatch = entry.slice(start, end);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cn-cuda-"));
      const source = path.join(fixture, "cuda.cpp");
      const executable = path.join(fixture, "cuda");
      fs.writeFileSync(source, `
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
  int (*device_attribute)(int*, int, int) = +[](int* out, int, int) {
    *out = 9;
    return 0;
  };
  int (*retain_primary)(void**, int) = +[](void** out, int) {
    ++counts.retain;
    *out = &handles[0];
    return 0;
  };
  int (*set_current)(void*) = +[](void*) { return 0; };
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
std::string rtc_error(const NvrtcApi&, int, void*, const char* message) { return message; }
namespace jit_cache {
uint64_t hash(const std::string&) { return 1; }
std::filesystem::path directory() { return {}; }
std::vector<char> read(const std::filesystem::path&) { return {}; }
void write(const std::filesystem::path&, const std::vector<char>&) {}
}
}
#include "cuda_jit.inc"
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
        } else if (id == 1) {
          check(!native && counts.generic == 1 && caught.empty() && counts.launch == 0 &&
                state.native_notice, "startup unavailability did not fall back");
        } else {
          check(id == 2 && native && counts.generic == 0 && caught.empty() &&
                counts.launch == 1 && counts.record == 1 && counts.query == 2,
                "native success changed");
        }
      }
      check(counts.release == (id == 1 ? 0u : 1u) &&
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
      ];
      try {
        for (const enabled of [true, false]) {
          const target = executable + (enabled ? "-enabled" : "-stub");
          const flags = enabled ? ["-DMOM_SYCL_HAS_CUDA=1"] : [];
          const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-pedantic", ...flags, "-I", directory, source, "-o", target], {encoding: "utf8"});
          assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
          const indexes = enabled ? cases.slice(0, 7).map((_name, index) => index) : [7];
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
      const ownerStart = header.indexOf("class HiprtcProgram {");
      const owner = ownerStart < 0 ? "" :
        header.slice(ownerStart, header.lastIndexOf("} // namespace mom")).trim();
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hiprtc-program-"));
      const source = path.join(fixture, "hiprtc_program.cpp");
      const executable = path.join(fixture, "hiprtc_program");
      try {
        fs.copyFileSync(path.join(__dirname, "native/hiprtc_program.cpp"), source);
        fs.writeFileSync(path.join(fixture, "owner.inc"), owner ? "namespace mom {\n" + owner + "\n}\n" : "");
        for (const algorithm of ["octopus", "walahash"]) {
          const caller = fs.readFileSync(path.join(root, algorithm, "amd_wmma.inc"), "utf8");
          const start = caller.indexOf("struct ");
          const end = caller.indexOf("  bool launch(");
          assert.ok(start >= 0 && end > start);
          fs.writeFileSync(path.join(fixture, algorithm + ".inc"), caller.slice(start, end) + "};\n");
        }
        const pearl = fs.readFileSync(path.join(root, "pearlhash/hip_jit.inc"), "utf8");
        const pearlEnd = pearl.indexOf("  void launch(");
        assert.ok(pearlEnd > 0);
        fs.writeFileSync(path.join(fixture, "pearl.inc"), pearl.slice(0, pearlEnd) + "#endif\n};\n");
        const flags = owner ? ["-DMOM_FIXTURE_HAS_OWNER=1"] : [];
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_HIP=1", ...flags, "-I", path.join(root, "pearlhash"),
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
        if (owner) {
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
      const identityFunction = header.indexOf(" hip_queue_identity(");
      const identityStart = header.lastIndexOf("inline ", identityFunction);
      const identityEnd = header.indexOf("// HIPRTC is optional", identityStart);
      const octopus = fs.readFileSync(path.join(root, "octopus/nvidia_tensor.inc"), "utf8");
      const classStart = octopus.indexOf("class OctopusSyclNativeSearch {");
      const contextStart = octopus.indexOf("struct NativeGemmContext {");
      const contextEnd = octopus.indexOf("\n};", contextStart);
      const wmma = fs.readFileSync(path.join(root, "octopus/amd_wmma.inc"), "utf8");
      const wmmaStart = wmma.indexOf("  bool ensure(");
      const wmmaEnd = wmma.indexOf("    hipDeviceProp_t properties{}", wmmaStart);
      assert.ok(identityStart > 0 && identityEnd > identityStart && classStart > contextEnd &&
      contextStart > 0 && contextEnd > contextStart && wmmaStart > 0 && wmmaEnd > wmmaStart);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-interop-"));
      const source = path.join(fixture, "hip_queue_identity.cpp");
      const executable = path.join(fixture, "hip_queue_identity");
      try {
        fs.copyFileSync(path.join(__dirname, "native/hip_queue_identity.cpp"), source);
        fs.writeFileSync(path.join(fixture, "identity.inc"),
          "namespace mom {\n" + header.slice(identityStart, identityEnd) + "}\n");
        fs.writeFileSync(path.join(fixture, "octopus.inc"), octopus.slice(classStart));
        fs.writeFileSync(path.join(fixture, "gemm-context.inc"), octopus.slice(contextStart, contextEnd + 3));
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
        await runGroupedCases(
          t, cases,
          ({target, stage, kind}) => {
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
      assert.equal(policy.selection("karlsenhashv2", "nvidia", "win32").key, "acpp-cuda");
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
      const minerCommand = require("./common/miner_command");
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
      const matrixPath = require.resolve("./common/gpu_test_matrix");
      const cachedMatrix = require.cache[matrixPath];
      delete require.cache[matrixPath];
      t.after(() => {
        if (cachedMatrix) {
          require.cache[matrixPath] = cachedMatrix;
        } else {
          delete require.cache[matrixPath];
        }
      });
      const discovery = require("./common/gpu_test_matrix").sharedDiscovery();
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
    test("PearlHash submitted native faults propagate without fallback", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl/pearlhash");
      const dispatch = fs.readFileSync(path.join(root, "dispatch.inc"), "utf8");
      const host = fs.readFileSync(path.join(root, "pearlhash.cpp"), "utf8");
      const invalidateStart = host.indexOf("  g_pf.valid = false;");
      const invalidateEnd = host.indexOf("  PearlHashState& st =", invalidateStart);
      const waitStart = host.indexOf("  if (search_backend == PearlHashSearchBackend::hip_jit ||");
      const uploadEnd = host.indexOf("  if (is_test && q.get_device().is_gpu())", waitStart);
      const drainStart = host.indexOf("\n  } catch (...)", waitStart);
      const waitEnd = drainStart < 0 ? uploadEnd : drainStart;
      const uploadStart = Math.min(host.indexOf("  if (!st.have_header ||"),
        host.indexOf("  uint8_t tgtLE[32];"));
      const readbackEnd = host.indexOf("  // search() retains", uploadEnd);
      const commitStart = host.indexOf("  if (!b.result->found)", waitEnd);
      const commitEnd = host.indexOf("\n}\n// Builds", commitStart);
      assert.ok(invalidateStart > 0 && invalidateEnd > invalidateStart &&
      waitStart > invalidateEnd && waitEnd > waitStart && commitStart > waitEnd && commitEnd > commitStart &&
      uploadStart > invalidateEnd && uploadEnd >= waitEnd && readbackEnd > uploadEnd);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-native-fault-"));
      const source = path.join(fixture, "pearlhash_fault.cpp");
      const executable = path.join(fixture, "pearlhash_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/pearlhash_fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "dispatch.inc"), dispatch);
        fs.writeFileSync(path.join(fixture, "invalidate.inc"), host.slice(invalidateStart, invalidateEnd));
        fs.writeFileSync(path.join(fixture, "wait.inc"), host.slice(waitStart, waitEnd));
        fs.writeFileSync(path.join(fixture, "upload.inc"), host.slice(uploadStart, uploadEnd));
        fs.writeFileSync(path.join(fixture, "readback.inc"), host.slice(uploadEnd, readbackEnd));
        fs.writeFileSync(path.join(fixture, "commit.inc"), host.slice(commitStart, commitEnd));
        const compiled = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
          "-pedantic", "-DMOM_SYCL_HAS_CUDA=1", "-DMOM_SYCL_HAS_HIP=1", source, "-o", executable],
        {encoding: "utf8"});
        assert.equal(compiled.error, undefined);
        assert.equal(compiled.signal, null);
        assert.equal(compiled.status, 0, compiled.stderr);
        const cases = [];
        for (const [hip, name] of ["CUDA", "HIP"].entries()) {
          cases.push({dispatch: 0, hip, stage: 0, stringError: 0, name: name + " wait succeeds"});
          cases.push({dispatch: 1, hip, stage: 0, stringError: 0, name: name + " native attempt succeeds"});
          cases.push({dispatch: 1, hip, stage: 1, stringError: 0, name: name + " unavailable JIT retains SYCL"});
          for (const [stringError, error] of ["std::exception", "std::string"].entries()) {
            cases.push({dispatch: 0, hip, stage: 7, stringError, name: name + " wait propagates " + error});
            for (const [stage, operation] of [[2, "queue wait"], [3, "preparation"], [6, "launch"]]) {
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
          "-pedantic", "-I" + fixture, path.join(__dirname, "native/autolykos_pipeline_selection.cpp"),
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
      const session = readSyclSource(path.join(root, "sycl/zhash/zhash_session.hpp"));
      const layout = readSyclSource(path.join(root, "sycl/zhash/zhash_session_layout.hpp"));
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
            path.join(__dirname, "native/zhash_schedule_selection.cpp"), "-o", executable], {encoding: "utf8"});
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
        "../sycl/equihash192_7/equihash192_7_direct_session.hpp"));
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-e192-arena-selection-"));
      try {
        const source = path.join(fixture, "fixture.cpp");
        fs.copyFileSync(path.join(__dirname, "native/equihash192_arena_selection.cpp"), source);
        for (const name of ["run_with_cache", "select_cuda_arena"]) {
          const definition = extractCppBlock(session, `  std::vector<Solution> ${name}(`);
          fs.writeFileSync(path.join(fixture, `${name}.inc`),
            "  template <bool CacheInput>\n" + definition.replaceAll("std::chrono::", "fixture_clock::"));
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
      const source = path.join(fixture, "octopus_dag_fault.cpp");
      const executable = path.join(fixture, "octopus_dag_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/octopus_dag_fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "state.inc"),
          extractCppBlock(octopus, "  template <typename T> void free_ptr") + "\n" +
        extractCppBlock(octopus, "  void ensure_dag(") + "\n" +
        extractCppBlock(octopus, "  void ensure_epoch(") + "\n");
        fs.writeFileSync(path.join(fixture, "result.inc"),
          extractCppBlock(octopus, "struct Result {") + ";\n");
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
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-etchash-probe-fault-"));
      const source = path.join(fixture, "etchash_probe_fault.cpp");
      const executable = path.join(fixture, "etchash_probe_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/etchash_probe_fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "probe.inc"),
          extract(state, "  bool use_inline_pair()", "  explicit EtchashState("));
        fs.writeFileSync(path.join(fixture, "result.inc"),
          extract(device, "struct EtchashResult {", "struct Uint2 {"));
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "selection.inc"),
          extract(entry, "    sycl_wait_and_throw(\n        state.use_inline_pair()", "\n  } else {"));
        fs.writeFileSync(path.join(fixture, "commit.inc"),
          extract(entry, "  const uint32_t count = state.result->count;", "\n}"));
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
      const pipeline = fs.readFileSync(path.join(root, "sycl_pipeline.inc"), "utf8");
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
              "-I" + fixture, "-I" + root, path.join(__dirname, "native/nexapow_field_arithmetic.cpp"),
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
      const pipeline = fs.readFileSync(path.join(root, "sycl_pipeline.inc"), "utf8");
      const library = fs.readFileSync(path.join(root, "../lib-internal.h"), "utf8");
      const section = (/** @type {string} */ source, /** @type {string} */ start,
      /** @type {string} */ end) => {
        const begin = source.indexOf(start);
        assert.ok(begin >= 0 && source.indexOf(start, begin + start.length) === -1, start);
        const stop = source.indexOf(end, begin + start.length);
        assert.ok(stop > begin, end);
        return source.slice(begin, stop);
      };
      const prefix = section(pipeline, "class NexaPowSyclSearch {", "  bool search(");
      const probe = fs.readFileSync(path.join(__dirname, "../sycl/nexapow/validation.cpp"), "utf8");
      const caller = extractCppBlock(host,
        "int nexapow(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,\n" +
      "            uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,\n" +
      "            bool is_test, bool is_benchmark, const std::string& dev) {");
      const support = section(host, "struct Result {", "\nstatic constexpr uint8_t") + "\n" +
      section(host, "inline bool np_equal(", "\n#ifndef MOM_NEXAPOW_HOST_TEST");
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nexapow-fault-"));
      const source = path.join(fixture, "nexapow_fault.cpp");
      const executable = path.join(fixture, "nexapow_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/nexapow_fault.cpp"), source);
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
      const source = path.join(fixture, "c30_async_fault.cpp");
      const executable = path.join(fixture, "c30_async_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/c30_async_fault.cpp"), source);
        fs.copyFileSync(path.join(root, "sycl/c30/c30.h"), path.join(fixture, "c30-types.inc"));
        fs.copyFileSync(path.join(root, "native/job-boundary.h"), path.join(fixture, "job-boundary.inc"));
        fs.writeFileSync(path.join(fixture, "solver.inc"),
          extract(c30, "struct Solver {", "\n} // namespace mom::c30"));
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
            "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {"));
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
        }
      } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
      }
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
      assert.equal(completeSearch.split("std::thread(").length, 2);
      const catchStart = execution.indexOf("      } catch (const std::string& err) {",
        execution.indexOf("switch (m_dev)"));
      const catchEnd = execution.indexOf("      if (loop_stats)", catchStart);
      assert.ok(catchStart > 0 && catchEnd > catchStart);
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-c29-submitted-fault-"));
      const source = path.join(fixture, "c29_submitted_fault.cpp");
      const executable = path.join(fixture, "c29_submitted_fault");
      try {
        fs.copyFileSync(path.join(__dirname, "native/c29_submitted_fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "search.inc"),
          completeSearch.replace("std::thread(", "TestThread(") + "\n#include \"entry.inc\"\n");
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
        extract(trim, "profile.mark(\"before tail wait\");", "profile.finish(trimmed_edge_count);") +
        "profile.finish(trimmed_edge_count);\n");
        fs.writeFileSync(path.join(fixture, "active-guard.inc"),
          extract(execution, "    if (!m_has_fn) {", "\n    { // A compute function is active"));
        fs.writeFileSync(path.join(fixture, "dispatch.inc"),
          extract(execution, "          case DEV::C29_GPU:", "          case DEV::C30_GPU:"));
        fs.writeFileSync(path.join(fixture, "compute-catch.inc"), execution.slice(catchStart, catchEnd));
        fs.writeFileSync(path.join(fixture, "accounting.inc"),
          extract(execution, "      uint64_t completed = m_batch;",
            "      if (m_dev == DEV::KAWPOW_GPU || m_dev == DEV::ETCHASH_GPU) {"));
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
        {name: "cn/gpu", file: "cn_gpu/cn_gpu.cpp", end: "  // Initial Keccak", uploads: 1},
        {name: "Etchash", file: "etchash/etchash.cpp", end: "  sycl::queue& q = state.queue;", uploads: 2},
        {name: "KawPow", file: "kawpow/kawpow.cpp", end: "  sycl::queue& q = state.queue;", uploads: 1},
        {name: "FishHash", file: "fishhash/fishhash.cpp", end: "  sycl::event search;", uploads: 2},
        {name: "KarlsenHashV2", file: "fishhash/fishhash.cpp", scope: "int karlsenhashv2(",
          end: "#if !defined(MOM_SYCL_PORTABLE_OPENCL)\n  const bool cooperative =",
          uploads: 2, focused: true, shared: true},
        {name: "Nexa monolithic", file: "nexapow/nexapow.cpp", scope: "  state.init_portable();",
          end: "  sycl_wait_and_throw(state.queue.memset(state.result,", uploads: 3, focused: true},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-host-transfer-fault-"));
      const source = path.join(fixture, "host_transfer_fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host_transfer_fault.cpp"), source);
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
      const source = path.join(fixture, "host_transfer_fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host_transfer_fault.cpp"), source);
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
      const session = fs.readFileSync(path.join(root, "zhash/zhash_session.hpp"), "utf8");
      /** @param {string} source @param {string} start @param {string} end */
      const extract = (source, start, end) => {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        assert.ok(first >= 0 && last > first, start);
        return source.slice(first, last);
      };
      const routes = [
        {name: "fast", copies: 4, failCopy: 3, body: extract(session,
          "      std::array<std::uint32_t, fast_candidate_roots * Spec::proof_indices> fast_leaves{};",
          "      std::copy_n(result_counts.begin()")},
        {name: "tail", copies: 2, failCopy: 2, body: extract(session,
          "        std::vector<std::uint32_t> leaves(",
          "        recover_roots(report, leaves.data(), valid.data());")},
      ];
      const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nested-readback-fault-"));
      const source = path.join(fixture, "host_transfer_fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host_transfer_fault.cpp"), source);
        fs.writeFileSync(path.join(fixture, "cleanup.inc"),
          extract(library, "inline void sycl_log_cleanup_exception(", "sycl::device get_dev("));
        fs.writeFileSync(path.join(fixture, "converted-catch.inc"),
          extract(session, "    } catch (const sycl::exception& error) {", "\nprivate:"));
        fs.writeFileSync(path.join(fixture, "error-report.inc"),
          extract(session, "  RunReport& device_error_report(", "  void recover_roots("));
        for (const route of routes) {
          await t.test(route.name, async (readbackTest) => {
            for (const fixed of [0, 1]) {
              fs.writeFileSync(path.join(fixture, "nested.inc"), fixed ? route.body :
                route.body.replace(/^\s*MomSyclHostTransferGuard[^\n]+\n/m, ""));
              const executable = path.join(fixture, "nested_fault_" + fixed);
              const compiled = spawnSync("c++", ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-fsanitize=address", "-fsanitize-address-use-after-scope",
                "-fno-omit-frame-pointer", ...(process.platform === "linux" ? ["-fno-pie", "-no-pie"] : []),
                "-DMOM_NESTED_FIXTURE=1", "-DMOM_NESTED_FAIL_COPY=" + route.failCopy,
                "-DMOM_NESTED_COPIES=" + route.copies, source, "-o", executable], {encoding: "utf8"});
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
                    assert.equal(result.signal, null);
                    if (!fixed && stage) {
                      assert.equal(result.status, 1, result.stdout + result.stderr);
                      assert.equal(result.stderr, "FAIL nested readback left its try-local destination pending\n");
                    } else {
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

    test("Nexa staged and Octopus entry faults retire uploads and invalidate retry caches", {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, async (t) => {
      const root = path.join(__dirname, "../sycl");
      const library = fs.readFileSync(path.join(root, "lib-internal.h"), "utf8");
      const octopus = readSyclSource(path.join(root, "octopus/octopus.cpp"));
      const nexa = fs.readFileSync(path.join(root, "nexapow/sycl_pipeline.inc"), "utf8");
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
      const source = path.join(fixture, "host_transfer_fault.cpp");
      try {
        fs.copyFileSync(path.join(__dirname, "native/host_transfer_fault.cpp"), source);
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
