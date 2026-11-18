"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {spawnSync} = require("node:child_process");
const {cudaRuntimeInputs, admitSyclSource, validateStaticSyclBackendCache, countHcfKernelGlobals, hcfReproPlan, hcfReproFailures} = require("../scripts/native-sycl-inputs");
const {parseHcfObject, inspectHcfObjects, assertUniqueHcfIds} = require("../scripts/native-hcf");
const {staticSyclSdkPlan, sdkStage, buildStaticSyclSdk, admitCudaLibraries, cmakeCompilerPath, compilerEnvironment,
  sdkHostEnvironment, verifyCompilerEnvironment, staticSyclValidationEnvironment, hostCompilerEvidence} = require("../scripts/build-native-sycl-sdk");
const {cudaBuildHeaders} = require("../scripts/native-gpu-build");
const {fileHash} = require("../scripts/build-source-inputs");
/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sycl-sdk-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  fs.mkdirSync(path.join(root, "lib64"));
  const libraries = ["cudart_static", "culibos"].map(name => {const file = path.join(root, "lib64/lib" + name + ".a"); fs.writeFileSync(file, "!<arch>\n"); return file;});
  const cache = libraries.map((file, i) => `CUDA_${["cudart_static", "culibos"][i]}_LIBRARY:FILEPATH=${file}`).join("\n");
  const calls = /** @type {string[][]} */ ([]);
  const run = (/** @type {string} */ command, /** @type {string[]} */ args) => {calls.push([command, ...args]); return "#define CUDA_VERSION 12060\n#define CUDART_VERSION 12060\n";};
  return {root, libraries, cache, calls, run};
}

/** @param {string} [id] @param {Buffer} [payload] @param {string} [metadata] */
function hcfBytes(id = "123", payload = Buffer.from([0x42, 0x43, 0xc0, 0xde, 1]), metadata = "") {
  const header = Buffer.from(`object-id=${id}\ngenerator=hipSYCL SSCP\nstart=0\nsize=${payload.length}\n${metadata}`);
  return Buffer.concat([Buffer.from("\0"), header, Buffer.from("__acpp_hcf_binary_appendix"), payload, Buffer.from("\0")]);
}

/** @param {string} name @param {string} id @param {string} deviceIRSha256 @param {string} objectSha256
 * @param {string} [normalizedHeaderSha256] */
function reproRecord(name, id, deviceIRSha256, objectSha256, normalizedHeaderSha256 = "header") {
  return {name, id, deviceIRSha256, objectSha256, normalizedHeaderSha256, kernelNameGlobals: 24};
}

test.describe("AdaptiveCpp HCF object reproducibility", () => {
  test("the CLI keeps the CPU host link distinct from the LLVM device compiler", () => {
    const script = fs.readFileSync(path.join(__dirname, "../scripts/native-sycl-inputs.js"), "utf8");
    assert.match(script, /CMAKE_CXX_COMPILER:\[\^=\]\+=/);
    assert.match(script, /runJitCacheIdentity\(hostCxx, flags, source, build, output, repo\)/);
    assert.match(script, /runHcfRepro\(cxx, flags, source, build, output, repo\)/);
  });
  test("the bounded HCF reader extracts a canonical ID and bitcode digest", () => {
    const parsed = parseHcfObject(hcfBytes());
    assert.equal(parsed.id, "123"); assert.equal(parsed.deviceIRBytes, 5);
    assert.match(parsed.objectSha256, /^[a-f0-9]{64}$/); assert.match(parsed.normalizedHeaderSha256, /^[a-f0-9]{64}$/);
  });
  test("marker-like strings inside device bitcode do not create a second HCF record", () => {
    const payload = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.from("object-id=9\n__acpp_hcf_binary_appendix")]);
    assert.equal(parseHcfObject(hcfBytes("123", payload)).id, "123");
  });
  test("nested HCF metadata trees retain their single bounded payload range", () => {
    const metadata = "{.images\n{.llvm-ir.global\nvariant=global-module\nformat=llvm-ir\n{.__binary\n}.__binary\n}.llvm-ir.global\n}.images\n{.kernels\n{.kernel\n{.parameters\n}.parameters\n}.kernel\n}.kernels\n";
    const parsed = parseHcfObject(hcfBytes("123", undefined, metadata));
    assert.equal(parsed.id, "123"); assert.equal(parsed.deviceIRBytes, 5);
  });
  test("an appendix outside the declared bitcode payload makes the object ambiguous", () => {
    const unrelated = Buffer.from("junk__acpp_hcf_binary_appendix");
    assert.throws(() => parseHcfObject(Buffer.concat([hcfBytes(), unrelated])));
  });
  test("the object inspector returns multiple top-level records and an empty list without markers", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hcf-multi-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const multi = path.join(root, "multi.o"), plain = path.join(root, "plain.o");
    fs.writeFileSync(multi, Buffer.concat([hcfBytes("123"), hcfBytes("456")])); fs.writeFileSync(plain, Buffer.from("ordinary COFF/ELF object"));
    assert.deepEqual(inspectHcfObjects(multi).map(record => record.id), ["123", "456"]);
    assert.deepEqual(inspectHcfObjects(plain), []);
  });
  test("the object inspector ignores embedded payload markers but rejects malformed top-level markers", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hcf-marker-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const nested = path.join(root, "nested.o"), malformed = path.join(root, "malformed.o");
    const payload = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.from("__acpp_hcf_binary_appendix")]);
    fs.writeFileSync(nested, hcfBytes("123", payload)); fs.writeFileSync(malformed, Buffer.concat([hcfBytes("123"), Buffer.from("__acpp_hcf_binary_appendix")]));
    assert.deepEqual(inspectHcfObjects(nested).map(record => record.id), ["123"]);
    assert.throws(() => inspectHcfObjects(malformed), /no object ID/);
  });
  test("the selected object union must be nonempty and have unique IDs even for equal payloads", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hcf-union-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const first = path.join(root, "first.o"), duplicate = path.join(root, "duplicate.o"), empty = path.join(root, "empty.o");
    fs.writeFileSync(first, hcfBytes("123")); fs.copyFileSync(first, duplicate); fs.writeFileSync(empty, Buffer.from("no records"));
    assert.equal(assertUniqueHcfIds([first, empty]).length, 1);
    assert.throws(() => assertUniqueHcfIds([first, duplicate]), /Duplicate HCF object ID 123/);
    assert.throws(() => assertUniqueHcfIds([empty]), /No HCF records/);
    assert.throws(() => assertUniqueHcfIds([]), /empty HCF object file list/);
  });
  test("per-object HCF record count is bounded", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hcf-count-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const file = path.join(root, "too-many.o");
    fs.writeFileSync(file, Buffer.concat(Array.from({length: 4097}, (_, index) => hcfBytes(String(index + 1)))));
    assert.throws(() => inspectHcfObjects(file), /Excessive HCF record count/);
  });
  test("the HCF reader rejects missing, ambiguous and malformed records", () => {
    assert.throws(() => parseHcfObject(Buffer.from("no HCF")), /Missing or malformed/);
    assert.throws(() => parseHcfObject(Buffer.concat([hcfBytes(), hcfBytes("456")])), /multiple records/);
    assert.throws(() => parseHcfObject(hcfBytes("123", undefined, "object-id=456\n")), /one bounded object-id field/);
    assert.throws(() => parseHcfObject(hcfBytes("123", undefined,
      "generator=hipSYCL SSCP\nobject-id=456\ngenerator=hipSYCL SSCP\n")));
    assert.throws(() => parseHcfObject(hcfBytes("0")), /object ID/);
    assert.throws(() => parseHcfObject(hcfBytes("0123")), /canonical/);
    assert.throws(() => parseHcfObject(hcfBytes("123", Buffer.from([0, 1, 2, 3]))), /LLVM bitcode/);
  });
  test("duplicate metadata and truncated bitcode fail closed", () => {
    assert.throws(() => parseHcfObject(hcfBytes("123", undefined, "size=5\n")), /one bounded size field/);
    const truncated = hcfBytes().subarray(0, hcfBytes().length - 2);
    assert.throws(() => parseHcfObject(truncated), /Truncated HCF payload/);
  });
  test("same-input objects require matching IDs, metadata, bitcode and complete bytes", () => {
    const records = [reproRecord("same-a", "1", "ir1", "object1"), reproRecord("same-b", "1", "ir1", "object1"),
      reproRecord("payload-v2", "2", "ir2", "object2", "larger-header")];
    assert.deepEqual(hcfReproFailures(records), []);
    const [sameA, sameB, changed] = records; assert.ok(sameA && sameB && changed);
    assert.ok(hcfReproFailures([sameA, {...sameB, objectSha256: "different"}, changed])
      .includes("identical HCF inputs differ in objectSha256"));
  });
  test("changed device content must change both stable ID and complete object", () => {
    const records = [reproRecord("same-a", "1", "ir1", "object1"), reproRecord("same-b", "1", "ir1", "object1"),
      reproRecord("payload-v2", "1", "ir1", "object1")];
    const failures = hcfReproFailures(records);
    assert.ok(failures.includes("changed HCF payload did not change id"));
    assert.ok(failures.includes("changed HCF payload did not change deviceIRSha256"));
    assert.ok(failures.includes("changed HCF payload did not change objectSha256"));
  });
  test("the regression requires at least 20 named kernel globals", () => {
    const records = [reproRecord("same-a", "1", "ir1", "object1"), reproRecord("same-b", "1", "ir1", "object1"),
      reproRecord("payload-v2", "2", "ir2", "object2")];
    const first = records[0]; assert.ok(first); first.kernelNameGlobals = 19;
    assert.ok(hcfReproFailures(records).includes("HCF fixture emitted fewer than 20 named kernel globals"));
  });
  test("symbol parsing counts unique initialized kernel-name objects only", () => {
    const named = "_Zkernel__acpp_sscp_kernel_name.initialized";
    const line = (/** @type {number} */ index, /** @type {string} */ type, /** @type {string} */ name) =>
      ` ${index}: 0000000000000000 4 ${type} LOCAL DEFAULT 2 ${name}`;
    assert.equal(countHcfKernelGlobals([line(1, "OBJECT", named), line(2, "OBJECT", named),
      line(3, "FUNC", named), line(4, "OBJECT", "unrelated")].join("\n")), 1);
    assert.throws(() => countHcfKernelGlobals("x".repeat((1 << 20) + 1)), /Excessive/);
  });
  test("compile plan uses the admitted plugin and namespace with unique compile-only outputs", () => {
    const plans = hcfReproPlan("/build/bin/clang++", ["-fplugin=/build/plugin.so", "-fpass-plugin=/build/plugin.so",
      "-include", "/source/include/hcf_registration.hpp", "-mllvm", "-acpp-sscp", "-Wl,--threads=4", "--ld-path=/build/ld.lld"],
    "/source", "/build", "/evidence", "/repo");
    assert.deepEqual(plans.map(plan => [plan.name, plan.value]), [["same-a", 1], ["same-b", 1], ["payload-v2", 2]]);
    assert.equal(new Set(plans.map(plan => plan.object)).size, 3);
    assert.equal(new Set(plans.map(plan => plan.depfile)).size, 3);
    const normalizeOutputs = (/** @type {{args: string[], object: string, depfile: string}} */ plan) =>
      plan.args.map(arg => arg === plan.object ? "<object>" : arg === plan.depfile ? "<depfile>" : arg);
    const [sameA, sameB, payloadV2] = plans; assert.ok(sameA && sameB && payloadV2);
    assert.deepEqual(normalizeOutputs(sameA), normalizeOutputs(sameB));
    const normalizePayload = (/** @type {{args: string[], object: string, depfile: string}} */ plan) =>
      normalizeOutputs(plan).map(arg => arg.replace(/^-DHCF_PROOF_VALUE=[12]$/, "-DHCF_PROOF_VALUE=<value>"));
    assert.deepEqual(normalizePayload(sameA), normalizePayload(payloadV2));
    for (const plan of plans) {
      assert.equal(plan.args.filter(arg => arg === "-c").length, 1);
      assert.ok(plan.args.includes("-fplugin=/build/plugin.so")); assert.ok(plan.args.includes("-fpass-plugin=/build/plugin.so"));
      assert.ok(plan.args.includes("-acpp-sscp-hcf-object-id-namespace=acpp-hip"));
      assert.ok(plan.args.includes(`-DHCF_PROOF_VALUE=${plan.value}`));
      assert.ok(plan.args.includes(path.join("/repo", "tests/reference/native-sycl-hcf-repro.cpp")));
      assert.equal(plan.args.some(arg => arg.startsWith("-Wl,") || arg.startsWith("--ld-path=")), false);
      assert.equal(plan.args[plan.args.indexOf("-o") + 1], plan.object);
      assert.equal(plan.args[plan.args.indexOf("-MF") + 1], plan.depfile);
    }
  });
});

test.describe("Pinned static SYCL source admission", {skip: process.platform !== "linux"}, () => {
  const revision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
  const patch = (/** @type {boolean} */ gpu) => ["adaptivecpp-static.patch", ...(gpu ? ["adaptivecpp-jit.patch"] : [])]
    .map(name => fs.readFileSync(path.resolve(__dirname, "../runtime", name), "utf8")).join("");
  /** Git input transcripts only; no SDK, compiler, hardware or network.
   * @param {string} [others] @param {string} [head] @param {string} [diff] */
  function source(others = "", head = revision, diff = patch(true)) {
    const calls = /** @type {string[][]} */ ([]);
    const execute = (/** @type {string} */ tool, /** @type {string[]} */ args) => {
      assert.equal(tool, "git"); calls.push(args);
      if (args.includes("rev-parse")) {return head + "\n";}
      if (args.includes("diff")) {return diff;}
      assert.ok(args.includes("ls-files")); return others;
    };
    return {calls, execute, admit: () => admitSyclSource("/source", "/output", true, execute)};
  }
  test("the reviewed GPU revision and patches remain admitted", () => {assert.equal(source().admit().revision, revision);});
  test("CPU-only admission needs only the reviewed static patch", () => {const f = source("", revision, patch(false)); assert.equal(admitSyclSource("/source", "/output", false, f.execute).revision, revision);});
  test("patch block ordering and empty context markers retain canonical equivalence", () => {
    const diff = patch(true).split(/(?=^diff --git )/m).filter(Boolean).reverse().join("").replace(/^ $/gm, "");
    assert.equal(source("", revision, diff).admit().revision, revision);
  });
  test("a different upstream revision fails before further queries", () => {const f = source("", "wrong"); assert.throws(f.admit, /pinned AdaptiveCpp/); assert.equal(f.calls.length, 1);});
  test("GPU source cannot borrow CPU-only patch admission", () => {assert.throws(source("", revision, patch(false)).admit, /reviewed/);});
  test("an extra tracked source change remains rejected", () => {assert.throws(source("", revision, patch(true) + "unreviewed change\n").admit, /reviewed/);});
  test("an untracked source header cannot hide behind a matching tracked diff", () => {assert.throws(source("include/shadow.hpp\n").admit, /reviewed/);});
  test("an untracked directory cannot add compiler inputs", () => {assert.throws(source("include/extra/\n").admit, /reviewed/);});
  test("any nonempty untracked transcript fails closed", () => {assert.throws(source("\n").admit, /reviewed/);});
  test("Git queries never request index updates or ignore exclusions", () => {
    const f = source(); f.admit();
    assert.equal(f.calls.length, 3);
    for (const args of f.calls) {assert.ok(args.includes("--no-optional-locks")); assert.equal(args.includes("--exclude-standard"), false);}
    assert.deepEqual(f.calls[2], ["--no-optional-locks", "-C", "/source", "ls-files", "--others", "--directory", "--"]);
  });
  test("DPC++ source admission applies the same ignored-file boundary", () => {
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/native-dpcpp-pins.js"), "utf8");
    assert.match(script, /"ls-files", "--others", "--directory", "--"\]\)\.length !== 0/);
    assert.doesNotMatch(script, /"--exclude-standard"/);
  });
  test("an ignored source file is still visible to the real Git admission query", t => {
    const f = fixture(t), git = (/** @type {string[]} */ args) => spawnSync("git", ["-C", f.root, ...args], {encoding: "utf8", timeout: 5000});
    assert.equal(git(["init", "-q"]).status, 0);
    fs.writeFileSync(path.join(f.root, ".gitignore"), "lib64/\n*.ignored\n");
    assert.equal(git(["add", ".gitignore"]).status, 0);
    fs.writeFileSync(path.join(f.root, "shadow.ignored"), "unreviewed header");
    assert.equal(git(["ls-files", "--others", "--exclude-standard"]).stdout, "");
    const baseline = source(), execute = (/** @type {string} */ tool, /** @type {string[]} */ args) => {
      if (!args.includes("ls-files")) {return baseline.execute(tool, args);}
      const result = git(args.slice(args.indexOf("ls-files")));
      assert.equal(result.status, 0); assert.match(result.stdout, /shadow\.ignored/); return result.stdout;
    };
    assert.throws(() => admitSyclSource(f.root, "/output", true, execute), /reviewed/);
  });
  test("a failed untracked query propagates without retry or admission", () => {
    const f = source(); let attempts = 0;
    assert.throws(() => admitSyclSource("/source", "/output", true, (tool, args) => {
      if (args.includes("ls-files")) {++attempts; throw new Error("Git query failed");}
      return f.execute(tool, args);
    }), /Git query failed/);
    assert.equal(attempts, 1);
  });
});

test.describe("AdaptiveCpp GPU runtime cache admission", () => {
  const combined = ["WITH_CUDA_BACKEND:BOOL=ON", "WITH_ROCM_BACKEND:BOOL=ON", "WITH_OPENCL_BACKEND:BOOL=OFF",
    "WITH_LEVEL_ZERO_BACKEND:BOOL=OFF", "WITH_VULKAN_BACKEND:BOOL=OFF"].join("\n") + "\n";
  const isolatedCuda = combined.replace("WITH_ROCM_BACKEND:BOOL=ON", "WITH_ROCM_BACKEND:BOOL=OFF"),
    isolatedHip = combined.replace("WITH_CUDA_BACKEND:BOOL=ON", "WITH_CUDA_BACKEND:BOOL=OFF");
  test("CUDA and HIP factories accept combined and isolated vendor caches", () => {
    for (const backend of ["cuda", "hip", "cuda-hip"]) {assert.doesNotThrow(() => validateStaticSyclBackendCache(combined, backend));}
    assert.doesNotThrow(() => validateStaticSyclBackendCache(isolatedCuda, "cuda"));
    assert.doesNotThrow(() => validateStaticSyclBackendCache(isolatedHip, "hip"));
  });
  test("the selected vendor cannot be disabled", () => {
    assert.throws(() => validateStaticSyclBackendCache(combined.replace("WITH_CUDA_BACKEND:BOOL=ON", "WITH_CUDA_BACKEND:BOOL=OFF"), "cuda"), /explicitly selected GPU/);
    assert.throws(() => validateStaticSyclBackendCache(combined.replace("WITH_ROCM_BACKEND:BOOL=ON", "WITH_ROCM_BACKEND:BOOL=OFF"), "hip"), /explicitly selected GPU/);
    assert.throws(() => validateStaticSyclBackendCache(isolatedHip, "cuda-hip"), /explicitly selected GPU/);
    assert.throws(() => validateStaticSyclBackendCache(isolatedCuda, "cuda-hip"), /explicitly selected GPU/);
  });
  test("the other vendor must still have an explicit known cache value", () => {
    for (const [backend, other] of [["cuda", "ROCM"], ["hip", "CUDA"]]) {
      for (const value of ["", `WITH_${other}_BACKEND:BOOL=unknown`]) {
        const cache = combined.replace(`WITH_${other}_BACKEND:BOOL=ON`, value);
        assert.throws(() => validateStaticSyclBackendCache(cache, backend || ""), /explicitly selected GPU/);
      }
    }
  });
  test("unrelated OpenCL, Level Zero and Vulkan backends remain disabled", () => {
    for (const name of ["OPENCL", "LEVEL_ZERO", "VULKAN"]) {
      const cache = combined.replace(`WITH_${name}_BACKEND:BOOL=OFF`, `WITH_${name}_BACKEND:BOOL=ON`);
      for (const backend of ["cuda", "hip", "cuda-hip"]) {
        assert.throws(() => validateStaticSyclBackendCache(cache, backend), /explicitly selected GPU/);
      }
    }
  });
  test("the OpenMP lane remains CPU-only", () => {
    const cpu = combined.replace("WITH_CUDA_BACKEND:BOOL=ON", "WITH_CUDA_BACKEND:BOOL=OFF")
      .replace("WITH_ROCM_BACKEND:BOOL=ON", "WITH_ROCM_BACKEND:BOOL=OFF");
    assert.doesNotThrow(() => validateStaticSyclBackendCache(cpu, "omp"));
    assert.throws(() => validateStaticSyclBackendCache(combined, "omp"), /explicitly selected GPU/);
  });
});

test.describe("Static SYCL pre-main setting isolation", {skip: process.platform !== "linux"}, () => {
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-acpp-init-"));
    const patch = fs.readFileSync(path.resolve(__dirname, "../runtime/adaptivecpp-static.patch"), "utf8");
    // Execute the actual removed/added statements, not a second implementation of the fix.
    const statements = (/** @type {string} */ file, /** @type {string} */ side, /** @type {string} */ hunkContext = "") => {
      const block = patch.split("diff --git a/" + file + " b/" + file + "\n")[1]?.split("diff --git ")[0];
      assert.ok(block, file);
      const hunks = block.split(/(?=^@@ )/m).filter(hunk => hunk.startsWith("@@ ")),
        selected = hunkContext ? hunks.filter(hunk => hunk.split("\n", 1)[0]?.includes(hunkContext)) : hunks;
      assert.equal(selected.length, 1, file + (hunkContext ? " " + hunkContext : ""));
      return (selected[0] || "").split("\n").filter(line => line.startsWith(side) && !line.startsWith(side.repeat(3)))
        .map(line => line.slice(1)).join("\n") + "\n";
    };
    for (let mode = 0; mode < 8; ++mode) {
      const snippets = /** @type {{file:string,name:string,hunkContext?:string}[]} */ ([
        {file: "include/hipSYCL/common/debug.hpp", name: "debug"},
        {file: "src/runtime/kernel_cache.cpp", name: "hcf", hunkContext: "hcf_cache::register_hcf_object"},
        {file: "include/hipSYCL/glue/persistent_runtime.hpp", name: "persistent"},
      ]);
      for (const [index, snippet] of snippets.entries()) {
        fs.writeFileSync(path.join(root, snippet.name + "-setting.inc"),
          statements(snippet.file, mode & (1 << index) ? "+" : "-", snippet.hunkContext));
      }
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", root,
        path.join(__dirname, "native/acpp-initialization.cpp"), "-o", path.join(root, String(mode))], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {string} mode @param {NodeJS.ProcessEnv} [extra] */
  function run(mode, extra = {}) {
    const env = {...process.env};
    for (const key of Object.keys(env)) {if (/^(ACPP_|HIPSYCL_|MOM_TEST_CONFIG_)/.test(key)) {delete env[key];}}
    const result = spawnSync(path.join(root, mode), [], {env: {...env, ...extra}, encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout.trim();
  }
  test("all three pre-main reads must avoid snapshotting worker visibility", () => {
    for (let mode = 0; mode < 7; ++mode) {assert.equal(run(String(mode)), "|2|");}
    assert.equal(run("7"), "hip|2|");
  });
  test("the two fixed reads retain explicit diagnostic values without locking the backend", () => {
    assert.equal(run("7", {ACPP_DEBUG_LEVEL: "4", ACPP_HCF_DUMP_DIRECTORY: "/dump"}), "hip|4|/dump");
  });
  test("both reads still use the common settings API, including legacy and configuration lookup", () => {
    assert.equal(run("7", {HIPSYCL_DEBUG_LEVEL: "3", HIPSYCL_HCF_DUMP_DIRECTORY: "/legacy"}), "hip|3|/legacy");
    assert.equal(run("7", {MOM_TEST_CONFIG_DEBUG_LEVEL: "1", MOM_TEST_CONFIG_HCF_DUMP_DIRECTORY: "/config"}), "hip|1|/config");
  });
  test("an invalid debug level retains its compile-time default", () => {assert.equal(run("7", {ACPP_DEBUG_LEVEL: "bad"}), "hip|2|");});
  test("explicit persistence still initializes the runtime, with current, legacy or configuration settings", () => {
    for (const key of ["ACPP_PERSISTENT_RUNTIME", "HIPSYCL_PERSISTENT_RUNTIME", "MOM_TEST_CONFIG_PERSISTENT_RUNTIME"]) {
      assert.equal(run("7", {[key]: "1", MOM_TEST_ASSERT_PERSISTENCE: "1"}), "|2|");
      assert.equal(run("7", {[key]: "0", MOM_TEST_ASSERT_PERSISTENCE: "0"}), "hip|2|");
    }
  });
});

test.describe("Static CUDA SDK header/runtime ABI admission", () => {
  test("matching12.6 headers resolve explicit regular static libraries", t => {const f = fixture(t), r = cudaRuntimeInputs(f.cache, "clang++", f.run); assert.equal(r.version, 12060); assert.deepEqual(r.libraries, f.libraries);});
  test("preprocessing checks selected headers without loading a GPU driver", t => {
    const f = fixture(t); cudaRuntimeInputs(f.cache, "clang++", f.run);
    assert.deepEqual(f.calls[0], ["clang++", "-E", "-dM", "-x", "c++", "-I", path.join(f.root, "include"), "-include", "cuda.h", "-include", "cuda_runtime_api.h", "/dev/null"]);
  });
  for (const macros of ["", "#define CUDA_VERSION 13020\n#define CUDART_VERSION 13020\n", "#define CUDA_VERSION 12060\n#define CUDART_VERSION 13020\n",
    "#define CUDA_VERSION 13020\n#define CUDART_VERSION 12060\n", "#define CUDA_VERSION 12060\n#define CUDART_VERSION 12060\n#define CUDART_VERSION 12060\n"]) {
    test("missing, mixed, newer or ambiguous header macros fail: " + JSON.stringify(macros), t => {
      const f = fixture(t); assert.throws(() => cudaRuntimeInputs(f.cache, "clang++", () => macros), /matching CUDA12.6/);
    });
  }
  test("missing archive metadata fails before the compiler", t => {const f = fixture(t); assert.throws(() => cudaRuntimeInputs("", "clang++", f.run), /static CUDA library/); assert.equal(f.calls.length, 0);});
  test("thin archives cannot borrow regular-archive admission", t => {const f = fixture(t); fs.writeFileSync(f.libraries[0] || "", "!<thin>\n"); assert.throws(() => cudaRuntimeInputs(f.cache, "clang++", f.run), /static CUDA library/);});
  test("runtime and support libraries cannot come from different SDK prefixes", t => {
    const f = fixture(t), other = path.join(f.root, "other"); fs.mkdirSync(other); const file = path.join(other, "libculibos.a"); fs.writeFileSync(file, "!<arch>\n");
    assert.throws(() => cudaRuntimeInputs(f.cache.replace(f.libraries[1] || "", file), "clang++", f.run), /one SDK/);
  });
  test("unreviewed archive bytes cannot pass the source-build pin despite valid macros", t => {const f = fixture(t); assert.throws(() => admitCudaLibraries(f.libraries), /reviewed12.6.77/);});
  test("a duplicate archive cannot substitute for the runtime/support pair", t => {const f = fixture(t); assert.throws(() => admitCudaLibraries([f.libraries[0] || "", f.libraries[0] || ""]), /reviewed12.6.77/);});
});

test.describe("Static SYCL SDK literal recipe and bounded lifecycle", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  const plan = () => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2);
  const host = {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"};
  test("the SDK recipe hashes the shared HCF probe reader", () => {
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/build-native-sycl-sdk.js"), "utf8");
    assert.match(script, /"scripts\/native-hcf\.js"/);
  });
  test("only the requested runtime/compiler targets are built", () => {assert.deepEqual(plan().commands[1]?.slice(-8), ["acpp-rt", "rt-backend-cuda", "rt-backend-omp", "acpp-common", "acpp-clang", "llvm-to-backend", "llvm-to-host", "llvm-to-ptx"]);});
  test("static runtime archives remain position-independent for the PIE application", () => {
    for (const backend of ["cuda", "hip", "cuda-hip"]) {
      for (const compilers of [undefined, host]) {
        const environment = {MOM_AOT_SYCL_BACKEND: backend, MOM_AOT_ROCM_STATIC_ROOT: "/rocm",
          MOM_AOT_LLD_ROOT: "/lld", ROCM_DEVICE_LIBS_PATH: "/device", MOM_AOT_COMGR_RECEIPT: "/comgr/result.json"};
        const configure = staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, environment, compilers).commands[0] || [];
        assert.ok(configure.includes("-DCMAKE_POSITION_INDEPENDENT_CODE=ON"), backend);
        assert.equal(configure.some(flag => /-no-pie|-fno-pic/i.test(flag)), false);
      }
    }
  });
  test("only the static Linux Intel OpenMP queue opts out of invalid IPO TLS IR", t => {
    const patch = fs.readFileSync(path.resolve(__dirname, "../runtime/adaptivecpp-static.patch"), "utf8"),
      block = patch.match(/^\+ {2}if\(ACPP_STATIC_RUNTIME AND NOT WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"\)\n(?:\+.*\n)+?\+ {2}endif\(\)/m)?.[0];
    assert.ok(block, "the pinned source patch must carry the narrow Intel/OpenMP IPO workaround");
    assert.match(block, /set_property\(SOURCE omp\/omp_queue\.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-lto\)/);
    const cmake = "/usr/bin/cmake", available = spawnSync(cmake, ["--version"], {encoding: "utf8", timeout: 10000, maxBuffer: 32768});
    if (available.error && "code" in available.error && available.error.code === "ENOENT") {t.skip("CMake is not installed"); return;}
    assert.equal(available.status, 0, available.stderr);
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-omp-ipo-policy-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    for (const [name, staticRuntime, windows, compiler, expected] of [
      ["intel-linux-static", "ON", "OFF", "IntelLLVM", "-ipo;-fno-lto"],
      ["intel-windows-static", "ON", "ON", "IntelLLVM", "-ipo"],
      ["intel-linux-shared", "OFF", "OFF", "IntelLLVM", "-ipo"],
      ["clang-linux-static", "ON", "OFF", "Clang", "-ipo"],
    ]) {
      const source = path.join(root, name || ""), build = path.join(source, "build");
      fs.mkdirSync(source);
      fs.writeFileSync(path.join(source, "CMakeLists.txt"), [
        "cmake_minimum_required(VERSION 3.16)", "project(MomOmpIpoPolicy NONE)",
        `set(ACPP_STATIC_RUNTIME ${staticRuntime})`, `set(WIN32 ${windows})`, `set(CMAKE_CXX_COMPILER_ID ${compiler})`,
        "set_property(SOURCE omp/omp_queue.cpp omp/omp_event.cpp PROPERTY COMPILE_OPTIONS -ipo)",
        block.split("\n").map(line => line.slice(1)).join("\n"),
        "get_source_file_property(queue omp/omp_queue.cpp COMPILE_OPTIONS)",
        "get_source_file_property(other omp/omp_event.cpp COMPILE_OPTIONS)",
        `if(NOT queue STREQUAL "${expected}" OR NOT other STREQUAL "-ipo")`,
        '  message(FATAL_ERROR "IPO workaround escaped its single-source/platform/compiler scope")', "endif()", "",
      ].join("\n"));
      const result = spawnSync(cmake, ["-S", source, "-B", build], {encoding: "utf8", timeout: 15000, maxBuffer: 1 << 20});
      assert.equal(result.status, 0, name + ": " + result.stdout + result.stderr);
    }
  });
  test("the explicit host pair changes CMake compilation but not the LLVM SDK identity", () => {
    const p = staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, host), configure = p.commands[0] || [];
    assert.deepEqual(p.hostCompilers, host); assert.equal(p.cxx, host.cxx);
    assert.ok(configure.includes("-DCMAKE_C_COMPILER=" + host.cc)); assert.ok(configure.includes("-DCMAKE_CXX_COMPILER=" + host.cxx));
    assert.ok(configure.includes("-DLLVM_DIR=/llvm/lib/cmake/llvm"));
    assert.equal(configure.some(flag => /-flto|INTERPROCEDURAL_OPTIMIZATION/i.test(flag)), false);
  });
  test("the Intel oneAPI host pair statically links shared and module compiler runtimes only", () => {
    const flags = ["-DCMAKE_SHARED_LINKER_FLAGS=-static-intel", "-DCMAKE_MODULE_LINKER_FLAGS=-static-intel"];
    for (const flag of flags) {
      assert.ok((staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, host).commands[0] || []).includes(flag));
    }
    for (const compilers of [undefined, {cc: "/llvm/bin/clang", cxx: "/llvm/bin/clang++"}, {cc: "/toolchain/gcc", cxx: "/toolchain/g++"},
      {cc: "/opt/intel/icx", cxx: "/opt/intel/icpx"},
      {cc: host.cc, cxx: "/opt/intel/oneapi/compiler/2026.0/bin/icpx"}]) {
      for (const flag of flags) {
        assert.equal((staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, compilers).commands[0] || []).includes(flag), false);
      }
    }
  });
  test("the default bootstrap plan still selects the pinned LLVM compiler pair", () => {
    assert.deepEqual(plan().hostCompilers, {cc: "/llvm/bin/clang", cxx: "/llvm/bin/clang++"});
    assert.ok(plan().commands[0]?.includes("-DCMAKE_CXX_COMPILER=/llvm/bin/clang++"));
  });
  test("host C/C++ compilers are an all-or-nothing literal pair", () => {
    const incomplete = /** @type {Array<{cc?: string, cxx?: string}>} */ ([{cc: "/opt/intel/icx"}, {cxx: "/opt/intel/icpx"}, {}]);
    for (const selected of incomplete) {
      assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, /** @type {{cc: string, cxx: string}} */ (selected)), /absolute, literal pair/);
    }
    for (const file of ["relative/icx", "/opt/intel/icx;bad", "/opt/intel/icx$bad", "/opt/intel/icx\\bad", "/opt/intel/icx\n"] ) {
      assert.equal(cmakeCompilerPath(file), false); assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, {cc: file, cxx: "/opt/intel/icpx"}), /absolute, literal pair/);
    }
  });
  test("output cannot contain or overwrite either selected compiler", () => {
    assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/opt/intel", 2, {},
      {cc: "/opt/intel/bin/icx", cxx: "/opt/intel/bin/icpx"}), /must not enter or enclose/);
    assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/", 2), /must not enter or enclose/);
    assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", path.resolve(__dirname, "../outside"), 2), /must not enter or enclose/);
  });
  test("Autoconf compiler names resolve exact tools through PATH, including paths with spaces", {skip: process.platform === "win32"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-autotools-compilers-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const ccDir = path.join(root, "Intel SDK with spaces"), cxxDir = path.join(root, "Intel's SDK with spaces");
    fs.mkdirSync(ccDir); fs.mkdirSync(cxxDir);
    const compilers = {cc: path.join(ccDir, "mom-test-cc"), cxx: path.join(cxxDir, "mom-test-cxx")};
    fs.writeFileSync(compilers.cc, "#!/bin/sh\nprintf 'selected-cc\\n'\n"); fs.chmodSync(compilers.cc, 0o755);
    fs.writeFileSync(compilers.cxx, "#!/bin/sh\nprintf 'selected-cxx\\n'\n"); fs.chmodSync(compilers.cxx, 0o755);
    assert.deepEqual(compilerEnvironment(compilers), {CC: "mom-test-cc", CXX: "mom-test-cxx"});
    const environment = sdkHostEnvironment({PATH: "/usr/bin:/bin", CC: "ambient", CXX: "ambient", CFLAGS: "bad"}, compilers);
    const pathEntries = (environment["PATH"] || "").split(path.delimiter);
    assert.equal(pathEntries[0], ccDir); assert.equal(pathEntries[1], cxxDir);
    assert.equal(environment["CFLAGS"], undefined);
    verifyCompilerEnvironment(compilers, environment);
    assert.throws(() => verifyCompilerEnvironment(compilers, {...environment, CC: compilers.cc}), /admitted command names/);
    assert.throws(() => verifyCompilerEnvironment(compilers, {...environment, CXX: "other-cxx"}), /admitted command names/);
    const autoconf = [
      'command -v "$CC"',
      'ac_compile=\'$CC\'; set X $ac_compile; ac_compiler=$2; eval "$ac_compiler --version"',
      'command -v "$CXX"',
      'ac_compile=\'$CXX\'; set X $ac_compile; ac_compiler=$2; eval "$ac_compiler --version"',
    ].join("\n");
    const result = spawnSync("/bin/sh", ["-c", autoconf], {env: {...process.env, ...environment}, encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(result.stdout.trim().split(/\r?\n/), [compilers.cc, "selected-cc", compilers.cxx, "selected-cxx"]);
    const quoted = spawnSync("/bin/sh", ["-c", autoconf], {env: {...process.env, PATH: "/usr/bin:/bin", CC: "'" + compilers.cc + "'", CXX: "'" + compilers.cxx + "'"}, encoding: "utf8", timeout: 5000});
    assert.notEqual(quoted.status, 0);
  });
  test("PATH compiler selection rejects ambiguous names and unsafe paths", () => {
    for (const compilers of [{cc: "/one/clang", cxx: "/two/clang"}, {cc: "/one:two/icx", cxx: "/two/icpx"},
      {cc: "/one/bad name", cxx: "/two/icpx"}]) {
      assert.throws(() => compilerEnvironment(compilers), /unique safe names/);
    }
  });
  test("selected host command names scrub ambient compiler settings and retain explicit CMake paths", () => {
    const environment = sdkHostEnvironment({CC: "bad", CXX: "bad", CFLAGS: "bad", CXXFLAGS: "bad", CMAKE_PREFIX_PATH: "bad"}, host);
    assert.equal(environment["CC"], "icx"); assert.equal(environment["CXX"], "icpx");
    assert.ok((environment["PATH"] || "").startsWith("/opt/intel"));
    assert.equal(environment["CFLAGS"], undefined); assert.equal(environment["CXXFLAGS"], undefined); assert.equal(environment["CMAKE_PREFIX_PATH"], undefined);
  });
  test("the existing CUDA validator receives CXX as a direct executable path", t => {
    const f = fixture(t), p = staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, {}, host);
    const nested = sdkHostEnvironment({MOM_AOT_OPENMP_LIBRARY: "/openmp.a"}, host);
    const consumer = staticSyclValidationEnvironment(nested, p.cxx);
    assert.equal(nested["CXX"], "icpx"); assert.equal(consumer["CXX"], host.cxx);
    cudaRuntimeInputs(f.cache, consumer["CXX"] || "", f.run);
    assert.equal(f.calls[0]?.[0], host.cxx);
  });
  test("selected host compiler versions and hashes are captured together", t => {
    const f = fixture(t), compilers = {cc: path.join(f.root, "icx"), cxx: path.join(f.root, "icpx")};
    fs.writeFileSync(compilers.cc, "fake icx"); fs.writeFileSync(compilers.cxx, "fake icpx");
    const calls = /** @type {string[][]} */ ([]), evidence = hostCompilerEvidence(compilers, (command, args, name) => {
      calls.push([command, ...args, name]); return command + " version 1\n";
    });
    assert.deepEqual(evidence.versions, {[compilers.cc]: compilers.cc + " version 1", [compilers.cxx]: compilers.cxx + " version 1"});
    assert.deepEqual(evidence.inputs, {[compilers.cc]: fileHash(compilers.cc), [compilers.cxx]: fileHash(compilers.cxx)});
    assert.deepEqual(calls, [[compilers.cc, "--version", "host-cc-version.log"], [compilers.cxx, "--version", "host-cxx-version.log"]]);
    assert.throws(() => hostCompilerEvidence(compilers, () => "  "), /invalid version/);
    assert.throws(() => hostCompilerEvidence(compilers, () => "clang\0broken"), /invalid version/);
    assert.throws(() => hostCompilerEvidence(compilers, () => "x".repeat((1 << 20) + 1)), /invalid version/);
    assert.throws(() => hostCompilerEvidence(compilers, command => {fs.appendFileSync(command, "changed"); return "compiler version\n";}), /changed during identity query/);
  });
  test("the CUDA source and actual toolkit lookup use the same explicit prefix", () => {const p = plan().commands[0] || []; assert.ok(p.includes("-DCUDA_TOOLKIT_ROOT_DIR=/cuda")); assert.ok(p.includes("-DCUDAToolkit_ROOT=/cuda"));});
  test("unused GPU backends and accelerated CPU/stdpar features are disabled", () => {const p = plan().commands[0] || []; for (const flag of ["-DWITH_ROCM_BACKEND=OFF", "-DWITH_OPENCL_BACKEND=OFF", "-DWITH_LEVEL_ZERO_BACKEND=OFF", "-DWITH_VULKAN_BACKEND=OFF", "-DWITH_STDPAR_COMPILER=OFF", "-DWITH_ACCELERATED_CPU=OFF"]) {assert.ok(p.includes(flag));}});
  test("the maintained resource generator owns embedded JIT data", () => {assert.ok(plan().commands[0]?.some(flag => flag === "-DACPP_RESOURCE_GENERATOR=" + path.resolve(__dirname, "../scripts/native-resources.js")));});
  test("static runtime and the SSCP compiler are mandatory", () => {const p = plan().commands[0] || []; assert.ok(p.includes("-DACPP_STATIC_RUNTIME=ON")); assert.ok(p.includes("-DWITH_SSCP_COMPILER=ON"));});
  for (const jobs of [0, 17, 1.5, NaN]) {test("invalid parallelism fails before CMake: " + jobs, () => {assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", jobs), /jobs/);});}
  test("SDK output cannot enter any source/toolchain root", () => {for (const output of ["/src", "/src/new", "/llvm/new", "/cuda/new"]) {assert.throws(() => staticSyclSdkPlan("/src", "/llvm", "/cuda", output, 2), /must not enter/);}});
  test("relative paths never depend on an incidental cwd", () => {assert.throws(() => staticSyclSdkPlan("src", "/llvm", "/cuda", "/output", 2), /absolute/);});
  test("an occupied SDK is not overwritten", {skip: process.platform !== "linux"}, t => {const f = fixture(t); assert.throws(() => buildStaticSyclSdk("/src", "/llvm", "/cuda", f.root, {MOM_AOT_BUILD_ONLY: "1"}), /existing SDKs/);});
  test("ordinary tests cannot accidentally start an SDK build", () => {assert.throws(() => buildStaticSyclSdk("/src", "/llvm", "/cuda", "/output", {}), /BUILD_ONLY/);});
  test("compiler output streams directly to the diagnostic fd", t => {
    const f = fixture(t), log = path.join(f.root, "stage.log"); sdkStage([process.execPath, "-e", 'process.stdout.write("fixture")'], log, process.env);
    assert.equal(fs.readFileSync(log, "utf8"), "fixture");
  });
  test("failure preserves its exact output and never retries", t => {
    const f = fixture(t), log = path.join(f.root, "stage.log");
    assert.throws(() => sdkStage([process.execPath, "-e", 'process.stderr.write("fixture failure"); process.exit(7)'], log, process.env), /inspect.*before retrying/);
    assert.equal(fs.readFileSync(log, "utf8"), "fixture failure");
  });
  test("failed subprocesses print only a bounded log tail while success remains quiet", t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/build-native-sycl-sdk.js"), limit = 16 * 1024;
    const log = path.join(f.root, "failed-stage.log"), fullLog = Buffer.concat([Buffer.alloc(80 * 1024, 0x78), Buffer.from("HIP_PREPROCESS_FAILURE\n")]);
    const producer = 'process.stderr.write(Buffer.alloc(80 * 1024, 0x78)); process.stderr.write("HIP_PREPROCESS_FAILURE\\n"); process.exit(7);';
    const failureChild = `const {sdkStage}=require(${JSON.stringify(script)}); try { sdkStage([process.execPath, "-e", ${JSON.stringify(producer)}], ${JSON.stringify(log)}, process.env); process.exitCode=2; } catch (error) { if (!/inspect.*before retrying/.test(error.message)) process.exitCode=3; }`;
    const failure = spawnSync(process.execPath, ["-e", failureChild], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 17});
    assert.equal(failure.status, 0, failure.stderr); assert.equal(failure.stdout, "");
    assert.deepEqual(fs.readFileSync(log), fullLog);
    const tail = fullLog.subarray(-limit), header = `[native-sdk] Failed stage; showing last ${limit} of ${fullLog.length} log bytes from ${log}:\n`;
    assert.deepEqual(Buffer.from(failure.stderr), Buffer.concat([Buffer.from(header), tail]));

    const successLog = path.join(f.root, "successful-stage.log"), successOutput = "successful stdout and stderr";
    const successProducer = 'process.stdout.write("successful stdout "); process.stderr.write("and stderr");';
    const successChild = `require(${JSON.stringify(script)}).sdkStage([process.execPath, "-e", ${JSON.stringify(successProducer)}], ${JSON.stringify(successLog)}, process.env);`;
    const success = spawnSync(process.execPath, ["-e", successChild], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 17});
    assert.equal(success.status, 0, success.stderr); assert.equal(success.stdout, ""); assert.equal(success.stderr, "");
    assert.equal(fs.readFileSync(successLog, "utf8"), successOutput);
  });
  test("earlier evidence cannot be overwritten by a repeated stage", t => {const f = fixture(t), log = path.join(f.root, "stage.log"); fs.writeFileSync(log, "preserve"); assert.throws(() => sdkStage([process.execPath, "-e", ""], log, process.env), /EEXIST/);});
  test("CLI arity rejects missing roots before source/toolchain access", () => {
    const r = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/build-native-sycl-sdk.js")], {encoding: "utf8", timeout: 5000});
    assert.equal(r.status, 1); assert.match(r.stderr, /Usage/); assert.equal(r.stdout, "");
  });
  test("CLI rejects a one-sided host compiler pair before source access", () => {
    const r = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/build-native-sycl-sdk.js"), "/src", "/llvm", "/cuda", "/output", "/host/icx"],
      {encoding: "utf8", timeout: 5000});
    assert.equal(r.status, 1); assert.match(r.stderr, /Usage/); assert.equal(r.stdout, "");
  });
});

test.describe("Combined static CUDA/HIP SDK preserves both backends", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  const env = {MOM_AOT_SYCL_BACKEND: "cuda-hip", MOM_AOT_ROCM_STATIC_ROOT: "/rocm", MOM_AOT_LLD_ROOT: "/lld", ROCM_DEVICE_LIBS_PATH: "/bitcode",
    MOM_AOT_COMGR_RECEIPT: "/comgr/result.json"};
  const plan = (environment = env, output = "/output") => staticSyclSdkPlan("/src", "/llvm", "/cuda", output, 2, environment);
  test("the combined profile enables CUDA and HIP rather than replacing one with the other", () => {
    const p = plan(); assert.equal(p.backend, "cuda-hip");
    for (const flag of ["-DWITH_CUDA_BACKEND=ON", "-DWITH_ROCM_BACKEND=ON"]) {assert.ok(p.commands[0]?.includes(flag));}
  });
  test("both runtime backends and device translators are explicit build targets", () => {
    const p = plan().commands[1] || [];
    for (const target of ["rt-backend-cuda", "rt-backend-hip", "llvm-to-ptx", "llvm-to-amdgpu"]) {assert.ok(p.includes(target));}
    assert.equal(p.includes("all"), false);
  });
  test("the shared runtime does not enable unselected OpenCL/LevelZero/Vulkan backends", () => {
    for (const name of ["OPENCL", "LEVEL_ZERO", "VULKAN"]) {assert.ok(plan().commands[0]?.includes("-DWITH_" + name + "_BACKEND=OFF"));}
  });
  test("coinstalled HIPConfig cannot import different headers or a shared HIP runtime", () => {
    const p = plan().commands[0] || [];
    for (const flag of ["-DCMAKE_DISABLE_FIND_PACKAGE_HIP=ON", "-DROCM_PATH=/rocm", "-DAMDHIP64_LIBRARY=/rocm/lib/libamdhip64.a", "-DHIPRTC_LIBRARY=/rocm/lib/libhiprtc.a"]) {assert.ok(p.includes(flag));}
  });
  test("device bitcode and matching LLD have explicit independent roots", () => {
    const p = plan().commands[0] || []; assert.ok(p.includes("-DROCM_DEVICE_LIBS_PATH=/bitcode")); assert.ok(p.includes("-DLLD_DIR=/lld/lib/cmake/lld"));
  });
  test("CUDA-only selection remains unchanged even with ROCm variables in the environment", () => {
    const p = plan({...env, MOM_AOT_SYCL_BACKEND: "cuda"}); assert.equal(p.hip, undefined); assert.ok(p.commands[0]?.includes("-DWITH_ROCM_BACKEND=OFF"));
    assert.equal(p.commands[1]?.includes("rt-backend-hip"), false);
  });
  test("an unsupported or ambiguous backend never falls back to CUDA", () => {
    for (const backend of ["all", "cuda-hip-opencl"]) {assert.throws(() => plan({...env, MOM_AOT_SYCL_BACKEND: backend}), /backend/);}
  });
  for (const name of ["MOM_AOT_ROCM_STATIC_ROOT", "MOM_AOT_LLD_ROOT", "ROCM_DEVICE_LIBS_PATH", "MOM_AOT_COMGR_RECEIPT"]) {
    test("missing or relative combined SDK root fails before configuration: " + name, () => {
      for (const value of ["", "relative"]) {assert.throws(() => plan({...env, [name]: value}), /absolute/);}
    });
  }
  test("the output cannot enter any added HIP/compiler/device-library root", () => {
    for (const output of ["/rocm/new", "/lld/new", "/bitcode/new"]) {assert.throws(() => plan(env, output), /must not enter/);}
  });
  test("SDK roots remain literal argv even when paths contain shell punctuation", () => {
    const p = plan({...env, MOM_AOT_ROCM_STATIC_ROOT: "/rocm space;false"}); assert.ok(p.commands[0]?.includes("-DROCM_PATH=/rocm space;false"));
  });
});

test.describe("Separate static HIP SDK", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  const env = {MOM_AOT_SYCL_BACKEND: "hip", MOM_AOT_ROCM_STATIC_ROOT: "/rocm", MOM_AOT_LLD_ROOT: "/lld",
    ROCM_DEVICE_LIBS_PATH: "/bitcode", MOM_AOT_COMGR_RECEIPT: "/comgr/result.json"};
  const plan = (environment = env) => staticSyclSdkPlan("/src", "/llvm", "/cuda", "/output", 2, environment);
  test("the HIP profile explicitly excludes CUDA and includes HIP", () => {
    const p = plan(); assert.equal(p.backend, "hip"); assert.equal(p.usesCuda, false);
    assert.ok(p.commands[0]?.includes("-DWITH_CUDA_BACKEND=OFF")); assert.ok(p.commands[0]?.includes("-DWITH_ROCM_BACKEND=ON"));
  });
  test("HIP compilation does not look up CUDA libraries or build NVIDIA targets", () => {
    const p = plan(); assert.equal(p.cache, "");
    assert.equal(p.commands[0]?.some(flag => /^-DCUDA/.test(flag)), false);
    assert.equal(p.commands[1]?.includes("rt-backend-cuda"), false); assert.equal(p.commands[1]?.includes("llvm-to-ptx"), false);
    assert.ok(p.commands[1]?.includes("rt-backend-hip")); assert.ok(p.commands[1]?.includes("llvm-to-amdgpu"));
  });
  test("HIP still requires the same explicit ROCm and compiler inputs as the combined build", () => {
    for (const name of ["MOM_AOT_ROCM_STATIC_ROOT", "MOM_AOT_LLD_ROOT", "ROCM_DEVICE_LIBS_PATH", "MOM_AOT_COMGR_RECEIPT"]) {
      assert.throws(() => plan({...env, [name]: ""}), /absolute/);
    }
    assert.ok(plan().commands[0]?.includes("-DCMAKE_DISABLE_FIND_PACKAGE_HIP=ON"));
  });
});

test.describe("Separate NVRTC compiler and CUDA host header ownership", () => {
  /** @param {import("node:test").TestContext} t */
  function headers(t) {
    const f = fixture(t), compiler = path.join(f.root, "compiler"), runtime = path.join(f.root, "runtime"), output = path.join(f.root, "output");
    for (const dir of [compiler, runtime, output]) {fs.mkdirSync(dir);}
    fs.writeFileSync(path.join(compiler, "nvrtc.h"), "// newer compiler API\n");
    fs.writeFileSync(path.join(compiler, "cuda.h"), "#define CUDA_VERSION 13020\n");
    fs.writeFileSync(path.join(runtime, "cuda.h"), "#define CUDA_VERSION 12060\n");
    const pack = path.join(f.root, "device-headers.inc"), requiredHeaders = ["cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"];
    for (const name of requiredHeaders) {const file = path.join(runtime, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "// fixture " + name + "\n");}
    const {source, ...metadata} = require("../scripts/native-jit-headers").headerResources(requiredHeaders.map(name => ({name, data: Buffer.from("modern " + name)})));
    fs.writeFileSync(pack, source); fs.writeFileSync(pack + ".json", JSON.stringify({...metadata, requiredHeaders}));
    const build = () => cudaBuildHeaders(compiler, output, pack, [runtime]);
    return {...f, compiler, runtime, output, pack, build};
  }
  test("only nvrtc.h enters the dedicated host API include directory", t => {const f = headers(t), r = f.build(); assert.deepEqual(fs.readdirSync(r.include), ["nvrtc.h"]);});
  test("the newer compiler cuda.h cannot shadow the older runtime header", t => {const f = headers(t), r = f.build(); assert.equal(fs.existsSync(path.join(r.include, "cuda.h")), false); assert.match(fs.readFileSync(path.join(f.runtime, "cuda.h"), "utf8"), /12060/);});
  test("copied API bytes exactly match the selected compiler", t => {const f = headers(t), r = f.build(); assert.equal(fileHash(path.join(r.include, "nvrtc.h")), fileHash(path.join(f.compiler, "nvrtc.h")));});
  test("device JIT keeps its independent modern header pack", t => {const f = headers(t); f.build(); assert.equal(fileHash(path.join(f.output, "nvrtc-headers.inc")), fileHash(f.pack));});
  test("both original API and device-header inputs have content receipts", t => {const f = headers(t), r = f.build(); assert.equal(r.files[f.pack], fileHash(f.pack)); assert.equal(r.files[path.join(f.compiler, "nvrtc.h")], fileHash(path.join(f.compiler, "nvrtc.h")));});
  test("missing API inputs fail before creating the host include directory", t => {const f = headers(t); fs.unlinkSync(path.join(f.compiler, "nvrtc.h")); assert.throws(f.build, /ENOENT/); assert.equal(fs.existsSync(path.join(f.output, "nvrtc-api")), false);});
  test("relative device-header packs cannot resolve through cwd", t => {const f = headers(t); assert.throws(() => cudaBuildHeaders(f.compiler, f.output, "relative.inc", [f.runtime]), /absolute/);});
  test("non-regular device headers fail without copying a directory", t => {const f = headers(t); assert.throws(() => cudaBuildHeaders(f.compiler, f.output, f.runtime, [f.runtime]), /regular file/);});
  test("a repeated build never overwrites earlier compiler-header evidence", t => {const f = headers(t); f.build(); assert.throws(f.build, /EEXIST/);});
  test("same-SDK fallback still uses the shared bounded header collector", t => {const f = headers(t), r = cudaBuildHeaders(f.compiler, f.output, undefined, [f.runtime]); assert.ok(fs.statSync(path.join(f.output, "nvrtc-headers.inc")).size > 0); assert.ok(r.headers);});
});
