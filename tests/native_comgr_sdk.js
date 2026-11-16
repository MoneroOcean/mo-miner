"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {comgrSdkPlan, comgrDeviceInputs, comgrDependencies, buildComgrSdk} = require("../scripts/build-native-comgr-sdk");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs");
const {comgrSdkInputs, verifyComgrSdkSelection, revision, recipeFiles} = require("../scripts/native-comgr-inputs");
const {rocmInputs, staticSyclInputs} = require("../scripts/probe-static-sycl");

/** Exact tiny SDK registry, not enumeration of a real compiler installation.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-comgr-sdk-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const config = path.join(root, "lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake"), data = path.join(root, "amdgcn/bitcode");
  fs.mkdirSync(path.dirname(config), {recursive: true}); fs.mkdirSync(data, {recursive: true});
  const names = ["ocml", "ockl", "opencl", "hip", "asanrtl"];
  // eslint-disable-next-line no-template-curly-in-string -- Literal upstream CMake variable, not JavaScript interpolation.
  const registry = (values = names) => values.map(name => 'IMPORTED_LOCATION "${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode/' + name + '.bc"').join("\n");
  fs.writeFileSync(config, registry());
  for (const name of names) {fs.writeFileSync(path.join(data, name + ".bc"), "BC:" + name);}
  return {root, config, data, names, registry};
}

test.describe("Native COMGR pinned snapshot and literal recipe", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  const host = {cc: "/opt/intel/icx", cxx: "/opt/intel/icpx"};
  /** @param {string} [output] @param {number} [jobs] @param {{cc: string, cxx: string}} [compilers] */
  const plan = (output = "/output", jobs = 2, compilers) => comgrSdkPlan("/source", "/llvm", "/lld", "/device", output, jobs, compilers);
  test("only pinned upstream COMGR Git objects enter the private source tree", () => {
    assert.equal(plan().commands[0]?.at(-1), "27682a16360e33e37c4f3cc6adf9a620733f8fe1:amd/comgr");
    assert.ok(plan().commands[0]?.includes("--prefix=amd/comgr/"));
    assert.ok(plan().commands[0]?.includes("archive"));
  });
  test("working-tree changes and untracked inputs are never copied", () => {assert.equal(plan().commands.some(args => args.includes("cp") || args.includes("checkout") || args.includes("clone")), false);});
  test("partial source clones cannot silently trigger a network fetch", () => {
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/build-native-comgr-sdk.js"), "utf8");
    assert.match(script, /env\["GIT_NO_LAZY_FETCH"\] = "1"/); assert.match(script, /env\["GIT_TERMINAL_PROMPT"\] = "0"/);
  });
  test("tar cannot restore foreign ownership or permissions", () => {assert.ok(plan().commands[1]?.includes("--no-same-owner")); assert.ok(plan().commands[1]?.includes("--no-same-permissions"));});
  test("both maintained patches are checked before application", () => {
    const checks = plan().commands.filter(command => command.includes("--check")); assert.equal(checks.length, 2);
    assert.ok(checks[0]?.at(-1)?.endsWith("rocm-comgr-llvm21.patch")); assert.ok(checks[1]?.at(-1)?.endsWith("rocm-comgr-resources.patch"));
  });
  test("the exact Clang resource header replaces recursive SDK discovery", () => {assert.ok(plan().commands[4]?.includes("-DCOMGR_OPENCL_C_HEADER=" + path.join("/llvm", "lib/clang/21/include/opencl-c.h")));});
  test("native static/resource policy and LLVM static linking cannot be omitted", () => {
    for (const flag of ["-DCOMGR_BUILD_SHARED_LIBS=OFF", "-DCOMGR_EMBEDDED_RESOURCES=ON", "-DLLVM_LINK_LLVM_DYLIB=OFF"]) {assert.ok(plan().commands[4]?.includes(flag));}
  });
  test("unrelated tests and host ROCm package detection are disabled", () => {assert.ok(plan().commands[4]?.includes("-DBUILD_TESTING=OFF")); assert.ok(plan().commands[4]?.includes("-DCMAKE_DISABLE_FIND_PACKAGE_ROCM=ON"));});
  test("host CFLAGS cannot silently make a nonportable native compiler", () => {
    for (const flag of ["-DCMAKE_C_FLAGS=", "-DCMAKE_CXX_FLAGS=", "-DCMAKE_EXE_LINKER_FLAGS="]) {assert.ok(plan().commands[4]?.includes(flag));}
  });
  test("only the original compiler component is built, without install or device execution", () => {
    assert.deepEqual(plan().commands[5]?.slice(-5), ["--target", "amd_comgr", "--", "-d", "keepdepfile"]);
    assert.equal(plan().commands.some(command => command.includes("--install") || command.includes("nvidia-smi") || command.includes("docker")), false);
  });
  test("an explicit host pair drives COMGR CMake while LLVM21 paths remain pinned", () => {
    const p = plan("/output", 2, host), configure = p.commands[4] || [];
    assert.deepEqual(p.hostCompilers, host);
    assert.ok(configure.includes("-DCMAKE_C_COMPILER=" + host.cc)); assert.ok(configure.includes("-DCMAKE_CXX_COMPILER=" + host.cxx));
    for (const flag of ["-DLLVM_DIR=/llvm/lib/cmake/llvm", "-DClang_DIR=/llvm/lib/cmake/clang", "-DLLD_DIR=/lld/lib/cmake/lld",
      "-DCOMGR_OPENCL_C_HEADER=/llvm/lib/clang/21/include/opencl-c.h"]) {assert.ok(configure.includes(flag));}
    assert.ok(configure.includes("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -ffunction-sections -fdata-sections"));
    assert.equal(configure.some(flag => /-flto|INTERPROCEDURAL_OPTIMIZATION/i.test(flag)), false);
  });
  test("COMGR uses the shared PATH-safe host command names without changing CMake compiler paths", () => {
    const compilers = {cc: "/Intel SDK/icx", cxx: "/Intel SDK/icpx"}, p = plan("/output", 2, compilers);
    assert.deepEqual(p.compilerEnvironment, {CC: "icx", CXX: "icpx"});
    assert.ok(p.commands[4]?.includes("-DCMAKE_C_COMPILER=" + compilers.cc));
    assert.ok(p.commands[4]?.includes("-DCMAKE_CXX_COMPILER=" + compilers.cxx));
  });
  test("default COMGR bootstrap compilers remain the selected LLVM Clang21 pair", () => {
    assert.deepEqual(plan().hostCompilers, {cc: "/llvm/bin/clang", cxx: "/llvm/bin/clang++"});
    assert.ok(plan().commands[4]?.includes("-DCMAKE_CXX_COMPILER=/llvm/bin/clang++"));
  });
  test("host compiler overrides require two absolute literal paths", () => {
    const incomplete = /** @type {Array<{cc?: string, cxx?: string}>} */ ([{cc: "/opt/intel/icx"}, {cxx: "/opt/intel/icpx"}, {}]);
    for (const selected of incomplete) {
      assert.throws(() => plan("/output", 2, /** @type {{cc: string, cxx: string}} */ (selected)), /absolute, literal pair/);
    }
    for (const file of ["relative/icx", "/opt/intel/icx;bad", "/opt/intel/icx$bad", "/opt/intel/icx\\bad", "/opt/intel/icx\n"]) {
      assert.throws(() => plan("/output", 2, {cc: file, cxx: "/opt/intel/icpx"}), /absolute, literal pair/);
    }
  });
  test("output cannot overlap a selected host compiler", () => {
    assert.throws(() => plan("/opt/intel", 2, {cc: "/opt/intel/bin/icx", cxx: "/opt/intel/bin/icpx"}), /must not enter or enclose/);
    assert.throws(() => plan("/", 2), /must not enter or enclose/);
    assert.throws(() => plan(path.resolve(__dirname, "../outside")), /must not enter or enclose/);
  });
  test("host selection leaves the embedded Clang identity check intact", () => {
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/build-native-comgr-sdk.js"), "utf8");
    assert.match(script, /version !== "21\.1\.8"/); assert.match(script, /clang version 21\\\.1\\\.8/);
  });
  test("producer receipts bind the shared host environment and LLVM admission helpers", () => {
    for (const file of ["scripts/build-native-sycl-sdk.js", "scripts/build-environment.js", "scripts/probe-native-jit.js", "scripts/native-audit.js"]) {
      assert.ok(recipeFiles.includes(file), file);
    }
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/build-native-comgr-sdk.js"), "utf8");
    assert.match(script, /verifyCompilerEnvironment\(plan\.hostCompilers, env\)/);
  });
  for (const jobs of [0, 17, 1.5, NaN]) {test("invalid jobs fail before SDK access: " + jobs, () => {assert.throws(() => plan("/output", jobs), /jobs/);});}
  test("source and all selected SDK roots reject nested output", () => {for (const root of ["/source", "/llvm", "/lld", "/device"]) {assert.throws(() => plan(root), /must not enter/); assert.throws(() => plan(root + "/build"), /must not enter/);}});
  test("relative paths never fall back to cwd", () => {assert.throws(() => comgrSdkPlan("relative", "/llvm", "/lld", "/device", "/output", 2), /absolute/);});
  test("paths with spaces and shell punctuation stay literal arguments", () => {assert.ok(comgrSdkPlan("/source space;false", "/llvm", "/lld", "/device", "/output", 2).commands[0]?.includes("/source space;false"));});
  test("build-only admission is required before reading a source tree", () => {assert.throws(() => buildComgrSdk("/missing", "/missing", "/missing", "/missing", "/new", {}), /BUILD_ONLY/);});
  test("existing evidence is never overwritten", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {const f = fixture(t); assert.throws(() => buildComgrSdk("/missing", "/missing", "/missing", "/missing", f.root, {MOM_AOT_BUILD_ONLY: "1"}), /never overwritten/);});
  test("CLI arity fails without starting tools or changing output", () => {
    const result = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/build-native-comgr-sdk.js")], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 1); assert.match(result.stderr, /Usage/); assert.equal(result.stdout, "");
  });
  test("CLI rejects a one-sided host compiler pair before source access", () => {
    const result = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/build-native-comgr-sdk.js"), "/source", "/llvm", "/lld", "/device", "/output", "/host/icx"],
      {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 1); assert.match(result.stderr, /Usage/); assert.equal(result.stdout, "");
  });
});

test.describe("COMGR and AdaptiveCpp compiler SDK identity", () => {
  test("a matching explicit LLVM/LLD/device package is accepted", t => {
    const f = component(t), c = comgrSdkInputs(f.receipt);
    verifyComgrSdkSelection(c, f.record.sdkRoots.llvm, f.record.sdkRoots.lld, f.data, "21.1.8");
  });
  for (const selected of ["llvm", "lld", "device", "version"]) {
    test("a different SDK is not accepted from equal-looking headers: " + selected, t => {
      const f = component(t), c = comgrSdkInputs(f.receipt);
      assert.throws(() => verifyComgrSdkSelection(c, selected === "llvm" ? f.root : f.record.sdkRoots.llvm,
        selected === "lld" ? f.root : f.record.sdkRoots.lld, selected === "device" ? f.root : f.data,
        selected === "version" ? "22.0.0" : "21.1.8"), /same LLVM\/LLD\/device SDKs/);
    });
  }
});

test.describe("Native COMGR embedded device-library content", () => {
  test("all imported bitcode files and the registry get content hashes", t => {const f = fixture(t), r = comgrDeviceInputs(f.root); assert.equal(Object.keys(r).length, 6); assert.equal(r[f.config], fileHash(f.config));});
  test("unreferenced files are not recursively inventoried", t => {const f = fixture(t); fs.mkdirSync(path.join(f.data, "unrelated")); assert.equal(Object.keys(comgrDeviceInputs(f.root)).length, 6);});
  for (const name of ["ocml", "ockl", "opencl", "hip", "asanrtl"]) {
    test("missing required builtin fails before CMake: " + name, t => {const f = fixture(t); fs.writeFileSync(f.config, f.registry(f.names.filter(value => value !== name))); assert.throws(() => comgrDeviceInputs(f.root), /Incomplete/);});
  }
  test("duplicate imported names are ambiguous", t => {const f = fixture(t); fs.writeFileSync(f.config, f.registry([...f.names, "hip"])); assert.throws(() => comgrDeviceInputs(f.root), /ambiguous/);});
  test("a missing selected library cannot pass from the registry alone", t => {const f = fixture(t); fs.unlinkSync(path.join(f.data, "hip.bc")); assert.throws(() => comgrDeviceInputs(f.root), /ENOENT/);});
  test("directories cannot impersonate bitcode files", t => {const f = fixture(t), file = path.join(f.data, "hip.bc"); fs.unlinkSync(file); fs.mkdirSync(file); assert.throws(() => comgrDeviceInputs(f.root), /regular file/);});
  test("backdated bitcode changes still alter content receipts", t => {const f = fixture(t), before = comgrDeviceInputs(f.root), file = path.join(f.data, "hip.bc"); fs.writeFileSync(file, "changed"); fs.utimesSync(file, 1, 1); assert.notDeepEqual(comgrDeviceInputs(f.root), before);});
  test("relative and excessive registries fail before arbitrary traversal", t => {const f = fixture(t); assert.throws(() => comgrDeviceInputs("relative"), /absolute/); fs.truncateSync(f.config, (1 << 20) + 1); assert.throws(() => comgrDeviceInputs(f.root), /excessive/);});
  test("parent-like names cannot escape the selected bitcode root", t => {const f = fixture(t); fs.writeFileSync(f.config, f.registry([...f.names, "..evil"])); assert.throws(() => comgrDeviceInputs(f.root), /ambiguous/);});
  test("new SDK bitcode members enter the receipt when the registry selects them", t => {const f = fixture(t); fs.writeFileSync(path.join(f.data, "new.bc"), "new"); fs.writeFileSync(f.config, f.registry([...f.names, "new"])); assert.equal(Object.keys(comgrDeviceInputs(f.root)).length, 7);});
});

test.describe("Native COMGR CMake dependency output admission", {skip: process.platform !== "linux"}, () => {
  const rows = (absolute = false) => Array.from({length: 10}, (_, i) => ({directory: "/build", output: (absolute ? "/build/" : "") + `CMakeFiles/comgr/${i}.o`}));
  test("relative CMake outputs remain admitted", () => {assert.equal(comgrDependencies("/build", rows())[0], "/build/CMakeFiles/comgr/0.o.d");});
  test("absolute CMake outputs map to identical dependency paths", () => {assert.deepEqual(comgrDependencies("/build", rows(true)), comgrDependencies("/build", rows()));});
  test("mixed absolute and relative forms describe the same build", () => {const entries = rows(); if (entries[0]) {entries[0].output = "/build/" + entries[0].output;} assert.deepEqual(comgrDependencies("/build", entries), comgrDependencies("/build", rows()));});
  for (const output of ["/other/file.o", "/build-other/file.o", "../file.o", "nested/../file.o", "./file.o", "file\\name.o", "bad\nname.o", "bad\0name.o", "file.d", ""]) {
    test("unsafe or non-object output is rejected: " + JSON.stringify(output), () => {const entries = rows(); if (entries[0]) {entries[0].output = output;} assert.throws(() => comgrDependencies("/build", entries), /COMGR/);});
  }
  test("objects from a different command directory fail", () => {const entries = rows(); if (entries[0]) {entries[0].directory = "/other";} assert.throws(() => comgrDependencies("/build", entries), /COMGR/);});
  test("duplicate output aliases cannot disguise a missing translation unit", () => {const entries = rows(); if (entries[1]) {entries[1].output = "/build/CMakeFiles/comgr/0.o";} assert.throws(() => comgrDependencies("/build", entries), /Duplicate/);});
  test("missing or excessive command graphs fail before dependency reads", () => {for (const entries of [null, {}, [], rows().slice(1), Array(129).fill({})]) {assert.throws(() => comgrDependencies("/build", entries), /Invalid/);}});
});

/** Tiny, locally trusted producer fixture; never a real SDK or executable compiler.
 * @param {import('node:test').TestContext} t */
function component(t) {
  const f = fixture(t), archive = path.join(f.root, "build/libamd_comgr.a"), header = path.join(f.root, "build/include/amd_comgr.h");
  const snapshot = path.join(f.root, "source"), receipt = path.join(f.root, "result.json"), sdkRoots = {llvm: path.join(f.root, "llvm"), lld: path.join(f.root, "lld"), device: f.root};
  const license = path.join(snapshot, "amd/comgr/LICENSE.txt"), notices = path.join(snapshot, "amd/comgr/NOTICES.txt");
  fs.mkdirSync(path.dirname(header), {recursive: true}); fs.mkdirSync(path.dirname(license), {recursive: true});
  for (const dir of [sdkRoots.llvm, sdkRoots.lld]) {fs.mkdirSync(dir);}
  fs.writeFileSync(archive, "!<arch>\nfixture"); fs.writeFileSync(header, "fixture header");
  fs.writeFileSync(license, "fixture license"); fs.writeFileSync(notices, "fixture notices");
  const dependency = path.join(sdkRoots.llvm, "dependency.h"); fs.writeFileSync(dependency, "fixture SDK input");
  const record = {version: 1, revision, llvmVersion: "21.1.8", archive, header, sha256: fileHash(archive), headerSha256: fileHash(header),
    license, notices, sdkRoots, sdkInputs: {[f.config]: fileHash(f.config)}, dependencies: {[dependency]: fileHash(dependency)},
    sourceState: sourceInputs(snapshot, ["amd/comgr"], []), recipe: sourceInputs(path.resolve(__dirname, ".."), [], recipeFiles),
    buildOnly: true, publicationQualified: false};
  const write = () => fs.writeFileSync(receipt, JSON.stringify(record)); write();
  return {...f, receipt, record, write, dependency, archive, header};
}

test.describe("Source-built COMGR SDK receipt consumption", () => {
  test("returns the exact source-built archive, header and dependency closure", t => {
    const f = component(t), r = comgrSdkInputs(f.receipt);
    assert.equal(r.archive, f.archive); assert.equal(r.header, f.header); assert.equal(r.files[f.dependency], fileHash(f.dependency));
    assert.equal(r.files[f.receipt], fileHash(f.receipt)); assert.equal(r.files[path.resolve(__dirname, "../runtime/rocm-comgr-resources.patch")], fileHash(path.resolve(__dirname, "../runtime/rocm-comgr-resources.patch")));
  });
  test("relative receipts cannot be selected through cwd", () => {assert.throws(() => comgrSdkInputs("relative.json"), /absolute/);});
  /** @type {Array<[string, string | boolean]>} */
  const invalid = [["revision", "wrong"], ["llvmVersion", "22.0.0"], ["buildOnly", false], ["publicationQualified", true]];
  for (const [key, value] of invalid) {
    test("mismatched producer identity is rejected: " + key, t => {
      const f = component(t); Reflect.set(f.record, key, value); f.write(); assert.throws(() => comgrSdkInputs(f.receipt), /pinned source/);
    });
  }
  test("an archive path outside the producer cannot masquerade as its output", t => {
    const f = component(t); f.record.archive = path.join(f.root, "other.a"); f.write(); assert.throws(() => comgrSdkInputs(f.receipt), /pinned source/);
  });
  test("changed archive bytes are not admitted from an old digest", t => {const f = component(t); fs.appendFileSync(f.archive, "changed"); assert.throws(() => comgrSdkInputs(f.receipt), /archive\/header/);});
  test("changed generated API headers are rejected", t => {const f = component(t); fs.appendFileSync(f.header, "changed"); assert.throws(() => comgrSdkInputs(f.receipt), /archive\/header/);});
  test("thin archives cannot borrow source-build provenance", t => {
    const f = component(t); fs.writeFileSync(f.archive, "!<thin>\n"); f.record.sha256 = fileHash(f.archive); f.write();
    assert.throws(() => comgrSdkInputs(f.receipt), /archive\/header/);
  });
  test("a backdated SDK dependency change invalidates admission", t => {
    const f = component(t); fs.writeFileSync(f.dependency, "changed"); fs.utimesSync(f.dependency, 1, 1);
    assert.throws(() => comgrSdkInputs(f.receipt), /SDK dependency changed/);
  });
  test("an omitted dependency map is not complete provenance", t => {const f = component(t); f.record.dependencies = {}; f.write(); assert.throws(() => comgrSdkInputs(f.receipt), /dependency receipt/);});
  test("a changed patched source is not admitted from timestamps", t => {const f = component(t); fs.appendFileSync(f.record.license, "changed"); assert.throws(() => comgrSdkInputs(f.receipt), /patched source/);});
  test("changed producer recipes demand a fresh build, never restamping", t => {
    const f = component(t); f.record.recipe.graph[recipeFiles[0] || ""] = "0".repeat(64); f.write(); assert.throws(() => comgrSdkInputs(f.receipt), /producer recipe/);
  });
  test("relative SDK roots cannot escape explicit compiler selection", t => {const f = component(t); f.record.sdkRoots.llvm = "relative"; f.write(); assert.throws(() => comgrSdkInputs(f.receipt), /pinned source/);});
  test("HIP consumers reject a missing source receipt before starting tools", () => {
    assert.throws(() => staticSyclInputs("/missing", "/missing", "/missing", "hip", {}), /MOM_AOT_COMGR_RECEIPT/);
  });
  test("the verified component replaces the prefix archive without mutating an SDK", t => {
    const f = component(t), sdk = path.join(f.root, "rocm"), include = path.join(sdk, "include/hip"), lib = path.join(sdk, "lib");
    fs.mkdirSync(include, {recursive: true}); fs.mkdirSync(path.join(lib, "cmake/hsa-runtime64"), {recursive: true});
    for (const name of ["hip_runtime_api.h", "hiprtc.h"]) {fs.writeFileSync(path.join(include, name), "fixture");}
    for (const name of ["amdhip64", "hiprtc", "hsakmt-staticdrm", "elf", "drm_amdgpu", "drm", "numa", "hsa-runtime64"]) {
      fs.writeFileSync(path.join(lib, "lib" + name + ".a"), "!<arch>\n");
    }
    fs.writeFileSync(path.join(lib, "cmake/hsa-runtime64/hsacore.so.link"), "hsa_a = hsa_b;\n");
    const selected = rocmInputs(sdk, f.receipt); assert.ok(selected.libraries.includes(f.archive));
    assert.equal(fs.existsSync(path.join(lib, "libamd_comgr.a")), false); assert.equal(selected.comgr?.files[f.receipt], fileHash(f.receipt));
  });
});
