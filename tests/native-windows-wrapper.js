"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");
const assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {coffSectionPayloads} = /** @type {{coffSectionPayloads: (file: string) => Array<{number: number, bytes: number, virtualSize: number, characteristics: number, sha256: string}>}} */ (require("../scripts/native-compiler-isolation"));
const {profileGateNames, windowsWrapperPlan, inspectWindowsWrapper, transformWindowsWrapper, extractSpecializationIds, verifyWindowsHostRetention, verifySpecializationIds, windowsDeviceLinkArgs} = require("../scripts/native-windows-wrapper");
const {deviceLinkArgs} = require("../scripts/build-native-gpu");
const gyp = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");

/** @typedef {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} WrapperTools */
/** @typedef {WrapperTools & {clang: string, bin: string}} Toolchain */
/** @typedef {{renamedSection: {from: string, to: string}, output: {sha256: string}, imageHashes: {before: Record<string, {sha256: string}>, after: Record<string, {sha256: string}>}}} WrapperManifest */

/** @param {string} file @returns {boolean} */
function regularFile(file) {
  return Boolean(file && fs.statSync(file, {throwIfNoEntry: false})?.isFile());
}

/** @param {string} prefix @param {string} name @returns {string} */
function specializationId(prefix, name) {return "uid" + prefix + "____" + name;}
/** @param {string} file @param {string | Buffer} contents */
function writeFixture(file, contents) {fs.writeFileSync(file, contents, {flag: "wx"});}
/** @param {string[]} ids @param {{defaultValue?: boolean, emptySection?: boolean}} [options] @returns {string} */
function specializationPropertyText(ids = [], options = {}) {
  const lines = ["[SYCL/device requirements]", "aspects=2|AA=="];
  if (options.emptySection || ids.length) {
    lines.push("[SYCL/specialization constants]");
    lines.push(...ids.map(id => id + "=2|AA=="));
  }
  if (options.defaultValue) {lines.push("[SYCL/specialization constants default values]", "all=2|AA==");}
  lines.push("[SYCL/misc properties]", "optLevel=1|2", "");
  return lines.join("\n");
}
/** @param {import("node:test").TestContext} t @param {string | Buffer} host @param {string | Buffer} device @param {string[]} [properties] */
function specializationFixture(t, host, device, properties = []) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-specialization-gate-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const hostPath = path.join(directory, "host.obj"), devicePath = path.join(directory, "device.obj");
  writeFixture(hostPath, host); writeFixture(devicePath, device);
  const propertyPaths = properties.map((contents, index) => {
    const file = path.join(directory, "image-" + index + ".prop"); writeFixture(file, contents); return file;
  });
  return {directory, hostPath, devicePath, propertyPaths};
}

/** @param {string[]} names @param {string} pathValue @returns {string | undefined} */
function discoverCommand(names, pathValue) {
  const lookup = spawnSync(process.platform === "win32" ? "where.exe" : "which", names, {encoding: "utf8", timeout: 5000,
    env: {...process.env, PATH: pathValue}});
  if (lookup.status !== 0 || typeof lookup.stdout !== "string") {return undefined;}
  const candidate = lookup.stdout.split(/\r?\n/)[0]?.trim();
  return candidate && path.isAbsolute(candidate) && regularFile(candidate) ? candidate : undefined;
}

/** @returns {Toolchain | undefined} */
function discoverToolchain() {
  const configured = process.env["MOM_TEST_LLVM_BIN"], suffix = process.platform === "win32" ? ".exe" : "";
  const bin = configured ? path.resolve(configured) : undefined, searchPath = bin ? bin + path.delimiter + (process.env["PATH"] || "") : (process.env["PATH"] || "");
  const command = (/** @type {string} */ name) => {
    if (!bin) {return discoverCommand([name + suffix, name], searchPath);}
    const candidate = path.join(bin, name + suffix);
    return regularFile(candidate) ? candidate : undefined;
  };
  const clang = command("clang") || command("clang-cl"), llvmObjcopy = command("llvm-objcopy"), llvmReadobj = command("llvm-readobj"), llvmNm = command("llvm-nm");
  if (!clang || !llvmObjcopy || !llvmReadobj || !llvmNm || ![clang, llvmObjcopy, llvmReadobj, llvmNm].every(regularFile)) {return undefined;}
  return {clang, llvmObjcopy, llvmReadobj, llvmNm, bin: bin || path.dirname(clang)};
}

const toolchain = discoverToolchain(), wala = (() => {
  const configured = process.env["MOM_TEST_WALA_WRAPPER"];
  return configured && path.isAbsolute(configured) && regularFile(configured) ? configured : undefined;
})();
const llvmSkip = toolchain ? false : "LLVM toolchain unavailable; install discoverable LLVM tools or set MOM_TEST_LLVM_BIN";
const walaSkip = !toolchain ? llvmSkip : wala ? false : "Wala audit input not configured; set MOM_TEST_WALA_WRAPPER";

/** @param {Toolchain} selected @returns {WrapperTools} */
function wrapperTools(selected) {
  return {llvmObjcopy: selected.llvmObjcopy, llvmReadobj: selected.llvmReadobj, llvmNm: selected.llvmNm};
}

/** @param {Toolchain} selected @param {string} file @param {string} cwd @returns {Array<{name: string, number: number}>} */
function readSectionNames(selected, file, cwd) {
  const result = spawnSync(selected.llvmReadobj, ["--sections", file], {cwd, encoding: "utf8", timeout: 30000});
  assert.equal(result.status, 0, result.error?.message || result.stderr);
  return [...(result.stdout || "").matchAll(/(?:^|\n)\s{2}Section \{\n([\s\S]*?)\n\s{2}\}/g)].map(match => {
    const block = match[1] || "", name = block.match(/^\s*Name:\s*(.*?)\s*$/m)?.[1]?.replace(/\s+\([0-9A-Fa-f ]+\)$/, "").trim(), number = Number(block.match(/^\s*Number:\s*(\d+)\s*$/m)?.[1]);
    assert.ok(name && Number.isSafeInteger(number) && number > 0, "LLVM section metadata must be complete");
    return {name, number};
  });
}

/** @param {import("node:test").TestContext} t @returns {{directory: string, original: string, retained: string, removedNumber: number, removedName: string}} */
function retentionFixture(t) {
  const selected = toolchain;
  if (!selected) {throw new Error("LLVM toolchain unexpectedly unavailable");}
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-host-retention-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const source = path.join(directory, "retention.c"), original = path.join(directory, "original.obj"), retained = path.join(directory, "retained.obj");
  fs.writeFileSync(source, [
    "__attribute__((section(\"__CLANG_OFFLOAD_BUNDLE__sycl-spir64\"), used)) static const unsigned char offload_data[4] = {1, 2, 3, 4};",
    "__attribute__((used, noinline)) int retained_code(int value) { return value + 7; }",
  ].join("\n") + "\n", {flag: "wx"});
  const env = {...process.env, PATH: selected.bin + path.delimiter + (process.env["PATH"] || "")}, clangName = path.basename(selected.clang).toLowerCase();
  const compileArgs = clangName.startsWith("clang-cl") ? ["/nologo", "/c", source, "/Fo" + original, "/clang:-target=x86_64-pc-windows-msvc", "/clang:-fno-stack-protector"] :
    ["-target", "x86_64-pc-windows-msvc", "-c", "-O0", "-fno-stack-protector", "-fno-asynchronous-unwind-tables", source, "-o", original];
  const compiled = spawnSync(selected.clang, compileArgs, {cwd: directory, encoding: "utf8", timeout: 30000, env});
  assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
  const removed = readSectionNames(selected, original, directory).find(section => section.name === "__CLANG_OFFLOAD_BUNDLE__sycl-spir64");
  if (!removed) {throw new Error("headerless fixture must contain its offload-like section");}
  const copied = spawnSync(selected.llvmObjcopy, ["--remove-section=" + removed.name, original, retained], {cwd: directory, encoding: "utf8", timeout: 30000});
  assert.equal(copied.status, 0, copied.error?.message || copied.stderr);
  assert.ok(fs.statSync(retained).isFile());
  assert.ok(coffSectionPayloads(original).length > coffSectionPayloads(retained).length);
  return {directory, original, retained, removedNumber: removed.number, removedName: removed.name};
}

/** @param {string} file */
function mutateTextPayload(file) {
  const data = fs.readFileSync(file), count = data.readUInt16LE(2), table = 20;
  for (let index = 0; index < count; index++) {
    const section = table + index * 40, name = data.subarray(section, section + 8).toString("ascii").replace(/\0.*$/, ""), bytes = data.readUInt32LE(section + 16), offset = data.readUInt32LE(section + 20);
    if (name === ".text" && bytes > 0 && offset > 0 && offset < data.length) {data[offset] = (data[offset] || 0) ^ 1; fs.writeFileSync(file, data); return;}
  }
  throw new Error("Headerless fixture has no mutable .text payload");
}

/** @returns {string} */
function fixtureSource() {
  return [
    "extern void sycl_register(void *) __asm__(\"__sycl_register_lib\");",
    "extern void sycl_unregister(void *) __asm__(\"__sycl_unregister_lib\");",
    "extern int register_exit(void (*)(void)) __asm__(\"atexit\");",
    "extern const char descriptor[] __asm__(\".sycl_offloading.descriptor\");",
    "void descriptor_reg(void) __asm__(\"sycl.descriptor_reg\");",
    "void descriptor_reg(void) {}",
    "void descriptor_unreg(void) __asm__(\"sycl.descriptor_unreg.atexit\");",
    "void descriptor_unreg(void) {}",
    "__attribute__((section(\".rdata\"), used)) const char descriptor[8] __asm__(\".sycl_offloading.descriptor\") = {0};",
    "__attribute__((section(\"__CLANG_OFFLOAD_BUNDLE__sycl-spir64\"), used)) const char image_data[8] __asm__(\".sycl_offloading.0.data\") = {1};",
    "__attribute__((section(\".tgtimg\"), used)) const char image_info[8] __asm__(\".sycl_offloading.0.info\") = {2};",
    "__attribute__((section(\".text.startup\"), used, noinline)) void wrapper_startup(void) {",
    "  sycl_unregister((void *)descriptor); sycl_register((void *)descriptor); register_exit(descriptor_unreg);",
    "}",
    "typedef void (*startup_fn)(void);",
    "__attribute__((section(\".CRT$XCA00001\"), used)) startup_fn const startup_pointer = descriptor_reg;",
  ].join("\n") + "\n";
}

test.describe("Windows SYCL wrapper transform", () => {
  test("profile and plan stay inside the maintained dispatcher contract", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-plan-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    assert.deepEqual(profileGateNames("dpcpp-opencl"), {key: "dpcpp_opencl", register: "mom_native_register_dpcpp_opencl", unregister: "mom_native_unregister_dpcpp_opencl"});
    assert.throws(() => profileGateNames("acpp-hip"), /registration profile/);
    const plan = windowsWrapperPlan({input: path.join(directory, "input.obj"), output: path.join(directory, "output.obj"), profile: "oneapi",
      tools: {llvmObjcopy: process.execPath, llvmReadobj: process.execPath, llvmNm: process.execPath}});
    assert.deepEqual(plan.args, ["--redefine-sym", "__sycl_register_lib=mom_native_register_oneapi", "--redefine-sym", "__sycl_unregister_lib=mom_native_unregister_oneapi"]);
  });

  test("real generated COFF fixture is transformed with source and image hashes", {skip: llvmSkip}, (t) => {
    const selected = toolchain;
    if (!selected) {throw new Error("LLVM toolchain unexpectedly unavailable");}
    const selectedTools = wrapperTools(selected);
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-coff-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = path.join(directory, "fixture.c"), input = path.join(directory, "fixture.obj"), output = path.join(directory, "fixture-xcu.obj"), manifestPath = path.join(directory, "manifest.json");
    fs.writeFileSync(source, fixtureSource(), {flag: "wx"});
    const env = {...process.env, PATH: selected.bin + path.delimiter + (process.env["PATH"] || "")}, clangName = path.basename(selected.clang).toLowerCase();
    const compileArgs = clangName.startsWith("clang-cl") ? ["/nologo", "/c", source, "/Fo" + input, "/clang:-target=x86_64-pc-windows-msvc", "/clang:-fno-stack-protector"] :
      ["-target", "x86_64-pc-windows-msvc", "-c", "-O0", "-fno-stack-protector", "-fno-asynchronous-unwind-tables", source, "-o", input];
    const compiled = spawnSync(selected.clang, compileArgs, {encoding: "utf8", timeout: 30000, env});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const beforeHash = fileHash(input), manifest = /** @type {WrapperManifest} */ (transformWindowsWrapper({input, output, profile: "dpcpp", sourceFiles: [source], manifestPath, tools: selectedTools}));
    assert.equal(fileHash(input), beforeHash);
    assert.equal(manifest.renamedSection.from, ".CRT$XCA00001"); assert.equal(manifest.renamedSection.to, ".CRT$XCU99999");
    for (const image of [".sycl_offloading.0.data", ".sycl_offloading.0.info"]) {
      assert.equal(manifest.imageHashes.before[image]?.sha256, manifest.imageHashes.after[image]?.sha256);
    }
    assert.equal(JSON.parse(fs.readFileSync(manifestPath, "utf8")).output.sha256, fileHash(output));
    assert.deepEqual(inspectWindowsWrapper(output, selectedTools, "output", "dpcpp").gates.register, "mom_native_register_dpcpp");
  });

  test("wrapper tool failures retain useful bounded diagnostics", t => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-tool-fault-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const input = path.join(directory, "fixture.obj"); fs.writeFileSync(input, "not inspected because the tool fails");
    const tools = {llvmObjcopy: process.execPath, llvmReadobj: process.execPath, llvmNm: process.execPath};
    assert.throws(() => inspectWindowsWrapper(input, tools, "input", "dpcpp"), error => {
      assert.ok(error instanceof Error); assert.match(error.message, /llvm-readobj failed.*--file-headers/s); assert.ok(error.message.length < 2300); return true;
    });
  });

  test("wrapper audit bounds scratch copies and streams section payloads", {skip: llvmSkip}, t => {
    const selected = toolchain; if (!selected) {throw new Error("LLVM toolchain unavailable");}
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-scratch-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = path.join(directory, "fixture.c"), input = path.join(directory, "fixture.obj"), output = path.join(directory, "output.obj");
    fs.writeFileSync(source, fixtureSource());
    const built = spawnSync(selected.clang, ["-target", "x86_64-pc-windows-msvc", "-c", "-O0", "-fno-stack-protector", source, "-o", input], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.stderr);
    const copy = fs.copyFileSync, read = fs.readFileSync;
    let maximumCopies = 0;
    // Instrument actual LLVM transforms: no large fixture, tmpfs mount or GPU is needed.
    fs.copyFileSync = (from, to, flags) => {
      copy(from, to, flags); const parent = path.dirname(to.toString());
      if (/^mom-windows-wrapper-[A-Za-z0-9]+$/.test(path.basename(parent))) {
        maximumCopies = Math.max(maximumCopies, fs.readdirSync(parent).filter(n => n.endsWith(".obj")).length);
      }
    };
    fs.readFileSync = /** @type {typeof fs.readFileSync} */ ((file, options) => {
      assert.doesNotMatch(file.toString(), /mom-windows-wrapper-[A-Za-z0-9]+[/\\][^/\\]+\.bin$/, "section payloads must be hashed in bounded chunks");
      return read(file, options);
    });
    try {transformWindowsWrapper({input, output, profile: "dpcpp", tools: wrapperTools(selected)});} finally {fs.copyFileSync = copy; fs.readFileSync = read;}
    assert.equal(maximumCopies, 1, "section count must not multiply the full-object scratch footprint");
  });

  test("the maintained Wala wrapper passes the same module API and image-symbol checks", {skip: walaSkip}, (t) => {
    const selected = toolchain, input = wala;
    if (!selected || !input) {throw new Error("Wala audit prerequisites unexpectedly unavailable");}
    const selectedTools = wrapperTools(selected);
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-wala-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const output = path.join(directory, "walahash-wrapper-xcu.obj");
    const beforeHash = fileHash(input), before = inspectWindowsWrapper(input, selectedTools, "input", "dpcpp");
    const manifest = /** @type {WrapperManifest} */ (transformWindowsWrapper({input, output, profile: "dpcpp", tools: selectedTools}));
    assert.equal(fileHash(input), beforeHash); assert.equal(before.imageSections[".sycl_offloading.0.data"], "__CLANG_OFFLOAD_BUNDLE__sycl-spir64");
    assert.equal(before.imageSections[".sycl_offloading.0.info"], ".tgtimg");
    assert.equal(manifest.imageHashes.before[".sycl_offloading.0.data"]?.sha256, manifest.imageHashes.after[".sycl_offloading.0.data"]?.sha256);
    assert.equal(manifest.imageHashes.before[".sycl_offloading.0.info"]?.sha256, manifest.imageHashes.after[".sycl_offloading.0.info"]?.sha256);
    assert.equal(inspectWindowsWrapper(output, selectedTools, "output", "dpcpp").gates.unregister, "mom_native_unregister_dpcpp");
  });

  test("real object mutations fail closed for unknown CRT and missing unregister", {skip: walaSkip}, (t) => {
    const selected = toolchain, input = wala;
    if (!selected || !input) {throw new Error("Wala audit prerequisites unexpectedly unavailable");}
    const selectedTools = wrapperTools(selected);
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wrapper-negative-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const renamed = path.join(directory, "renamed.obj"), missing = path.join(directory, "missing.obj");
    transformWindowsWrapper({input, output: renamed, profile: "dpcpp", tools: selectedTools});
    assert.throws(() => inspectWindowsWrapper(renamed, selectedTools, "input", "dpcpp"), /CRT/);
    const copied = spawnSync(selected.llvmObjcopy, ["--redefine-sym", "__sycl_unregister_lib=wrong_unregister", input, missing], {encoding: "utf8", timeout: 30000});
    assert.equal(copied.status, 0, copied.error?.message || copied.stderr);
    assert.throws(() => inspectWindowsWrapper(missing, selectedTools, "input", "dpcpp"), /startup|imports/);
  });
});

test.describe("Windows SYCL device-link recipe", () => {
  const object = path.resolve("/build/1.o"), output = path.resolve("/build/device.o");

  test("reuses maintained optimization, size_t, split and post-link controls", () => {
    const linux = deviceLinkArgs(gyp, undefined, [object], output, "oneapi");
    const windows = windowsDeviceLinkArgs(gyp, undefined, [object], output, "oneapi");
    assert.deepEqual(windows, ["--target=x86_64-pc-windows-msvc", "-mno-incremental-linker-compatible",
      ...linux.filter((flag) => flag !== "-fPIC")]);
    assert.equal(windows.filter(flag => flag === "-mno-incremental-linker-compatible").length, 1);
    for (const flag of ["-std=c++20", "-O3", "-fno-sycl-id-queries-fit-in-int", "-fsycl-device-code-split=per_kernel",
      "-fsycl-targets=spir64", "-fsycl-link", "-nolibsycl"]) {assert.ok(windows.includes(flag), flag);}
    assert.ok(linux.includes("-fPIC"));
    assert.equal(windows[0], "--target=x86_64-pc-windows-msvc");
    assert.ok(!windows.includes("-fPIC"));
    assert.doesNotMatch(windows.join(" "), /(?:^|\s)-save-temps(?:=|\s|$)/);
    assert.doesNotMatch(windows.join(" "), /(?:^|\s)-l(?:sycl|sycl9)(?:\s|$)/);
    assert.doesNotMatch(linux.join(" "), /--target=x86_64-pc-windows-msvc/);
  });

  test("inherits maintained path and profile validation", () => {
    assert.throws(() => windowsDeviceLinkArgs(gyp, undefined, ["relative/object.o"], output, "oneapi"), /absolute/);
    assert.throws(() => windowsDeviceLinkArgs(gyp, undefined, [object, object], output, "oneapi"), /unique absolute/);
    assert.throws(() => windowsDeviceLinkArgs(gyp, undefined, [object], output, "unknown"), /compiler profile/);
  });
});

test.describe("Windows host retention payload gate", {skip: llvmSkip}, () => {
  test("accepts removal of exactly one headerless offload-like section", (t) => {
    const selected = retentionFixture(t), receipt = verifyWindowsHostRetention(selected.original, selected.retained, [selected.removedNumber]);
    assert.equal(receipt.input.sha256, fileHash(selected.original));
    assert.equal(receipt.output.sha256, fileHash(selected.retained));
    assert.equal(receipt.removedSectionCount, 1);
    assert.deepEqual(receipt.removedSectionNumbers, [selected.removedNumber]);
    assert.equal(receipt.originalSectionCount, receipt.retainedSectionCount + 1);
  });

  test("rejects a mutation of a kept code section", (t) => {
    const selected = retentionFixture(t), mutated = path.join(selected.directory, "mutated.obj");
    fs.copyFileSync(selected.retained, mutated);
    mutateTextPayload(mutated);
    assert.throws(() => verifyWindowsHostRetention(selected.original, mutated, [selected.removedNumber]), /changed a kept section payload or flags/);
  });

  test("rejects duplicate, out-of-range, and mismatched removal counts", (t) => {
    const selected = retentionFixture(t);
    assert.throws(() => verifyWindowsHostRetention(selected.original, selected.retained, [selected.removedNumber, selected.removedNumber]), /unique positive integers/);
    assert.throws(() => verifyWindowsHostRetention(selected.original, selected.retained, [0]), /unique positive integers/);
    assert.throws(() => verifyWindowsHostRetention(selected.original, selected.retained, [coffSectionPayloads(selected.original).length + 1]), /out of range/);
    assert.throws(() => verifyWindowsHostRetention(selected.original, selected.retained, []), /section count mismatch/);
  });
});

test.describe("Windows SYCL specialization-ID regression gate", () => {
  const cecPrefix = "cec57e5d403672c0", fcPrefix = "fc6601124dd1a01c";
  const program = (/** @type {string} */ prefix) => specializationId(prefix, "ZN10mom_testL7programE");
  const dag = (/** @type {string} */ prefix) => specializationId(prefix, "ZN10mom_testL3dagE");

  test("rejects the concrete host cec/device fc UID mismatch", (t) => {
    const files = specializationFixture(t, Buffer.from(program(cecPrefix) + "\0" + dag(cecPrefix)), Buffer.from(program(fcPrefix) + "\0" + dag(fcPrefix)));
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath), /host and device objects/);
  });

  test("accepts a matching pair and returns bounded hash evidence", (t) => {
    const ids = [program(fcPrefix), dag(fcPrefix)], files = specializationFixture(t, Buffer.from(ids.join("\0")), Buffer.from(ids.join("\0")), [specializationPropertyText(ids)]);
    const receipt = verifySpecializationIds(files.hostPath, files.devicePath, files.propertyPaths);
    assert.equal(receipt.matched, true);
    assert.deepEqual(receipt.specializationIds.host, ids.slice().sort());
    assert.equal(receipt.host.bytes, Buffer.byteLength(ids.join("\0")));
    assert.match(receipt.host.sha256, /^[0-9a-f]{64}$/);
  });

  test("rejects a missing device ID", (t) => {
    const files = specializationFixture(t, program(cecPrefix) + "\0" + dag(cecPrefix), program(cecPrefix));
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath), /host and device objects/);
  });

  test("rejects an extra device ID", (t) => {
    const files = specializationFixture(t, program(fcPrefix), program(fcPrefix) + "\0" + dag(fcPrefix));
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath), /host and device objects/);
  });

  test("rejects property keys missing from the device set", (t) => {
    const ids = [program(fcPrefix), dag(fcPrefix)], files = specializationFixture(t, ids.join("\0"), ids.join("\0"), [specializationPropertyText([program(fcPrefix)])]);
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath, files.propertyPaths), /device object and property keys/);
  });

  test("rejects property keys not present in the device set", (t) => {
    const ids = [program(fcPrefix), dag(fcPrefix)], files = specializationFixture(t, program(fcPrefix), program(fcPrefix), [specializationPropertyText(ids)]);
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath, files.propertyPaths), /device object and property keys/);
  });

  test("ignores default payload values when comparing specialization identities", (t) => {
    const id = program(fcPrefix), files = specializationFixture(t, id, id, [specializationPropertyText([id], {defaultValue: true})]);
    const receipt = verifySpecializationIds(files.hostPath, files.devicePath, files.propertyPaths);
    assert.deepEqual(receipt.specializationIds.properties, [id]);
  });

  test("accepts an unchanged empty-TU pair without property files", (t) => {
    const files = specializationFixture(t, Buffer.from("ordinary host bytes\0"), Buffer.from("ordinary device bytes\0"));
    const receipt = verifySpecializationIds(files.hostPath, files.devicePath);
    assert.deepEqual(receipt.specializationIds, {host: [], device: [], properties: []});
  });

  test("scans NUL-padded objects without splitting every padding byte into an array", (t) => {
    const padded = Buffer.alloc(8 << 20), id = program(fcPrefix);
    padded.write(id, padded.length - id.length - 1, "ascii");
    const files = specializationFixture(t, padded, id);
    assert.deepEqual(verifySpecializationIds(files.hostPath, files.devicePath).host.ids, [id]);
  });

  test("deduplicates repeated IDs across objects and property files", (t) => {
    const id = program(fcPrefix), files = specializationFixture(t, id + "\0" + id, id, [specializationPropertyText([id, id]), specializationPropertyText([id])]);
    const receipt = verifySpecializationIds(files.hostPath, files.devicePath, files.propertyPaths);
    assert.deepEqual(receipt.specializationIds.host, [id]);
    assert.deepEqual(receipt.specializationIds.properties, [id]);
  });

  test("rejects unsupported or malformed stable names", (t) => {
    const files = specializationFixture(t, "uid" + cecPrefix + "____Qunsupported\0", "ordinary");
    assert.throws(() => extractSpecializationIds(files.hostPath), /Malformed or unsupported specialization ID/);
    const malformed = specializationFixture(t, "uid" + cecPrefix + "____ZN10mom_testL3bad!\0", "ordinary");
    assert.throws(() => extractSpecializationIds(malformed.hostPath), /Malformed or unsupported specialization ID/);
  });

  test("rejects a truncated or empty specialization section", (t) => {
    const truncated = specializationFixture(t, "ordinary", "ordinary", ["[SYCL/specialization constants\n"]);
    assert.throws(() => verifySpecializationIds(truncated.hostPath, truncated.devicePath, truncated.propertyPaths), /section header|Truncated/);
    const empty = specializationFixture(t, "ordinary", "ordinary", [specializationPropertyText([], {emptySection: true})]);
    assert.throws(() => verifySpecializationIds(empty.hostPath, empty.devicePath, empty.propertyPaths), /Empty specialization property section/);
  });

  test("rejects malformed records, unsupported names, and IDs outside the section", (t) => {
    const id = program(fcPrefix), badPayload = specializationFixture(t, id, id, ["[SYCL/specialization constants]\n" + id + "=2|\n"]);
    assert.throws(() => verifySpecializationIds(badPayload.hostPath, badPayload.devicePath, badPayload.propertyPaths), /Malformed specialization property record/);
    const badName = specializationFixture(t, id, id, ["[SYCL/specialization constants]\nuid" + fcPrefix + "____Qunsupported=2|AA==\n"]);
    assert.throws(() => verifySpecializationIds(badName.hostPath, badName.devicePath, badName.propertyPaths), /Malformed specialization property record/);
    const conflicting = specializationFixture(t, id, id, ["[SYCL/specialization constants]\n" + id + "=2|AA==\n" + id + "=2|AQ==\n"]);
    assert.throws(() => verifySpecializationIds(conflicting.hostPath, conflicting.devicePath, conflicting.propertyPaths), /Conflicting specialization property record/);
    const outside = specializationFixture(t, id, id, ["[SYCL/device requirements]\n" + id + "=2|AA==\n"]);
    assert.throws(() => verifySpecializationIds(outside.hostPath, outside.devicePath, outside.propertyPaths), /outside its property section/);
  });

  test("rejects an oversized regular input before reading it", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-specialization-large-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const host = path.join(directory, "host.obj"), device = path.join(directory, "device.obj");
    writeFixture(device, "ordinary");
    const descriptor = fs.openSync(host, "w");
    try {fs.ftruncateSync(descriptor, 256 * 1024 * 1024 + 1);} finally {fs.closeSync(descriptor);}
    assert.throws(() => verifySpecializationIds(host, device), /bounded regular file|bounded input size/);
  });

  test("rejects non-regular property paths and relative paths", (t) => {
    const id = program(fcPrefix), files = specializationFixture(t, id, id);
    assert.throws(() => verifySpecializationIds(files.hostPath, files.devicePath, [files.directory]), /bounded regular file/);
    assert.throws(() => verifySpecializationIds("relative-host.obj", files.devicePath), /distinct absolute/);
  });

  const configuredProperties = (() => {
    const value = process.env["MOM_TEST_SPECIALIZATION_PROPERTIES"];
    if (value === undefined) {return [];}
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : undefined;
    } catch {return undefined;}
  })();
  const configuredHost = process.env["MOM_TEST_SPECIALIZATION_HOST"], configuredDevice = process.env["MOM_TEST_SPECIALIZATION_DEVICE"];
  test("accepts an explicitly configured real artifact fixture", {skip: configuredHost === undefined && configuredDevice === undefined ? "real artifact paths not configured" : false}, () => {
    if (!configuredHost || !configuredDevice || !configuredProperties || configuredProperties.some(file => typeof file !== "string")) {
      throw new Error("MOM_TEST_SPECIALIZATION_* must provide absolute host, device, and JSON property paths");
    }
    const receipt = verifySpecializationIds(configuredHost, configuredDevice, configuredProperties);
    assert.equal(receipt.matched, true);
  });
});
