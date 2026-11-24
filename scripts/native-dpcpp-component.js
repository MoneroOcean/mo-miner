"use strict";

// Install only the original runtime/link closure, not compiler tools, test executables or
// build trees. The actual Ninja link edge owns archive membership; no second library list.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, readBuildJsonSnapshot} = require("./build-source-inputs");
const {pins} = require("./native-dpcpp-pins");
const {verifyLinuxSysrootRecord} = require("./native-linux-sysroot");
const {isStaticArchive} = require("./native-audit");
const {nativeSdkEnvironment} = require("./build-environment");
const objectNames = ["cuda-adapter.o", "level-zero-v2-adapter.o", "source-jit.o"];
const requiredLibraries = ["libmom-sycl-static.a", "libur_loader.a", "libur_adapter_level_zero.a", "libur_adapter_opencl.a", "libze_loader.a", "libhwloc.a"];
/** @param {unknown} value @returns {value is string} */
const absolute = value => typeof value === "string" && path.isAbsolute(value) && value.length <= 4096 &&
  !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);

/** Read Ninja's dependency graph, not its shell commands. Ignore order-only build targets;
 * only real link inputs can become SDK payloads. The two fixture objects are not SDK code.
 * @param {string} text @param {string} build @param {string} jit */
function dpcppRuntimeGraph(text, build, jit) {
  if (Buffer.byteLength(text) > (64 << 10) || !absolute(build) || !absolute(jit)) {throw new Error("Invalid bounded DPC++ link graph");}
  const lines = text.trimEnd().split(/\r?\n/), archives = [], objects = new Set(), fixtures = [];
  if (lines.shift() !== "native-dpcpp-core:" || !/^ {2}input: CXX_EXECUTABLE_LINKER__native-dpcpp-core_Release$/.test(lines.shift() || "")) {throw new Error("Unrecognized DPC++ SDK link edge");}
  let outputs = false;
  for (const line of lines) {
    if (line === "  outputs:") {if (outputs) {throw new Error("Duplicate DPC++ graph outputs");} outputs = true; continue;}
    if (outputs) {if (line !== "    all") {throw new Error("Unexpected DPC++ graph output");} continue;}
    if (/^ {4}\|\| [^\r\n\0]+$/.test(line)) {continue;}
    const implicit = line.match(/^ {4}\| (.+)$/), input = implicit?.[1] || line.match(/^ {4}(.+)$/)?.[1];
    if (!input || !absolute(path.resolve(build, input)) || /[\\\0]/.test(input) || input.split("/").some(p => p === "..")) {throw new Error("Invalid DPC++ graph input");}
    if (!implicit && input === jit) {if (objects.has("source-jit.o")) {throw new Error("Duplicate DPC++ source JIT");} objects.add("source-jit.o"); continue;}
    if (!implicit && /^CMakeFiles\/native-dpcpp-core\.dir\/.+\/(?:native-dpcpp-static|driver-library)\.cpp\.o$/.test(input)) {fixtures.push(input); continue;}
    if (path.isAbsolute(input)) {throw new Error("DPC++ runtime graph contains an unexpected external input");}
    if (implicit && /^[A-Za-z0-9_./-]+\.a$/.test(input)) {archives.push(input); continue;}
    const object = {"cuda-native/adapter.o": "cuda-adapter.o", "level-zero-v2-native/adapter.o": "level-zero-v2-adapter.o"};
    const name = Reflect.get(object, input);
    if (!implicit || !name || objects.has(name)) {throw new Error("Unrecognized or duplicate DPC++ runtime object");} objects.add(name);
  }
  const names = archives.map(n => path.basename(n));
  if (!outputs || fixtures.length !== 2 || new Set(fixtures).size !== 2 || archives.length > 32 || new Set(names).size !== names.length ||
      requiredLibraries.some(n => !names.includes(n)) || !isDeepStrictEqual([...objects].sort(), [...objectNames].sort())) {throw new Error("Incomplete DPC++ runtime/JIT graph");}
  return {archives: archives.map(n => path.join(build, n)), objects: {
    "cuda-adapter.o": path.join(build, "cuda-native/adapter.o"), "level-zero-v2-adapter.o": path.join(build, "level-zero-v2-native/adapter.o"), "source-jit.o": jit}};
}

/** Namespace-isolated helpers are ordinary x86-64 relocatable objects, never application
 * libraries or executable payloads. @param {string} file */
function dpcppObject(file) {
  const data = Buffer.alloc(20), fd = fs.openSync(file, "r");
  try {return fs.readSync(fd, data, 0, data.length, 0) === data.length && data.subarray(0, 6).toString("hex") === "7f454c460201" && data.readUInt16LE(16) === 1 && data.readUInt16LE(18) === 62;} finally {fs.closeSync(fd);}
}

/** Rehash the installed closure on each build. This is local provenance continuity, not a
 * signature authenticating an arbitrary SDK download. No tools/driver/GPU are executed.
 * @param {string} root */
function dpcppComponentInputs(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root) {throw new Error("Select a canonical DPC++ component root");}
  const receipt = path.join(root, "component.json"), {value: r, sha256: receiptSha256} = readBuildJsonSnapshot(receipt);
  if (r.version !== 1 || r.kind !== "dpcpp-linux" || !isDeepStrictEqual(r.pins, pins) || !r.files || typeof r.files !== "object" || Array.isArray(r.files) ||
      !Array.isArray(r.libraries) || r.libraries.length > 32 || r.libraries.some((/** @type {unknown} */ n) => typeof n !== "string" || !/^lib\/[A-Za-z0-9_.-]+\.a$/.test(n)) ||
      new Set(r.libraries).size !== r.libraries.length || requiredLibraries.some(n => !r.libraries.includes("lib/" + n))) {throw new Error("Invalid complete DPC++ component receipt");}
  const libraries = /** @type {string[]} */ (r.libraries), expected = [...libraries, ...objectNames.map(n => "objects/" + n)];
  if (!isDeepStrictEqual(Object.keys(r.files).sort(), [...expected].sort())) {throw new Error("DPC++ component membership differs");}
  const files = /** @type {Record<string, string>} */ ({[receipt]: receiptSha256});
  for (const name of expected) {
    const file = path.join(root, name), hash = r.files[name], stat = fs.lstatSync(file);
    if (!stat.isFile() || fs.realpathSync(file) !== file || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash) || fileHash(file) !== hash ||
        (name.startsWith("lib/") ? !isStaticArchive(file) : !dpcppObject(file))) {throw new Error("DPC++ component bytes/type changed: " + name);}
    files[file] = hash;
  }
  for (const directory of ["lib", "objects"]) {
    const names = fs.readdirSync(path.join(root, directory));
    if (!isDeepStrictEqual(names.sort(), expected.filter(n => n.startsWith(directory + "/")).map(n => path.basename(n)).sort())) {throw new Error("Unrecorded DPC++ component file");}
  }
  if (fileHash(receipt) !== receiptSha256) {throw new Error("DPC++ component receipt changed during admission");}
  const host = path.join(root, "lib/libmom-sycl-static.a");
  return {links: ["-Wl,--whole-archive", host, "-Wl,--no-whole-archive", ...libraries.filter(n => n !== "lib/libmom-sycl-static.a").map(n => path.join(root, n)),
    "-Wl,-Bstatic", "-lstdc++fs", "-Wl,-Bdynamic"], sdkObjects: objectNames.map(n => ({object: path.join(root, "objects", n)})), files};
}

/** Export a completed original SDK with the SAME Ninja binary that produced it. Running a
 * mismatched Ninja may rewrite its build log even for a query: reject it before invocation.
 * @param {string} producer @param {string} ninja @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function exportDpcppComponent(producer, ninja, output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1" || ![producer, ninja, output].every(absolute)) {throw new Error("DPC++ component export requires Linux, explicit paths and MOM_AOT_BUILD_ONLY=1");}
  const {value: r, sha256: sourceSha256} = readBuildJsonSnapshot(producer), root = path.dirname(fs.realpathSync(producer)), build = path.join(root, "build");
  if (r.version !== 1 || r.target !== "linux" || r.buildOnly !== true || r.executed !== false || !isDeepStrictEqual(r.pins, pins) ||
      r.archive !== path.join(build, "libmom-sycl-static.a") || fileHash(r.archive) !== r.sha256 || !r.inputs?.[ninja] || fileHash(ninja) !== r.inputs[ninja]) {throw new Error("DPC++ producer or matching Ninja identity differs");}
  verifyLinuxSysrootRecord(r.sysroot, "DPC++ static SDK");
  const before = readBuildJson(path.join(root, "before.json")), commands = before.configure;
  const prefix = "-DMOM_DPCPP_JIT_OBJECT=", jit = Array.isArray(commands) ? commands.filter(n => typeof n === "string" && n.startsWith(prefix)) : [];
  const jitObject = jit.length === 1 ? jit[0].slice(prefix.length) : "";
  if (!absolute(jitObject) || r.inputs[jitObject] !== fileHash(jitObject)) {throw new Error("Missing original source-JIT input");}
  if (fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("DPC++ component output must be new");}
  const target = path.join(fs.realpathSync(path.dirname(output)), path.basename(output)), relative = path.relative(root, target), reverse = path.relative(target, root);
  if (target !== output) {throw new Error("Select a canonical DPC++ component output");}
  const nested = (/** @type {string} */ n) => !n || (!path.isAbsolute(n) && n !== ".." && !n.startsWith(".." + path.sep));
  if (nested(relative) || nested(reverse)) {throw new Error("DPC++ component output overlaps its producer");}
  const query = spawnSync(ninja, ["-C", build, "-t", "query", "native-dpcpp-core"], {env: nativeSdkEnvironment(environment), encoding: "utf8", timeout: 30000, maxBuffer: 64 << 10});
  if (query.status !== 0 || query.error || query.signal || query.stderr) {throw new Error("DPC++ Ninja link query failed or warned");}
  const graph = dpcppRuntimeGraph(query.stdout, build, jitObject);
  fs.mkdirSync(output, {mode: 0o700}); fs.mkdirSync(path.join(output, "lib")); fs.mkdirSync(path.join(output, "objects"));
  const files = /** @type {Record<string, string>} */ ({}), libraries = [];
  for (const [source, name] of [...graph.archives.map(file => [file, "lib/" + path.basename(file)]), ...Object.entries(graph.objects).map(([name, file]) => [file, "objects/" + name])]) {
    if (!source || !name || !fs.lstatSync(source).isFile() || fs.realpathSync(source) !== source) {throw new Error("DPC++ exported input is not a canonical regular file");}
    const hash = fileHash(source), destination = path.join(output, name); fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
    if (fileHash(destination) !== hash || fileHash(source) !== hash) {throw new Error("DPC++ export input changed during copy");}
    files[name] = hash; if (name.startsWith("lib/")) {libraries.push(name);}
  }
  if (fileHash(producer) !== sourceSha256 || fileHash(ninja) !== r.inputs[ninja]) {throw new Error("DPC++ export inputs changed");}
  fs.writeFileSync(path.join(output, "component.json"), JSON.stringify({version: 1, kind: "dpcpp-linux", pins, libraries, files,
    producer: {path: producer, sha256: sourceSha256}, graph: query.stdout, buildOnly: true, publicationQualified: false}, null, 2) + "\n", {flag: "wx"});
  return dpcppComponentInputs(output);
}

module.exports = {dpcppRuntimeGraph, dpcppComponentInputs, exportDpcppComponent};
if (require.main === module) {
  try {
    const [producer, ninja, output, extra] = process.argv.slice(2);
    if (!producer || !ninja || !output || extra) {throw new Error("Usage: native-dpcpp-component.js /producer/result.json /matching/ninja /new/component");}
    const r = exportDpcppComponent(producer, ninja, output); console.log("[native-sdk] Exported complete static DPC++ component (" + Object.keys(r.files).length + " bound files); not runtime-qualified.");
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
