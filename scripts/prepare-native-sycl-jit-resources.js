"use strict";

// Build-only GNU source-JIT resource admission. The upstream DPC++ generator still
// consumes the resulting overlay; this script only admits and stages its explicit inputs.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {absolutePath} = require("./build-native-release");
const {fileHash, readBuildJson, readBuildText, buildDependencies, buildFileInputs, completeBuildInputs} = require("./build-source-inputs");
const {validName} = require("./native-jit-headers");
const {jitSourceInputs} = require("./native-jit-sdk");
const {sdkSourceFiles} = require("./prepare-native-sdk-sources");
const {isDeepStrictEqual} = require("node:util");

const ALGORITHMS = ["kawpow", "firopow", "evrprogpow", "meowpow"];
const BACKWARD_VIRTUAL = "include/mom-cxx/backward";
const MAX_FILES = 16384, MAX_BYTES = 64 << 20, MAX_SOURCE = 16 << 20, MAX_DEPFILES = 4 << 20;
const FORBIDDEN = /(?:^|[/\\])(?:msvc(?:[._-]|$)|msvcp(?:[._-]|$)|ucrt(?:[._-]|$)|vcruntime(?:[._-]|$)|api-ms-win(?:[._-]|$))|\.(?:dll|exe|obj|pdb|ilk)$/i;
const LEGACY_BUILD_INPUT = "tests/native/dpcpp-jit/CMakeLists.txt";
const LEGACY_BUILD_INPUT_SHA256 = "66a631feb1d8fab961f84d7e73e981a84e897e08d9c265fece512abe628ea021";
// The CMake fixture remains in the complete SDK recipe, but its Windows-only PCH policy
// cannot affect the standalone source emitter or the resource payload it produces.
const SOURCE_INPUTS = Object.freeze(jitSourceInputs.filter(name => name !== LEGACY_BUILD_INPUT));

/** Same collector, target-specific GNU ABI and search roots. Linux also needs common
 * glibc headers; Windows alone must suppress host-vector aliases in OpenCL headers.
 * @param {string | undefined} target */
function resourceTarget(target = "x86_64-w64-windows-gnu") {
  if (!["x86_64-w64-windows-gnu", "x86_64-unknown-linux-gnu"].includes(target)) {throw new Error("Unsupported GNU JIT resource target");}
  const windows = target === "x86_64-w64-windows-gnu";
  return {target, platform: windows ? "windows-gnu" : "linux-gnu", undefines: windows ? ["__SSE__", "__SSE2__", "__MMX__", "__AVX__"] : [],
    roots: ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target", ...(windows ? [] : ["include/mom-system"])]};
}

/** @typedef {{version: number, fixture: string, fixtureSha256: string, compiler: string, compilerSha256: string,
 * inputs: Record<string, string>}} Receipt
 * @typedef {{version: number, files: Array<{source: string, virtual: string, sha256: string}>}} BaseManifest
 * @typedef {{version: number, target?: string, fixture: string, clang: string, receipt: string, gnuRoot: string,
 * baseRoot: string, noticeRoot: string, syclInclude: string, clangBuiltinInclude: string,
 * baseManifest: string, headerRoots: Array<{path: string, virtual: string}>,
 * required: {spirv32: {virtual: string, sha256: string}, spirv64: {virtual: string, sha256: string},
 * libdevice: {virtual: string, sha256: string}}, notices: Array<{source: string, virtual: string, sha256: string}>}} Profile */

/** @param {unknown} value @param {string} label @returns {string} */
function requirePath(value, label) {
  if (!absolutePath(value)) {throw new Error("Invalid absolute " + label);}
  return /** @type {string} */ (value);
}

/** @param {unknown} value @param {string} label @returns {string} */
function relativeName(value, label) {
  if (typeof value !== "string" || !validName(value) || value.includes(":") || value.includes("\\") ||
      value.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Invalid relative " + label);
  }
  return value;
}

/** @param {string} root @param {string} file @returns {boolean} */
function within(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative));
}

/** @param {string} value @returns {string} */
function foldedName(value) {return value.toLowerCase();}

/** @param {{virtual: string}} a @param {{virtual: string}} b @returns {number} */
function compareVirtual(a, b) {return a.virtual < b.virtual ? -1 : a.virtual > b.virtual ? 1 : 0;}

/** @param {string} file @param {string} label */
function regular(file, label) {
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile()) {throw new Error(label + " is not a regular file: " + file);}
  return stat;
}

/** @param {unknown} value @param {string} label */
function hashValue(value, label) {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/i.test(value)) {throw new Error("Invalid " + label + " hash");}
}

/** @param {unknown} value @returns {Profile} */
function validateProfile(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Invalid GNU JIT profile");}
  const profile = /** @type {Profile} */ (value);
  if (profile.version !== 1) {throw new Error("Invalid GNU JIT profile");}
  const selected = resourceTarget(profile.target);
  requirePath(profile.fixture, "fixture"); requirePath(profile.clang, "clang"); requirePath(profile.receipt, "receipt");
  requirePath(profile.gnuRoot, "GNU package root"); requirePath(profile.baseRoot, "base resource root");
  requirePath(profile.noticeRoot, "notice root"); requirePath(profile.syclInclude, "SYCL include root");
  requirePath(profile.clangBuiltinInclude, "Clang builtin include root"); requirePath(profile.baseManifest, "base manifest");
  if (!Array.isArray(profile.headerRoots) || profile.headerRoots.length !== selected.roots.length) {
    throw new Error("GNU JIT profile needs exactly " + (selected.roots.length === 3 ? "three" : "four") + " header roots");
  }
  for (const [index, root] of profile.headerRoots.entries()) {
    requirePath(root.path, "header root");
    if (root.virtual !== selected.roots[index]) {throw new Error("Unexpected GNU header root layout");}
  }
  if (!Array.isArray(profile.notices) || profile.notices.length < 1 || profile.notices.length > 256) {
    throw new Error("GNU JIT profile needs explicit notice files");
  }
  for (const notice of profile.notices) {
    relativeName(notice.source, "notice source"); relativeName(notice.virtual, "notice name"); hashValue(notice.sha256, "notice");
  }
  if (!profile.required || typeof profile.required !== "object" || Array.isArray(profile.required)) {throw new Error("Missing required device resources");}
  const required = profile.required;
  for (const [name, resource] of [["spirv32", required.spirv32], ["spirv64", required.spirv64], ["libdevice", required.libdevice]]) {
    if (!resource || typeof resource !== "object" || Array.isArray(resource)) {throw new Error("Missing required " + name + " resource");}
    relativeName(resource.virtual, "required resource"); hashValue(resource.sha256, "required resource");
  }
  return profile;
}

/** Admit only the SDK's explicitly hashed base files, never a recursively discovered SDK tree.
 * GNU headers are added later from generated-source dependencies, not borrowed from the host.
 * @param {Profile} profile @param {string} root @returns {{root: string, bytes: number, entries: Array<{source: string, virtual: string, target: string, sha256: string, bytes: number}>}} */
function readBaseManifest(profile, root) {
  const manifestFile = requirePath(profile.baseManifest, "base manifest");
  const value = /** @type {BaseManifest} */ (readBuildJson(manifestFile));
  if (!value || value.version !== 1 || !Array.isArray(value.files) || !value.files.length || value.files.length > MAX_FILES) {
    throw new Error("Invalid explicit base resource manifest");
  }
  const entries = [], virtuals = new Set(), foldedVirtuals = new Set(), sources = new Set(), foldedSources = new Set();
  let total = 0;
  for (const row of value.files) {
    const item = /** @type {{source: string, virtual: string, sha256: string}} */ (row);
    const source = relativeName(item.source, "base source"), virtual = relativeName(item.virtual, "base virtual path");
    hashValue(item.sha256, "base");
    const folded = foldedName(virtual);
    const foldedSource = foldedName(source);
    if (FORBIDDEN.test(source) || FORBIDDEN.test(virtual) || virtuals.has(virtual) || foldedVirtuals.has(folded) ||
        sources.has(source) || foldedSources.has(foldedSource)) {
      throw new Error("Forbidden or duplicate base resource entry");
    }
    const logical = path.resolve(root, source);
    const target = fs.realpathSync(logical);
    if (!within(root, target)) {throw new Error("Base resource symlink escapes its package root");}
    const stat = regular(target, "Base resource");
    const sha256 = fileHash(target);
    if (sha256.toLowerCase() !== item.sha256.toLowerCase()) {throw new Error("Base resource hash mismatch: " + source);}
    total += stat.size;
    if (total > MAX_BYTES) {throw new Error("Base resources exceed bounds");}
    virtuals.add(virtual); foldedVirtuals.add(folded); sources.add(source); foldedSources.add(foldedSource);
    const entry = {source, virtual, target, sha256, bytes: stat.size};
    entries.push(entry);
  }
  entries.sort(compareVirtual);
  return {root, bytes: total, entries};
}

/** Bind the executable source emitter and its maintained inputs to the SDK build receipt.
 * This admission cannot itself prove how an arbitrary caller produced that trusted receipt.
 * @param {Profile} profile @param {string} repo */
function admitReceipt(profile, repo) {
  const fixture = requirePath(profile.fixture, "source fixture"), clang = requirePath(profile.clang, "host clang");
  const fixtureReal = fs.realpathSync(fixture), clangReal = fs.realpathSync(clang);
  const receipt = /** @type {Receipt} */ (readBuildJson(requirePath(profile.receipt, "SDK receipt")));
  if (!receipt || receipt.version !== 1 || receipt.fixture !== fixture || receipt.compiler !== clang ||
      typeof receipt.fixtureSha256 !== "string" || typeof receipt.compilerSha256 !== "string" || !receipt.inputs ||
      typeof receipt.inputs !== "object" || Array.isArray(receipt.inputs)) {throw new Error("SDK receipt does not pin the source fixture and compiler");}
  hashValue(receipt.fixtureSha256, "fixture"); hashValue(receipt.compilerSha256, "compiler");
  if (fileHash(fixtureReal).toLowerCase() !== receipt.fixtureSha256.toLowerCase() ||
      fileHash(clangReal).toLowerCase() !== receipt.compilerSha256.toLowerCase()) {throw new Error("Pinned source fixture/compiler changed");}
  const inputs = receipt.inputs, paths = Object.keys(inputs);
  if (!paths.length || paths.length > MAX_FILES) {throw new Error("SDK receipt has no bounded inputs");}
  const actual = /** @type {Record<string, string>} */ ({});
  for (const file of paths) {
    const expected = inputs[file];
    if (typeof expected !== "string") {throw new Error("Invalid SDK receipt input hash");}
    requirePath(file, "SDK receipt input"); hashValue(expected, "SDK receipt input");
    const real = fs.realpathSync(file), sha256 = fileHash(real);
    if (sha256.toLowerCase() !== expected.toLowerCase()) {throw new Error("SDK receipt input changed: " + file);}
    actual[real] = sha256;
  }
  for (const relative of jitSourceInputs) {
    const file = path.resolve(repo, relative), expected = inputs[file];
    if (typeof expected !== "string" || actual[fs.realpathSync(file)]?.toLowerCase() !== expected.toLowerCase()) {
      throw new Error("SDK receipt does not pin maintained source input: " + relative);
    }
  }
  return {fixture: fixtureReal, clang: clangReal, receipt, inputs: actual};
}

/** @param {string} file @param {string} virtual @param {string} stage */
function copyFile(file, virtual, stage) {
  const destination = path.join(stage, virtual);
  fs.mkdirSync(path.dirname(destination), {recursive: true});
  fs.copyFileSync(file, destination);
}

/** @param {string} command @param {string[]} args @param {string} cwd @param {string} label @param {BufferEncoding | null} encoding */
function run(command, args, cwd, label, encoding = "utf8") {
  regular(command, "Build tool");
  const env = {...process.env};
  for (const name of ["CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "INCLUDE", "LIB", "LIBRARY_PATH", "MOM_NATIVE_RUNTIME"]) {
    delete env[name];
  }
  const result = /** @type {import("node:child_process").SpawnSyncReturns<string | Buffer>} */
    (spawnSync(command, args, {cwd, env, encoding, timeout: 300000, maxBuffer: MAX_SOURCE}));
  if (result.error || result.signal || result.status !== 0) {
    // Failed staging is removed below; retain bounded compiler diagnostics in the build log.
    const diagnostic = result.stderr?.toString().slice(-8192) || result.error?.message || "";
    throw new Error(`Build-only ${label} failed (${result.signal || result.status}): ${diagnostic}`, {cause: result.error});
  }
  return result;
}

/** Build one closed resource overlay without executing a GPU or editing an installed SDK.
 * Keep the base compiler/device resources explicit; discover only GNU headers used by all
 * maintained source variants. Recheck inputs before atomic promotion of the private staging tree.
 * @param {unknown} profile @param {string} output */
function prepare(profile, output) {
  const repo = path.resolve(__dirname, ".."), outputPath = requirePath(output, "output");
  if (fs.existsSync(outputPath)) {throw new Error("Output already exists");}
  const admitted = validateProfile(profile), gnuRoot = fs.realpathSync(requirePath(admitted.gnuRoot, "GNU package root"));
  const selected = resourceTarget(admitted.target);
  const baseRoot = fs.realpathSync(requirePath(admitted.baseRoot, "base resource root"));
  const noticeRoot = fs.realpathSync(requirePath(admitted.noticeRoot, "notice root"));
  if (!fs.statSync(gnuRoot).isDirectory()) {throw new Error("GNU package root is not a directory");}
  if (!fs.statSync(baseRoot).isDirectory()) {throw new Error("Base resource root is not a directory");}
  if (!fs.statSync(noticeRoot).isDirectory()) {throw new Error("Notice root is not a directory");}
  if (within(gnuRoot, baseRoot) || within(baseRoot, gnuRoot) ||
      within(noticeRoot, gnuRoot) && !within(gnuRoot, noticeRoot) ||
      within(noticeRoot, baseRoot) && !within(baseRoot, noticeRoot)) {
    throw new Error("GNU, base and notice roots must be explicit non-overlapping package roots");
  }
  const outputParent = fs.realpathSync(path.dirname(outputPath)), outputCandidate = path.join(outputParent, path.basename(outputPath));
  for (const root of [repo, gnuRoot, baseRoot, noticeRoot]) {
    if (within(root, outputCandidate)) {throw new Error("Output must be outside every admitted input root");}
  }
  const manifest = readBaseManifest(admitted, baseRoot), receipt = admitReceipt(admitted, repo);
  const manifestByVirtual = new Map(manifest.entries.map(entry => [entry.virtual, entry]));
  const required = admitted.required;
  const requiredChecks = [{name: "spirv32", resource: required.spirv32}, {name: "spirv64", resource: required.spirv64},
    {name: "libdevice", resource: required.libdevice}];
  for (const {name, resource} of requiredChecks) {
    const entry = manifestByVirtual.get(resource.virtual);
    if (!entry || entry.sha256.toLowerCase() !== resource.sha256.toLowerCase()) {throw new Error("Required " + name + " resource is absent or changed");}
  }
  const noticeEntries = [], noticeVirtuals = new Set();
  for (const notice of admitted.notices) {
    const source = relativeName(notice.source, "notice source"), virtual = relativeName(notice.virtual, "notice name");
    const foldedVirtual = foldedName(virtual);
    if (FORBIDDEN.test(source) || FORBIDDEN.test(virtual) || noticeVirtuals.has(foldedVirtual)) {throw new Error("Forbidden or duplicate notice path");}
    const target = fs.realpathSync(path.resolve(noticeRoot, source));
    if (!within(noticeRoot, target)) {throw new Error("Notice symlink escapes its admitted package root");}
    const stat = regular(target, "Notice file"), sha256 = fileHash(target);
    if (sha256.toLowerCase() !== String(notice.sha256).toLowerCase()) {throw new Error("Notice hash mismatch: " + source);}
    noticeVirtuals.add(foldedVirtual);
    noticeEntries.push({source, virtual, target, sha256, bytes: stat.size});
  }
  const noticeBytes = noticeEntries.reduce((total, entry) => total + entry.bytes, 0);
  if (manifest.bytes + noticeBytes > MAX_BYTES) {throw new Error("Base resources and notices exceed bounds");}
  noticeEntries.sort(compareVirtual);
  const roots = admitted.headerRoots.map(root => {
    const real = fs.realpathSync(requirePath(root.path, "header root"));
    if (!within(gnuRoot, real) || !fs.statSync(real).isDirectory()) {throw new Error("GNU header root is outside package root");}
    return {path: real, virtual: root.virtual};
  });
  const backward = path.join(roots[0]?.path || "", "backward");
  if (!fs.statSync(backward, {throwIfNoEntry: false})?.isDirectory() || !within(gnuRoot, fs.realpathSync(backward))) {
    throw new Error("Derived GNU backward include root is missing or escapes the package root");
  }
  const compilerRoots = [...roots, {path: fs.realpathSync(backward), virtual: BACKWARD_VIRTUAL}];
  const compilerSearchRoots = [...compilerRoots].sort((a, b) => b.path.length - a.path.length);
  const syclInclude = fs.realpathSync(admitted.syclInclude), clangBuiltinInclude = fs.realpathSync(admitted.clangBuiltinInclude);
  if (!within(baseRoot, syclInclude) || !fs.statSync(syclInclude).isDirectory()) {throw new Error("SYCL include root is outside the base resource root");}
  if (!within(baseRoot, clangBuiltinInclude) || !fs.statSync(clangBuiltinInclude).isDirectory()) {throw new Error("Clang builtin include root is outside the base resource root");}
  const initialFiles = new Set([requirePath(admitted.receipt, "SDK receipt"), requirePath(admitted.baseManifest, "base manifest"),
    requirePath(admitted.fixture, "source fixture"), requirePath(admitted.clang, "host clang"), ...Object.keys(receipt.inputs), ...manifest.entries.map(entry => entry.target),
    ...noticeEntries.map(entry => entry.target)]);
  const initial = buildFileInputs([...initialFiles].map(file => fs.realpathSync(file)));
  const receiptFileSha256 = fileHash(admitted.receipt), baseManifestFileSha256 = fileHash(admitted.baseManifest);
  const startedAt = Date.now();
  const parent = path.dirname(outputPath), staging = fs.mkdtempSync(path.join(parent, ".mom-sycl-jit-resource-"));
  try {
    const overlay = path.join(staging, "overlay"), notices = path.join(staging, "notices"), work = path.join(staging, "work");
    fs.mkdirSync(overlay); fs.mkdirSync(notices); fs.mkdirSync(work);
    for (const entry of manifest.entries) {copyFile(entry.target, entry.virtual, overlay);}
    for (const entry of noticeEntries) {copyFile(entry.target, entry.virtual, notices);}
    const generated = [], depfiles = [], dependencyPaths = new Set();
    for (const algorithm of ALGORITHMS) {
      const source = path.join(work, algorithm + ".cpp");
      const emitted = /** @type {Buffer} */ (run(receipt.fixture, ["--emit-source", algorithm], repo, "source emission", null).stdout);
      if (!Buffer.isBuffer(emitted) || !emitted.length || emitted.length > MAX_SOURCE || emitted.includes(0)) {throw new Error("Invalid generated source");}
      fs.writeFileSync(source, emitted, {flag: "wx"});
      const again = /** @type {Buffer} */ (run(receipt.fixture, ["--emit-source", algorithm], repo, "source determinism", null).stdout);
      if (!Buffer.isBuffer(again) || !emitted.equals(again)) {throw new Error("Generated source is not deterministic");}
      generated.push({algorithm, source, sha256: fileHash(source), bytes: emitted.length});
    }
    const clangArgsBase = ["--target=" + selected.target, "-fsycl", "-fsycl-device-only", ...selected.undefines.flatMap(name => ["-U" + name]), "-nostdinc", "-nostdinc++", "-std=c++17", "-x", "c++", "-fsyntax-only",
      "-isystem", syclInclude, "-isystem", clangBuiltinInclude,
      ...compilerRoots.flatMap(root => ["-isystem", root.path])];
    for (const item of generated) {
      const depfile = path.join(work, item.algorithm + ".d");
      run(receipt.clang, [...clangArgsBase, "-MD", "-MF", depfile, item.source], repo, "dependency probe");
      depfiles.push(depfile);
      const dependencies = buildDependencies(readBuildText(depfile, MAX_DEPFILES), repo);
      if (!dependencies?.length) {throw new Error("Incomplete generated-source dependency graph");}
      for (const dependency of dependencies) {
        if (path.resolve(dependency) === path.resolve(item.source)) {continue;}
        dependencyPaths.add(dependency);
      }
    }
    const byLogical = new Map(manifest.entries.map(entry => [path.resolve(baseRoot, entry.source), entry]));
    const byTarget = new Map();
    for (const entry of manifest.entries) {
      const list = byTarget.get(entry.target) || []; list.push(entry); byTarget.set(entry.target, list);
    }
    const virtualEntries = new Map(manifest.entries.map(entry => [foldedName(entry.virtual), entry]));
    /** @type {Array<{root: string, source: string, virtual: string, target: string, sha256: string, bytes: number}>} */
    const gnuEntries = [], used = new Set();
    let stagedBytes = manifest.bytes + noticeBytes;
    /** @param {string} dependency @param {string} target */
    function admitGnuDependency(dependency, target) {
      const logical = path.resolve(dependency);
      for (const root of compilerSearchRoots) {
        const candidate = within(root.path, logical) ? logical : target;
        if (!within(root.path, candidate)) {continue;}
        const relative = path.relative(root.path, candidate).split(path.sep).join("/");
        if (!relative || !validName(relative)) {throw new Error("Invalid GNU dependency path");}
        const sourcePath = path.relative(gnuRoot, logical).split(path.sep).join("/");
        if (!validName(sourcePath) || !within(gnuRoot, target)) {throw new Error("GNU dependency escapes its package root");}
        const expected = receipt.inputs[target];
        if (!expected) {throw new Error("GNU dependency is absent from the pinned SDK receipt: " + sourcePath);}
        const sha256 = fileHash(target);
        if (sha256.toLowerCase() !== expected.toLowerCase()) {throw new Error("GNU dependency changed: " + sourcePath);}
        const virtual = root.virtual + "/" + relative, folded = foldedName(virtual), prior = virtualEntries.get(folded);
        if (prior) {
          if (prior.sha256.toLowerCase() !== sha256.toLowerCase()) {throw new Error("Conflicting case-folded resource path: " + virtual);}
          used.add(prior.virtual); return;
        }
        const entry = {root: "gnu", source: sourcePath, virtual, target, sha256, bytes: fs.statSync(target).size};
        stagedBytes += entry.bytes;
        if (stagedBytes > MAX_BYTES) {throw new Error("Combined base, notices and GNU headers exceed bounds");}
        virtualEntries.set(folded, entry); gnuEntries.push(entry); used.add(virtual);
        return;
      }
      throw new Error("Generated source dependency is outside the explicit GNU/base roots: " + dependency);
    }
    for (const dependency of dependencyPaths) {
      const logical = path.resolve(dependency), target = fs.realpathSync(logical);
      const direct = byLogical.get(logical), aliases = byTarget.get(target) || [];
      const entry = direct || (aliases.length === 1 ? aliases[0] : undefined);
      if (entry && within(baseRoot, target)) {used.add(entry.virtual); continue;}
      admitGnuDependency(dependency, target);
    }
    for (const entry of gnuEntries) {copyFile(entry.target, entry.virtual, overlay);}
    completeBuildInputs(depfiles, repo, staging, startedAt, initial);
    fs.rmSync(work, {recursive: true, force: true});
    const manifestOutput = {
      version: 1, platform: selected.platform, target: selected.target, undefines: selected.undefines,
      fixtureSha256: receipt.receipt.fixtureSha256, compilerSha256: receipt.receipt.compilerSha256,
      receiptSha256: receiptFileSha256, baseManifestSha256: baseManifestFileSha256,
      sourceInputs: Object.fromEntries(SOURCE_INPUTS.map(relative => [relative, receipt.inputs[fs.realpathSync(path.resolve(repo, relative))]])),
      required: admitted.required,
      generated: generated.map(({algorithm, sha256, bytes}) => ({algorithm, sha256, bytes})), dependencies: [...used].sort(),
      files: [...manifest.entries.map(entry => ({root: "base", source: entry.source, virtual: entry.virtual, sha256: entry.sha256, bytes: entry.bytes})),
        ...gnuEntries.map(({root, source, virtual, sha256, bytes}) => ({root, source, virtual, sha256, bytes}))].sort(compareVirtual),
      notices: noticeEntries.map(({source, virtual, sha256, bytes}) => ({source, virtual, sha256, bytes})),
      publicationQualified: false,
    };
    fs.writeFileSync(path.join(staging, "manifest.json"), JSON.stringify(manifestOutput, null, 2) + "\n", {flag: "wx"});
    if (fs.existsSync(outputPath)) {throw new Error("Output appeared during preparation");}
    fs.renameSync(staging, outputPath);
    return manifestOutput;
  } catch (error) {
    fs.rmSync(staging, {recursive: true, force: true});
    throw error;
  }
}

/** Admit complete generated resources before a long compiler build, then recheck afterward.
 * This verifies target, current maintained source and exact payload bytes, not package origin
 * or licensing. No emitter, compiler or driver executes while inspecting this artifact.
 * @param {string} directory @param {string} target */
function verifyJitResources(directory, target) {
  requirePath(directory, "JIT resource directory");
  const selected = resourceTarget(target), file = path.join(directory, "manifest.json"), repo = path.resolve(__dirname, "..");
  if (!fs.lstatSync(directory).isDirectory() || !fs.lstatSync(file).isFile()) {throw new Error("JIT resources require regular metadata and a real directory");}
  const manifestHash = fileHash(file), manifest = readBuildJson(file);
  if (fileHash(file) !== manifestHash || manifest?.version !== 1 || manifest.target !== selected.target || manifest.platform !== selected.platform ||
      manifest.publicationQualified !== false || !isDeepStrictEqual(manifest.undefines, selected.undefines)) {throw new Error("Wrong-target or changing JIT resources");}
  const sourceNames = manifest.sourceInputs && typeof manifest.sourceInputs === "object" && !Array.isArray(manifest.sourceInputs) ?
    Object.keys(manifest.sourceInputs).sort() : [];
  const currentNames = [...SOURCE_INPUTS].sort(), legacyNames = [...SOURCE_INPUTS, LEGACY_BUILD_INPUT].sort();
  const currentFootprint = isDeepStrictEqual(sourceNames, currentNames);
  const legacyFootprint = isDeepStrictEqual(sourceNames, legacyNames) && manifest.sourceInputs[LEGACY_BUILD_INPUT] === LEGACY_BUILD_INPUT_SHA256;
  if (!currentFootprint && !legacyFootprint) {throw new Error("Incomplete JIT resource source footprint");}
  for (const name of SOURCE_INPUTS) {if (manifest.sourceInputs[name] !== fileHash(path.join(repo, name))) {throw new Error("JIT resource source is stale: " + name);}}
  /** @type {Record<string,string>} */ const inputs = {[file]: manifestHash};
  let bytes = 0;
  for (const [kind, rows, limit] of /** @type {Array<[string, unknown, number]>} */ ([["overlay", manifest.files, MAX_FILES], ["notices", manifest.notices, 256]])) {
    if (!Array.isArray(rows) || !rows.length || rows.length > limit || !fs.lstatSync(path.join(directory, kind)).isDirectory()) {throw new Error("Incomplete JIT resource " + kind);}
    const names = new Set(), entries = [];
    for (const row of rows) {
      const name = relativeName(row?.virtual, "resource payload"); hashValue(row?.sha256, "resource payload");
      if (FORBIDDEN.test(name) || names.has(foldedName(name)) || !Number.isSafeInteger(row.bytes) || row.bytes < 0 || row.bytes > MAX_BYTES - bytes) {throw new Error("Invalid or duplicate JIT resource payload");}
      names.add(foldedName(name)); bytes += row.bytes;
      const payload = path.join(directory, kind, name);
      const stat = fs.lstatSync(payload);
      if (!stat.isFile() || stat.size !== row.bytes) {throw new Error("JIT resource payload is not regular or its size changed: " + name);}
      entries.push([name, row.sha256.toLowerCase()]); inputs[payload] = row.sha256.toLowerCase();
    }
    if (!isDeepStrictEqual(Object.fromEntries(entries), sdkSourceFiles(path.join(directory, kind)))) {throw new Error("JIT resource payload membership or bytes changed: " + kind);}
  }
  if (!manifest.required || Object.keys(manifest.required).sort().join(",") !== "libdevice,spirv32,spirv64") {throw new Error("Incomplete required JIT bitcode");}
  for (const row of Object.values(manifest.required)) {
    const name = relativeName(row?.virtual, "required bitcode"), payload = path.join(directory, "overlay", name);
    hashValue(row.sha256, "required bitcode");
    if (inputs[payload] !== row.sha256.toLowerCase() || fs.statSync(payload).size < 16) {throw new Error("Required JIT bitcode changed or missing");}
    const fd = fs.openSync(payload, "r"), magic = Buffer.alloc(4);
    try {fs.readSync(fd, magic, 0, 4, 0);} finally {fs.closeSync(fd);}
    if (!magic.equals(Buffer.from([0x42, 0x43, 0xc0, 0xde]))) {throw new Error("Required JIT resource is not LLVM bitcode");}
  }
  if (!Array.isArray(manifest.generated) || manifest.generated.length !== ALGORITHMS.length) {throw new Error("Incomplete JIT source variants");}
  for (const [index, row] of manifest.generated.entries()) {
    if (row?.algorithm !== ALGORITHMS[index] || !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > MAX_SOURCE) {throw new Error("Invalid generated JIT source variant");}
    hashValue(row.sha256, "generated source");
  }
  if (fileHash(file) !== manifestHash) {throw new Error("JIT resource manifest changed during admission");}
  return {target: selected.target, directory, overlay: path.join(directory, "overlay"), manifestHash, inputs};
}
module.exports = {prepare, validateProfile, resourceTarget, verifyJitResources, resourceSourceInputs: SOURCE_INPUTS};

if (require.main === module) {
  try {
    const args = process.argv.slice(2), profileIndex = args.indexOf("--profile"), outputIndex = args.indexOf("--output");
    if (args.length !== 4 || profileIndex < 0 || outputIndex < 0 || profileIndex === outputIndex || !args[profileIndex + 1] || !args[outputIndex + 1]) {
      throw new Error("Usage: prepare-native-sycl-jit-resources.js --profile <file> --output <new-directory>");
    }
    const profileArgument = args[profileIndex + 1], outputArgument = args[outputIndex + 1];
    if (!profileArgument || !outputArgument) {throw new Error("Missing profile or output path");}
    const profilePath = requirePath(profileArgument, "profile");
    prepare(readBuildJson(profilePath), outputArgument);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error); process.exitCode = 1;
  }
}
