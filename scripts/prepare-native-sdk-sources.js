"use strict";

// Materialize original pinned SDK objects plus reviewed patches, never a private prebuilt
// archive or edits from an Engineering checkout. Source preparation needs no VM or GPU.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {pins} = require("./probe-static-dpcpp");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {sdkSourceEnvironment} = require("./build-environment");
const {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");

const components = Object.freeze({
  dpcpp: {revision: pins.source, paths: ["sycl", "unified-runtime", "sycl-jit/jit-compiler/include"], leaves: ["LICENSE.TXT"], patches: ["dpcpp-static-adapters.patch", "dpcpp-static-windows.patch", "dpcpp-static-opencl.patch", "dpcpp-static-windows-cuda.patch", "dpcpp-static-jit.patch"]},
  emhash: {revision: pins.emhash, paths: [], leaves: [], patches: []},
  umf: {revision: pins.umf, paths: [], leaves: [], patches: ["umf-static-level-zero.patch", "umf-static-windows.patch"]},
  levelZero: {revision: pins.levelZero, paths: [], leaves: [], patches: ["level-zero-static-complete.patch", "level-zero-static-windows.patch"]},
  computeHeaders: {revision: pins.computeHeaders, paths: ["level_zero/include"], leaves: ["LICENSE.md"], patches: []},
  hwloc: {revision: pins.hwloc, paths: [], leaves: [], patches: []},
  openclLoader: {revision: pins.openclLoader, paths: [], leaves: [], patches: ["opencl-static-windows.patch"]},
  openclHeaders: {revision: pins.openclHeaders, paths: [], leaves: [], patches: []},
});
// The compiler needs original implementation bodies, not just runtime/JIT ABI headers.
// Omit tests/docs and unused ML utilities (which contain source symlinks); the compiler
// recipe must disable their CMake targets. Original TableGen host-tool sources remain present.
const compilerPaths = ["llvm/cmake", "llvm/include", "llvm/lib", "llvm/tools", "llvm/utils/TableGen", "llvm/utils/PerfectShuffle",
  "llvm/projects", "llvm/runtimes", "llvm/resources", "clang/cmake", "clang/include", "clang/lib", "clang/tools", "clang/utils", "clang/runtime", "clang/examples",
  "cmake", "sycl/cmake", "sycl/include", "sycl/source", "sycl-jit", "llvm-spirv", "libdevice", "third-party", "unified-runtime/include", "libc"];
const compilerLeaves = ["LICENSE.TXT", "llvm/CMakeLists.txt", "llvm/LICENSE.TXT", "clang/CMakeLists.txt", "clang/LICENSE.TXT", "sycl/CMakeLists.txt", "sycl/LICENSE.TXT"];
const compilerIntrinsics = {revision: "4fc83e12979096db72c129bd432238d5ca397e4d", paths: /** @type {string[]} */ ([]), leaves: /** @type {string[]} */ ([]), patches: /** @type {string[]} */ ([])};
const compilerSpirvHeaders = {revision: "5c50cbd25a40f8b60e44a2ccc2f1ba3c9e0d0299", paths: ["include", "LICENSES"], leaves: ["LICENSE", "REUSE.toml"], patches: /** @type {string[]} */ ([])};
const repo = path.resolve(__dirname, ".."), own = (/** @type {string} */ name) => path.join(repo, name);
/** Both targets consume the same original repositories. Keep the Windows patch order
 * unchanged; never apply Windows-only build adaptations to a Linux source cohort.
 * @param {string} target @returns {typeof components & {vcIntrinsics?: typeof compilerIntrinsics, spirvHeaders?: typeof compilerSpirvHeaders}} */
function sdkComponents(target) {
  if (!["linux", "windows", "compiler"].includes(target)) {throw new Error("SDK source target must be linux or windows for the runtime, or compiler for shared JIT sources");}
  const compiler = target === "compiler";
  const selected = compiler ? {...components, vcIntrinsics: compilerIntrinsics, spirvHeaders: compilerSpirvHeaders} : components;
  return /** @type {typeof components & {vcIntrinsics?: typeof compilerIntrinsics, spirvHeaders?: typeof compilerSpirvHeaders}} */ (Object.fromEntries(Object.entries(selected).map(([name, component]) =>
    [name, {...component, paths: [...(compiler && name === "dpcpp" ? compilerPaths : component.paths)],
      leaves: [...(compiler && name === "dpcpp" ? compilerLeaves : component.leaves)],
      patches: compiler ? (name === "dpcpp" ? ["dpcpp-static-jit.patch"] : []) :
        component.patches.filter(patch => target === "windows" || !patch.includes("-windows"))}])));
}
/** Runtime preparation needs only the host bridge hunks; compiler preparation needs the
 * reviewed Clang and source-JIT changes too. @param {string} target @param {string} patch */
function sdkPatchScope(target, patch) {
  preparationComponents(target);
  // Stage only patch-declared edits/additions so exact HEAD-diff admission sees new headers.
  // --index also preserves the full index on older Git versions where --intent-to-add did not.
  if (target === "acpp") {return ["--index"];}
  return patch === own("runtime/dpcpp-static-jit.patch") && target !== "compiler" ? ["--include=sycl/*"] : [];
}
/** AdaptiveCpp's existing consumer checks Git history/diff, unlike the exported DPC++ trees.
 * Keep that metadata in a private clone; the original sources and reviewed patches stay shared.
 * @param {string} target
 * @returns {Record<string, {revision: string, paths: string[], leaves: string[], patches: string[]}>} */
function preparationComponents(target) {
  if (target === "acpp") {
    const {syclRevision} = require("./probe-static-sycl");
    return {acpp: {revision: syclRevision, paths: [], leaves: [], patches: ["adaptivecpp-static.patch", "adaptivecpp-jit.patch"]}};
  }
  return sdkComponents(target);
}
/** @param {unknown} name @returns {name is string} */
const safe = name => typeof name === "string" && path.isAbsolute(name) && !Array.from(name).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);

/** The consumer reuses the exact target's recipe, not a receipt's patch/file selection.
 * @param {string} target */
function sdkSourceRecipe(target) {
  return sourceInputs(repo, [], ["scripts/prepare-native-sdk-sources.js", target === "acpp" ? "scripts/probe-static-sycl.js" : "scripts/probe-static-dpcpp.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
    ...(target === "acpp" ? ["scripts/native-hcf.js"] : []),
    ...new Set(Object.values(preparationComponents(target)).flatMap(component => component.patches.map(name => "runtime/" + name)))]);
}

/** One complete original-source cohort; outputs must not be existing paths, ancestors or
 * children of any selected repository. Resolve parents to reject symlink aliases too.
 * @param {unknown} input @param {string} output @param {Record<string, unknown>} [expected] */
function validateSourceRoots(input, output, expected = components) {
  if (!input || typeof input !== "object" || Array.isArray(input) ||
      Object.keys(input).sort().join(",") !== Object.keys(expected).sort().join(",") || !safe(output) || fs.existsSync(output)) {
    throw new Error("Select all " + Object.keys(expected).length + " pinned SDK repositories and a new absolute output directory");
  }
  const target = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  /** @type {Record<string, string>} */ const roots = {};
  for (const [name, value] of Object.entries(input)) {
    if (!safe(value) || !fs.lstatSync(value).isDirectory()) {throw new Error("SDK source root must be a real directory: " + name);}
    const root = fs.realpathSync(value);
    const related = (/** @type {string} */ a, /** @type {string} */ b) => {const relative = path.relative(a, b); return !relative || (!path.isAbsolute(relative) && !relative.startsWith(".." + path.sep) && relative !== "..");};
    if (related(root, target) || related(target, root)) {throw new Error("SDK output overlaps a selected source root");}
    roots[name] = root;
  }
  return roots;
}

/** Stable regular files only: reject source links before an include can escape the new tree.
 * Archive size, file count and total bytes bound both hashing and later source inventories.
 * @param {string} root @param {boolean} [gitCheckout] */
function sdkSourceFiles(root, gitCheckout = false) {
  /** @type {Record<string, string>} */ const files = {};
  let count = 0, bytes = 0;
  /** @param {string} directory */
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      if (gitCheckout && directory === root && entry.name === ".git") {continue;}
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) {visit(file); continue;}
      if (!entry.isFile() || ++count > 100000 || (bytes += fs.statSync(file).size) > (1 << 30)) {throw new Error("SDK source inventory is not bounded regular files");}
      files[path.relative(root, file).split(path.sep).join("/")] = fileHash(file);
    }
  }
  visit(root);
  return files;
}

/** One bounded, offline query policy for preparation and read-only Git-checkout admission.
 * @param {NodeJS.ProcessEnv} env @param {string} command @param {string[]} args @param {string} label */
function sourceQuery(env, command, args, label) {
  const selected = command === "git" ? ["-c", "core.attributesFile=/dev/null", ...args] : args;
  const result = spawnSync(command, selected, {env, encoding: "utf8", timeout: 30000, maxBuffer: 16 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error("SDK source input query failed: " + label + ": " + (result.stderr || "").slice(0, 4096), {cause: result.error});}
  return result.stdout;
}

/** Preserve the original SDK's exact Git/patch admission without changing files or receipts.
 * @param {string} root @param {string} output @param {NodeJS.ProcessEnv} env */
function verifyAcppCheckout(root, output, env) {
  const source = fs.lstatSync(root), git = path.join(root, ".git"), stat = fs.lstatSync(git);
  if (!source.isDirectory() || source.isSymbolicLink()) {throw new Error("AdaptiveCpp source must be a real working directory");}
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.existsSync(path.join(git, "objects/info/alternates"))) {
    throw new Error("AdaptiveCpp source must retain an independent regular Git directory");
  }
  const {admitSyclSource} = require("./probe-static-sycl");
  admitSyclSource(root, output, true, (command, args) => sourceQuery(env, command, args, "AdaptiveCpp checkout"));
}

/** Prove the exported bytes, not just an archive command: Git attributes may omit or
 * substitute members. Match every regular file to the pinned tree's original blob identity.
 * Stream large files with bounded memory; reject mutation during verification.
 * @param {string} root @param {Record<string, string>} files @param {string} tree */
function verifySdkExport(root, files, tree) {
  const entries = tree.split("\0").filter(Boolean), expected = new Set();
  for (const entry of entries) {
    if (/^160000 commit [0-9a-f]{40}\t/.test(entry)) {continue;} // Uninitialized, unused submodule directories are not source files.
    const match = entry.match(/^100(?:644|755) blob ([0-9a-f]{40})\t(.+)$/), name = match?.[2];
    if (!match || !name || name.split("/").some(part => !part || part === "." || part === "..") || name.includes("\\") || expected.has(name) || !files[name]) {throw new Error("SDK export membership differs from its pinned tree");}
    expected.add(name);
    const fd = fs.openSync(path.join(root, name), fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
    try {
      const before = fs.fstatSync(fd, {bigint: true}), hash = createHash("sha1").update("blob " + before.size + "\0"), buffer = Buffer.allocUnsafe(64 << 10);
      let bytes = 0n;
      for (;;) {const count = fs.readSync(fd, buffer, 0, buffer.length, null); if (!count) {break;} bytes += BigInt(count); if (bytes > before.size) {throw new Error("SDK export changed during verification");} hash.update(buffer.subarray(0, count));}
      const after = fs.fstatSync(fd, {bigint: true});
      if (!before.isFile() || bytes !== before.size || after.size !== before.size || after.mtimeNs !== before.mtimeNs || after.ctimeNs !== before.ctimeNs || hash.digest("hex") !== match[1]) {throw new Error("SDK export bytes differ from its pinned blob: " + name);}
    } finally {fs.closeSync(fd);}
  }
  if (expected.size !== Object.keys(files).length) {throw new Error("SDK export has files outside its pinned tree");}
}

/** Build output from Git objects, not the mutable working tree. No network/lazy fetching,
 * submodule initialization, installed-SDK mutation, compilation or publication is implicit.
 * @param {string} target @param {unknown} input @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function prepareSdkSources(target, input, output, environment = process.env) {
  const selectedComponents = preparationComponents(target), gitCheckout = target === "acpp";
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("SDK source preparation requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const roots = validateSourceRoots(input, output, selectedComponents), env = sdkSourceEnvironment(environment);
  if (gitCheckout) {env["GIT_ALLOW_PROTOCOL"] = "file";}
  const recipe = sdkSourceRecipe(target);
  /** @param {string} command @param {string[]} args @param {string} label */
  function query(command, args, label) {
    return sourceQuery(env, command, args, label);
  }
  // Reject Git replacement objects/lazy fetches and symlinks before creating any output.
  /** @type {string[]} */ const failures = [];
  for (const [name, component] of Object.entries(selectedComponents)) {
    try {
      const root = roots[name] || "", prefix = ["-C", root];
      if (query("git", [...prefix, "cat-file", "-t", component.revision], name).trim() !== "commit" ||
          query("git", [...prefix, "ls-tree", "-r", component.revision, "--", ...component.paths, ...component.leaves], name).split("\n").some(line => line.startsWith("120000 "))) {
        throw new Error("SDK sources need the pinned commit and no archive symlinks: " + name);
      }
    } catch (error) {failures.push(error instanceof Error ? error.message : "SDK source admission failed: " + name);}
  }
  if (failures.length) {throw new Error(failures.join("\n"));}
  // Keep this explicit pin synchronized with the original compiler's own dependency;
  // never let CMake discover/fetch a moving VC checkout during the compiler build.
  if (target === "compiler") {
    if (query("git", ["-C", roots["dpcpp"] || "", "show", pins.source + ":llvm/lib/SYCLLowerIR/CMakeLists.txt"], "compiler intrinsic pin")
      .match(/set\(LLVMGenXIntrinsics_GIT_TAG ([0-9a-f]{40})\)/)?.[1] !== compilerIntrinsics.revision) {throw new Error("Compiler intrinsic pin differs from the original DPC++ source");}
    // Disconnected FetchContent can configure successfully with an empty header tree;
    // explicitly admit the translator's own pinned headers instead of accepting that false green.
    if (query("git", ["-C", roots["dpcpp"] || "", "show", pins.source + ":llvm-spirv/spirv-headers-tag.conf"], "compiler SPIR-V pin").trim() !== compilerSpirvHeaders.revision) {
      throw new Error("Compiler SPIR-V header pin differs from the original translator source");
    }
  }
  fs.mkdirSync(output);
  /** @type {Record<string, unknown>} */ const prepared = {};
  for (const [name, component] of Object.entries(selectedComponents)) {
    try {
      const root = path.join(output, name), archives = [];
      if (gitCheckout) {
        query("git", ["clone", "--no-checkout", "--no-hardlinks", "--dissociate", "--template=", "--", roots[name] || "", root], name + " clone");
        query("git", ["-C", root, "checkout", "--detach", component.revision], name + " checkout");
      } else {fs.mkdirSync(root);}
      // Archive selected tree objects directly. A commit-wide archive can eagerly read
      // unrelated missing blobs even with a pathspec, defeating sparse/offline SDK inputs.
      for (const selected of gitCheckout ? [] : component.paths.length ? component.paths : [""]) {
        const archive = path.join(output, name + "-" + archives.length + ".tar");
        query("git", ["-C", roots[name] || "", "archive", "--format=tar", "--output=" + archive,
          ...(selected ? ["--prefix=" + selected + "/"] : []), component.revision + (selected ? ":" + selected : "")], name + " archive");
        if (fs.statSync(archive).size > (512 << 20)) {throw new Error("SDK source archive exceeds bounds: " + name);}
        query("tar", ["--extract", "--file=" + archive, "--directory=" + root, "--no-same-owner", "--no-same-permissions"], name + " unpack");
        archives.push({path: archive, sha256: fileHash(archive)});
      }
      for (const selected of component.leaves) {
        fs.writeFileSync(path.join(root, selected), query("git", ["-C", roots[name] || "", "show", component.revision + ":" + selected], name + " license"), {flag: "wx"});
      }
      const original = sdkSourceFiles(root, gitCheckout), patches = component.patches.map(name => own("runtime/" + name));
      verifySdkExport(root, original, query("git", ["-C", roots[name] || "", "ls-tree", "-rz", "--full-tree", component.revision, "--", ...component.paths, ...component.leaves], name + " original tree"));
      // Windows verifies the pre-applied upstream fix. Linux's original UMF configure
      // instead patches a private build-time hwloc snapshot, never this frozen input.
      if (name === "hwloc" && target === "windows") {patches.push(path.join(output, "umf/cmake/fix_coverity_issues.patch"));}
      for (const patch of patches) {
        // Runtime-only cohorts take the host bridge hunks; compiler cohorts need the
        // same reviewed patch in full, including its Clang/source-JIT adaptations.
        const scope = sdkPatchScope(target, patch);
        query("git", ["-C", root, "apply", ...scope, "--check", "--", patch], name + " patch check");
        query("git", ["-C", root, "apply", ...scope, "--", patch], name + " patch");
      }
      if (gitCheckout) {verifyAcppCheckout(root, output, env);}
      const files = sdkSourceFiles(root, gitCheckout);
      prepared[name] = {root, revision: component.revision, paths: component.paths, leaves: component.leaves, archives,
        // Large compiler cohorts retain their original inventory's digest, not a second
        // full copy of every path/hash. The post-patch file inventory remains explicit.
        original: target === "compiler" ? {files: Object.keys(original).length, sha256: createHash("sha256").update(JSON.stringify(original)).digest("hex")} : original,
        patches: Object.fromEntries(patches.map(file => [file, fileHash(file)])), files};
    } catch (error) {
      const message = error instanceof Error ? error.message : "SDK source preparation failed: " + name;
      prepared[name] = {error: message}; failures.push(message);
    }
  }
  verifySourceInputs(recipe, sdkSourceRecipe(target), "SDK source preparation");
  const result = {version: 2, target, status: failures.length ? "FAIL" : "PASS", prepared, recipe, buildOnly: true, compiled: false, publicationQualified: false,
    scope: `Pinned original ${target} SDK source preparation plus reviewed patches. Not compiled SDK provenance, runtime or release qualification.`};
  fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  if (failures.length) {throw new Error(failures.join("\n"));}
  console.log(`[native-sdk] Prepared original ${target} SDK sources; no compilation or GPU execution.`);
  return result;
}

/** Consumers must recheck complete current source membership, patches and recipe before
 * and after compilation. Never bless already-built archives by updating their input hashes.
 * @param {string} file @param {string} target */
function verifyPreparedSdkSources(file, target) {
  const selectedComponents = preparationComponents(target), gitCheckout = target === "acpp";
  const receipt = readBuildJson(file);
  if (receipt?.version !== 2 || receipt.target !== target || receipt.status !== "PASS" || receipt.compiled !== false ||
      receipt.buildOnly !== true || receipt.publicationQualified !== false ||
      !receipt.prepared || Object.keys(receipt.prepared).sort().join(",") !== Object.keys(selectedComponents).sort().join(",")) {throw new Error("Incomplete or wrong-target SDK source cohort");}
  verifySourceInputs(receipt.recipe, sdkSourceRecipe(target), "SDK source recipe");
  for (const [name, component] of Object.entries(selectedComponents)) {
    const selected = receipt.prepared[name];
    if (!selected || selected.revision !== component.revision || !safe(selected.root) ||
        !isDeepStrictEqual(selected.paths, component.paths) || !isDeepStrictEqual(selected.leaves, component.leaves)) {throw new Error("SDK sources changed: " + name);}
    if (gitCheckout) {verifyAcppCheckout(selected.root, path.dirname(file), {...sdkSourceEnvironment(process.env), GIT_ALLOW_PROTOCOL: "file"});}
    if (!isDeepStrictEqual(selected.files, sdkSourceFiles(selected.root, gitCheckout))) {throw new Error("SDK sources changed: " + name);}
    const patches = component.patches.map(name => own("runtime/" + name));
    if (name === "hwloc" && target === "windows") {patches.push(path.join(receipt.prepared.umf.root, "cmake/fix_coverity_issues.patch"));}
    if (!isDeepStrictEqual(Object.keys(selected.patches || {}).sort(), patches.sort())) {throw new Error("SDK source patch cohort changed: " + name);}
    for (const [patch, hash] of Object.entries(selected.patches || {})) {if (fileHash(patch) !== hash) {throw new Error("SDK patch changed: " + patch);}}
  }
  return receipt;
}

if (require.main === module) {
  try {
    const [target, input, output, extra] = process.argv.slice(2);
    if (!target || !input || !output || extra) {throw new Error("Usage: prepare-native-sdk-sources.js linux|windows|compiler|acpp /absolute/source-roots.json /new/absolute/output");}
    prepareSdkSources(target, readBuildJson(input), output);
  } catch (error) {console.error(error instanceof Error ? error.message : "SDK source preparation failed"); process.exitCode = 1;}
}
module.exports = {sdkComponents, sdkPatchScope, validateSourceRoots, sdkSourceEnvironment, sdkSourceFiles, verifySdkExport, prepareSdkSources, verifyPreparedSdkSources};
