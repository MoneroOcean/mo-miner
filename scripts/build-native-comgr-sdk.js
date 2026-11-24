"use strict";

// Build the native-only AMD compiler component from pinned Git objects, not a mutable
// working tree. The same maintained patches supply its resource/lifetime policy.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {buildPathsOverlap, fileHash, sourceInputs, verifySourceInputs, readBuildText, readBuildJson, buildFileInputs, completeBuildInputs, nativeBuildJobs} = require("./build-source-inputs");
const {sdkStage, selectHostCompilers, compilerEnvironment, sdkHostEnvironment, verifyCompilerEnvironment, hostCompilerEvidence} = require("./build-native-sycl-sdk");
const {absolutePath} = require("./build-native-release");
const {lldInputs} = require("./native-jit-link");
const {isStaticArchive} = require("./native-audit");
const {revision, patches, recipeFiles} = require("./native-comgr-inputs");
const {requireLinuxSysroot, linuxSysrootRecord} = require("./native-linux-sysroot");

/** Literal commands only. The private source/build/install roots cannot overwrite source
 * or SDK inputs; no configure step may select a shared runtime or run a GPU workload.
 * @param {string} source @param {string} llvm @param {string} lld @param {string} device
 * @param {string} output @param {number} jobs @param {{cc: string, cxx: string}} [selectedHostCompilers] */
function comgrSdkPlan(source, llvm, lld, device, output, jobs, selectedHostCompilers) {
  if (![source, llvm, lld, device, output].every(absolutePath) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Native COMGR needs absolute source/SDK/output roots and jobs in1..16");
  }
  const hostCompilers = selectHostCompilers(llvm, selectedHostCompilers), repo = path.resolve(__dirname, "..");
  for (const input of [source, llvm, lld, device, repo, ...Object.values(hostCompilers)]) {
    if (buildPathsOverlap(input, output)) {throw new Error("COMGR output must not enter or enclose source, SDK or compiler inputs");}
  }
  const snapshot = path.join(output, "source"), build = path.join(output, "build"), archive = path.join(output, "source.tar");
  const header = path.join(llvm, "lib/clang/21/include/opencl-c.h");
  return {snapshot, build, archive, header, commands: [
    ["git", "--no-optional-locks", "-C", source, "archive", "--format=tar", "--prefix=amd/comgr/", "--output=" + archive, revision + ":amd/comgr"],
    ["tar", "--extract", "--file", archive, "--directory", snapshot, "--no-same-owner", "--no-same-permissions"],
    ...patches.map(file => ["git", "--no-optional-locks", "-C", snapshot, "apply", "--check", path.join(repo, file)]),
    ["cmake", "-S", path.join(snapshot, "amd/comgr"), "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_C_COMPILER=" + hostCompilers.cc, "-DCMAKE_CXX_COMPILER=" + hostCompilers.cxx,
      "-DCMAKE_INSTALL_PREFIX=" + path.join(output, "install"), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
      "-DCMAKE_C_FLAGS=", "-DCMAKE_CXX_FLAGS=", "-DCMAKE_EXE_LINKER_FLAGS=",
      "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -ffunction-sections -fdata-sections",
      "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -ffunction-sections -fdata-sections",
      "-DLLVM_DIR=" + path.join(llvm, "lib/cmake/llvm"), "-DClang_DIR=" + path.join(llvm, "lib/cmake/clang"),
      "-DLLD_DIR=" + path.join(lld, "lib/cmake/lld"), "-DAMDDeviceLibs_DIR=" + path.join(device, "lib/cmake/AMDDeviceLibs"),
      "-DCOMGR_OPENCL_C_HEADER=" + header, "-DCOMGR_BUILD_SHARED_LIBS=OFF", "-DCOMGR_EMBEDDED_RESOURCES=ON",
      "-DLLVM_LINK_LLVM_DYLIB=OFF", "-DCOMGR_DISABLE_SPIRV=ON", "-DBUILD_TESTING=OFF", "-DCMAKE_DISABLE_FIND_PACKAGE_ROCM=ON"],
    ["cmake", "--build", build, "--parallel", String(jobs), "--target", "amd_comgr", "--", "-d", "keepdepfile"],
  ], hostCompilers, compilerEnvironment: compilerEnvironment(hostCompilers)};
}

/** The pinned SDK's CMake registry supplies device-library membership. Do not evaluate CMake
 * text in JS or scan unrelated SDKs; every literal imported bitcode file gets a content hash.
 * @param {string} root */
function comgrDeviceInputs(root) {
  if (!absolutePath(root)) {throw new Error("AMD device-library root must be absolute");}
  const config = path.join(root, "lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake"), text = readBuildText(config, 1 << 20);
  const names = [...text.matchAll(/IMPORTED_LOCATION "\$\{AMD_DEVICE_LIBS_PREFIX\}\/amdgcn\/bitcode\/([A-Za-z0-9_.-]+\.bc)"/g)].map(match => match[1] || "");
  if (names.length < 5 || names.length > 256 || new Set(names).size !== names.length ||
      !["ocml.bc", "ockl.bc", "opencl.bc", "hip.bc", "asanrtl.bc"].every(name => names.includes(name)) ||
      names.some(name => name.startsWith(".") || name.includes(".."))) {throw new Error("Incomplete or ambiguous AMD device-library imports");}
  return buildFileInputs([config, ...names.map(name => path.join(root, "amdgcn/bitcode", name))]);
}

/** CMake emits absolute outputs in some versions and relative outputs in others. Admit
 * either spelling only inside this exact build tree, never paths supplied by Make expansion.
 * @param {string} build @param {unknown} entries */
function comgrDependencies(build, entries) {
  if (!absolutePath(build) || !Array.isArray(entries) || entries.length < 10 || entries.length > 128) {throw new Error("Invalid COMGR compiler dependency graph");}
  const files = entries.map(row => {
    if (typeof row?.output !== "string" || row.directory !== build || /[\\\r\n\0]/.test(row.output) ||
        row.output.split("/").some((/** @type {string} */ part) => part === "." || part === "..") || !row.output.endsWith(".o")) {throw new Error("Invalid COMGR object dependency path");}
    const object = path.resolve(build, row.output), relative = path.relative(build, object);
    if (!relative || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) {throw new Error("COMGR object leaves its build tree");}
    return object + ".d";
  });
  if (new Set(files).size !== files.length) {throw new Error("Duplicate COMGR object dependencies");}
  return files;
}

/** Build only a fresh private component; no host installation, downloads, GPU queries or VM.
 * The caller must enforce a host memory/swap envelope, just as for the other SDK producers.
 * @param {string} source @param {string} llvm @param {string} lld @param {string} device
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {{cc: string, cxx: string}} [selectedHostCompilers] */
function buildComgrSdk(source, llvm, lld, device, output, environment = process.env, selectedHostCompilers) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Native COMGR compilation requires Linuxx64 and MOM_AOT_BUILD_ONLY=1");
  }
  if (!absolutePath(output) || fs.existsSync(output)) {throw new Error("Select a new COMGR output; existing SDKs/evidence are never overwritten");}
  // Resolve aliases before the containment check; an output-parent symlink cannot enter an SDK.
  const roots = [source, llvm, lld, device].map(root => {if (!absolutePath(root)) {throw new Error("SDK roots must be absolute");} return fs.realpathSync(root);});
  const destination = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  const plan = comgrSdkPlan(roots[0] || "", roots[1] || "", roots[2] || "", roots[3] || "", destination,
    nativeBuildJobs(environment), selectedHostCompilers);
  const repo = path.resolve(__dirname, ".."), canonicalHostCompilers = Object.values(plan.hostCompilers).map(file => fs.realpathSync(file));
  if ([...roots, repo, ...canonicalHostCompilers].some(input => buildPathsOverlap(input, destination))) {throw new Error("COMGR output overlaps a canonical source, SDK or compiler input");}
  const env = sdkHostEnvironment(environment, plan.hostCompilers), sysroot = requireLinuxSysroot(env);
  verifyCompilerEnvironment(plan.hostCompilers, env);
  // Git-object selection must not inherit an unrelated process's checkout/index override.
  for (const key of Object.keys(env).filter(name => /^GIT_(?:DIR|WORK_TREE|INDEX_FILE|OBJECT_DIRECTORY|ALTERNATE_OBJECT_DIRECTORIES|CONFIG.*)$/.test(name))) {delete env[key];}
  // A partial LLVM clone may lazy-fetch huge unrelated trees even for an archive pathspec.
  // Select the component tree directly and fail for missing objects, never fetch implicitly.
  env["GIT_NO_LAZY_FETCH"] = "1"; env["GIT_TERMINAL_PROMPT"] = "0"; env["GIT_OPTIONAL_LOCKS"] = "0";
  const recipe = sourceInputs(repo, [], recipeFiles);
  fs.mkdirSync(destination); fs.mkdirSync(plan.snapshot);
  /** Small input probes only; compiler output uses the shared streaming lifecycle helper.
   * @param {string} command @param {string[]} args @param {string} name */
  function query(command, args, name) {
    const result = spawnSync(command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    fs.writeFileSync(path.join(destination, name + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Native COMGR input query failed: " + name, {cause: result.error});}
    return result.stdout.trim();
  }
  const [selectedSource = "", selectedLlvm = "", selectedLld = "", selectedDevice = ""] = roots;
  if (query("git", ["--no-optional-locks", "-C", selectedSource, "rev-parse", revision + "^{commit}"], "revision") !== revision) {throw new Error("Missing pinned COMGR Git revision");}
  const version = query(path.join(selectedLlvm, "bin/llvm-config"), ["--version"], "llvm-version");
  if (version !== "21.1.8") {throw new Error("Native COMGR requires the matching reviewed LLVM/Clang21.1.8 SDK");}
  const clangVersion = query(path.join(selectedLlvm, "bin/clang"), ["--version"], "clang-version");
  if (!/\bclang version 21\.1\.8\b/.test(clangVersion)) {throw new Error("COMGR Clang version differs from its selected LLVM SDK");}
  const {inputs: hostCompilerInputs, versions: compilerVersions} = hostCompilerEvidence(plan.hostCompilers, query);
  const sdkInputs = {...comgrDeviceInputs(selectedDevice), ...buildFileInputs([
    ...["clang", "clang++", "llvm-config"].map(name => path.join(selectedLlvm, "bin", name)), plan.header,
    path.join(path.dirname(plan.header), "opencl-c-base.h"), ...lldInputs(selectedLld, version)]), ...hostCompilerInputs};
  fs.writeFileSync(path.join(destination, "plan.json"), JSON.stringify({revision, version, hostCompilers: plan.hostCompilers,
    hostCompilerInputs, compilerEnvironment: plan.compilerEnvironment, compilerVersions, commands: plan.commands, sdkInputs, recipe}, null, 2) + "\n", {flag: "wx"});
  /** @type {ReturnType<typeof sourceInputs> | undefined} */
  let sourceState;
  const startedAt = Date.now();
  for (const [index, command] of plan.commands.entries()) {
    console.log("[native-comgr] stage " + index);
    sdkStage(command, path.join(destination, "stage-" + index + ".log"), env);
    // Check and apply each patch before checking the next; both touch the compiler TU.
    if (index >= 2 && index < 2 + patches.length) {
      sdkStage(command.filter(arg => arg !== "--check"), path.join(destination, "apply-" + index + ".log"), env);
      if (index === 1 + patches.length) {
        sourceState = sourceInputs(plan.snapshot, ["amd/comgr"], []);
        fs.writeFileSync(path.join(destination, "source-inputs-before.json"), JSON.stringify({startedAt, sourceState}, null, 2) + "\n", {flag: "wx"});
      }
    }
  }
  const archive = path.join(plan.build, "libamd_comgr.a"), header = path.join(plan.build, "include/amd_comgr.h");
  if (!isStaticArchive(archive)) {throw new Error("COMGR output is not a regular static archive");}
  for (const [file, sha] of Object.entries(sdkInputs)) {if (fileHash(file) !== sha) {throw new Error("COMGR SDK input changed during compilation: " + file);}}
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "COMGR SDK recipe");
  if (!sourceState) {throw new Error("Missing patched COMGR source receipt");}
  verifySourceInputs(sourceState, sourceInputs(plan.snapshot, ["amd/comgr"], []), "Patched COMGR source");
  const dependencies = completeBuildInputs(comgrDependencies(plan.build, readBuildJson(path.join(plan.build, "compile_commands.json"))), repo, destination, startedAt, sdkInputs);
  fs.writeFileSync(path.join(destination, "result.json"), JSON.stringify({version: 1, revision, llvmVersion: version, archive,
    sdkRoots: {llvm: selectedLlvm, lld: selectedLld, device: selectedDevice}, hostCompilers: plan.hostCompilers, hostCompilerInputs, compilerVersions,
    sha256: fileHash(archive), bytes: fs.statSync(archive).size, header, headerSha256: fileHash(header), sourceState, sdkInputs, dependencies, recipe,
    license: path.join(plan.snapshot, "amd/comgr/LICENSE.txt"), notices: path.join(plan.snapshot, "amd/comgr/NOTICES.txt"),
    sysroot: linuxSysrootRecord(sysroot), buildOnly: true, publicationQualified: false,
    scope: "Complete pinned COMGR source snapshot + maintained LLVM21/static-resource patches. Static component build only; full SDK provisioning/provenance, application integration, JIT/GPU/performance and Windows qualification remain separate gates."}, null, 2) + "\n", {flag: "wx"});
  console.log("[native-comgr] Built " + archive + "; NOT executed or publication-qualified.");
}

if (require.main === module) {
  try {
    const [source, llvm, lld, device, output, cc, cxx, extra] = process.argv.slice(2);
    if (!source || !llvm || !lld || !device || !output || extra || Boolean(cc) !== Boolean(cxx)) {
      throw new Error("Usage: build-native-comgr-sdk.js /ROCm-LLVM-Git /LLVM21 /LLD21 /AMD-device-libs /new-output [/host-cc /host-cxx]");
    }
    buildComgrSdk(source, llvm, lld, device, output, process.env, cc && cxx ? {cc, cxx} : undefined);
  } catch (error) {console.error(error instanceof Error ? error.message : "Native COMGR build failed"); process.exitCode = 1;}
}
module.exports = {comgrSdkPlan, comgrDeviceInputs, comgrDependencies, buildComgrSdk};
