"use strict";

// Original ROCm device libraries and HIPRTC headers must follow the release target's ABI,
// not the build host. This CPU-only producer never installs SDKs or initializes a driver.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {revision} = require("./native-comgr-inputs");
const {sdkStage, selectHostCompilers, sdkHostEnvironment, verifyCompilerEnvironment, hostCompilerEvidence} = require("./build-native-sycl-sdk");
const {sdkSourceEnvironment, sdkSourceFiles, verifySdkExport} = require("./prepare-native-sdk-sources");
const {hasExactBuildFields, sourceInputs, verifySourceInputs, fileHash, readBuildJson, completeBuildInputs, nativeBuildJobs} = require("./build-source-inputs");
const {comgrDeviceInputs} = require("./build-native-comgr-sdk");
const clrRevision = "5b591b2ff64aac4c1c08be879b0d5f89fd713d50";
const repo = path.resolve(__dirname, ".."), recipes = ["scripts/build-native-rocm-resources.js",
  "scripts/native-comgr-inputs.js", "scripts/build-native-comgr-sdk.js", "scripts/build-native-sycl-sdk.js",
  "scripts/prepare-native-sdk-sources.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"];
/** @typedef {{version: number, target: string, source: string, clr: string, llvm: string, cmake: string, hipHeaders: string, hostCompilers?: {cc: string, cxx: string}}} Config */
/** CMake and the upstream assembler generator embed paths literally. Reject syntax that
 * could turn a selected path into an expression; subprocess arguments are never shell text.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && path.isAbsolute(value) && !/[;"$\\\r\n\0]/.test(value);

/** @param {unknown} value @returns {Config} */
function validateConfig(value) {
  const c = /** @type {Config} */ (value), keys = ["clr", "cmake", "hipHeaders", "llvm", "source", "target", "version"];
  if (!c || c.version !== 1 || !["linux", "win32"].includes(c.target) ||
      (!hasExactBuildFields(c, keys) && !hasExactBuildFields(c, [...keys, "hostCompilers"])) ||
      ![c.source, c.clr, c.llvm, c.cmake, c.hipHeaders].every(literal) || /\s/.test(c.hipHeaders)) {
    throw new Error("ROCm resources need an explicit Linux/Windows target and literal absolute source/tool/header paths");
  }
  if (!Object.values(selectHostCompilers(c.llvm, c.hostCompilers)).every(literal)) {throw new Error("Host compilers need literal absolute paths");}
  return c;
}

/** Resolve only the pinned upstream HIPRTC header list, not arbitrary CMake expressions.
 * Both common HIP and CLR headers are supplied by the selected complete header package.
 * @param {string} text @param {string} headers */
function hiprtcHeaders(text, headers) {
  // Upstream separate_arguments(HIPRTC_HEADER_LIST) splits spaces even inside paths.
  // Refuse that layout rather than silently hashing a different set of header fragments.
  if (!literal(headers) || /\s/.test(headers)) {throw new Error("HIPRTC headers need a literal absolute path without whitespace");}
  const block = text.match(/^set\(HIPRTC_HEADERS\s*\n([\s\S]*?)^\)/m)?.[1];
  if (!block) {throw new Error("Missing original HIPRTC header list");}
  const rows = block.trim().split(/\r?\n/).map(row => row.trim());
  if (rows.length < 16 || rows.length > 128) {throw new Error("Unexpected HIPRTC header membership");}
  return rows.map(row => {
    const match = row.match(/^\$\{(?:HIP_COMMON_INCLUDE_DIR\}|PROJECT_SOURCE_DIR\}\/include)\/(hip\/[A-Za-z0-9_./-]+\.h)$/);
    const name = match?.[1];
    if (!name || name.split("/").some(part => !part || part === "." || part === "..")) {throw new Error("Nonliteral HIPRTC header member");}
    return path.join(headers, name);
  });
}

/** Same upstream Windows wchar flag and HIP host auxiliary target used by native builds.
 * A device-only HIP preprocessor still gets size_t from that host target.
 * @param {Config} c @param {string} output @param {number} jobs */
function resourcePlan(c, output, jobs) {
  validateConfig(c);
  if (!literal(output) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {throw new Error("Select a literal output and jobs in 1..16");}
  const source = path.join(output, "source"), build = path.join(output, "build"), install = path.join(output, "install");
  const clang = path.join(c.llvm, "bin/clang"), target = c.target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu";
  // prepare-builtins is a host executable linked to the selected LLVM archives. Its
  // driver must supply that SDK's host runtime; device commands still use imported clang.
  const hostCompilers = selectHostCompilers(c.llvm, c.hostCompilers),
    hipRuntimeWrapper = path.join(c.llvm, "lib/clang/21/include/__clang_hip_runtime_wrapper.h");
  return {source, build, install, clang, target, hostCompilers, hipRuntimeWrapper,
    configure: [c.cmake, "-S", path.join(source, "device"), "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_C_COMPILER=" + hostCompilers.cc, "-DCMAKE_CXX_COMPILER=" + hostCompilers.cxx,
      "-DLLVM_DIR=" + path.join(c.llvm, "lib/cmake/llvm"), "-DClang_DIR=" + path.join(c.llvm, "lib/cmake/clang"),
      "-DCMAKE_INSTALL_PREFIX=" + install, "-DCMAKE_DISABLE_FIND_PACKAGE_ROCM=ON",
      "-DCLANG_OPTIONS_APPEND=" + (c.target === "win32" ? "-fshort-wchar" : "")],
    compile: [c.cmake, "--build", build, "--parallel", String(jobs), "--target", "rocm-device-libs"],
    // Use only the explicitly selected header package; do not discover a host HIP runtime.
    preprocess: [clang, "--target=" + target, "-O3", "-std=c++17", "-nogpulib", "-nogpuinc", "--hip-version=7.1", "-isystem", c.hipHeaders,
      "--cuda-device-only", "-D__HIPCC_RTC__", "-D__HIP_PLATFORM_AMD__", "-DHIP_VERSION_MAJOR=7", "-DHIP_VERSION_MINOR=1",
      // -nogpuinc suppresses the automatic HIP prelude; restore only this compiler-owned RTC wrapper.
      "-include", hipRuntimeWrapper, "-x", "hip", path.join(output, "hipRTC_header.h"), "-E", "-P", "-Xclang", "-dependency-file", "-Xclang", path.join(output, "hipRTC.d"),
      "-Xclang", "-MT", "-Xclang", "hipRTC", "-Xclang", "-sys-header-deps", "-o", path.join(output, "hipRTC")],
  };
}

/** Private original-source generation; outputs carry content receipts, not an execution
 * or publication claim. Caller enforces memory/swap limits. No tool download is implicit.
 * @param {unknown} input @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildResources(input, output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("ROCm resource production requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const c = validateConfig(input), plan = resourcePlan(c, output, nativeBuildJobs(environment));
  if (fs.existsSync(output)) {throw new Error("Select a new output; existing evidence is never overwritten");}
  const destination = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  for (const root of [repo, c.source, c.clr, c.llvm, c.hipHeaders, ...Object.values(plan.hostCompilers)]) {
    const relative = path.relative(fs.realpathSync(root), destination);
    if (!relative || !relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative)) {throw new Error("ROCm output overlaps an input root");}
  }
  const env = sdkHostEnvironment(sdkSourceEnvironment(environment), plan.hostCompilers), recipe = sourceInputs(repo, [], recipes);
  verifyCompilerEnvironment(plan.hostCompilers, env);
  fs.mkdirSync(output); fs.mkdirSync(plan.source);
  const commands = /** @type {string[][]} */ ([]), inputs = /** @type {Record<string, string>} */ ({});
  /** @param {string[]} command @param {string} name */
  function stage(command, name) {commands.push(command); console.log("[rocm-resources] " + name); sdkStage(command, path.join(output, name + ".log"), env);}
  /** @param {string} command @param {string[]} args */
  function query(command, args) {
    const r = spawnSync(command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: 16 << 20});
    if (r.error || r.signal || r.status !== 0) {throw new Error("ROCm resource input query failed: " + command, {cause: r.error});}
    return r.stdout;
  }
  if (!/\bclang version 21\.1\.8\b/.test(query(plan.clang, ["--version"]))) {throw new Error("Select the matching LLVM21.1.8 compiler");}
  const wrapper = fs.lstatSync(plan.hipRuntimeWrapper);
  if (!wrapper.isFile() || wrapper.isSymbolicLink() || wrapper.nlink !== 1 || fs.realpathSync(plan.hipRuntimeWrapper) !== plan.hipRuntimeWrapper) {
    throw new Error("Selected LLVM HIP runtime wrapper must be a canonical singly-linked regular file");
  }
  inputs[plan.hipRuntimeWrapper] = fileHash(plan.hipRuntimeWrapper);
  const {inputs: hostCompilerInputs, versions: compilerVersions} = hostCompilerEvidence(plan.hostCompilers, query);
  Object.assign(inputs, hostCompilerInputs);
  for (const file of [c.cmake, ...["clang", "clang++", "llvm-mc", "llvm-dis", "llvm-link", "opt"].map(name => path.join(c.llvm, "bin", name))]) {inputs[file] = fileHash(file);}
  for (const [name, git, rev, tree] of [["device", c.source, revision, "amd/device-libs"], ["hiprtc", c.clr, clrRevision, "hipamd/src/hiprtc"]]) {
    const snapshot = path.join(plan.source, name || ""), archive = path.join(output, name + ".tar");
    fs.mkdirSync(snapshot);
    stage(["git", "--no-optional-locks", "-C", git || "", "archive", "--format=tar", "--output=" + archive, rev + ":" + tree], name + "-archive");
    stage(["tar", "--extract", "--file", archive, "--directory", snapshot, "--no-same-owner", "--no-same-permissions"], name + "-extract");
    verifySdkExport(snapshot, sdkSourceFiles(snapshot), query("git", ["-C", git || "", "ls-tree", "-rz", rev + ":" + tree]));
  }
  const source = sdkSourceFiles(plan.source), startedAt = Date.now();
  stage(plan.configure, "configure"); stage(plan.compile, "compile");
  stage([c.cmake, "--install", plan.build], "install");
  const generator = path.join(plan.source, "hiprtc/cmake/HIPRTC.cmake"), header = path.join(output, "hipRTC_header.h"), preprocessed = path.join(output, "hipRTC");
  fs.writeFileSync(path.join(output, "generator.cmake"), `set(WIN32 ${c.target === "win32" ? "TRUE" : "FALSE"})\ninclude("${generator}")\ngenerate_hiprtc_header("${header}")\ngenerate_hiprtc_mcin("${output}/hipRTC_header.mcin" "${preprocessed}")\n`, {flag: "wx"});
  stage([c.cmake, "-P", path.join(output, "generator.cmake")], "hiprtc-generate");
  stage(plan.preprocess, "hiprtc-preprocess");
  const headers = hiprtcHeaders(fs.readFileSync(path.join(plan.source, "hiprtc/CMakeLists.txt"), "utf8"), c.hipHeaders);
  for (const file of headers) {inputs[file] = fileHash(file);}
  stage([c.cmake, "-DHIPRTC_ADD_MACROS=1", "-DHIPRTC_HEADERS=" + headers.join(" "), "-DHIPRTC_PREPROCESSED_FILE=" + preprocessed, "-P", generator], "hiprtc-append");
  // Compile the actual generated header, not a regex substitute: Linux size_t embedded
  // in a Windows header fails here before it can reach any GPU or application link.
  const probe = path.join(output, "abi.cpp");
  fs.writeFileSync(probe, 'static_assert(__is_same(size_t, __SIZE_TYPE__), "HIPRTC size_t ABI");\n', {flag: "wx"});
  stage([plan.clang, "--target=" + plan.target, "--offload-arch=gfx900", "--cuda-device-only", "-nogpulib", "-nogpuinc",
    "-D__HIPCC_RTC__", "-D__HIP_PLATFORM_AMD__", "-x", "hip", "-std=c++17", "-fsyntax-only", "-include", preprocessed, probe], "hiprtc-abi");
  const object = path.join(output, c.target === "win32" ? "hipRTC_header.obj" : "hipRTC_header.o");
  stage([path.join(c.llvm, "bin/llvm-mc"), "-o", object, path.join(output, "hipRTC_header.mcin"), "--filetype=obj", "--triple=" + plan.target], "hiprtc-object");
  const device = comgrDeviceInputs(plan.install), expected = c.target === "win32" ? "2" : "4";
  for (const name of ["ockl", "ocml", "hip"]) {
    if (!new RegExp('!"wchar_size", i32 ' + expected + "}").test(query(path.join(c.llvm, "bin/llvm-dis"), [path.join(plan.install, "amdgcn/bitcode", name + ".bc"), "-o", "-"]))) {throw new Error("ROCm device library wchar ABI mismatch: " + name);}
  }
  const dependencies = completeBuildInputs([path.join(output, "hipRTC.d")], repo, output, startedAt, inputs);
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("ROCm resource input changed: " + file);}}
  if (JSON.stringify(source) !== JSON.stringify(sdkSourceFiles(plan.source))) {throw new Error("Original ROCm resource sources changed during compilation");}
  verifySourceInputs(recipe, sourceInputs(repo, [], recipes), "ROCm resource recipe");
  const result = {version: 1, target: c.target, revision, clrRevision, recipe, source, inputs, dependencies, commands, device,
    hostCompilers: plan.hostCompilers, hostCompilerInputs, compilerVersions,
    header: {path: preprocessed, sha256: fileHash(preprocessed)}, object: {path: object, sha256: fileHash(object)},
    buildOnly: true, publicationQualified: false, scope: "Pinned original device libraries and HIPRTC builtins with matching target ABI. Not a complete SDK, application link or GPU/runtime qualification."};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[rocm-resources] Built original " + c.target + " resources; application/runtime qualification remains separate.");
  return result;
}
if (require.main === module) {
  try {
    const [config, output, extra] = process.argv.slice(2);
    if (!config || !output || extra) {throw new Error("Usage: build-native-rocm-resources.js /config.json /new-output");}
    buildResources(readBuildJson(config), output);
  } catch (error) {console.error(error instanceof Error ? error.message : "ROCm resources failed"); process.exitCode = 1;}
}
module.exports = {validateConfig, hiprtcHeaders, resourcePlan, buildResources};
