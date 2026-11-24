"use strict";

// Build the original pinned ROCr and CLR/HIP static runtime with an explicitly selected host
// compiler. ROCr also builds its embedded OpenCL image kernels; this producer never initializes a GPU.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {hasExactBuildFields, buildPathsOverlap, fileHash, sourceInputs, verifySourceInputs, buildFileInputs, completeBuildInputs, readBuildJson, readBuildText, nativeBuildJobs} = require("./build-source-inputs");
const {sdkStage, selectHostCompilers, sdkHostEnvironment, verifyCompilerEnvironment, hostCompilerEvidence} = require("./build-native-sycl-sdk");
const {absolutePath} = require("./build-native-release");
const {isStaticArchive} = require("./native-audit");
const {comgrSdkInputs} = require("./native-comgr-inputs");
const {sdkSourceFiles, verifySdkExport} = require("./prepare-native-sdk-sources");
const {rocmInputs} = require("./native-sycl-inputs");
const {linuxSysrootPackages, requireLinuxSysroot, linuxSysrootRecord} = require("./native-linux-sysroot");
const {libdrmPin, nativeLibdrmInputs} = require("./build-native-libdrm");

const pins = Object.freeze({
  clr: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50",
  hip: "993b610351d78993091c0057fcf67636169b7467",
  rocr: "4e4d7abeb786e903ac2a46815ae2565de331cbd4",
});
const archiveNames = Object.freeze({amdhip64: "libamdhip64.a", hiprtc: "libhiprtc.a",
  hsa: "libhsa-runtime64.a", hsakmt: "libhsakmt-staticdrm.a"});
const hsakmtInstallArchiveName = "libhsakmt.a";
const repo = path.resolve(__dirname, "..");
const recipeFiles = ["scripts/build-native-rocm-host-runtime.js", "scripts/build-native-sycl-sdk.js", "scripts/build-native-comgr-sdk.js",
  "scripts/native-comgr-inputs.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-native-release.js", "scripts/build-source-inputs.js",
  "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js",
  "scripts/native-linux-sysroot.js", "scripts/build-native-libdrm.js"];
const configKeys = ["comgrReceipt", "dependencies", "hostCompilers", "libdrm", "pkgConfigDir", "sources", "tools", "version"];
const sourceKeys = ["clr", "hip", "rocr"], toolKeys = ["ar", "cmake", "dpkgQuery", "git", "llvmObjcopy", "ninja", "perl", "pkgConfig", "python3", "ranlib", "tar", "xxd"];
const dependencyKeys = ["drm", "drmAmdgpu", "elf", "numa"];
const xxdPin = Object.freeze({path: "/usr/bin/xxd", package: "xxd", packageVersion: "2:9.1.0016-1ubuntu7.20",
  version: "xxd 2023-10-25 by Juergen Weigert et al."});
const hipPchScript = "hipamd/src/hip_embed_pch.sh";
const hipPchScriptSha256 = "da0c5b74d9ad2eae4a8ca1fd4e454c69bee2dab6691940d798cb45d2deba52fa";
const rocrBlitCmake = "runtime/hsa-runtime/image/blit_src/CMakeLists.txt";
const rocrBlitCmakeSha256 = "711e3834f5a1d3644dfa419c376b83cc9da207d138825c64492e5f683752c9d4";
const pythonPin = Object.freeze({executable: "/usr/bin/python3.12", version: "3.12.3",
  parserVersion: "2.7.4",
  packagePath: "/usr/local/lib/python3.12/dist-packages/CppHeaderParser/__init__.py",
  packageSha256: "1b2436c131e71bfb0d395f6ab1f5f3a6c491d23233e776c026fba4fcf3570d60",
  modulePath: "/usr/local/lib/python3.12/dist-packages/CppHeaderParser/CppHeaderParser.py",
  moduleSha256: "b09cb7802c4860468987272feea715395b45d0d3ad1b0747450a8880f141a9bd",
  sourceUrl: "https://files.pythonhosted.org/packages/3c/ba/d8d168a4b54cae66eaf13d1d9197ca9349c94653815e061f79e7eed86c01/CppHeaderParser-2.7.4.tar.gz",
  sourceSha256: "382b30416d95b0a5e8502b214810dcac2a56432917e2651447d3abe253e3cc42", sourceBytes: 54445,
  plyVersion: "3.11", plyPackageVersion: "3.11-6", plyInitPath: "/usr/lib/python3/dist-packages/ply/__init__.py",
  plyLexPath: "/usr/lib/python3/dist-packages/ply/lex.py"});
const packageNames = Object.freeze({elf: "libelf.a", drm: "libdrm.a", drmAmdgpu: "libdrm_amdgpu.a", numa: "libnuma.a"});
// libelf/libnuma come from the glibc 2.31 sysroot packages; libdrm is built from its pinned source release.
const packageOwners = Object.freeze({elf: "libelf-dev", drm: "libdrm", drmAmdgpu: "libdrm", numa: "libnuma-dev"});
// Header-only GL/EGL/X11 declarations for CLR's always-compiled GL interop; staged privately so the build
// never adds the host /usr/include (and its newer glibc headers) to a compile line.
const glHeaderDirectories = Object.freeze(["EGL", "GL", "KHR", "X11"]);
/** @type {Array<keyof RocmSourcePaths>} */ const sourceNames = ["clr", "hip", "rocr"];
/** @type {Array<keyof RocmDependencyPaths>} */ const dependencyFileNames = ["drm", "drmAmdgpu", "elf", "numa"];
/** @type {Array<keyof RocmTools>} */ const toolNames = ["ar", "cmake", "dpkgQuery", "git", "ninja", "perl", "pkgConfig", "python3", "ranlib", "tar", "xxd"];
/** @type {Array<keyof typeof archiveNames>} */ const runtimeArchiveNames = ["amdhip64", "hiprtc", "hsa", "hsakmt"];

/** CMake values are argv, not shell text; reject characters with list or variable semantics.
 * @param {unknown} value @returns {value is string} */
function literalPath(value) {return absolutePath(value) && !/[;"$\\]/.test(value);}

/** Keep the actual HIP PCH scratch directory unique while giving Clang a stable stdin filename.
 * @param {string} source @returns {string} */
function useStableHipPchStdin(source) {
  let patched = source;
  for (const wave of [32, 64]) {
    const original = `-x hip $tmp/hip_pch.h -E >$tmp/pch_wave${wave}.cui &&`;
    const replacement = `-x hip -E - <$tmp/hip_pch.h >$tmp/pch_wave${wave}.cui &&`;
    if (patched.split(original).length !== 2) {throw new Error("Pinned HIP PCH source has unexpected preprocessor commands");}
    patched = patched.replace(original, replacement);
  }
  return patched;
}

/** Patch only the exact exported upstream helper; never repair a source checkout.
 * @param {string} sourceRoot */
function patchExportedHipPchScript(sourceRoot) {
  const file = path.join(sourceRoot, hipPchScript);
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fileHash(file) !== hipPchScriptSha256) {
    throw new Error("Pinned exported HIP PCH helper changed; refusing source patch");
  }
  const source = fs.readFileSync(file, "utf8"), patched = useStableHipPchStdin(source);
  fs.writeFileSync(file, patched);
  return {path: "source/clr/" + hipPchScript, sourceSha256: hipPchScriptSha256, patchedSha256: fileHash(file), preprocessorInput: "stdin"};
}

/** @param {string} source */
function useExplicitRocrDeviceLibPath(source) {
  const dollar = "$", original = `COMMAND clang ${dollar}{CLANG_ARG_LIST}`,
    replacement = `COMMAND clang "--rocm-device-lib-path=${dollar}{MOM_ROCM_DEVICE_LIBS_PATH}" ${dollar}{CLANG_ARG_LIST}`;
  if (source.split(original).length !== 2) {throw new Error("Pinned ROCr image-kernel command has changed");}
  return source.replace(original, replacement);
}

/** Patch only the exact exported upstream image-kernel command; the path comes from the admitted
 * source-built COMGR device-library receipt, never a host ROCm search. @param {string} sourceRoot */
function patchExportedRocrDeviceLibPath(sourceRoot) {
  const file = path.join(sourceRoot, rocrBlitCmake);
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fileHash(file) !== rocrBlitCmakeSha256) {
    throw new Error("Pinned exported ROCr image-kernel CMake source changed; refusing source patch");
  }
  const source = fs.readFileSync(file, "utf8"), patched = useExplicitRocrDeviceLibPath(source);
  fs.writeFileSync(file, patched);
  return {path: "source/rocr/" + rocrBlitCmake, sourceSha256: rocrBlitCmakeSha256, patchedSha256: fileHash(file),
    option: "--rocm-device-lib-path", cacheVariable: "MOM_ROCM_DEVICE_LIBS_PATH"};
}

/** @param {unknown} input */
function validateConfig(input) {
  const c = /** @type {RocmHostConfig | null} */ (input);
  if (!c || c.version !== 1 || !hasExactBuildFields(c, configKeys) ||
      !hasExactBuildFields(c.sources, sourceKeys) || !hasExactBuildFields(c.tools, toolKeys) ||
      !hasExactBuildFields(c.dependencies, dependencyKeys) || !hasExactBuildFields(c.hostCompilers, ["cc", "cxx"]) ||
      c.tools.python3 !== pythonPin.executable || c.tools.xxd !== xxdPin.path ||
      ![c.comgrReceipt, c.libdrm, c.pkgConfigDir, ...Object.values(c.sources), ...Object.values(c.tools),
        ...Object.values(c.dependencies), ...Object.values(c.hostCompilers)].every(literalPath)) {
    throw new Error("ROCm host runtime needs exact pinned roots, tool paths, static OS archives and a COMGR receipt");
  }
  return /** @type {RocmHostConfig} */ (c);
}

/** @typedef {{clr: string, hip: string, rocr: string}} RocmSourcePaths */
/** @typedef {{ar: string, cmake: string, dpkgQuery: string, git: string, llvmObjcopy: string, ninja: string, perl: string, pkgConfig: string, python3: string, ranlib: string, tar: string, xxd: string}} RocmTools */
/** @typedef {{drm: string, drmAmdgpu: string, elf: string, numa: string}} RocmDependencyPaths */
/** @typedef {{version: number, sources: RocmSourcePaths, tools: RocmTools, hostCompilers: {cc: string, cxx: string},
 * dependencies: RocmDependencyPaths, libdrm: string, pkgConfigDir: string, comgrReceipt: string}} RocmHostConfig */
/** @typedef {{llvm?: string, comgrPackage?: string, device?: string}} RocmSdkRoots */
/** @typedef {{name: "rocr" | "clr", cache: string, database: string, depfiles: string[], commandCount: number}} ConfiguredBuild */
/** @typedef {{path: string, bytes: number, sha256: string}} ArchiveEvidence */
/** @typedef {{home: string, tmp: string, cleanup: () => void}} PrivateBuildDirectories */

/** One literal build plan; host compilers and original ROCm assembly tools remain distinct.
 * The glibc sysroot supplies libc/NUMA/libelf headers; GL headers come from the private overlay.
 * @param {unknown} input @param {string} output @param {number} jobs @param {RocmSdkRoots} [sdkRoots]
 * @param {{root: string, toolchain: string}} [sysroot] */
function rocmHostPlan(input, output, jobs, sdkRoots = /** @type {RocmSdkRoots} */ ({}), sysroot = {root: "<SYSROOT>", toolchain: "<SYSROOT-TOOLCHAIN>"}) {
  const c = validateConfig(input);
  if (!literalPath(output) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Native ROCm host runtime needs a literal output and jobs in1..16");
  }
  const llvm = sdkRoots.llvm || "<COMGR-LLVM>", comgrPackage = sdkRoots.comgrPackage || "<COMGR-PACKAGE>",
    device = sdkRoots.device || "<COMGR-DEVICE-LIBS>", deviceLibPath = path.join(device, "amdgcn/bitcode");
  if ((!literalPath(llvm) && llvm !== "<COMGR-LLVM>") || (!literalPath(comgrPackage) && comgrPackage !== "<COMGR-PACKAGE>") ||
      (!literalPath(device) && device !== "<COMGR-DEVICE-LIBS>") ||
      Boolean(sdkRoots.llvm) !== Boolean(sdkRoots.comgrPackage) || Boolean(sdkRoots.llvm) !== Boolean(sdkRoots.device)) {
    throw new Error("ROCm host runtime needs literal selected COMGR LLVM, package and device-library roots");
  }
  const hostCompilers = selectHostCompilers("", c.hostCompilers), sources = c.sources, tools = c.tools;
  const imageLlvmRoot = path.dirname(path.dirname(tools.llvmObjcopy));
  const imageLlvmPackage = path.join(imageLlvmRoot, "lib/cmake/llvm");
  const clangPackage = path.join(llvm, "lib/cmake/clang");
  const source = {clr: path.join(output, "source/clr"), hip: path.join(output, "source/hip"), rocr: path.join(output, "source/rocr")};
  const build = {rocr: path.join(output, "build/rocr"), clr: path.join(output, "build/clr")};
  const rocrInstall = path.join(output, "rocr-install"), sdk = path.join(output, "sdk");
  const comgrHeaderInclude = path.join(output, "comgr-install-include");
  const comgrHeader = path.join(comgrHeaderInclude, "amd_comgr", "amd_comgr.h");
  const glHeaderInclude = path.join(output, "gl-include"), systemInclude = path.join(sysroot.root, "usr/include");
  const clrCxxFlags = '-I"' + comgrHeaderInclude + '" -isystem "' + glHeaderInclude + '"';
  const comgrPrefix = path.dirname(path.dirname(path.dirname(comgrPackage)));
  // ROCclr's quiet 2.9 probe rejects our genuine 3.0 package and clears amd_comgr_DIR;
  // keep its required 3.0 fallback ahead of coinstalled /opt/rocm in CMake's search path.
  const cmakePrefixPath = rocrInstall + ";" + comgrPrefix;
  const inputPaths = [...Object.values(sources), ...Object.values(tools), ...Object.values(c.dependencies), c.comgrReceipt,
    c.pkgConfigDir, ...Object.values(hostCompilers), imageLlvmRoot, imageLlvmPackage,
    ...(sdkRoots.llvm ? [sdkRoots.llvm, comgrPackage, device, comgrPrefix, clangPackage, deviceLibPath] : []), repo];
  if (inputPaths.some(inputPath => buildPathsOverlap(inputPath, output))) {throw new Error("ROCm output must not overlap a source, tool, SDK, receipt or repository input");}
  const common = ["-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + tools.ninja, "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=" + hostCompilers.cc, "-DCMAKE_CXX_COMPILER=" + hostCompilers.cxx,
    "-DCMAKE_TOOLCHAIN_FILE=" + sysroot.toolchain, "-DPython3_EXECUTABLE=" + tools.python3,
    "-DGIT_EXECUTABLE=" + tools.git, "-DPERL_EXECUTABLE=" + tools.perl,
    "-DCMAKE_AR=" + tools.ar, "-DCMAKE_RANLIB=" + tools.ranlib, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_C_FLAGS=",
    "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -ffunction-sections -fdata-sections",
    "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -ffunction-sections -fdata-sections",
    "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_INSTALL_INCLUDEDIR=include", "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_TESTING=OFF", "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE",
    "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON", "-DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON"];
  const rocrConfigure = [tools.cmake, "-S", source.rocr, "-B", build.rocr, ...common, "-DCMAKE_CXX_FLAGS=",
    "-DCMAKE_INSTALL_PREFIX=" + rocrInstall, "-DBUILD_ROCR=ON", "-DBUILD_THUNK_VIRTIO=OFF", "-DROCM_CCACHE_BUILD=OFF",
    "-DEXPORT_TO_USER_PACKAGE_REGISTRY=OFF", "-DENABLE_LDCONFIG=OFF",
    "-DLLVM_DIR=" + imageLlvmPackage, "-DClang_DIR=" + clangPackage,
    "-DMOM_ROCM_DEVICE_LIBS_PATH=" + deviceLibPath,
    "-DHSA_ASSEMBLER=" + path.join(llvm, "bin/clang-21"), "-DHSA_OBJCOPY=" + tools.llvmObjcopy,
    "-DPKG_CONFIG_EXECUTABLE=" + tools.pkgConfig, "-DLIBELF_INCLUDE_DIRS=" + systemInclude,
    "-DLIBELF_LIBRARIES=" + c.dependencies.elf, "-DNUMA=" + c.dependencies.numa,
    "-DNUMA_LIBRARIES=" + c.dependencies.numa];
  const rocrBuild = [tools.cmake, "--build", build.rocr, "--parallel", String(jobs), "--target", "hsa-runtime64_static", "hsakmt", "hsakmt-staticdrm", "--", "-d", "keepdepfile"];
  const rocrInstallBinary = [tools.cmake, "--install", build.rocr, "--prefix", rocrInstall, "--component", "binary"];
  const rocrInstallDev = [tools.cmake, "--install", build.rocr, "--prefix", rocrInstall, "--component", "dev"];
  const clrConfigure = [tools.cmake, "-S", source.clr, "-B", build.clr, ...common, "-DCMAKE_CXX_FLAGS=" + clrCxxFlags,
    "-DCMAKE_INSTALL_PREFIX=" + sdk, "-DCMAKE_PREFIX_PATH=" + cmakePrefixPath, "-DROCM_INSTALL_PATH=" + rocrInstall,
    "-DLLVM_DIR=" + path.join(llvm, "lib/cmake/llvm"), "-DClang_DIR=" + path.join(llvm, "lib/cmake/clang"),
    "-DLLD_DIR=" + path.join(llvm, "lib/cmake/lld"),
    "-DCLR_BUILD_HIP=ON", "-DCLR_BUILD_OCL=OFF", "-DHIP_PLATFORM=amd", "-DBUILD_SHARED_LIBS=OFF",
    "-DHIP_COMMON_DIR=" + source.hip, "-DHIPCC_BIN_DIR=", "-DROCCLR_PATH=" + path.join(source.clr, "rocclr"),
    "-DAMD_OPENCL_PATH=" + path.join(source.clr, "opencl"), "-DAMD_OPENCL_INCLUDE_DIR=" + path.join(source.clr, "opencl/khronos/headers/opencl2.2/CL"),
    "-DHIP_LLVM_ROOT=" + imageLlvmRoot,
    "-Damd_comgr_DIR=" + comgrPackage, "-Dhsa-runtime64_DIR=" + path.join(rocrInstall, "lib/cmake/hsa-runtime64"),
    "-DNUMA_INCLUDE_DIR=" + systemInclude, "-DNUMA_LIBRARIES=" + c.dependencies.numa,
    "-DLIBELF_INCLUDE_DIRS=" + systemInclude, "-DLIBELF_LIBRARIES=" + c.dependencies.elf,
    // FindOpenGL only gates configuration; the static archives never link these host dispatch libraries.
    "-DOPENGL_INCLUDE_DIR=" + glHeaderInclude, "-DOPENGL_EGL_INCLUDE_DIR=" + glHeaderInclude, "-DOPENGL_GLX_INCLUDE_DIR=" + glHeaderInclude,
    "-DOPENGL_opengl_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenGL.so", "-DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so",
    "-DCMAKE_DISABLE_FIND_PACKAGE_ROCM=ON"];
  const clrBuild = [tools.cmake, "--build", build.clr, "--parallel", String(jobs), "--target", "amdhip64", "hiprtc", "--", "-d", "keepdepfile"];
  const clrInstallBinary = [tools.cmake, "--install", build.clr, "--prefix", sdk, "--component", "binary"];
  const clrInstallDev = [tools.cmake, "--install", build.clr, "--prefix", sdk, "--component", "dev"];
  return {sources, tools, hostCompilers, source, build, rocrInstall, sdk, comgrHeaderInclude, comgrHeader, clrCxxFlags, glHeaderInclude,
    llvm, comgrPackage, comgrPrefix, cmakePrefixPath, device, deviceLibPath,
    imageLlvmRoot, imageLlvmPackage, clangPackage, jobs,
    dependencies: c.dependencies, pkgConfigDir: c.pkgConfigDir,
    commands: {rocrConfigure, rocrBuild, rocrInstallBinary, rocrInstallDev, clrConfigure, clrBuild, clrInstallBinary, clrInstallDev}};
}

/** Require a canonical regular path before CMake or a package query can observe it.
 * @param {string} file @param {string} label @param {boolean} [executable] */
function regularPath(file, label, executable = false) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || fs.realpathSync(file) !== file || (executable && !(stat.mode & 0o111))) {
    throw new Error("ROCm " + label + " must be a canonical regular" + (executable ? " executable" : " file") + ": " + file);
  }
  return file;
}

/** @param {string} directory @param {string} label */
function realDirectory(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || fs.realpathSync(directory) !== directory) {throw new Error("ROCm " + label + " must be a canonical directory: " + directory);}
  return directory;
}

/** Keep compiler identity probes private before the final output tree is created.
 * @param {string} parent @returns {PrivateBuildDirectories} */
function privateBuildDirectories(parent) {
  const root = fs.mkdtempSync(path.join(parent, ".rocm-host-private-"));
  try {
    fs.chmodSync(root, 0o700);
    const home = path.join(root, "home"), tmp = path.join(root, "tmp");
    fs.mkdirSync(home, {mode: 0o700}); fs.mkdirSync(tmp, {mode: 0o700});
    return {home, tmp, cleanup() {
      const stat = fs.lstatSync(root, {throwIfNoEntry: false});
      if (!stat) {return;}
      if (!stat.isDirectory() || fs.realpathSync(root) !== root || path.dirname(root) !== parent) {
        throw new Error("ROCm private preflight directory changed before cleanup: " + root);
      }
      fs.rmSync(root, {recursive: true});
    }};
  } catch (error) {
    fs.rmSync(root, {recursive: true, force: true});
    throw error;
  }
}

/** @template T @param {string} parent @param {(temporary: PrivateBuildDirectories) => T} run @returns {T} */
function withPrivateBuildDirectories(parent, run) {
  const temporary = privateBuildDirectories(parent);
  try {
    return run(temporary);
  } finally {temporary.cleanup();}
}

/** Keep all process-level compiler and Git hooks out of the build. Selected executables still
 * run by absolute path; PATH only exposes their directories and the pinned image's OS tools.
 * @param {NodeJS.ProcessEnv} environment @param {RocmHostConfig} config @param {ReturnType<typeof rocmHostPlan>} plan
 * @param {string} output @param {string} llvm @param {Pick<PrivateBuildDirectories, "home" | "tmp">} [temporary] */
function hostEnvironment(environment, config, plan, output, llvm, temporary) {
  const env = sdkHostEnvironment(environment, plan.hostCompilers);
  // LLVM and Intel tools load pinned shared runtimes from the activated SDK path.
  for (const key of Object.keys(env)) {
    if (/^(?:CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|OBJC_INCLUDE_PATH|LIBRARY_PATH|PKG_CONFIG_PATH|PKG_CONFIG_LIBDIR|PKG_CONFIG_SYSROOT_DIR)$/i.test(key) ||
        /^XDG_/.test(key)) {delete env[key];}
  }
  const directories = [...Object.values(plan.hostCompilers).map(file => path.dirname(file)),
    path.dirname(config.tools.cmake), path.dirname(config.tools.ninja), path.dirname(config.tools.git), path.dirname(config.tools.tar),
    path.dirname(config.tools.pkgConfig), path.dirname(config.tools.python3), path.dirname(config.tools.dpkgQuery), path.dirname(config.tools.ar), path.dirname(config.tools.ranlib),
    path.dirname(config.tools.perl), path.dirname(config.tools.xxd), path.join(llvm, "bin")];
  env["PATH"] = [...new Set([...directories, "/usr/bin", "/bin"])].join(path.delimiter);
  env["PKG_CONFIG_LIBDIR"] = config.pkgConfigDir;
  env["PKG_CONFIG_PATH"] = "";
  env["HOME"] = temporary?.home || path.join(output, "private-home");
  env["TMPDIR"] = temporary?.tmp || path.join(output, "tmp");
  env["LC_ALL"] = "C";
  env["PYTHONDONTWRITEBYTECODE"] = "1";
  env["GIT_TERMINAL_PROMPT"] = "0";
  return env;
}

/** @param {string} command @param {string[]} args @param {NodeJS.ProcessEnv} env @param {string} label @param {number} [maximum] @param {boolean} [includeStderr] */
function query(command, args, env, label, maximum = 1 << 20, includeStderr = false) {
  const result = spawnSync(command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: maximum});
  const output = typeof result.stdout === "string" ? result.stdout + (includeStderr ? result.stderr || "" : "") : result.stdout;
  if (result.error || result.signal || result.status !== 0 || typeof output !== "string" || output.length > maximum) {
    throw new Error("ROCm host runtime input query failed: " + label + (result.stderr ? ": " + result.stderr.slice(0, 2048) : ""), {cause: result.error});
  }
  return output;
}

/** Capture package ownership from the selected image, never infer origin from a copied archive.
 * @param {string} file @param {string} ownersText @param {string} statusText @param {string} sha256 @param {string} expectedPackage */
function parsePackageOrigin(file, ownersText, statusText, sha256, expectedPackage) {
  const owners = ownersText.trimEnd().split(/\r?\n/);
  if (owners.length !== 1 || !owners[0]) {throw new Error("Static ROCm dependency must have exactly one OS package owner: " + file);}
  const match = owners[0]?.match(/^(.+?):\s+(.+)$/), packageId = match?.[1], ownedPath = match?.[2];
  if (!packageId || ownedPath !== file || !/^[A-Za-z0-9.+-]+(?::[A-Za-z0-9_-]+)?$/.test(packageId) ||
      !/^[A-Za-z0-9.+-]+$/.test(expectedPackage) || packageId.split(":")[0] !== expectedPackage) {
    throw new Error("Static ROCm dependency package ownership is ambiguous: " + file);
  }
  const row = statusText.trimEnd().split("\t"), name = row[0], architecture = row[1];
  const expectedId = packageId.includes(":") ? name + ":" + architecture : name;
  if (row.length !== 4 || name !== expectedPackage || expectedId !== packageId ||
      (!packageId.includes(":") && !["all", "amd64"].includes(architecture || "")) || !row[2] || !row[3]?.trim().startsWith("ii") ||
      !/^[a-f0-9]{64}$/.test(sha256)) {
    throw new Error("Static ROCm dependency package is not installed: " + packageId);
  }
  return {path: file, sha256, package: name || "", architecture: architecture || "", version: row[2] || "", status: row[3]?.trim() || ""};
}

/** @param {string} file @param {NodeJS.ProcessEnv} env @param {string} dpkgQuery @param {string} expectedPackage */
function packageOrigin(file, env, dpkgQuery, expectedPackage) {
  const owners = query(dpkgQuery, ["-S", "--", file], env, "static archive package owner").trim();
  const packageId = owners.match(/^(.+?):\s+/)?.[1] || "";
  if (!packageId) {throw new Error("Static ROCm dependency package ownership is ambiguous: " + file);}
  const field = String.fromCharCode(36), format = ["Package", "Architecture", "Version", "db:Status-Abbrev"]
    .map(name => field + "{" + name + "}").join("\\t") + "\\n";
  const status = query(dpkgQuery, ["-W", "-f=" + format, packageId], env,
    "static archive package status");
  return parsePackageOrigin(file, owners, status, fileHash(file), expectedPackage);
}

/** Static OS-library inputs are admitted by their producer, not by host package ownership: libelf/libnuma
 * by the pinned sysroot tree, libdrm by its source-built SDK receipt.
 * @param {keyof RocmDependencyPaths} name @param {string} file */
function staticInputOrigin(name, file) {
  const owner = packageOwners[name], sysrootPackage = linuxSysrootPackages.find(row => row.package === owner);
  return {path: file, sha256: fileHash(file), package: owner, architecture: "amd64",
    version: sysrootPackage ? sysrootPackage.version : libdrmPin.version, status: sysrootPackage ? "sysroot" : "source"};
}

/** Hash the host GL/EGL/KHR/X11 header trees (no links, bounded) for the private include overlay. */
function glHeaderInputs() {
  /** @type {Record<string, string>} */ const files = {};
  const visit = (/** @type {string} */ directory) => {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const file = path.join(directory, entry.name);
      if (Object.keys(files).length > 4096) {throw new Error("Host GL header tree exceeds its bound");}
      if (entry.isDirectory()) {visit(file);} else if (entry.isFile() && /\.h$/.test(entry.name)) {files[file] = fileHash(file);} else if (!entry.isFile()) {
        throw new Error("Host GL header tree contains a link or special file: " + file);
      }
    }
  };
  for (const name of glHeaderDirectories) {visit(realDirectory(path.join("/usr/include", name), "host " + name + " header directory"));}
  return files;
}

/** Record and pin only the Python inputs used to generate HIP profiling headers.
 * @param {string} interpreter @param {NodeJS.ProcessEnv} env @param {string} dpkgQuery */
function pythonBuildInput(interpreter, env, dpkgQuery) {
  const code = [
    "import hashlib,importlib,json,pathlib,sys,ply,ply.lex",
    "package=importlib.import_module('CppHeaderParser')",
    "parser=importlib.import_module('CppHeaderParser.CppHeaderParser')",
    "def item(path):",
    " p=pathlib.Path(path).resolve(); return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}",
    "print(json.dumps({'executable':str(pathlib.Path(sys.executable).resolve()),'version':sys.version.split()[0],",
    " 'package':item(package.__file__),'parser':item(parser.__file__),'parserVersion':parser.__version__,",
    " 'plyVersion':ply.__version__,'plyInit':item(ply.__file__),'plyLex':item(ply.lex.__file__)}))",
  ].join("\n");
  const found = /** @type {{executable: string, version: string, package: {path: string, sha256: string}, parser: {path: string, sha256: string},
   parserVersion: string, plyVersion: string, plyInit: {path: string, sha256: string}, plyLex: {path: string, sha256: string}}} */ (
      JSON.parse(query(interpreter, ["-c", code], env, "Python profiling-header dependencies")));
  const expected = {executable: interpreter, version: pythonPin.version, parserVersion: pythonPin.parserVersion,
    packagePath: pythonPin.packagePath, packageSha256: pythonPin.packageSha256,
    parserPath: pythonPin.modulePath, parserSha256: pythonPin.moduleSha256,
    plyVersion: pythonPin.plyVersion, plyInitPath: pythonPin.plyInitPath, plyLexPath: pythonPin.plyLexPath};
  const observed = {executable: found.executable, version: found.version, parserVersion: found.parserVersion,
    packagePath: found.package.path, packageSha256: found.package.sha256,
    parserPath: found.parser.path, parserSha256: found.parser.sha256,
    plyVersion: found.plyVersion, plyInitPath: found.plyInit.path, plyLexPath: found.plyLex.path};
  for (const [name, value] of Object.entries(expected)) {
    if (Reflect.get(observed, name) !== value) {throw new Error("ROCm Python profiling-header input changed: " + name);}
  }
  const files = [found.executable, found.package.path, found.parser.path, found.plyInit.path, found.plyLex.path];
  for (const file of files) {regularPath(file, "Python profiling-header input", file === interpreter);}
  const plyOrigins = [packageOrigin(found.plyInit.path, env, dpkgQuery, "python3-ply"),
    packageOrigin(found.plyLex.path, env, dpkgQuery, "python3-ply")];
  if (plyOrigins.some(origin => origin.version !== pythonPin.plyPackageVersion)) {
    throw new Error("ROCm PLY package version changed");
  }
  return {interpreter: {path: found.executable, version: found.version, sha256: fileHash(interpreter)},
    parser: {name: "CppHeaderParser", version: found.parserVersion, packagePath: found.package.path, packageSha256: found.package.sha256,
      modulePath: found.parser.path, moduleSha256: found.parser.sha256,
      sourceDistribution: {url: pythonPin.sourceUrl, sha256: pythonPin.sourceSha256, bytes: pythonPin.sourceBytes}},
    ply: {version: found.plyVersion, init: {path: found.plyInit.path, sha256: found.plyInit.sha256, package: plyOrigins[0]},
      lex: {path: found.plyLex.path, sha256: found.plyLex.sha256, package: plyOrigins[1]}}};
}

/** CMake 3.20+ writes an output path for every compilation database row. Each preserved Ninja
 * depfile must belong to this build and have the matching object output.
 * @param {string} build @param {unknown} entries */
function compileDepfiles(build, entries) {
  if (!absolutePath(build) || !Array.isArray(entries) || entries.length < 8 || entries.length > 20000) {
    throw new Error("Invalid ROCm compile database");
  }
  const files = /** @type {unknown[]} */ (entries).map(row => {
    if (!row || typeof row !== "object" || Array.isArray(row)) {throw new Error("Invalid ROCm compile output path");}
    const directory = Reflect.get(row, "directory"), output = Reflect.get(row, "output");
    if (directory !== build || typeof output !== "string" || /[\\\r\n\0]/.test(output) ||
        output.split(/[\\/]/).some((part) => part === "." || part === "..") || !output.endsWith(".o")) {
      throw new Error("Invalid ROCm compile output path");
    }
    const object = path.isAbsolute(output) ? path.normalize(output) : path.resolve(build, output);
    const relative = path.relative(build, object);
    if (!relative || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error("ROCm compile output leaves its build tree");
    }
    return object + ".d";
  });
  if (new Set(files).size !== files.length) {throw new Error("Duplicate ROCm compile dependency outputs");}
  return files;
}

/** @param {string} file @param {string} build @param {ReturnType<typeof rocmHostPlan>["hostCompilers"]} compilers */
function readCompileDatabase(file, build, compilers) {
  const maximum = 32 << 20;
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || fs.realpathSync(file) !== file || stat.size < 2 || stat.size > maximum) {
    throw new Error("Missing or excessive ROCm compile database: " + file);
  }
  const entries = /** @type {unknown} */ (readBuildJson(file, maximum));
  const depfiles = compileDepfiles(build, entries);
  for (const row of /** @type {unknown[]} */ (entries)) {
    if (!row || typeof row !== "object" || Array.isArray(row)) {throw new Error("Invalid ROCm compiler command");}
    const commandText = Reflect.get(row, "command");
    if (typeof commandText !== "string" || commandText.length > (1 << 20)) {throw new Error("Invalid ROCm compiler command");}
    if (/(?:^|\s)-D\s*(?:COMGR_DYN_DLL|ROCR_DYN_DLL)(?:=\S+)?(?:\s|$)/.test(commandText)) {
      throw new Error("ROCm host runtime must not compile dynamic DLL lookup shims");
    }
    const command = commandText.match(/^"([^"\r\n]+)"|^(\S+)/)?.slice(1).find(Boolean);
    if (![compilers.cc, compilers.cxx].includes(command || "")) {throw new Error("ROCm host source was not compiled by the selected Intel compiler");}
  }
  return {entries: /** @type {Array<{directory: string, output: string, command: string}>} */ (entries), depfiles};
}

/** @param {string} cacheFile @param {string} name @param {string} value */
function requireCache(cacheFile, name, value) {
  const cache = readBuildText(cacheFile, 8 << 20), rows = [...cache.matchAll(new RegExp(`^${name}:[^=]+=(.*)$`, "gm"))];
  if (rows.length !== 1 || rows[0]?.[1] !== value) {throw new Error("ROCm CMake selected an unexpected " + name);}
}

/** Build only a new private prefix. The caller enforces the memory, CPU, offline and UID envelope. *
 * @param {unknown} input @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {string} [configFile] */
function buildRocmHostRuntime(input, output, environment = process.env, configFile) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Native ROCm host runtime compilation requires Linuxx64 and MOM_AOT_BUILD_ONLY=1");
  }
  const c = validateConfig(input);
  if (!literalPath(output) || fs.lstatSync(output, {throwIfNoEntry: false}) ||
      !fs.statSync(path.dirname(output), {throwIfNoEntry: false})?.isDirectory()) {
    throw new Error("Select a new ROCm host runtime output; existing evidence is never overwritten");
  }
  realDirectory(path.dirname(output), "output parent");
  const destination = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  if (configFile !== undefined) {regularPath(configFile, "config");}
  const configuredRoots = [...Object.values(c.sources), ...Object.values(c.tools).map(file => path.dirname(file)),
    ...Object.values(c.dependencies).map(file => path.dirname(file)), c.pkgConfigDir, path.dirname(c.hostCompilers.cc),
    path.dirname(c.hostCompilers.cxx), path.dirname(c.comgrReceipt), ...(configFile ? [configFile] : []), repo];
  if (configuredRoots.some(inputPath => buildPathsOverlap(inputPath, destination))) {
    throw new Error("ROCm output or source roots overlap a configured input");
  }
  for (const source of Object.values(c.sources)) {realDirectory(source, "source root");}
  realDirectory(c.pkgConfigDir, "pkg-config directory");
  for (const file of [...Object.values(c.tools), ...Object.values(c.dependencies), ...Object.values(c.hostCompilers)]) {
    regularPath(file, "tool or input", Object.values(c.tools).includes(file) || Object.values(c.hostCompilers).includes(file));
  }
  regularPath(c.comgrReceipt, "COMGR receipt");
  const sysroot = requireLinuxSysroot(sdkHostEnvironment(environment, c.hostCompilers)), libdrm = nativeLibdrmInputs(c.libdrm);
  if (c.dependencies.drm !== libdrm.drm || c.dependencies.drmAmdgpu !== libdrm.drmAmdgpu || c.pkgConfigDir !== libdrm.pkgConfigDir ||
      c.dependencies.elf !== path.join(sysroot.root, "usr/lib/x86_64-linux-gnu", packageNames.elf) ||
      c.dependencies.numa !== path.join(sysroot.root, "usr/lib/x86_64-linux-gnu", packageNames.numa)) {
    throw new Error("ROCm static OS inputs must be the sysroot libelf/libnuma and the source-built libdrm SDK");
  }
  const comgr = comgrSdkInputs(c.comgrReceipt), llvm = comgr.roots.llvm;
  const comgrPackage = path.join(path.dirname(comgr.archive), "lib/cmake/amd_comgr");
  const plan = rocmHostPlan(c, destination, nativeBuildJobs(environment),
    {llvm, comgrPackage, device: comgr.roots.device}, {root: sysroot.root, toolchain: sysroot.toolchain});
  const protectedRoots = [...Object.values(c.sources), ...Object.values(c.tools).map(file => path.dirname(file)),
    ...Object.values(c.dependencies).map(file => path.dirname(file)), c.pkgConfigDir,
    path.dirname(c.hostCompilers.cc), path.dirname(c.hostCompilers.cxx), path.dirname(comgr.receipt),
    comgr.roots.llvm, comgr.roots.lld, comgr.roots.device, plan.imageLlvmPackage, repo].map(file => fs.realpathSync(file));
  if (protectedRoots.some(inputPath => buildPathsOverlap(inputPath, destination)) ||
      Object.values(c.sources).some((root, index, rows) => rows.slice(index + 1).some(other => buildPathsOverlap(root, other)))) {
    throw new Error("ROCm output or source roots overlap a canonical input");
  }
  realDirectory(plan.deviceLibPath, "admitted COMGR device-library bitcode directory");
  for (const file of Object.values(c.dependencies)) {if (!isStaticArchive(file)) {throw new Error("ROCm OS input is not a regular static archive: " + file);}}
  return withPrivateBuildDirectories(path.dirname(destination), temporary =>
    buildRocmHostRuntimeWithTemporaryDirectories(c, destination, environment, configFile, comgr, llvm, plan, temporary,
      {sysroot: linuxSysrootRecord(sysroot), libdrm: {receipt: libdrm.receipt, receiptSha256: libdrm.receiptSha256}}));
}

/** @param {RocmHostConfig} c @param {string} destination @param {NodeJS.ProcessEnv} environment @param {string | undefined} configFile
 * @param {ReturnType<typeof comgrSdkInputs>} comgr @param {string} llvm @param {ReturnType<typeof rocmHostPlan>} plan
 * @param {PrivateBuildDirectories} temporary
 * @param {{sysroot: ReturnType<typeof linuxSysrootRecord>, libdrm: {receipt: string, receiptSha256: string}}} producers */
function buildRocmHostRuntimeWithTemporaryDirectories(c, destination, environment, configFile, comgr, llvm, plan, temporary, producers) {
  const comgrPackage = path.join(path.dirname(comgr.archive), "lib/cmake/amd_comgr");
  const env = hostEnvironment(environment, c, plan, destination, llvm, temporary);
  verifyCompilerEnvironment(plan.hostCompilers, env);
  const python = pythonBuildInput(c.tools.python3, env, c.tools.dpkgQuery);
  /** @type {Record<keyof RocmSourcePaths, string>} */ const sourceTrees = {clr: "", hip: "", rocr: ""};

  for (const name of sourceNames) {
    const revision = pins[name], root = c.sources[name];
    if (query(c.tools.git, ["--no-optional-locks", "-C", root, "rev-parse", revision + "^{commit}"], env, name + " revision").trim() !== revision) {
      throw new Error("ROCm source is not at its pinned " + name + " commit");
    }
    const tree = query(c.tools.git, ["-c", "core.attributesFile=/dev/null", "--no-optional-locks", "-C", root,
      "ls-tree", "-rz", "--full-tree", revision], env, name + " source tree", 64 << 20);
    if (tree.split("\0").some(entry => /^(?:120000 blob|160000 commit) /.test(entry))) {
      throw new Error("Pinned ROCm sources must not contain symlinks or unpinned submodules: " + name);
    }
    sourceTrees[name] = tree;
  }
  const llvmVersion = query(path.join(llvm, "bin/llvm-config"), ["--version"], env, "LLVM version").trim();
  if (llvmVersion !== "21.1.8") {throw new Error("ROCm host runtime must use the matching COMGR LLVM 21.1.8 tools");}
  const rocmClangAlias = path.join(llvm, "bin/clang"), rocmClang = path.join(llvm, "bin/clang-21"), rocmObjcopy = c.tools.llvmObjcopy;
  const clangAlias = fs.lstatSync(rocmClangAlias, {throwIfNoEntry: false});
  if (!clangAlias?.isSymbolicLink() || fs.readlinkSync(rocmClangAlias) !== "clang-21" || fs.realpathSync(rocmClangAlias) !== rocmClang) {
    throw new Error("COMGR LLVM clang alias must resolve only to the adjacent clang-21 binary");
  }
  regularPath(rocmClang, "ROCm assembler", true); regularPath(rocmObjcopy, "ROCm objcopy", true);
  const comgrPackageFiles = ["amd_comgr-config.cmake", "amd_comgr-config-version.cmake", "amd_comgr-targets.cmake"].map(name => path.join(comgrPackage, name));
  for (const file of comgrPackageFiles) {regularPath(file, "COMGR package input");}
  realDirectory(plan.imageLlvmRoot, "image LLVM root");
  realDirectory(plan.imageLlvmPackage, "image LLVM CMake package input");
  const hipPchTools = Object.fromEntries(["clang", "llvm-mc"].map(name => {
    const file = path.join(plan.imageLlvmRoot, "bin", name), realpath = fs.realpathSync(file), relative = path.relative(plan.imageLlvmRoot, realpath);
    regularPath(realpath, "HIP PCH tool", true);
    if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) {
      throw new Error("HIP PCH tool escaped the selected image LLVM root: " + name);
    }
    const version = query(file, ["--version"], env, "HIP PCH LLVM tool version").trim();
    if (!/\b21\.1\.8\b/.test(version)) {throw new Error("HIP PCH tools must match LLVM 21.1.8: " + name);}
    const stat = fs.lstatSync(file);
    return [name, {path: file, realpath, linkTarget: stat.isSymbolicLink() ? fs.readlinkSync(file) : null,
      sha256: fileHash(realpath), version}];
  }));
  const packageTrees = ["llvm", "clang", "lld", "amd_comgr"].map(name => {
    const root = name === "amd_comgr" ? comgrPackage : path.join(llvm, "lib/cmake", name);
    realDirectory(root, "CMake package input"); return {root, files: sdkSourceFiles(root)};
  });
  packageTrees.push({root: plan.imageLlvmPackage, files: sdkSourceFiles(plan.imageLlvmPackage)});
  const osOrigins = /** @type {Record<keyof RocmDependencyPaths, ReturnType<typeof staticInputOrigin>>} */ (
    Object.fromEntries(dependencyFileNames.map(name => [name, staticInputOrigin(name, c.dependencies[name])])));
  const glHeaders = glHeaderInputs();
  const xxdOrigin = packageOrigin(c.tools.xxd, env, c.tools.dpkgQuery, xxdPin.package);
  if (xxdOrigin.version !== xxdPin.packageVersion) {throw new Error("ROCm xxd package version changed");}
  const pkgFiles = ["libdrm.pc", "libdrm_amdgpu.pc"].map(name => path.join(c.pkgConfigDir, name));
  for (const file of pkgFiles) {regularPath(file, "libdrm pkg-config input");}
  const pkgDirectories = /** @type {Record<string, string>} */ (Object.fromEntries(["libdrm", "libdrm_amdgpu"].map(name => [name,
    query(c.tools.pkgConfig, ["--variable=pcfiledir", name], env, name + " pkg-config directory").trim()])));
  if (Object.values(pkgDirectories).some(directory => directory !== c.pkgConfigDir)) {throw new Error("pkg-config selected an unadmitted libdrm package directory");}
  /** @type {Array<[string, keyof RocmDependencyPaths]>} */ const pkgLibraries = [["libdrm", "drm"], ["libdrm_amdgpu", "drmAmdgpu"]];
  for (const [name, archiveKey] of pkgLibraries) {
    const directory = query(c.tools.pkgConfig, ["--variable=libdir", name], env, name + " library directory").trim();
    if (!path.isAbsolute(directory) || path.resolve(directory, packageNames[archiveKey]) !== c.dependencies[archiveKey]) {
      throw new Error("pkg-config library directory does not identify the admitted OS archive: " + name);
    }
  }
  /** @type {Record<string, string>} */ const toolVersions = {};
  for (const name of toolNames) {
    const file = c.tools[name];
    toolVersions[file] = query(file, name === "xxd" ? ["-v"] : ["--version"], env, name + " version", 1 << 20, name === "xxd").trim();
  }
  /** @type {Array<[string, string]>} */ const rocmTools = [["rocmClang", rocmClang], ["rocmObjcopy", rocmObjcopy]];
  for (const [name, file] of rocmTools) {
    toolVersions[file] = query(file, ["--version"], env, name + " version").trim();
  }
  if (![rocmClang, rocmObjcopy].every(file => /(?:clang version|LLVM version)\s+21\.1\.8\b/.test(toolVersions[file] || ""))) {
    throw new Error("ROCm assembler and objcopy must match the admitted LLVM 21.1.8 toolchain");
  }
  if (toolVersions[c.tools.python3] !== "Python " + pythonPin.version) {throw new Error("ROCm host runtime must use the pinned Python interpreter");}
  if (toolVersions[c.tools.xxd] !== xxdPin.version) {throw new Error("ROCm host runtime must use the pinned xxd tool");}
  const compiler = hostCompilerEvidence(plan.hostCompilers, (command, args, name) => query(command, args, env, name));
  if (!Object.values(compiler.versions).every(version => /Intel\(R\).*2026\.2/.test(version))) {
    throw new Error("ROCm host runtime requires the selected Intel oneAPI 2026.2 C and C++ compilers");
  }
  const comgrHeaderSha256 = comgr.files[comgr.header];
  if (!comgrHeaderSha256 || fileHash(comgr.header) !== comgrHeaderSha256) {throw new Error("Selected COMGR header is missing from its verified input receipt");}
  const comgrHeaderOverlay = {source: comgr.header, destination: plan.comgrHeader, includeRoot: plan.comgrHeaderInclude, sha256: comgrHeaderSha256};
  const recipe = sourceInputs(repo, [], recipeFiles), inputFiles = new Set([
    ...(configFile ? [configFile] : []), ...Object.keys(comgr.files), ...comgrPackageFiles,
    ...pkgFiles, ...Object.values(c.tools), python.interpreter.path, python.parser.packagePath, python.parser.modulePath,
    python.ply.init.path, python.ply.lex.path, ...Object.values(plan.hostCompilers), rocmClangAlias, rocmClang, rocmObjcopy,
    ...Object.values(hipPchTools).map(tool => tool.realpath), ...Object.keys(glHeaders),
    ...Object.values(c.dependencies)]);
  const sdkInputs = {...buildFileInputs([...inputFiles]), ...compiler.inputs,
    ...Object.fromEntries(packageTrees.flatMap(tree => Object.entries(tree.files).map(([name, hash]) => [path.join(tree.root, name), hash])))};
  if (fs.lstatSync(destination, {throwIfNoEntry: false})) {
    throw new Error("Select a new ROCm host runtime output; existing evidence is never overwritten");
  }
  fs.mkdirSync(destination); fs.mkdirSync(path.join(destination, "source"));
  fs.mkdirSync(path.join(destination, "source/clr")); fs.mkdirSync(path.join(destination, "source/hip")); fs.mkdirSync(path.join(destination, "source/rocr"));
  fs.mkdirSync(path.join(destination, "private-home")); fs.mkdirSync(path.join(destination, "tmp"));
  env["HOME"] = path.join(destination, "private-home"); env["TMPDIR"] = path.join(destination, "tmp");
  temporary.cleanup();
  // COMGR's build export exposes a flat header, unlike its nested installed layout; stage only
  // the receipt-authenticated header privately so CLR's amd_comgr/ include resolves unchanged.
  fs.mkdirSync(path.dirname(plan.comgrHeader), {recursive: true});
  fs.copyFileSync(comgr.header, plan.comgrHeader, fs.constants.COPYFILE_EXCL);
  regularPath(plan.comgrHeader, "private COMGR install-layout header");
  if (fileHash(plan.comgrHeader) !== comgrHeaderOverlay.sha256) {throw new Error("Private COMGR header differs from its authenticated source");}
  for (const [file, sha256] of Object.entries(glHeaders)) {
    const copy = path.join(plan.glHeaderInclude, path.relative("/usr/include", file));
    fs.mkdirSync(path.dirname(copy), {recursive: true});
    fs.copyFileSync(file, copy, fs.constants.COPYFILE_EXCL);
    if (fileHash(copy) !== sha256) {throw new Error("Private GL header overlay differs from its host source: " + file);}
  }
  /** @param {string[]} command @param {string} name */
  function stage(command, name) {console.log("[rocm-host] " + name); sdkStage(command, path.join(destination, name + ".log"), env);}
  /** @type {{path: string, sourceSha256: string, patchedSha256: string, preprocessorInput: string} | undefined} */
  let hipPchSourcePatch;
  /** @type {{path: string, sourceSha256: string, patchedSha256: string, option: string, cacheVariable: string} | undefined} */
  let rocrDeviceLibPatch;
  for (const name of sourceNames) {
    const revision = pins[name], sourceArchive = path.join(destination, "source-" + name + ".tar"), root = c.sources[name];
    stage([c.tools.git, "-c", "core.attributesFile=/dev/null", "--no-optional-locks", "-C", root, "archive", "--format=tar",
      "--output=" + sourceArchive, revision], name + "-archive");
    if (fs.statSync(sourceArchive).size > (512 << 20)) {throw new Error("Pinned ROCm source archive exceeds bounds: " + name);}
    stage([c.tools.tar, "--extract", "--file", sourceArchive, "--directory", path.join(destination, "source", name), "--no-same-owner", "--no-same-permissions"], name + "-extract");
    const files = sdkSourceFiles(path.join(destination, "source", name));
    verifySdkExport(path.join(destination, "source", name), files, sourceTrees[name] || "");
    if (name === "clr") {hipPchSourcePatch = patchExportedHipPchScript(path.join(destination, "source", name));}
    if (name === "rocr") {rocrDeviceLibPatch = patchExportedRocrDeviceLibPath(path.join(destination, "source", name));}
  }
  if (!hipPchSourcePatch) {throw new Error("Pinned CLR export did not apply the HIP PCH path mapping");}
  if (!rocrDeviceLibPatch) {throw new Error("Pinned ROCr export did not bind image kernels to admitted device libraries");}
  const before = {pins, sourceRoots: c.sources, sourceTrees: Object.fromEntries(Object.entries(pins).map(([name, revision]) => [name, revision])),
    comgrReceipt: c.comgrReceipt, comgr: {archive: comgr.archive, llvmVersion, roots: comgr.roots}, rocrDeviceLibPath: plan.deviceLibPath,
    rocrDeviceLibPatch, comgrHeaderOverlay, glHeaderInclude: plan.glHeaderInclude, ...producers,
    hipPchTools,
    rocmTools: {clang: {alias: rocmClangAlias, target: fs.readlinkSync(rocmClangAlias), path: rocmClang, sha256: fileHash(rocmClang)},
      objcopy: {path: rocmObjcopy, sha256: fileHash(rocmObjcopy)}}, hostCompilers: plan.hostCompilers,
    hostCompilerInputs: compiler.inputs, compilerVersions: compiler.versions, toolVersions, toolOrigins: {xxd: xxdOrigin}, python, osOrigins,
    pkgConfigDir: c.pkgConfigDir, pkgFiles, packageTrees, sdkInputs, recipe, commands: plan.commands};
  /** @type {Record<keyof RocmSourcePaths, ArchiveEvidence>} */ const sourceArchives = {clr: {path: "", bytes: 0, sha256: ""},
    hip: {path: "", bytes: 0, sha256: ""}, rocr: {path: "", bytes: 0, sha256: ""}};
  for (const name of sourceNames) {
    const file = path.join(destination, "source-" + name + ".tar");
    sourceArchives[name] = {path: file, bytes: fs.statSync(file).size, sha256: fileHash(file)};
  }
  const sourceState = sourceInputs(destination, ["source/clr", "source/hip", "source/rocr"], []);
  /** @param {string} name @param {string} command @param {string[]} args */
  const queryLog = (name, command, args) => fs.writeFileSync(path.join(destination, name + ".log"), query(command, args, env, name), {flag: "wx"});
  /** @type {Array<[string, string]>} */ const identityTools = [["cmake", c.tools.cmake], ["ninja", c.tools.ninja], ["pkg-config", c.tools.pkgConfig],
    ["dpkg-query", c.tools.dpkgQuery], ["python3", c.tools.python3], ["llvm-config", path.join(llvm, "bin/llvm-config")]];
  for (const [name, command] of identityTools) {
    queryLog(name + "-identity", command, ["--version"]);
  }
  fs.writeFileSync(path.join(destination, "before.json"), JSON.stringify({startedAt: Date.now(), before, sourceState, sourceArchives,
    packageDirectories: pkgDirectories}, null, 2) + "\n", {flag: "wx"});

  const startedAt = Date.now();
  /** @type {ConfiguredBuild[]} */ const configured = [];
  /** @type {Array<{name: keyof typeof plan.build, configure: string[], build: string[]}>} */ const buildStages = [
    {name: "rocr", configure: plan.commands.rocrConfigure, build: plan.commands.rocrBuild},
    {name: "clr", configure: plan.commands.clrConfigure, build: plan.commands.clrBuild}];
  for (const {name, configure, build} of buildStages) {
    stage(configure, name + "-configure");
    const buildPath = plan.build[name];
    const cacheFile = path.join(buildPath, "CMakeCache.txt");
    requireCache(cacheFile, "CMAKE_C_COMPILER", plan.hostCompilers.cc);
    requireCache(cacheFile, "CMAKE_CXX_COMPILER", plan.hostCompilers.cxx);
    requireCache(cacheFile, "Python3_EXECUTABLE", c.tools.python3);
    requireCache(cacheFile, "CMAKE_AR", c.tools.ar);
    requireCache(cacheFile, "CMAKE_RANLIB", c.tools.ranlib);
    requireCache(cacheFile, "CMAKE_MAKE_PROGRAM", c.tools.ninja);
    requireCache(cacheFile, "CMAKE_LIBRARY_ARCHITECTURE", "x86_64-linux-gnu");
    requireCache(cacheFile, "GIT_EXECUTABLE", c.tools.git); requireCache(cacheFile, "PERL_EXECUTABLE", c.tools.perl);
    requireCache(cacheFile, "CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    requireCache(cacheFile, "BUILD_SHARED_LIBS", "OFF");
    requireCache(cacheFile, "CMAKE_HOME_DIRECTORY", path.join(destination, "source", name));
    if (name === "rocr") {
      requireCache(cacheFile, "HSA_ASSEMBLER", rocmClang); requireCache(cacheFile, "HSA_OBJCOPY", rocmObjcopy);
      requireCache(cacheFile, "LLVM_DIR", plan.imageLlvmPackage); requireCache(cacheFile, "Clang_DIR", plan.clangPackage);
      requireCache(cacheFile, "MOM_ROCM_DEVICE_LIBS_PATH", plan.deviceLibPath);
      requireCache(cacheFile, "LIBELF_LIBRARIES", c.dependencies.elf); requireCache(cacheFile, "NUMA", c.dependencies.numa);
    } else {
      requireCache(cacheFile, "HIP_COMMON_DIR", path.join(destination, "source/hip"));
      requireCache(cacheFile, "HIP_PLATFORM", "amd"); requireCache(cacheFile, "amd_comgr_DIR", comgrPackage);
      requireCache(cacheFile, "CMAKE_CXX_FLAGS", plan.clrCxxFlags);
      requireCache(cacheFile, "LLVM_DIR", path.join(llvm, "lib/cmake/llvm")); requireCache(cacheFile, "Clang_DIR", plan.clangPackage);
      requireCache(cacheFile, "hsa-runtime64_DIR", path.join(plan.rocrInstall, "lib/cmake/hsa-runtime64"));
      requireCache(cacheFile, "NUMA_LIBRARIES", c.dependencies.numa); requireCache(cacheFile, "LIBELF_LIBRARIES", c.dependencies.elf);
    }
    const database = readCompileDatabase(path.join(buildPath, "compile_commands.json"), buildPath, plan.hostCompilers);
    configured.push({name, cache: cacheFile, database: path.join(buildPath, "compile_commands.json"), depfiles: database.depfiles,
      commandCount: database.entries.length});
    stage(build, name + "-build");
    if (name === "rocr") {
      stage(plan.commands.rocrInstallBinary, "rocr-install-binary");
      stage(plan.commands.rocrInstallDev, "rocr-install-dev");
    }
  }
  /** @type {Record<keyof typeof archiveNames, string>} */ const builtPaths = {hsa: path.join(plan.build.rocr, "runtime/hsa-runtime", archiveNames.hsa),
    hsakmt: path.join(plan.build.rocr, "libhsakmt", archiveNames.hsakmt),
    amdhip64: path.join(plan.build.clr, "hipamd/lib", archiveNames.amdhip64),
    hiprtc: path.join(plan.build.clr, "hipamd/lib", archiveNames.hiprtc)};
  const archiveBuildBefore = /** @type {Record<keyof typeof archiveNames, ArchiveEvidence>} */ (Object.fromEntries(runtimeArchiveNames.map(name => {
    const file = builtPaths[name];
    if (!isStaticArchive(file)) {throw new Error("ROCm upstream build did not produce a regular static archive: " + name);}
    return [name, {path: file, bytes: fs.statSync(file).size, sha256: fileHash(file)}];
  })));
  const hsakmtInstallArchive = path.join(plan.build.rocr, "libhsakmt", "archive", hsakmtInstallArchiveName);
  if (!isStaticArchive(hsakmtInstallArchive)) {throw new Error("ROCr upstream build did not produce its component-install archive: hsakmt");}
  const hsakmtInstallArchiveBefore = {path: hsakmtInstallArchive, bytes: fs.statSync(hsakmtInstallArchive).size, sha256: fileHash(hsakmtInstallArchive)};
  stage(plan.commands.clrInstallBinary, "clr-install-binary");
  stage(plan.commands.clrInstallDev, "clr-install-dev");

  const sdkLib = path.join(plan.sdk, "lib"), hsaLib = path.join(plan.rocrInstall, "lib");
  /** @param {string} from @param {string} to */
  const copy = (from, to) => {regularPath(from, "staged archive"); fs.copyFileSync(from, to, fs.constants.COPYFILE_EXCL);};
  fs.mkdirSync(sdkLib, {recursive: true});
  /** @type {Array<[string, string]>} */ const stagedHsaArchives = [[archiveNames.hsa, path.join(hsaLib, archiveNames.hsa)],
    [archiveNames.hsakmt, path.join(hsaLib, archiveNames.hsakmt)]];
  for (const [name, sourceFile] of stagedHsaArchives) {
    copy(sourceFile, path.join(sdkLib, name));
  }
  const aliasFile = path.join(hsaLib, "cmake/hsa-runtime64/hsacore.so.link");
  regularPath(aliasFile, "HSA alias script"); fs.mkdirSync(path.join(sdkLib, "cmake/hsa-runtime64"), {recursive: true});
  fs.copyFileSync(aliasFile, path.join(sdkLib, "cmake/hsa-runtime64/hsacore.so.link"), fs.constants.COPYFILE_EXCL);
  const aliases = readBuildText(path.join(sdkLib, "cmake/hsa-runtime64/hsacore.so.link"), 1 << 20);
  if (!/^(?:hsa_[a-z0-9_]+ = hsa_[a-z0-9_]+;\r?\n)+$/.test(aliases)) {throw new Error("ROCm HSA alias script is malformed");}
  /** @type {Record<keyof RocmDependencyPaths, ReturnType<typeof staticInputOrigin> & {staged: string, stagedSha256: string}>} */
  const stagedDependencies = /** @type {Record<keyof RocmDependencyPaths, ReturnType<typeof staticInputOrigin> & {staged: string, stagedSha256: string}>} */ ({});
  for (const name of dependencyFileNames) {
    const sourceFile = c.dependencies[name], staged = path.join(sdkLib, packageNames[name]); copy(sourceFile, staged);
    const sha256 = fileHash(staged);
    if (sha256 !== osOrigins[name].sha256) {throw new Error("Staged OS archive differs from its admitted input: " + name);}
    stagedDependencies[name] = {...osOrigins[name], staged, stagedSha256: sha256};
  }
  const produced = /** @type {Record<keyof typeof archiveNames, ArchiveEvidence>} */ (Object.fromEntries(runtimeArchiveNames.map(name => {
    const file = archiveNames[name];
    const archive = path.join(sdkLib, file);
    if (!isStaticArchive(archive)) {throw new Error("ROCm output is not a regular static archive: " + archive);}
    const built = archiveBuildBefore[name];
    if (built && fileHash(archive) !== built.sha256) {throw new Error("ROCm staged archive differs from its upstream build output: " + name);}
    return [name, {path: archive, bytes: fs.statSync(archive).size, sha256: fileHash(archive)}];
  })));
  const missingHeaders = ["hip_runtime.h", "hip_runtime_api.h", "hiprtc.h", "hip_version.h"].filter(name =>
    !fs.existsSync(path.join(plan.sdk, "include/hip", name)));
  if (missingHeaders.length) {throw new Error("ROCm staged prefix is missing public HIP headers: " + missingHeaders.join(", "));}
  const consumerAdmission = rocmInputs(plan.sdk, c.comgrReceipt);

  const depfiles = configured.flatMap(row => row.depfiles);
  for (const file of depfiles) {regularPath(file, "compiler depfile");}
  const dependencies = completeBuildInputs(depfiles, repo, destination, startedAt, sdkInputs);
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeFiles), "ROCm host runtime recipe");
  verifySourceInputs(sourceState, sourceInputs(destination, ["source/clr", "source/hip", "source/rocr"], []), "Pinned ROCm source snapshots");
  for (const [file, hash] of Object.entries(sdkInputs)) {if (fileHash(file) !== hash) {throw new Error("ROCm host runtime input changed during compilation: " + file);}}
  for (const tool of Object.values(hipPchTools)) {
    const stat = fs.lstatSync(tool.path, {throwIfNoEntry: false});
    if (fs.realpathSync(tool.path) !== tool.realpath || (tool.linkTarget !== null &&
        (!stat?.isSymbolicLink() || fs.readlinkSync(tool.path) !== tool.linkTarget)) || fileHash(tool.realpath) !== tool.sha256) {
      throw new Error("HIP PCH LLVM tool changed during compilation: " + tool.path);
    }
  }
  regularPath(plan.comgrHeader, "private COMGR install-layout header");
  if (fileHash(plan.comgrHeader) !== comgrHeaderOverlay.sha256) {throw new Error("Private COMGR header changed during compilation");}
  if (fs.readlinkSync(rocmClangAlias) !== "clang-21" || fs.realpathSync(rocmClangAlias) !== rocmClang) {
    throw new Error("COMGR LLVM clang alias changed during compilation");
  }
  for (const [name, record] of Object.entries(sourceArchives)) {
    if (fileHash(record.path) !== record.sha256) {throw new Error("ROCm source archive changed during compilation: " + name);}
  }
  for (const [name, record] of Object.entries(archiveBuildBefore)) {
    if (fileHash(record.path) !== record.sha256) {throw new Error("ROCm built archive changed after its stage guard: " + name);}
  }
  if (fileHash(hsakmtInstallArchiveBefore.path) !== hsakmtInstallArchiveBefore.sha256) {
    throw new Error("ROCr component-install archive changed after its stage guard: hsakmt");
  }
  for (const [name, record] of Object.entries(produced)) {
    if (fileHash(record.path) !== record.sha256) {throw new Error("ROCm archive changed after staging: " + name);}
  }
  for (const [name, record] of Object.entries(stagedDependencies)) {
    if (fileHash(record.staged) !== record.stagedSha256) {throw new Error("Staged OS archive changed after copying: " + name);}
  }
  const result = {version: 1, pins, sources: sourceState, sourceArchives, hostCompilers: plan.hostCompilers,
    hostCompilerInputs: compiler.inputs, compilerVersions: compiler.versions, toolVersions, toolOrigins: {xxd: xxdOrigin}, osOrigins,
    python, hipPchTools, hipPchSourcePatch, rocrDeviceLibPath: plan.deviceLibPath, rocrDeviceLibPatch, comgrHeaderOverlay,
    comgr: {receipt: c.comgrReceipt, archive: comgr.archive, llvmVersion, roots: comgr.roots},
    sdkInputs, dependencies, configured, archivesBefore: archiveBuildBefore, archivesAfter: produced,
    installerOnlyArchives: {hsakmt: hsakmtInstallArchiveBefore}, stagedDependencies, consumerAdmission,
    prefix: plan.sdk, recipe, ...producers, glHeaderInclude: plan.glHeaderInclude, buildOnly: true, executed: false, publicationQualified: false,
    scope: "Original pinned ROCr and CLR/HIP static host runtime archives built with Intel oneAPI 2026.2. No GPU execution, runtime or release qualification."};
  fs.writeFileSync(path.join(destination, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[rocm-host] Built original ROCr and CLR/HIP static archives; NOT executed or publication-qualified.");
  return result;
}

if (require.main === module) {
  try {
    const [configPath, output, extra] = process.argv.slice(2);
    if (!configPath || !output || extra || !absolutePath(configPath)) {throw new Error("Usage: build-native-rocm-host-runtime.js /config.json /new-output");}
    regularPath(configPath, "config");
    buildRocmHostRuntime(readBuildJson(configPath), output, process.env, configPath);
  } catch (error) {console.error(error instanceof Error ? error.message : "Native ROCm host runtime build failed"); process.exitCode = 1;}
}

module.exports = {pins, archiveNames, hsakmtInstallArchiveName, xxdPin, packageOwners, staticInputOrigin, validateConfig, rocmHostPlan, hostEnvironment, withPrivateBuildDirectories, packageOrigin, parsePackageOrigin, compileDepfiles, readCompileDatabase,
  useStableHipPchStdin, patchExportedHipPchScript, useExplicitRocrDeviceLibPath, patchExportedRocrDeviceLibPath,
  buildRocmHostRuntime};
