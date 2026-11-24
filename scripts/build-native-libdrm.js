"use strict";

// Build the pinned libdrm/libdrm_amdgpu static archives that the ROCm host runtime links. Distro
// archives are compiled against the build host's glibc, and focal's libdrm-dev ships no static
// archive and predates amdgpu_device_get_fd, so the original upstream release is built against
// the glibc 2.31 sysroot instead.
const fs = require("node:fs"), path = require("node:path");
const {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, sourceInputs, verifySourceInputs, readBuildJsonSnapshot, nativeBuildJobs} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {runLoggedSdkStage} = require("./native-sdk-stage");
const {isStaticArchive} = require("./native-audit");
const {linuxSysrootRelease, requireLinuxSysroot, linuxSysrootRecord, verifyLinuxSysrootRecord} = require("./native-linux-sysroot");

const repo = path.resolve(__dirname, "..");
const libdrmPin = Object.freeze({version: "2.4.125", fileName: "libdrm-2.4.125.tar.xz", url: linuxSysrootRelease + "libdrm-2.4.125.tar.xz",
  sha256: "d4bae92797a50f81a93524762e0410a49cd84cfa0f997795bc0172ac8fb1d96a", bytes: 485592});
const recipeLeaves = ["scripts/build-native-libdrm.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
  "scripts/native-audit.js", "scripts/native-linux-sysroot.js", "scripts/native-sdk-stage.js"];
const toolNames = Object.freeze(["cc", "meson", "ninja", "tar"]);
// The runtime ASIC table stays at the distribution path the AMD driver stack installs.
const amdgpuIds = "/usr/share/libdrm/amdgpu.ids";
// Only libdrm core and amdgpu; every other driver, test program and udev hook stays disabled.
const mesonOptions = Object.freeze(["--buildtype=release", "--libdir=lib", "--includedir=include", "--datadir=/usr/share",
  "-Ddefault_library=static", "-Damdgpu=enabled", "-Dintel=disabled", "-Dradeon=disabled", "-Dnouveau=disabled",
  "-Dvmwgfx=disabled", "-Domap=disabled", "-Dexynos=disabled", "-Dfreedreno=disabled", "-Dtegra=disabled", "-Dvc4=disabled",
  "-Detnaviv=disabled", "-Dcairo-tests=disabled", "-Dman-pages=disabled", "-Dvalgrind=disabled", "-Dfreedreno-kgsl=false",
  "-Dinstall-test-programs=false", "-Dudev=false", "-Dtests=false"]);

/** @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.resolve(value) === value &&
  !/[\s;"'$`\\]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);

/** @param {unknown} value @returns {{cc: string, meson: string, ninja: string, tar: string}} */
function validateTools(value) {
  const tools = /** @type {{cc: string, meson: string, ninja: string, tar: string}} */ (value);
  if (!hasExactBuildFields(tools, toolNames) ||
      !Object.values(tools).every(literal) || path.basename(tools.cc) !== "icx") {
    throw new Error("libdrm SDK needs explicit absolute icx, meson, ninja and tar tools");
  }
  return tools;
}

/** @param {string} output @param {string} archive @param {unknown} tools @param {number} jobs */
function libdrmSdkPlan(output, archive, tools, jobs) {
  const selected = validateTools(tools);
  if (!literal(output) || !literal(archive) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("libdrm SDK needs literal absolute paths and jobs in 1..16");
  }
  const source = path.join(output, "source"), build = path.join(output, "build"), sdk = path.join(output, "sdk");
  return {source, build, sdk, extract: [selected.tar, "-xJf", archive, "-C", source, "--strip-components=1", "--no-same-owner", "--no-same-permissions"],
    setup: [selected.meson, "setup", build, source, "--prefix=" + sdk, ...mesonOptions],
    compile: [selected.ninja, "-C", build, "-j", String(jobs)],
    install: [selected.meson, "install", "-C", build, "--no-rebuild", "--tags", "devel"]};
}

/** Inventory the installed static SDK; links and special files are rejected. @param {string} sdk */
function sdkFiles(sdk) {
  /** @type {Record<string, string>} */ const files = {};
  const visit = (/** @type {string} */ directory) => {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const file = path.join(directory, entry.name);
      if (Object.keys(files).length > 256 || entry.isSymbolicLink()) {throw new Error("libdrm SDK tree is unbounded or linked");}
      if (entry.isDirectory()) {visit(file);} else if (entry.isFile()) {files[path.relative(sdk, file)] = fileHash(file);} else {
        throw new Error("libdrm SDK member is not a regular file: " + file);
      }
    }
  };
  visit(sdk);
  for (const name of ["lib/libdrm.a", "lib/libdrm_amdgpu.a", "lib/pkgconfig/libdrm.pc", "lib/pkgconfig/libdrm_amdgpu.pc",
    "include/xf86drm.h", "include/libdrm/amdgpu.h", "include/libdrm/amdgpu_drm.h"]) {
    if (!files[name]) {throw new Error("libdrm SDK is missing " + name);}
  }
  return files;
}

/** Build the pinned upstream archive into a new output with explicit tools and the selected sysroot.
 * @param {string} archive @param {string} output @param {unknown} tools @param {NodeJS.ProcessEnv} [environment] */
function buildNativeLibdrm(archive, output, tools, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("libdrm SDK production requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  const selected = validateTools(tools), jobs = nativeBuildJobs(environment), plan = libdrmSdkPlan(output, archive, selected, jobs);
  if (fs.lstatSync(output, {throwIfNoEntry: false}) || fs.realpathSync(path.dirname(output)) !== path.dirname(output)) {
    throw new Error("libdrm SDK output must be a new canonical directory");
  }
  if (fs.statSync(archive).size !== libdrmPin.bytes || fileHash(archive) !== libdrmPin.sha256) {throw new Error("libdrm source archive differs from its pin");}
  const env = nativeSdkEnvironment(environment), sysroot = requireLinuxSysroot(env);
  for (const key of ["MAKEFLAGS", "MESONINTROSPECT", "LDFLAGS", "CFLAGS", "CPPFLAGS"]) {delete env[key];}
  Object.assign(env, {CC: selected.cc, PATH: [...new Set([path.dirname(selected.cc), path.dirname(selected.meson), path.dirname(selected.ninja), "/usr/bin", "/bin"])].join(":"),
    LC_ALL: "C"});
  const toolHashes = Object.fromEntries(Object.entries(selected).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  const recipe = sourceInputs(repo, [], recipeLeaves);
  fs.mkdirSync(output, {mode: 0o700});
  for (const directory of [plan.source, path.join(output, "logs")]) {fs.mkdirSync(directory);}
  for (const stage of /** @type {const} */ (["extract", "setup", "compile", "install"])) {
    runLoggedSdkStage(plan[stage], output, env, output, stage, "libdrm SDK", 20 * 60 * 1000);
  }
  const files = sdkFiles(plan.sdk);
  for (const name of ["libdrm.a", "libdrm_amdgpu.a"]) {
    if (!isStaticArchive(path.join(plan.sdk, "lib", name))) {throw new Error("libdrm output is not a static archive: " + name);}
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "libdrm SDK producer");
  if (fileHash(archive) !== libdrmPin.sha256 || Object.values(toolHashes).some(row => fileHash(row.path) !== row.sha256)) {
    throw new Error("libdrm source archive or tools changed during the build");
  }
  const result = {version: 1, libdrmVersion: libdrmPin.version, archive: {path: archive, sha256: libdrmPin.sha256, bytes: libdrmPin.bytes},
    tools: toolHashes, plan, amdgpuIds, sysroot: linuxSysrootRecord(sysroot), recipe, files, buildOnly: true, executed: false,
    publicationQualified: false};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built static libdrm " + libdrmPin.version + " (core and amdgpu) against the glibc sysroot; nothing executed.");
  return nativeLibdrmInputs(output);
}

/** Admit the exact installed static libdrm SDK. @param {string} root */
function nativeLibdrmInputs(root) {
  if (!literal(root) || fs.realpathSync(root) !== root) {throw new Error("Select the canonical libdrm SDK root");}
  const receipt = path.join(root, "result.json"), {value: record, sha256: receiptSha256} = readBuildJsonSnapshot(receipt), sdk = path.join(root, "sdk");
  if (!record || record.version !== 1 || record.libdrmVersion !== libdrmPin.version || record.archive?.sha256 !== libdrmPin.sha256 ||
      record.amdgpuIds !== amdgpuIds || record.buildOnly !== true || record.publicationQualified !== false ||
      !isDeepStrictEqual(record.files, sdkFiles(sdk))) {
    throw new Error("libdrm SDK receipt or installed files differ; rebuild it in a new output");
  }
  verifyLinuxSysrootRecord(record.sysroot, "libdrm SDK");
  if (fileHash(receipt) !== receiptSha256) {throw new Error("libdrm SDK receipt changed during admission");}
  return {root, sdk, drm: path.join(sdk, "lib", "libdrm.a"), drmAmdgpu: path.join(sdk, "lib", "libdrm_amdgpu.a"),
    pkgConfigDir: path.join(sdk, "lib", "pkgconfig"), include: path.join(sdk, "include"), files: record.files, receipt, receiptSha256};
}

module.exports = {libdrmPin, toolNames, libdrmSdkPlan, buildNativeLibdrm, nativeLibdrmInputs, libdrmBuildRecipe: () => sourceInputs(repo, [], recipeLeaves)};

if (require.main === module) {
  try {
    const [archive, output, cc, meson, ninja, tar, extra] = process.argv.slice(2);
    if (!archive || !output || !cc || !meson || !ninja || !tar || extra) {
      throw new Error("Usage: MOM_AOT_BUILD_ONLY=1 MOM_LINUX_SYSROOT=/sysroot node scripts/build-native-libdrm.js /libdrm.tar.xz /new/sdk /icx /meson /ninja /tar");
    }
    buildNativeLibdrm(archive, output, {cc, meson, ninja, tar});
  } catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}
