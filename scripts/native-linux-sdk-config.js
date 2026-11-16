"use strict";

const fs = require("node:fs");
const path = require("node:path");
const childProcess = require("node:child_process");

const repo = path.resolve(__dirname, "..");

/** @typedef {{llvmSdk: string, images: {handoffRoot: string}, acppSdk: {sdkRoot: string}, acpp: {preparedCheckout: string},
 * dpcpp: {dpcppRoot: string, cudaRoot: string}, jit: {staticComponent: string}, hermes: {sdkRoot: string},
 * rocm: {sdkRoot: string, runtimeOutput: string}, openmp: {sdkRoot: string}, zstd: {sdkRoot: string}, opensslSdk: string, ioSdk: string}} LinuxNativeSdkConfigState */

/** Query the pinned Intel driver's own LLD and preserve its ld.lld invocation alias.
 * @param {string} cxx
 * @param {(compiler: string, args: string[]) => {error?: Error, signal?: string | null, status: number | null, stdout: string, stderr?: string}} [query] */
function linuxNativeSdkLldPath(cxx, query = (compiler, args) => childProcess.spawnSync(compiler, args,
  {encoding: "utf8", timeout: 10000, maxBuffer: 16 * 1024})) {
  if (!path.isAbsolute(cxx) || path.resolve(cxx) !== cxx || path.basename(cxx) !== "icpx" ||
      path.basename(path.dirname(cxx)) !== "bin") {throw new Error("Pinned Intel CXX path is not a canonical bin/icpx executable");}
  const compilerRoot = path.dirname(path.dirname(cxx)), result = query(cxx, ["-print-prog-name=ld.lld"]), output = result.stdout;
  if (result.error || result.signal || result.status !== 0 || result.stderr || typeof output !== "string") {
    throw new Error(`Pinned Intel compiler LLD query failed (exit ${result.status}, signal ${result.signal || "none"})` +
      (result.stderr ? ": " + result.stderr.slice(-2048) : ""), {cause: result.error});
  }
  if (!/^[^\r\n]+\n?$/.test(output)) {throw new Error("Pinned Intel compiler LLD query did not return a single absolute path");}
  const linker = output.endsWith("\n") ? output.slice(0, -1) : output;
  if (!path.isAbsolute(linker) || path.resolve(linker) !== linker || path.basename(linker) !== "ld.lld" ||
      Array.from(linker).some(char => char.charCodeAt(0) < 32 || (char.charCodeAt(0) >= 127 && char.charCodeAt(0) <= 159))) {
    throw new Error("Pinned Intel compiler LLD query did not return a single absolute ld.lld path");
  }
  /** @param {string} root @param {string} file */
  const inside = (root, file) => {
    const relative = path.relative(root, file);
    return Boolean(relative) && relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
  };
  try {
    const realRoot = fs.realpathSync(compilerRoot), realLinker = fs.realpathSync(linker), stat = fs.statSync(realLinker);
    if (!inside(compilerRoot, linker) || !inside(realRoot, realLinker) || !stat.isFile()) {
      throw new Error("target is outside the compiler SDK or not a regular file");
    }
    fs.accessSync(linker, fs.constants.X_OK);
  } catch (error) {
    throw new Error("Pinned Intel compiler ld.lld is not executable within the pinned compiler SDK", {cause: error});
  }
  return linker;
}

/** The release frontdoor's exact source-composed Linux shape; no Core/factory receipts are fabricated.
 * @param {LinuxNativeSdkConfigState} state @param {{cc: string, cxx: string}} tools @param {string} [linker] */
function linuxNativeSdkDescriptorConfig(state, tools, linker = linuxNativeSdkLldPath(tools.cxx)) {
  if (!path.isAbsolute(linker) || path.basename(linker) !== "ld.lld") {throw new Error("Native application linker must be the selected absolute ld.lld");}
  const image = path.join(state.images.handoffRoot, "cuda-images.inc"), acppBuild = path.join(state.acppSdk.sdkRoot, "build"),
    acppEnv = {CXX: tools.cxx, LLVM_CONFIG: path.join(state.llvmSdk, "bin/llvm-config"),
      MOM_AOT_OPENMP_LIBRARY: path.join(state.openmp.sdkRoot, "sdk/libomp.a"),
      MOM_AOT_ROCM_STATIC_ROOT: path.join(state.rocm.runtimeOutput, "sdk"),
      MOM_AOT_COMGR_RECEIPT: path.join(state.rocm.sdkRoot, "result.json"), MOM_AOT_LLD_ROOT: state.llvmSdk};
  return {version: 1,
    cpu: {cache: path.join(repo, "build/cache/oneapi"), prepareCache: true, env: {CXX: tools.cxx}},
    gpu: {
      oneapi: {args: [state.dpcpp.dpcppRoot], env: {MOM_AOT_ONEAPI_CXX: tools.cxx}},
      dpcpp: {args: [state.dpcpp.dpcppRoot, state.dpcpp.cudaRoot], env: {MOM_AOT_CUDA_IMAGES: image}},
      "dpcpp-opencl": {args: [state.dpcpp.dpcppRoot]},
      "acpp-cuda": {args: [state.acpp.preparedCheckout, acppBuild], env: {...acppEnv, MOM_AOT_CUDA_IMAGES: image}},
      "acpp-hip": {args: [state.acpp.preparedCheckout, acppBuild], env: acppEnv},
    },
    sdkSources: {acpp: path.join(state.acppSdk.sdkRoot, "result.json"), hermes: path.join(state.hermes.sdkRoot, "result.json"),
      rocm: path.join(state.rocm.runtimeOutput, "result.json"), cudaImages: image, openmpSdk: state.openmp.sdkRoot, zstd: state.zstd.sdkRoot},
    controller: {source: path.join(state.hermes.sdkRoot, "source"), build: path.join(state.hermes.sdkRoot, "build"), openssl: state.opensslSdk,
      io: state.ioSdk, cc: tools.cc, nvml: path.join(state.dpcpp.cudaRoot, "include/nvml.h")},
    sdk: {defaultCompiler: "oneapi", cxx: tools.cxx, flags: [], libraries: [],
      // Host archives contain Intel LTO; the embedded JIT's older LLVM linker cannot read it.
      links: ["--ld-path=" + linker, "-Wl,--threads=2", "-Wl,--thinlto-jobs=2",
        "-Wl,-T," + path.join(repo, "runtime/static-constructors.ld"), path.join(state.dpcpp.dpcppRoot, "lib/libsycl-devicelib-host.a")],
      sdkObjects: [], dpcppComponent: state.jit.staticComponent},
  };
}

module.exports = {linuxNativeSdkLldPath, linuxNativeSdkDescriptorConfig};
