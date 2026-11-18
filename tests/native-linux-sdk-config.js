"use strict";

const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const recipe = require("../scripts/native-linux-sdk-config");

const repo = path.resolve(__dirname, "..");
const state = {
  llvmSdk: "/work/llvm-sdk",
  images: {handoffRoot: "/work/cuda-image-handoff"},
  acppSdk: {sdkRoot: "/work/acpp-sdk"},
  acpp: {preparedCheckout: "/work/acpp-source"},
  dpcpp: {dpcppRoot: "/work/dpcpp", cudaRoot: "/work/cuda"},
  gnu: {root: "/work/gnu", name: "gnu-inputs", sdkTar: "/work/gnu.tar"},
  jit: {staticComponent: "/work/static-dpcpp-component"},
  hermes: {source: "/work/hermes-source", sdkRoot: "/work/hermes-sdk"},
  rocm: {sdkRoot: "/work/rocm-sdk", runtimeOutput: "/work/rocm-runtime"},
  openmp: {sdkRoot: "/work/openmp-sdk"},
  zstd: {sdkRoot: "/work/zstd-sdk"},
  opensslSdk: "/work/openssl-sdk",
  ioSdk: "/work/io-sdk",
};
const tools = {cc: "/tools/icx", cxx: "/tools/icpx"};

test.describe("Shared Linux native SDK descriptor recipe", {skip: process.platform !== "linux"}, () => {
  test("builds the exact Linux source-composed SDK config from explicit state without CI env", () => {
    const linker = "/tools/compiler/ld.lld", image = "/work/cuda-image-handoff/cuda-images.inc",
      acppBuild = "/work/acpp-sdk/build", acppEnv = {CXX: tools.cxx, LLVM_CONFIG: "/work/llvm-sdk/bin/llvm-config",
        MOM_AOT_OPENMP_LIBRARY: "/work/openmp-sdk/sdk/libomp.a", MOM_AOT_ROCM_STATIC_ROOT: "/work/rocm-runtime/sdk",
        MOM_AOT_COMGR_RECEIPT: "/work/rocm-sdk/result.json", MOM_AOT_LLD_ROOT: "/work/llvm-sdk"};
    const expected = {version: 1,
      cpu: {cache: path.join(repo, "build/cache/oneapi"), prepareCache: true, env: {CXX: tools.cxx}},
      gpu: {
        oneapi: {args: ["/work/dpcpp"], env: {MOM_AOT_ONEAPI_CXX: tools.cxx}},
        dpcpp: {args: ["/work/dpcpp", "/work/cuda"], env: {MOM_AOT_CUDA_IMAGES: image}},
        "dpcpp-opencl": {args: ["/work/dpcpp"]},
        "acpp-cuda": {args: ["/work/acpp-source", acppBuild], env: {...acppEnv, MOM_AOT_CUDA_IMAGES: image}},
        "acpp-hip": {args: ["/work/acpp-source", acppBuild], env: acppEnv},
      },
      sdkSources: {acpp: "/work/acpp-sdk/result.json", hermes: "/work/hermes-sdk/result.json", rocm: "/work/rocm-runtime/result.json",
        cudaImages: image, openmpSdk: "/work/openmp-sdk", zstd: "/work/zstd-sdk"},
      controller: {source: "/work/hermes-sdk/source", build: "/work/hermes-sdk/build", openssl: "/work/openssl-sdk",
        io: "/work/io-sdk", cc: tools.cc, nvml: "/work/cuda/include/nvml.h"},
      sdk: {defaultCompiler: "oneapi", cxx: tools.cxx, flags: [], libraries: [],
        links: ["--ld-path=" + linker, "-Wl,--threads=2", "-Wl,--thinlto-jobs=2",
          "-Wl,-T," + path.join(repo, "runtime/static-constructors.ld"), "/work/dpcpp/lib/libsycl-devicelib-host.a"],
        sdkObjects: [], dpcppComponent: "/work/static-dpcpp-component"},
    };
    const githubKeys = Object.keys(process.env).filter(key => key.startsWith("GITHUB_") || key === "RUNNER_TEMP"),
      savedEnvironment = Object.fromEntries(githubKeys.map(key => [key, process.env[key]]));
    for (const key of githubKeys) {delete process.env[key];}
    try {assert.deepEqual(recipe.linuxNativeSdkDescriptorConfig(state, tools, linker), expected);} finally {Object.assign(process.env, savedEnvironment);}
  });

  test("CI keeps re-exporting the shared recipe functions", () => {
    const ci = require("../scripts/prepare-native-ci");
    assert.equal(ci.linuxNativeSdkDescriptorConfig, recipe.linuxNativeSdkDescriptorConfig);
    assert.equal(ci.linuxNativeSdkLldPath, recipe.linuxNativeSdkLldPath);
  });
});
