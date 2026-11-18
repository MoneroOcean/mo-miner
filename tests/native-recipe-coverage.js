"use strict";
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");

// A producer's recipe hash decides when a cached SDK output is stale. Every module it requires
// directly must either be listed in that recipe or be excluded here because it only admits,
// hashes or verifies inputs that are pinned separately and cannot shape the produced bytes.
const scripts = path.join(__dirname, "..", "scripts");
// Shared hashing, path, process and CI-job helpers check or run inputs; they never choose them.
const sharedHelpers = new Set(["build-source-inputs", "native-ci-job", "native-paths", "native-process"]);
/** @type {Record<string, Record<string, string>>} */
const producers = {
  "build-native-llvm-sdk": {},
  "build-native-windows-llvm-sdk": {},
  "build-native-windows-comgr-sdk": {},
  "build-native-windows-hiprtc-sdk": {},
  "build-native-windows-acpp-sdk": {},
  "build-native-windows-sdk": {},
  "build-native-rocm-host-runtime": {},
  "build-native-rocm-resources": {},
  "build-native-jit-sdk": {},
  "build-native-hermes": {},
  "build-native-io": {},
  "build-native-openssl": {},
  "build-native-zstd": {},
  "build-native-libdrm": {},
  "native-linux-llvm-tools": {},
  "native-linux-llvm-sdk-handoff": {},
  "native-rocm-hip-headers": {},
  "native-comgr-inputs": {},
  "prepare-native-windows-sdk": {},
  "prepare-native-jit-base": {},
  "prepare-native-cudart-windows": {},
  "pack-native-executable": {},
  "build-native-release": {
    "native-audit": "post-link audit of the produced executable; it rejects but never changes bytes",
    "build-native-zstd": "SDK receipt reader; the zstd SDK files are hashed into sdkInputs",
    "build-native-openssl": "SDK receipt reader; the OpenSSL SDK files are hashed into sdkInputs",
  },
  "build-native-sycl-sdk": {
    "build-native-release": "absolute-path predicate only",
    "native-comgr-inputs": "admits the COMGR SDK selection, whose files are hashed as inputs",
  },
  "build-native-openmp-sdk": {
    "fetch-native-sdk-sources": "acquisition is admitted by exact archive pins; only compilation policy is in the recipe",
  },
  "native-windows-jit-resources": {
    "prepare-native-gnu-headers": "admits the pinned Windows GNU header handoff",
    "native-ci-linux-sources": "CI work-root paths of the pinned same-run inputs",
    "native-dpcpp-cuda-inputs": "admits the externally pinned DPC++/CUDA input handoff",
  },
  "prepare-native-amdhip-windows": {
    "build-local-native-sdk": "local workspace admission",
    "build-local-native-rocm-sdk": "local ROCm source path layout",
    "build-native-llvm-sdk": "admits the pinned LLVM tool output it runs",
  },
};

test.describe("SDK producer recipes", () => {
  test("list or explicitly exclude every direct local require", () => {
    for (const [producer, exclusions] of Object.entries(producers)) {
      const source = fs.readFileSync(path.join(scripts, producer + ".js"), "utf8");
      const required = new Set([...source.matchAll(/require\("\.\/([a-z0-9-]+)"\)/g)].map(match => match[1] || ""));
      for (const module of required) {
        assert.ok(source.includes(`"scripts/${module}.js"`) || sharedHelpers.has(module) || Object.hasOwn(exclusions, module),
          `${producer} requires ${module} but neither lists it in its recipe nor excludes it with a reason`);
      }
      for (const module of Object.keys(exclusions)) {
        assert.ok(required.has(module), `${producer} no longer requires excluded ${module}; drop the exclusion`);
      }
    }
  });
});
