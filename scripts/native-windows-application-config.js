"use strict";

// Compose already-admitted inputs; this is not a substitute for the source readers.
const path = require("node:path");
const {isDeepStrictEqual} = require("node:util");
const {validateWindowsLlvmSdkConfig, validateWindowsControllerConfig} = require("./build-native-windows-controller");

const hash = /^[a-f0-9]{64}$/;

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function object(value) {return Boolean(value && typeof value === "object" && !Array.isArray(value));}

/** @param {unknown} value @param {"io" | "hermes"} kind @param {string} runAttempt */
function admittedHandoff(value, kind, runAttempt) {
  if (!object(value) || value["kind"] !== kind || value["runAttempt"] !== runAttempt || typeof value["handoff"] !== "string" ||
      !path.isAbsolute(value["handoff"]) || path.resolve(value["handoff"]) !== value["handoff"] ||
      typeof value["handoffSha256"] !== "string" || !hash.test(value["handoffSha256"]) || !object(value["files"])) {
    throw new Error("Windows application config requires the same-run admitted " + kind + " handoff");
  }
}

/** Compose admitted reader results. The Windows Hermes support reader omits shermaot, so bind it
 * separately to the source-built Linux Hermes reader result rather than treating it as a handoff member.
 * @param {unknown} input */
function composeWindowsApplicationConfig(input) {
  if (!object(input)) {throw new Error("Invalid Windows application config composition input");}
  const row = input, runAttempt = row["runAttempt"];
  if (typeof runAttempt !== "string" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(runAttempt) ||
      !object(row["tools"]) || !object(row["io"]) || !object(row["hermes"]) || !object(row["hermesAot"]) ||
      !object(row["llvmDescriptor"]) || !Array.isArray(row["allowedImports"])) {
    throw new Error("Windows application config input is incomplete");
  }
  const expectedInputKeys = ["allowedImports", "hermes", "hermesAot", "io", "llvmDescriptor", "runAttempt", "tools"],
    hasApplication = Object.hasOwn(row, "application");
  if (!isDeepStrictEqual(Object.keys(row).sort(), [...expectedInputKeys, ...(hasApplication ? ["application"] : [])].sort()) ||
      hasApplication && !object(row["application"])) {throw new Error("Windows application config input has an unexpected shape");}
  admittedHandoff(row["io"], "io", runAttempt); admittedHandoff(row["hermes"], "hermes", runAttempt);
  const llvm = validateWindowsLlvmSdkConfig(row["llvmDescriptor"]), io = /** @type {{libraries?: unknown}} */ (row["io"]),
    hermes = /** @type {{hermes?: unknown,intelRuntime?: unknown}} */ (row["hermes"]),
    hermesAot = /** @type {{artifacts?: unknown,files?: unknown}} */ (row["hermesAot"]);
  if (!object(io.libraries) || !object(hermes.hermes) || !object(hermes.intelRuntime)) {
    throw new Error("Windows support readers returned incomplete inputs");
  }
  const runtime = /** @type {{libRoot?: unknown,libraries?: unknown}} */ (hermes.intelRuntime),
    runtimeLibraries = runtime.libraries;
  if (typeof runtime.libRoot !== "string" || !path.isAbsolute(runtime.libRoot) || !object(runtimeLibraries) || !Object.keys(runtimeLibraries).length ||
      Object.values(runtimeLibraries).some(file => typeof file !== "string" || !llvm.windowsSdk.msvcLibraries.includes(file)) ||
      !llvm.windowsSdk.libraryRoots.includes(runtime.libRoot)) {
    throw new Error("Windows LLVM descriptor does not retain the admitted Hermes Intel runtime inputs");
  }
  const tools = row["tools"], shermaot = tools["shermaot"], llvmReadobj = tools["llvmReadobj"];
  const hasObjdump = Object.hasOwn(tools, "llvmObjdump");
  if (!isDeepStrictEqual(Object.keys(tools).sort(), ["llvmReadobj", "shermaot", ...(hasObjdump ? ["llvmObjdump"] : [])].sort()) ||
      typeof shermaot !== "string" || typeof llvmReadobj !== "string" || hasObjdump && typeof tools["llvmObjdump"] !== "string") {
    throw new Error("Windows application config needs explicit shermaot and llvm-readobj paths");
  }
  if (!object(hermesAot["artifacts"]) || !object(hermesAot["files"]) || hermesAot["artifacts"]["compiler"] !== shermaot ||
      typeof hermesAot["files"][shermaot] !== "string" || !hash.test(hermesAot["files"][shermaot])) {
    throw new Error("Selected shermaot is not bound to the admitted Linux Hermes compiler receipt");
  }
  const config = {
    version: 1,
    tools: {...llvm.tools, shermaot, llvmReadobj, ...(hasObjdump ? {llvmObjdump: tools["llvmObjdump"]} : {})},
    hermes: {...hermes.hermes, shermaot},
    windowsSdk: {...llvm.windowsSdk, allowedImports: [...row["allowedImports"]]},
    libraries: {...io.libraries},
    ...(Object.hasOwn(row, "application") ? {application: row["application"]} : {}),
  };
  return validateWindowsControllerConfig(config);
}

module.exports = {composeWindowsApplicationConfig};
