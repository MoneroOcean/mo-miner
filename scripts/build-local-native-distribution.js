"use strict";

// Local release assembly uses the same candidate, dependency audit and archive producers as CI.
// SDK preparation stays separate so a JavaScript edit never rebuilds vendor compilers.
const fs = require("node:fs"), path = require("node:path");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {checkNativeNoticeReview} = require("./check-native-notices");
const {prepareNativePackingTool} = require("./prepare-native-packing-tool");
const local = require("./build-local-native-sdk");

/** Build, compress, audit and package a prepared SDK on Linux without running the miner.
 * The shared release producer selects ELF or PE from the admitted descriptor; WSL uses
 * this same assembly path for Windows instead of maintaining another packaging recipe.
 * Callers own the build lock and select a fresh output; unknown evidence is never cleared.
 * @param {{workspace:string,configFile:string,tools:ReturnType<typeof local.localTools>}} sdk
 * @param {string} output @param {NodeJS.ProcessEnv} [environment]
 * @param {{verifyScope?:()=>unknown,review?:typeof checkNativeNoticeReview,
 * preparePacking?:typeof prepareNativePackingTool,build?:typeof import('./build-release').buildNativeArtifact,
 * audit?:typeof import('./check-native-release').auditNativeRelease,
 * package?:typeof import('./package-native-release').packageNativeRelease}} [dependencies] */
function buildLocalNativeDistribution(sdk, output, environment = process.env, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local distribution assembly requires Linux x64");}
  const repo = fs.realpathSync(path.resolve(__dirname, ".."));
  const relative = sdk && typeof sdk.workspace === "string" ? path.relative(sdk.workspace, output) : "";
  if (!sdk || !path.isAbsolute(sdk.workspace) || fs.realpathSync(sdk.workspace) !== sdk.workspace ||
      !path.isAbsolute(sdk.configFile) || fs.realpathSync(sdk.configFile) !== sdk.configFile ||
      !path.isAbsolute(output) || path.resolve(output) !== output ||
      !relative || relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) ||
      fs.realpathSync(path.dirname(output)) !== path.dirname(output) || /[\r\n\0]/.test(output) ||
      [output, output + ".unpacked"].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Local distribution requires an admitted SDK and fresh workspace-owned output");
  }
  local.admitWorkspace(sdk.workspace, repo);
  const version = require("./build-release").releaseVersion(environment),
    notices = path.join(repo, "THIRD-PARTY-NOTICES.txt"), review = (dependencies.review || checkNativeNoticeReview)(),
    config = readBuildJson(sdk.configFile), configHash = fileHash(sdk.configFile);
  if (!review.notice || review.notice.sha256 !== fileHash(notices) ||
      !config.notices || config.notices.path !== notices || config.notices.sha256 !== review.notice.sha256) {
    throw new Error("Local SDK descriptor must bind the current reviewed distribution notices");
  }
  const env = {...environment, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2"},
    packing = (dependencies.preparePacking || prepareNativePackingTool)(path.join(sdk.workspace, "packing-tool"), env),
    artifact = (dependencies.build || require("./build-release").buildNativeArtifact)(sdk.configFile, output, packing, env),
    entrypoint = path.join(output, "entrypoint.json"),
    audit = dependencies.audit || require("./check-native-release").auditNativeRelease;
  audit(entrypoint);
  const archive = (dependencies.package || require("./package-native-release").packageNativeRelease)(
    entrypoint, sdk.tools.cmake, version);
  audit(entrypoint, {archive: {path: archive.archive.path, tool: sdk.tools.cmake}});
  if (fileHash(sdk.configFile) !== configHash || fileHash(artifact.artifact.path) !== artifact.artifact.sha256 ||
      fileHash(archive.archive.path) !== archive.archive.sha256 || fileHash(notices) !== review.notice.sha256) {
    throw new Error("Local distribution inputs changed during assembly");
  }
  const sidecar = archive.archive.path + ".sha256";
  fs.writeFileSync(sidecar, `${archive.archive.sha256}  ${path.basename(archive.archive.path)}\n`, {flag: "wx", mode: 0o600});
  return {archive: archive.archive, sidecar, executable: artifact.artifact,
    buildOnly: true, runtimeTested: false, publicationQualified: false};
}

module.exports = {buildLocalNativeDistribution};
