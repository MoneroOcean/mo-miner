"use strict";

// This is a reviewed-notice drift check, not an SDK admission or legal-completeness claim.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"),
  {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readBuildText} = require("./build-source-inputs");
const {nativeNoticeFile} = require("./package-native-release");
const {nativePathSource} = require("./native-bundle");
const {gitSourceComponents} = require("./fetch-native-sdk-sources");
const {ioPins} = require("./build-native-io"), {opensslPin} = require("./build-native-openssl"),
  {zstdPin} = require("./build-native-zstd"), {hermesPins} = require("./build-native-hermes"),
  {archiveName, archiveSha256} = require("./native-llvm-sdk"), {nvrtcCohortInfo} = require("./probe-static-nvrtc");

const repo = path.resolve(__dirname, ".."), manifestDefault = path.join(repo, "runtime/native-notice-review.json"),
  noticeDefault = path.join(repo, "THIRD-PARTY-NOTICES.txt"), maxMetadataBytes = 8 << 20;
/** @type {("linux" | "windows" | "compiler" | "hermes" | "acpp" | "rocm")[]} */
const gitTargets = ["linux", "windows", "compiler", "hermes", "acpp", "rocm"];
const runtimeCatalogs = {
  linuxCudart: "linux-cudart.json",
  windowsHostSdk: "windows-host-sdk.json", windowsIntelRuntime: "windows-intel-runtime.json",
  windowsCudart: "windows-cudart.json", windowsAmdhip: "windows-amdhip.json",
};
const windowsSdkRedistributionEvidence = Object.freeze({
  windowsHostSdk: {
    sdkVersion: "10.0.26100.0",
    sdkLicense: {asset: "runtime/license-evidence/sdk_license.rtf",
      installPath: "Program Files (x86)/Windows Kits/10/Licenses/10.0.26100.0/sdk_license.rtf",
      bytes: 248573, sha256: "0f4a26ac9dc50066f8a1bfeaaf3f092d1b9e4791df5487b9f3c31e7c3dc4d7f5"},
    sdkThirdPartyNotices: {asset: "runtime/license-evidence/sdk_third_party_notices.rtf",
      installPath: "Program Files (x86)/Windows Kits/10/Licenses/10.0.26100.0/sdk_third_party_notices.rtf",
      bytes: 22279, sha256: "ebb6640a8246b39b70322c5cec9aad7a71e86ea7b86c9c604d3a7f3ff1b30e89"},
    redist: {licenseReference: "http://go.microsoft.com/fwlink/?LinkId=524842",
      url: "https://learn.microsoft.com/en-us/legal/windows-sdk/redist"},
    ucrtLibrary: {catalog: "runtime/windows-host-sdk.json", package: "ucrt-headers-libraries-sources",
      sourceMember: "Program Files/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64/libucrt.lib",
      outputMember: "sdk/lib/ucrt/libucrt.lib", sha256: "2b247d6a27c22c520c34134b8acf2da3366175a4690886927a51929f37725fdd"},
    acquisitionBasis: {product: "Microsoft Visual Studio Community 2022",
      eulaUrl: "https://visualstudio.microsoft.com/wp-content/uploads/2021/11/Visual-Studio-2022-Community-License-EN.docx",
      eulaBytes: 62859, eulaSha256: "41a207b10c8ab91d0d2f10a854715f73dca54509581692d2fe179aa3ffcb8540"},
  },
});
/** @typedef {{component:string,member:string,sha256:string}} LlvmLicenseMember */
/** @typedef {{bytes:number,member:string,sha256:string}} LinuxCudartLibrary */
/** @typedef {{version:number,target:string,vendor:string,releaseLabel:string,vendorVersion:string,
 * manifest:{releaseDate:string,url:string,sha256:string,entry:string},
 * archive:{fileName:string,url:string,origin:string,bytes:number,sha256:string},
 * libraries:{cudartStatic:LinuxCudartLibrary,culibos:LinuxCudartLibrary},
 * license:{name:string,manifestPath:string,member:string,noticeComponent:string,bytes:number,sha256:string}}} LinuxCudartNoticeCatalog */
/** @typedef {{buildNode:{version:string,pathSourceSha256:string},git:Record<string,unknown>,
 * archives:{io:unknown,openssl:{version:string},zstd:{version:string},hermes:unknown,
 * llvm:{archiveName:string,archiveSha256:string,licenseMembers:LlvmLicenseMember[]}},
 * cutlass:{version:string,archiveSha256:string,licenseSha256:string},
 * nvrtc:{linux12_6:{cohort:string},linux13_3:{version:string}},
 * runtime:Record<string,unknown>&{linuxCudart:LinuxCudartNoticeCatalog},
 * packingTool:{version:string,source:{notices:{member:string}[]}},
 * bundledModules:Record<string,{version:string,integrity:string,license:string}>}} NativeNoticeDependencyPins */
const llvmLicenseMembers = Object.freeze([
  {component: "LLVM Clang subproject license (llvm-project-21.1.8.src/clang/LICENSE.TXT)",
    member: "llvm-project-21.1.8.src/clang/LICENSE.TXT", sha256: "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47"},
  {component: "LLVM LLD subproject license (llvm-project-21.1.8.src/lld/LICENSE.TXT)",
    member: "llvm-project-21.1.8.src/lld/LICENSE.TXT", sha256: "f7891568956e34643eb6a0db1462db30820d40d7266e2a78063f2fe233ece5a0"},
]);
// Actual controller bundle inputs: direct runtime polyfills plus their bundled transitive packages.
const bundledModules = ["buffer", "base64-js", "ieee754", "events", "string_decoder", "safe-buffer", "core-js-pure"];

/** Bounded data reads reject aliases and verify both the read path and reopened file stayed stable.
 * @param {string} file @param {string} label @param {number} limit */
function readUtf8(file, label, limit) {
  if (typeof file !== "string" || !path.isAbsolute(file) || /[\0\r\n]/.test(file)) {throw new Error(label + " path must be absolute");}
  const canonical = fs.realpathSync(file);
  if (canonical !== file || !Number.isSafeInteger(limit) || limit < 1 || limit > maxMetadataBytes) {
    throw new Error(label + " must use a canonical absolute path and a bounded size");
  }
  const text = readBuildText(file, limit), bytes = Buffer.from(text, "utf8"), sha256 = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length > limit || fileHash(file) !== sha256 || fs.realpathSync(file) !== canonical) {
    throw new Error(label + " changed while reading or is not valid UTF-8");
  }
  return text;
}

/** Node path builtin source is embedded by nativeAliases; this is a build-tool pin, not a production Node dependency. */
function buildNodePin() {
  const {source} = nativePathSource();
  return {version: process.version, pathSourceSha256: createHash("sha256").update(source).digest("hex")};
}

/** Read only the three reviewed CUTLASS installer pins; script-file hashes are not version pins. */
function cutlassPins() {
  const text = readUtf8(path.join(repo, "scripts/install-cutlass.sh"), "CUTLASS installer", 1 << 20);
  /** @param {string} name */
  const literal = name => {
    const rows = [...text.matchAll(new RegExp("^" + name + "=([^\\r\\n]+)$", "gm"))];
    if (rows.length !== 1) {throw new Error("CUTLASS installer must have one " + name + " pin");}
    return rows[0]?.[1] || "";
  };
  const version = literal("MOM_CUTLASS_VERSION"), archiveSha256 = literal("MOM_CUTLASS_SHA256"),
    licenseSha256 = literal("MOM_CUTLASS_LICENSE_SHA256");
  const urlTemplate = /^MOM_CUTLASS_URL="https:\/\/github\.com\/NVIDIA\/cutlass\/archive\/refs\/tags\/\$\{MOM_CUTLASS_VERSION\}\.tar\.gz"$/gm;
  if (!/^v\d+\.\d+\.\d+$/.test(version) || !/^[0-9a-f]{64}$/.test(archiveSha256) ||
      !/^[0-9a-f]{64}$/.test(licenseSha256) || [...text.matchAll(urlTemplate)].length !== 1) {
    throw new Error("CUTLASS installer pins or URL template are unsupported");
  }
  return {version, archiveSha256, licenseSha256};
}

/** @param {unknown} value @returns {unknown} */
function stableJson(value) {
  if (typeof value === "string") {
    if (path.isAbsolute(value) || /^[A-Za-z]:[\\/]/.test(value) || /^\\\\/.test(value)) {
      throw new Error("Native notice pins must not contain absolute host paths");
    }
    return value;
  }
  if (value === null || typeof value === "boolean" || typeof value === "number") {return value;}
  if (Array.isArray(value)) {return value.map(stableJson);}
  if (value && typeof value === "object") {
    const entries = Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
    return Object.fromEntries(entries.map(([key, child]) => [key, stableJson(child)]));
  }
  throw new Error("Native notice pin source contains a non-JSON value");
}

/** Preserve exact installed Windows SDK terms as review evidence without adding them to the app notice body.
 * @param {string} manifestPath @param {unknown} value */
function verifyWindowsSdkRedistributionEvidence(manifestPath, value) {
  if (!isDeepStrictEqual(value, windowsSdkRedistributionEvidence)) {
    throw new Error("Windows SDK redistribution evidence differs from the reviewed source metadata");
  }
  const runtimeDirectory = path.dirname(manifestPath);
  if (path.basename(runtimeDirectory) !== "runtime") {throw new Error("Windows SDK redistribution evidence requires a runtime review manifest path");}
  const readEvidenceAsset = (/** @type {{asset:string,bytes:number,sha256:string}} */ pin, /** @type {string} */ label) => {
    const projectRoot = path.dirname(runtimeDirectory), file = path.resolve(projectRoot, pin.asset), relative = path.relative(projectRoot, file);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error(label + " path escapes the repository evidence root");
    }
    let text;
    try {text = readUtf8(file, label, 1 << 20);} catch (error) {
      throw new Error(label + " is missing or unreadable", {cause: error});
    }
    const bytes = Buffer.from(text, "utf8");
    if (bytes.length !== pin.bytes || createHash("sha256").update(bytes).digest("hex") !== pin.sha256) {
      throw new Error(label + " differs from its reviewed byte pin");
    }
  };
  const evidence = /** @type {typeof windowsSdkRedistributionEvidence} */ (value).windowsHostSdk;
  readEvidenceAsset(evidence.sdkLicense, "Windows SDK license evidence");
  readEvidenceAsset(evidence.sdkThirdPartyNotices, "Windows SDK third-party notice evidence");

  const catalog = JSON.parse(readUtf8(path.join(repo, evidence.ucrtLibrary.catalog), "Windows host SDK catalog", 2 << 20));
  const sourcePrefix = path.posix.join("Program Files/Windows Kits/10/Lib", evidence.sdkVersion, "ucrt/x64"),
    selection = catalog.selections?.filter((/** @type {{package?:string,kind?:string,sourcePrefix?:string,destinationPrefix?:string}} */ row) =>
      row.package === evidence.ucrtLibrary.package && row.kind === "tree" && row.sourcePrefix === sourcePrefix && row.destinationPrefix === "sdk/lib/ucrt");
  if (!Array.isArray(selection) || selection.length !== 1 ||
      evidence.ucrtLibrary.sourceMember !== path.posix.join(sourcePrefix, "libucrt.lib") ||
      evidence.ucrtLibrary.outputMember !== "sdk/lib/ucrt/libucrt.lib") {
    throw new Error("Windows SDK UCRT member does not match its catalog selection");
  }
}

/** Verify exact archive-member bodies retained under their reviewed notice headings.
 * @param {string} noticeText @param {{component:string,member:string,sha256:string}[]} members */
function verifyNoticeLicenseMembers(noticeText, members) {
  const separator = "================================================================================\n";
  for (const member of members) {
    const marker = separator + "Components: " + member.component + "\n\n", first = noticeText.indexOf(marker),
      bodyStart = first + marker.length, next = noticeText.indexOf("\n" + separator + "Components: ", bodyStart),
      body = first < 0 || noticeText.indexOf(marker, first + marker.length) >= 0 ? "" : noticeText.slice(bodyStart, next < 0 ? noticeText.length : next);
    let matched = false;
    for (let trailing = 0; trailing <= 2 && trailing <= body.length; trailing++) {
      if (trailing > 0 && !body.endsWith("\n".repeat(trailing))) {continue;}
      const candidate = Buffer.from(body.slice(0, body.length - trailing), "utf8");
      if (createHash("sha256").update(candidate).digest("hex") === member.sha256) {matched = true; break;}
    }
    if (!matched) {throw new Error("Third-party notices do not retain the pinned license member: " + member.member);}
  }
}

/** @param {NativeNoticeDependencyPins["runtime"]} runtime @returns {LlvmLicenseMember[]} */
function noticeLicenseMembers(runtime) {
  const license = runtime.linuxCudart.license;
  if (typeof license.noticeComponent !== "string" || !/^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$/.test(license.noticeComponent) ||
      typeof license.member !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._+/-]*$/.test(license.member) ||
      typeof license.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(license.sha256)) {
    throw new Error("Unknown Linux CUDA runtime license catalog");
  }
  return [...llvmLicenseMembers, {component: license.noticeComponent, member: license.member, sha256: license.sha256}];
}

/** @returns {NativeNoticeDependencyPins} */
function dependencyPins() {
  const git = Object.fromEntries(gitTargets.map(target => [target, gitSourceComponents(target)]));
  const lock = JSON.parse(readUtf8(path.join(repo, "package-lock.json"), "Package lock", maxMetadataBytes));
  if (lock.lockfileVersion !== 3 || !lock.packages || typeof lock.packages !== "object") {
    throw new Error("Native runtime package pins require package-lock v3");
  }
  const packages = Object.fromEntries(bundledModules.map(name => {
    const entry = lock.packages["node_modules/" + name];
    if (!entry || typeof entry.version !== "string" || !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(entry.version) ||
        typeof entry.integrity !== "string" || !/^sha512-[A-Za-z0-9+/]+={0,2}$/.test(entry.integrity) ||
        typeof entry.license !== "string" || !entry.license) {
      throw new Error("Missing reviewed runtime bundle package pin: " + name);
    }
    return [name, {version: entry.version, integrity: entry.integrity, license: entry.license}];
  }));
  const runtime = Object.fromEntries(Object.entries(runtimeCatalogs).map(([name, file]) => [name,
    JSON.parse(readUtf8(path.join(repo, "runtime", file), "Runtime catalog " + name, 2 << 20))]));
  if (Object.values(runtime).some(catalog => !catalog || typeof catalog !== "object" || Array.isArray(catalog) ||
      !Number.isSafeInteger(catalog.version))) {throw new Error("Unknown native runtime catalog schema");}
  const packingTool = JSON.parse(readUtf8(path.join(repo, "scripts/native-packing-tool.json"), "Native packing tool catalog", 1 << 20));
  if (!packingTool || typeof packingTool !== "object" || Array.isArray(packingTool) ||
      typeof packingTool.version !== "string" || !/^\d+\.\d+\.\d+$/.test(packingTool.version) ||
      !Array.isArray(packingTool.source?.notices) || !packingTool.source.notices.some((/** @type {{member?: unknown}} */ row) =>
    typeof row?.member === "string" && row.member.includes("lzma-sdk"))) {
    throw new Error("Unknown native packing-tool notice catalog");
  }
  return /** @type {NativeNoticeDependencyPins} */ (stableJson({
    buildNode: buildNodePin(),
    git,
    archives: {io: ioPins, openssl: opensslPin, zstd: zstdPin, hermes: hermesPins,
      llvm: {archiveName, archiveSha256, licenseMembers: llvmLicenseMembers}},
    cutlass: cutlassPins(),
    nvrtc: {linux12_6: nvrtcCohortInfo("linux", "linux-12.6"), linux13_3: nvrtcCohortInfo("linux", "linux")},
    runtime, packingTool,
    bundledModules: packages,
  }));
}

/** Match tracked notice bytes and the explicitly reviewed semantic pin snapshot.
 * @param {string} [manifestPath] @param {string} [noticePath] */
function checkNativeNoticeReview(manifestPath = manifestDefault, noticePath = noticeDefault) {
  const manifest = JSON.parse(readUtf8(manifestPath, "Native notice review", 2 << 20));
  if (typeof noticePath !== "string" || !path.isAbsolute(noticePath) || /[\0\r\n]/.test(noticePath)) {
    throw new Error("Third-party notices path must be absolute");
  }
  const notice = nativeNoticeFile(noticePath);
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest) ||
      !hasExactBuildFields(manifest, ["dependencies", "notice", "redistributionEvidence", "version"]) || manifest.version !== 1 ||
      !manifest.notice || typeof manifest.notice !== "object" || Array.isArray(manifest.notice) ||
      !hasExactBuildFields(manifest.notice, ["bytes", "sha256"]) ||
      !Number.isSafeInteger(manifest.notice.bytes) || manifest.notice.bytes < 1 ||
      typeof manifest.notice.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.notice.sha256)) {
    throw new Error("Native notice review manifest has an unsupported schema");
  }
  if (notice.bytes !== manifest.notice.bytes || notice.sha256 !== manifest.notice.sha256) {
    throw new Error("Tracked third-party notices differ from the reviewed byte pin");
  }
  verifyWindowsSdkRedistributionEvidence(manifestPath, manifest.redistributionEvidence);
  const noticeText = readUtf8(noticePath, "Third-party notices", 8 << 20);
  const dependencies = dependencyPins();
  if (!isDeepStrictEqual(manifest.dependencies, dependencies)) {
    throw new Error("Native dependency pins differ from the reviewed notice snapshot");
  }
  verifyNoticeLicenseMembers(noticeText, noticeLicenseMembers(dependencies.runtime));
  return {version: 1, notice, redistributionEvidence: manifest.redistributionEvidence, dependencies};
}

module.exports = {dependencyPins, checkNativeNoticeReview};
if (require.main === module) {
  try {checkNativeNoticeReview(); console.log("Native notice review snapshot and notice bytes match.");} catch (error) {
    console.error(error instanceof Error ? error.message : "Native notice review failed"); process.exitCode = 1;
  }
}
