"use strict";

// Build-only GNU target headers for source JIT. Enumerate installed package file lists,
// not a host include-tree crawl; the emitter pins every staged input.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {validName} = require("./native-jit-headers");
const maxFiles = 16384;
const linuxTarget = "x86_64-unknown-linux-gnu", windowsTarget = "x86_64-w64-windows-gnu";
/** @type {Readonly<Record<string,Readonly<{resolvedSourcePath:string,ownershipPath:string}>>>} */
const copyrightAliases = Object.freeze({
  "libstdc++-13-dev": Object.freeze({resolvedSourcePath: "/usr/share/doc/gcc-13-base/copyright", ownershipPath: "/usr/share/doc/libstdc++-13-dev"}),
  "g++-mingw-w64-x86-64-posix": Object.freeze({resolvedSourcePath: "/usr/share/doc/gcc-mingw-w64-base/copyright",
    ownershipPath: "/usr/share/doc/g++-mingw-w64-x86-64-posix"}),
});
/** @type {[string,string][]} */
const linuxRootMappings = [
  ["x86_64-linux-gnu/c++/13", "include/mom-cxx-target"],
  ["c++/13", "include/mom-cxx"],
  ["x86_64-linux-gnu", "include/mom-system-target"],
  ["", "include/mom-system"],
];
/** @typedef {{packages:Array<{name:string,architecture:string}>,maxBytes:number,virtualRoots:string[],requireBackward:boolean,packageOwnersByRoot?:Record<string,string>,resolvedPackagesByRoot?:Record<string,string>}} GnuProfile */
/** @typedef {{source:string,virtual:string,package:string,resolvedPackage?:string,sha256:string,bytes:number}} HeaderFile */
/** @typedef {{version:number,target:string,root:string,packages:Array<{name:string,version:string,architecture:string}>,files:HeaderFile[],bytes:number,producerSha256:string,buildOnly:boolean,compiled:boolean,executed:boolean,publicationQualified:boolean}} GnuHeadersResult */
/** @type {Record<string,GnuProfile>} */
const profiles = {
  [linuxTarget]: {
    packages: [
      {name: "libstdc++-13-dev", architecture: "amd64"},
      {name: "libc6-dev", architecture: "amd64"},
      {name: "linux-libc-dev", architecture: "amd64"},
    ],
    maxBytes: 64 * 1024 * 1024,
    virtualRoots: ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target", "include/mom-system"],
    requireBackward: true,
  },
  [windowsTarget]: {
    packages: [
      {name: "mingw-w64-common", architecture: "all"},
      {name: "mingw-w64-x86-64-dev", architecture: "all"},
      {name: "g++-mingw-w64-x86-64-posix", architecture: "amd64"},
    ],
    maxBytes: 128 * 1024 * 1024,
    virtualRoots: ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target"],
    requireBackward: true,
    packageOwnersByRoot: {
      "include/mom-cxx": "g++-mingw-w64-x86-64-posix",
      "include/mom-cxx-target": "g++-mingw-w64-x86-64-posix",
      "include/mom-system-target": "mingw-w64-x86-64-dev",
    },
    resolvedPackagesByRoot: {"include/mom-system-target": "mingw-w64-common"},
  },
};
/** @typedef {{name:string, version:string, architecture:string, files:string[]}} Package */
/** @typedef {{logicalRoot:string, physicalRoot:string, virtualRoot:string, realRoots:string[], resolvedPackage?:string, packageOwner?:string}} HeaderRoot */

/** @param {string} target @returns {GnuProfile} */
function profileFor(target) {
  if (!Object.hasOwn(profiles, target)) {throw new Error("Unsupported GNU header target: " + target);}
  const profile = profiles[target];
  if (!profile) {throw new Error("Unsupported GNU header target: " + target);}
  return profile;
}

/** Preserve Linux's historical relative-path mapping. @param {string} relative */
function virtualHeader(relative) {
  if (!validName(relative)) {throw new Error("Invalid GNU header source path");}
  for (const [source, virtual] of linuxRootMappings) {
    if (!source || relative.startsWith(source + "/")) {
      return virtual + "/" + (source ? relative.slice(source.length + 1) : relative);
    }
  }
  throw new Error("Unmapped GNU header");
}

/** Resolve an absolute package-list path under a fixture sysroot or the live root.
 * @param {string} sysroot @param {string} source */
function underSysroot(sysroot, source) {
  return sysroot === "/" ? source : path.join(sysroot, source.slice(1));
}

/** @param {string} root @param {string} candidate */
function within(root, candidate) {
  return candidate === root || candidate.startsWith(root + path.sep);
}

/** @param {string} includeRoot @returns {HeaderRoot[]} */
function linuxRoots(includeRoot) {
  return linuxRootMappings.map(([relative, virtualRoot]) => {
    const logicalRoot = relative ? path.join(includeRoot, ...relative.split("/")) : includeRoot;
    return {logicalRoot, physicalRoot: logicalRoot, virtualRoot, realRoots: [logicalRoot]};
  });
}

/** @param {string} sysroot @returns {HeaderRoot[]} */
function windowsRoots(sysroot) {
  const mingwInclude = "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include";
  const targetSystem = "/usr/x86_64-w64-mingw32/include";
  const commonHeaders = "/usr/share/mingw-w64/include";
  /** @param {string} logicalRoot @param {string} virtualRoot @param {string[]} [realRoots] @param {string} [resolvedPackage] @param {string} [packageOwner] @returns {HeaderRoot} */
  const root = (logicalRoot, virtualRoot, realRoots = [logicalRoot], resolvedPackage, packageOwner) => ({
    logicalRoot, physicalRoot: underSysroot(sysroot, logicalRoot), virtualRoot,
    realRoots: realRoots.map(realRoot => underSysroot(sysroot, realRoot)),
    ...(resolvedPackage ? {resolvedPackage} : {}),
    ...(packageOwner ? {packageOwner} : {}),
  });
  // The target C++ subtree is nested under the common C++ directory, so it must
  // be selected first and retain its own virtual include root.
  return [
    root(path.join(mingwInclude, "c++/x86_64-w64-mingw32"), "include/mom-cxx-target", undefined, undefined, "g++-mingw-w64-x86-64-posix"),
    root(path.join(mingwInclude, "c++"), "include/mom-cxx", undefined, undefined, "g++-mingw-w64-x86-64-posix"),
    root(targetSystem, "include/mom-system-target", [targetSystem, commonHeaders], "mingw-w64-common", "mingw-w64-x86-64-dev"),
  ];
}

/** @param {Package[]} records @param {string} target */
function validatePackages(records, target) {
  const expected = profileFor(target).packages;
  if (!Array.isArray(records) || records.length !== expected.length || records.some((record, index) => {
    const packageSpec = expected[index];
    return !packageSpec || record.name !== packageSpec.name || record.architecture !== packageSpec.architecture ||
      typeof record.version !== "string" || !/^[A-Za-z0-9:.+~-]{1,128}$/.test(record.version) ||
      !Array.isArray(record.files) || record.files.length > maxFiles;
  })) {
    const archText = target === linuxTarget ? " amd64" : "";
    throw new Error("GNU headers require the explicit installed" + archText + " package records for " + target);
  }
}

/** @param {HeaderRoot[]} roots */
function validateRoots(roots) {
  const canonical = new Set();
  for (const root of roots) {
    for (const candidate of [root.physicalRoot, ...root.realRoots]) {
      if (!path.isAbsolute(candidate) || path.resolve(candidate) !== candidate) {
        throw new Error("GNU header source roots must be canonical directories");
      }
      canonical.add(candidate);
      if (!fs.lstatSync(candidate, {throwIfNoEntry: false})) {continue;}
      if (fs.realpathSync(candidate) !== candidate || !fs.lstatSync(candidate).isDirectory() || !fs.statSync(candidate).isDirectory()) {
        throw new Error("GNU header source roots must be canonical directories");
      }
    }
  }
  return [...canonical];
}

/** @param {string} output @param {string[]} sourceRoots */
function validateOutput(output, sourceRoots) {
  if (!path.isAbsolute(output) || path.resolve(output) !== output || fs.lstatSync(output, {throwIfNoEntry: false}) ||
      sourceRoots.some(root => output === root || within(root, output) || within(output, root)) ||
      fs.realpathSync(path.dirname(output)) !== path.dirname(output)) {
    throw new Error("GNU header roots must be canonical, separate and newly staged");
  }
}

/** @param {Package[]} records @param {HeaderRoot[]} roots @param {string} sysroot @param {string} target */
function collectFiles(records, roots, sysroot, target) {
  const profile = profileFor(target), ownerByPath = new Map();
  for (const record of records) {
    for (const source of record.files) {
      // dpkg lists the package's root directory as '/.'. It owns no header;
      // discard only that sentinel, never normalize arbitrary package paths.
      if (source === "/.") {continue;}
      if (typeof source !== "string" || !path.isAbsolute(source) || path.resolve(source) !== source) {
        throw new Error("Malformed GNU package file path");
      }
      const owners = ownerByPath.get(source) || [];
      owners.push(record.name);
      ownerByPath.set(source, owners);
    }
  }

  // Longest prefix first keeps target-specific C++ headers out of the common root.
  const orderedRoots = [...roots].sort((a, b) => b.logicalRoot.length - a.logicalRoot.length);
  const files = new Map(); let bytes = 0;
  for (const record of records) {
    for (const source of record.files) {
      const selected = orderedRoots.find(root => within(root.logicalRoot, source) && source !== root.logicalRoot);
      if (!selected) {continue;}
      if (selected.packageOwner && record.name !== selected.packageOwner) {
        throw new Error("GNU header path is not owned by the selected package: " + source);
      }
      const physical = underSysroot(sysroot, source), stat = fs.statSync(physical);
      if (stat.isDirectory()) {continue;}
      const real = fs.realpathSync(physical);
      if (!stat.isFile() || !selected.realRoots.some(root => within(root, real))) {
        throw new Error("GNU header is special or escapes its package-owned include roots");
      }
      const relative = path.relative(selected.logicalRoot, source).split(path.sep).join("/");
      if (!validName(relative)) {throw new Error("Malformed GNU package file path");}
      const virtual = selected.virtualRoot + "/" + relative;
      if (files.has(virtual)) {throw new Error("Duplicate GNU package header");}
      if (files.size >= maxFiles || bytes + stat.size > profile.maxBytes) {
        throw new Error("GNU header count or bytes exceed bounds: " + virtual + " (files=" + files.size + ", bytes=" + bytes + ")");
      }
      bytes += stat.size;
      let resolvedPackage;
      if (selected.resolvedPackage && real === physical) {
        throw new Error("GNU target headers require package-owned alias symlinks");
      }
      if (selected.resolvedPackage) {
        const realLogical = sysroot === "/" ? real : "/" + path.relative(sysroot, real).split(path.sep).join("/");
        const owners = ownerByPath.get(realLogical) || [];
        if (!owners.includes(selected.resolvedPackage)) {
          throw new Error("GNU header symlink target is absent from its selected package file list");
        }
        if (record.name !== selected.resolvedPackage) {resolvedPackage = selected.resolvedPackage;}
      }
      files.set(virtual, {source: physical, virtual, package: record.name,
        ...(resolvedPackage ? {resolvedPackage} : {}), sha256: fileHash(real), bytes: stat.size});
    }
  }
  for (const virtualRoot of profile.virtualRoots) {
    if (![...files.keys()].some(name => name.startsWith(virtualRoot + "/"))) {
      const count = profile.virtualRoots.length === 3 ? "three" : "four";
      throw new Error("GNU header package set lacks one of the " + count + " target roots: " + virtualRoot);
    }
  }
  if (profile.requireBackward && ![...files.keys()].some(name => name.startsWith("include/mom-cxx/backward/"))) {
    throw new Error("GNU header package set lacks backward headers");
  }
  return {files, bytes};
}

/** Stage only package-owned headers. Symlinks are materialized only when their resolved
 * regular-file target is inside an explicitly selected root. @param {string} target
 * @param {string} output @param {Package[]} records @param {{includeRoot?:string,sysroot?:string}} [options] @returns {GnuHeadersResult} */
function prepareGnuHeaders(target, output, records, options = {}) {
  const sysroot = options.sysroot || "/";
  if (!path.isAbsolute(sysroot) || path.resolve(sysroot) !== sysroot || fs.realpathSync(sysroot) !== sysroot ||
      !fs.lstatSync(sysroot).isDirectory()) {throw new Error("GNU package sysroot must be canonical");}
  const roots = target === linuxTarget ? linuxRoots(options.includeRoot || "/usr/include") : target === windowsTarget ? windowsRoots(sysroot) : null;
  if (!roots) {throw new Error("Unsupported GNU header target: " + target);}
  validatePackages(records, target);
  const sourceRoots = validateRoots(roots);
  validateOutput(output, sourceRoots);
  const {files, bytes} = collectFiles(records, roots, sysroot, target);
  fs.mkdirSync(output, {mode: 0o700});
  const ordered = [...files.values()].sort((a, b) => a.virtual < b.virtual ? -1 : a.virtual > b.virtual ? 1 : 0);
  for (const row of ordered) {
    const destination = path.join(output, row.virtual);
    fs.mkdirSync(path.dirname(destination), {recursive: true});
    fs.copyFileSync(row.source, destination, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(destination, 0o644);
    if (fileHash(destination) !== row.sha256 || fileHash(row.source) !== row.sha256) {throw new Error("GNU header changed during staging");}
  }
  const targetHeaders = ordered.map(row => path.join(output, row.virtual));
  fs.writeFileSync(path.join(output, "target-headers.json"), JSON.stringify(targetHeaders, null, 2) + "\n", {flag: "wx"});
  const result = {version: 1, target, root: output,
    packages: records.map(({name, version, architecture}) => ({name, version, architecture})),
    files: ordered, bytes, producerSha256: fileHash(__filename), buildOnly: true, compiled: false, executed: false, publicationQualified: false};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  return verifyGnuHeaders(output, target);
}

/** Preserve the existing Linux producer API. @param {string} includeRoot @param {string} output @param {Package[]} records */
function prepareLinuxGnuHeaders(includeRoot, output, records) {
  if (!path.isAbsolute(includeRoot) || path.resolve(includeRoot) !== includeRoot || fs.realpathSync(includeRoot) !== includeRoot ||
      !fs.lstatSync(includeRoot).isDirectory()) {throw new Error("GNU header include root must be canonical");}
  return prepareGnuHeaders(linuxTarget, output, records, {includeRoot});
}

/** @param {string} output @param {string} target @returns {GnuHeadersResult} */
function verifyGnuHeaders(output, target) {
  const profile = profileFor(target);
  const record = readBuildJson(path.join(output, "result.json")), targets = readBuildJson(path.join(output, "target-headers.json"));
  if (fs.realpathSync(output) !== output || !fs.lstatSync(output).isDirectory() || record.version !== 1 || record.root !== output ||
      record.target !== target || record.producerSha256 !== fileHash(__filename) || record.buildOnly !== true ||
      record.compiled !== false || record.executed !== false || record.publicationQualified !== false || !Array.isArray(record.files) ||
      !record.files.length || record.files.length > maxFiles || !Array.isArray(record.packages) || !Array.isArray(targets) ||
      targets.length !== record.files.length) {throw new Error("Malformed GNU target-header handoff");}
  validatePackages(record.packages.map((/** @type {Package} */ row) => ({...row, files: []})), target);
  let bytes = 0; const names = new Set();
  for (const [index, row] of record.files.entries()) {
    const expectedOwner = profile.packageOwnersByRoot && Object.entries(profile.packageOwnersByRoot)
      .find(([virtualRoot]) => row.virtual.startsWith(virtualRoot + "/"))?.[1];
    const expectedResolvedPackage = profile.resolvedPackagesByRoot && Object.entries(profile.resolvedPackagesByRoot)
      .find(([virtualRoot]) => row.virtual.startsWith(virtualRoot + "/"))?.[1];
    if (!validName(row.virtual) || !profile.virtualRoots.some(virtual => row.virtual.startsWith(virtual + "/")) ||
        names.has(row.virtual) || !profile.packages.some(item => item.name === row.package) ||
        (expectedOwner !== undefined && row.package !== expectedOwner) ||
        (expectedResolvedPackage !== undefined && row.resolvedPackage !== expectedResolvedPackage) ||
        (expectedResolvedPackage === undefined && row.resolvedPackage !== undefined) ||
        (row.resolvedPackage !== undefined && !profile.packages.some(item => item.name === row.resolvedPackage)) ||
        !/^[a-f0-9]{64}$/.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 0 ||
        (bytes += row.bytes) > profile.maxBytes) {throw new Error("Invalid GNU target-header inventory");}
    names.add(row.virtual);
    const file = path.join(output, row.virtual);
    if (targets[index] !== file || fs.realpathSync(file) !== file || !fs.lstatSync(file).isFile() ||
        fs.statSync(file).size !== row.bytes || fileHash(file) !== row.sha256) {throw new Error("GNU target-header content or path changed");}
  }
  for (const virtualRoot of profile.virtualRoots) {
    if (![...names].some(name => name.startsWith(virtualRoot + "/"))) {throw new Error("GNU target-header inventory lacks a required root");}
  }
  if (profile.requireBackward && ![...names].some(name => name.startsWith("include/mom-cxx/backward/"))) {
    throw new Error("GNU target-header inventory lacks backward compatibility headers");
  }
  if (!Number.isSafeInteger(record.bytes) || bytes !== record.bytes) {throw new Error("GNU target-header byte count changed");}
  return record;
}

/** @param {string} output @returns {any} Legacy Linux reader keeps its existing dynamic receipt shape. */
function verifyLinuxGnuHeaders(output) {return verifyGnuHeaders(output, linuxTarget);}
/** @param {string} sysroot @param {string} output @param {Package[]} records */
function prepareWindowsGnuHeaders(sysroot, output, records) {
  return prepareGnuHeaders(windowsTarget, output, records, {sysroot});
}
/** @param {string} output @returns {GnuHeadersResult} */
function verifyWindowsGnuHeaders(output) {return verifyGnuHeaders(output, windowsTarget);}

/** Query only the target's installed packages. No filesystem crawl. @param {string} [target] @returns {Package[]} */
function installedPackages(target = linuxTarget) {
  const specs = profileFor(target).packages;
  return specs.map(({name}) => {
    const query = (/** @type {string[]} */ args) => {
      const result = spawnSync("/usr/bin/dpkg-query", args, {encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20, env: {PATH: "/usr/bin:/bin", LC_ALL: "C"}});
      if (result.error || result.signal || result.status !== 0) {throw new Error("GNU package query failed: " + name, {cause: result.error});}
      return result.stdout.trim();
    };
    // dpkg's fields are not JavaScript interpolation.
    // eslint-disable-next-line no-template-curly-in-string
    const [status, version, architecture] = query(["-W", "-f=${db:Status-Status}\t${Version}\t${Architecture}", name]).split("\t");
    if (status !== "installed" || !version || !architecture) {throw new Error("GNU package is not installed: " + name);}
    return {name, version, architecture, files: query(["-L", name]).split("\n")};
  });
}

/** Return only the two package-owned Debian doc-directory aliases admitted by this helper.
 * @param {string} packageName */
function packageCopyrightAlias(packageName) {
  return Object.hasOwn(copyrightAliases, packageName) ? copyrightAliases[packageName] : undefined;
}

/** Retain a bounded package-owned copyright body without substituting another package's terms.
 * @param {{root:string}} state @param {string} packageName @param {string} packageVersion
 * @returns {{package:string,version:string,sourcePath:string,path:string,sha256:string,resolvedSourcePath?:string,ownershipPath?:string}} */
function capturePackageCopyright(state, packageName, packageVersion) {
  if (!/^[a-z0-9][a-z0-9+.-]{1,100}$/.test(packageName) || !/^[A-Za-z0-9:.+~-]{1,128}$/.test(packageVersion)) {
    throw new Error("Invalid package copyright identity");
  }
  const sourcePath = "/usr/share/doc/" + packageName + "/copyright", destination = path.join(state.root, "licenses", packageName, "copyright"),
    real = fs.realpathSync(sourcePath), sourceStat = fs.statSync(real);
  if (!real.startsWith("/usr/share/doc/") || !sourceStat.isFile() || sourceStat.size < 1 || sourceStat.size > (2 << 20)) {
    throw new Error("Installed package copyright file is absent, escaped or excessive");
  }
  const ownershipPath = fs.lstatSync(path.dirname(sourcePath)).isSymbolicLink() ? path.dirname(sourcePath) : sourcePath;
  const owner = spawnSync("/usr/bin/dpkg-query", ["-S", ownershipPath], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (owner.error || owner.signal || owner.status !== 0 ||
      !owner.stdout.trim().split(/\r?\n/).some(line => line === packageName + ": " + ownershipPath ||
        line === packageName + ":amd64: " + ownershipPath)) {
    throw new Error("Copyright path is not owned by the selected installed package: " + packageName, {cause: owner.error});
  }
  fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o755});
  fs.copyFileSync(sourcePath, destination, fs.constants.COPYFILE_EXCL);
  const sha256 = fileHash(sourcePath);
  if (fileHash(destination) !== sha256) {throw new Error("Copied copyright data differs from its installed package source");}
  return {package: packageName, version: packageVersion, sourcePath, path: destination, sha256,
    ...(real !== sourcePath ? {resolvedSourcePath: real, ownershipPath} : {})};
}

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {return false;}
  const actual = Object.keys(value).sort(), expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

/** Re-admit a transported target-specific header inventory and its exact package terms.
 * @param {{root:string,target?:"x86_64-unknown-linux-gnu" | "x86_64-w64-windows-gnu"}} state
 * @param {"x86_64-unknown-linux-gnu" | "x86_64-w64-windows-gnu"} [target]
 * @returns {{headers:ReturnType<typeof verifyGnuHeaders>,notices:Array<{package:string,version:string,sourcePath:string,path:string,sha256:string,resolvedSourcePath?:string,ownershipPath?:string}>}} */
function auditGnuHeaders(state, target = state.target || linuxTarget) {
  const headers = verifyGnuHeaders(state.root, target), copyrightFile = path.join(state.root, "copyrights.json"), notices = readBuildJson(copyrightFile);
  if (!fs.lstatSync(copyrightFile).isFile() || fs.realpathSync(copyrightFile) !== copyrightFile ||
      !Array.isArray(notices) || notices.length !== headers.packages.length) {throw new Error("GNU copyright handoff is incomplete");}
  for (const [index, notice] of notices.entries()) {
    const selected = headers.packages[index];
    if (!selected) {throw new Error("GNU copyright package row is missing");}
    const sourcePath = "/usr/share/doc/" + selected.name + "/copyright",
      destination = path.join(state.root, "licenses", selected.name, "copyright"), alias = notice?.resolvedSourcePath !== undefined,
      aliasSpec = packageCopyrightAlias(selected.name), keys = ["package", "version", "sourcePath", "path", "sha256", ...(alias ? ["resolvedSourcePath", "ownershipPath"] : [])];
    if (!hasExactKeys(notice, keys) || notice.package !== selected.name || notice.version !== selected.version || notice.sourcePath !== sourcePath ||
        notice.path !== destination || !/^[a-f0-9]{64}$/.test(notice.sha256 || "") || !fs.lstatSync(destination).isFile() ||
        fs.realpathSync(destination) !== destination || fs.statSync(destination).size > (2 << 20) || fileHash(destination) !== notice.sha256 ||
        alias && (!aliasSpec || notice.resolvedSourcePath !== aliasSpec.resolvedSourcePath || notice.ownershipPath !== aliasSpec.ownershipPath)) {
      throw new Error("GNU copyright content, package or path changed");
    }
  }
  return {headers, notices};
}

module.exports = {prepareLinuxGnuHeaders, verifyLinuxGnuHeaders, prepareWindowsGnuHeaders, verifyWindowsGnuHeaders,
  prepareGnuHeaders, verifyGnuHeaders, virtualHeader, installedPackages, packageCopyrightAlias, capturePackageCopyright, auditGnuHeaders};
if (require.main === module) {
  try {
    const [mode, output, target, extra] = process.argv.slice(2);
    const selectedTarget = target || linuxTarget;
    if (process.platform !== "linux" || process.arch !== "x64" || process.env["MOM_AOT_BUILD_ONLY"] !== "1" || !output ||
        extra || !["prepare", "verify"].includes(mode || "") || (target && !Object.hasOwn(profiles, target))) {
      throw new Error("Usage (Linux x64, MOM_AOT_BUILD_ONLY=1): prepare-native-gnu-headers.js prepare|verify /absolute/output [x86_64-unknown-linux-gnu|x86_64-w64-windows-gnu]");
    }
    const result = mode === "prepare" ? prepareGnuHeaders(selectedTarget, output, installedPackages(selectedTarget),
      selectedTarget === linuxTarget ? {includeRoot: "/usr/include"} : {sysroot: "/"}) : verifyGnuHeaders(output, selectedTarget);
    console.log(JSON.stringify({target: selectedTarget, files: result.files.length, bytes: result.bytes, buildOnly: true, publicationQualified: false}));
  } catch (error) {console.error(error); process.exitCode = 1;}
}
