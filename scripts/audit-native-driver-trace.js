"use strict";

const fs = require("node:fs"), path = require("node:path");
const {inspectDrivers, systemFile} = require("./inspect-native-driver-deps");
const {auditNativeTrace, auditNativeHeaders, isLibraryPath} = require("./native-audit");

/** Literal ELF searches retain dot components: lexical normalization could hide a symlink.
 * The inspector independently validates every search root before this helper is used.
 * @param {string[]} search @param {string} origin @param {string} name */
function librarySearches(search, origin, name) {
  return search.flatMap(value => value.split(":"))
    .flatMap(value => {
      const directory = value.replace(/\$\{ORIGIN\}|\$ORIGIN(?=\/|$)/g, origin);
      const prefix = directory ? directory.replace(/\/$/, "") + "/" : "";
      return [prefix + name, ...[4, 3, 2].map(level => prefix + `glibc-hwcaps/x86-64-v${level}/` + name)];
    });
}

/** Failed lookups include dlopen entrypoints, ordinary loader paths and the literal alias
 * used for $ORIGIN (e.g. /lib versus /usr/lib). Only metadata-derived names are considered;
 * successful opens still require the independently inspected, root-owned ELF closure.
 * @param {{defaultSearch: string[], roots: {requested: string}[], lateRoots: {requested: string}[],
 * libraries: {file: string, soname: string | null, needed: string[], search: string[]}[]}} closure */
function driverSearches(closure) {
  const misses = new Set(), names = new Set([...closure.roots, ...closure.lateRoots].map(root => path.basename(root.requested)));
  for (const library of closure.libraries) {
    if (library.soname) {names.add(library.soname);}
    for (const name of library.needed) {
      names.add(name);
      for (const origin of new Set([path.dirname(library.file), ...closure.defaultSearch])) {
        for (const file of librarySearches(library.search, origin, name)) {misses.add(file);}
      }
    }
  }
  for (const name of names) {for (const file of librarySearches(closure.defaultSearch, "/", name)) {misses.add(file);}}
  return misses;
}

/** A complete process trace is required, not an asserted cwd. Track Linux clone/fork cwd
 * inheritance, including CLONE_FS sharing, and reject descriptor-based/unknown cwd changes.
 * @param {string} launchDirectory */
function directoryTracker(launchDirectory) {
  if (!path.posix.isAbsolute(launchDirectory) || path.posix.normalize(launchDirectory) !== launchDirectory) {
    throw new Error("Driver trace requires its exact absolute launch directory");
  }
  /** @type {Map<string, {cwd: string}>} */
  const processes = new Map();
  let launches = 0;
  return (/** @type {string} */ line) => {
    const row = line.match(/^(\d+)\s+(.*)$/);
    if (!row) {throw new Error("Missing process identity in driver trace");}
    const pid = row[1] || "", call = row[2] || "";
    if (/^(?:clone3?|fork|vfork|chdir|fchdir)\(/.test(call) && /unfinished|resumed/.test(call)) {
      throw new Error("Incomplete driver process/cwd trace");
    }
    if (call.startsWith("execve(")) {
      if (!launches) {processes.set(pid, {cwd: launchDirectory});}
      const state = processes.get(pid);
      const environment = call.match(/\], \[((?:"(?:[^"\\]|\\.)*"(?:, )?)*)\]\)\s+= 0$/)?.[1];
      if (!state || !/\s= 0$/.test(call) || (launches && (state.cwd !== "/" ||
          !environment || !/(?:^|, )"MOM_CLUSTER_WORKER=1"(?:, |$)/.test(environment)))) {
        throw new Error("Native worker exec lacks proven neutral cwd");
      }
      launches++;
    }
    const state = processes.get(pid);
    if (!state) {throw new Error("Untraced process ancestry in driver trace");}
    if (call.startsWith("fchdir(")) {throw new Error("Descriptor-based driver cwd change");}
    if (call.startsWith("chdir(")) {
      if (!/^chdir\("\/"\)\s+= 0$/.test(call)) {throw new Error("Unexpected driver cwd change");}
      state.cwd = "/";
    }
    const clone = call.match(/^(?:clone3?|fork|vfork)\(.*\)\s+= ([1-9][0-9]*)$/);
    if (clone) {
      if (processes.has(clone[1] || "")) {throw new Error("Reused process identity in driver trace");}
      processes.set(clone[1] || "", /\bCLONE_FS\b/.test(call) ? state : {...state});
    }
    return state.cwd;
  };
}

/** Qualify driver-transitive libraries separately from application imports. Every successful
 * library open must resolve to the freshly hashed, root-owned ELF closure; every tolerated
 * miss must be an exact metadata-derived search (or the named absent Intel sysman helper).
 * Non-library activity still passes the unchanged launch/header/JIT audits. No SO-name blanket
 * exemption, caller-provided allowlist, external compiler, or application import is admitted.
 * @param {string} trace @param {string} executable @param {string[]} roots
 * @param {{lateLibraries?: string[], launchDirectory: string, cacheRoots?: string[], processes?: number}} options */
function auditDriverTrace(trace, executable, roots, options) {
  if (!trace || trace.length > (64 << 20)) {throw new Error("Missing or excessive driver trace");}
  const closure = inspectDrivers(roots, {lateLibraries: options.lateLibraries ?? [], workingDirectory: "/"});
  const expected = new Set(closure.libraries.map(library => library.file)), misses = driverSearches(closure);
  // This optional Intel management helper is absent on the qualified host. Only failed OS
  // searches are allowed; installing it requires adding its actual driver closure first.
  if (roots.some(file => /\/libze_intel_gpu(?:_legacy1)?\.so\.1$/.test(file))) {
    for (const file of librarySearches(closure.defaultSearch, "/", "libigdml.so.1")) {misses.add(file);}
  }
  // Intel OpenCL lazily probes VA media sharing, which mining does not require. Permit
  // only missing OS searches in the audited filesystem; a successful load still needs
  // its own inspected closure. Run this audit in the recorded runtime's filesystem view.
  if (roots.some(file => /\/libigdrcl(?:_legacy1)?\.so$/.test(file))) {
    for (const file of librarySearches(closure.defaultSearch, "/", "libva.so.2")) {misses.add(file);}
  }
  // Qualification deliberately disables inherited library search; libc still probes that
  // nonexistent directory before its OS cache. This is not an application-library exception.
  for (const name of ["libm.so.6", "libc.so.6"]) {
    for (const file of librarySearches(["/unavailable"], "/", name)) {misses.add(file);}
  }
  const directory = directoryTracker(options.launchDirectory), remainder = [], opened = new Set(), resolvedOpened = new Set();
  let searches = 0;
  for (const line of trace.trimEnd().split("\n")) {
    const cwd = directory(line), match = line.match(/^\d+\s+(?:openat2?|open|creat)\([^\n]*?"([^"\n]+)"/);
    const file = match?.[1];
    if (!file || !isLibraryPath(file)) {remainder.push(line); continue;}
    if (file.includes("\\") || !/\bO_RDONLY\b/.test(line) || /\bO_(?:WRONLY|RDWR|CREAT|TRUNC)\b/.test(line)) {
      throw new Error("Malformed or writable driver library open: " + file);
    }
    const success = /\s= [0-9]+$/.test(line);
    if (success) {
      const resolved = path.posix.isAbsolute(file) ? systemFile(file) : null;
      if (!resolved || !expected.has(resolved)) {
        throw new Error("Library is outside the inspected system-driver closure: " + file);
      }
      if (!/^lib[cm]\.so\./.test(path.basename(file)) && cwd !== "/") {
        throw new Error("Driver library loaded before proven neutral cwd: " + file);
      }
      opened.add(file);
      resolvedOpened.add(resolved);
    } else {
      if (!/\s= -1 ENOENT(?: |$)/.test(line) || !misses.has(file) || fs.existsSync(path.resolve(cwd, file)) ||
          (!file.startsWith("/unavailable/") && cwd !== "/")) {
        throw new Error("Unproven driver library search: " + file);
      }
      searches++;
    }
  }
  const rest = remainder.join("\n");
  auditNativeTrace(rest, executable, options.processes ?? 2);
  auditNativeHeaders(rest, [], options.cacheRoots ?? []);
  if (!closure.roots.some(root => resolvedOpened.has(root.resolved))) {
    throw new Error("No inspected GPU driver was actually opened");
  }
  return {closure, opened: [...opened].sort(), searches,
    scope: "Exact observed system-driver library closure, cwd/process inheritance and non-library source/cache/launch audit. Not full algorithm, performance, OS portability or release qualification."};
}

module.exports = {librarySearches, driverSearches, directoryTracker, auditDriverTrace};
