"use strict";

const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {fileHash} = require("./build-source-inputs");
const {coffSectionPayloads, rewriteCoffDirectives, stageCoffDirectives} = require("./native-compiler-isolation");
const archiveBytes = 256 << 20, memberBytes = 64 << 20, diagnosticBytes = 256 << 20;
/** @param {string} text */
function hasControls(text) {return Array.from(text).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);}

/** Read a bounded archive/object region exactly; never trust member lengths as allocations.
 * @param {number} fd @param {number} offset @param {number} size */
function read(fd, offset, size) {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 || size > memberBytes || offset + size > fs.fstatSync(fd).size) {
    throw new Error("COFF archive read exceeds bounds");
  }
  const data = Buffer.alloc(size);
  for (let done = 0; done < size;) {
    const bytes = fs.readSync(fd, data, done, size - done, offset + done);
    if (!bytes) {throw new Error("Truncated COFF archive");} done += bytes;
  }
  return data;
}

/** Enumerate bounded regular ar containers without trusting their index or extracting paths.
 * @param {number} fd */
function* rawMembers(fd) {
  const stat = fs.fstatSync(fd), size = stat.size;
  if (!stat.isFile() || size > archiveBytes || read(fd, 0, 8).toString() !== "!<arch>\n") {throw new Error("Expected a bounded regular COFF archive");}
  for (let offset = 8; offset < size;) {
    const header = read(fd, offset, 60), token = header.toString("ascii", 0, 16).trim(), text = header.toString("ascii", 48, 58).trim();
    const bytes = Number(text), start = offset + 60;
    if (!/^\d+$/.test(text) || header.toString("ascii", 58) !== "`\n" || !Number.isSafeInteger(bytes) || bytes > memberBytes || start + bytes + bytes % 2 > size) {
      throw new Error("Malformed COFF archive member boundary");
    }
    yield {token, offset: start, bytes}; offset = start + bytes + bytes % 2;
  }
}

/** Resolve member names only as evidence. Thin/BSD formats remain unsupported.
 * @param {number} fd */
function* members(fd) {
  let names = Buffer.alloc(0), count = 0;
  for (const {token, offset: start, bytes} of rawMembers(fd)) {
    if (token === "//") {names = read(fd, start, bytes);} else if (!["/", "/SYM64/"].includes(token)) {
      let name = token.replace(/\/$/, "");
      if (/^\/\d+$/.test(token)) {
        const index = Number(token.slice(1)); let end = index;
        if (!Number.isSafeInteger(index) || index >= names.length) {throw new Error("Invalid COFF archive long name");}
        while (end < names.length && names[end] !== 0 && names[end] !== 10) {end++;}
        if (end === names.length) {throw new Error("Unterminated COFF archive long name");}
        name = names.toString("utf8", index, end).replace(/\/$/, "");
      }
      if (!name || name.length > 4096 || hasControls(name) || token.startsWith("#1/") || ++count > 100000) {throw new Error("Unsupported COFF archive member");}
      yield {name, offset: start, bytes};
    }
  }
  if (!count) {throw new Error("COFF archive contains no objects");}
}

/** Hash one bounded file region without retaining the region in memory.
 * @param {number} fd @param {number} offset @param {number} bytes @param {Buffer} buffer */
function hashRegion(fd, offset, bytes, buffer) {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(bytes) || offset < 0 || bytes < 0 || offset + bytes > fs.fstatSync(fd).size) {
    throw new Error("COFF archive hash exceeds bounds");
  }
  const hash = createHash("sha256");
  for (let position = 0; position < bytes;) {
    const wanted = Math.min(buffer.length, bytes - position), size = fs.readSync(fd, buffer, 0, wanted, offset + position);
    if (!size) {throw new Error("Truncated COFF archive hash input");}
    hash.update(buffer.subarray(0, size)); position += size;
  }
  return hash.digest("hex");
}

/** Reject aliases and symlinked inputs, then compare the opened regular file with its path.
 * @param {string} file @param {"file" | "directory"} kind */
function canonicalInput(file, kind) {
  if (typeof file !== "string" || !file || file.length > 4096 || !path.isAbsolute(file) || hasControls(file)) {throw new Error("COFF archive inputs must be canonical absolute paths");}
  const resolved = path.resolve(file), real = fs.realpathSync(file), lstat = fs.lstatSync(file);
  const samePath = process.platform === "win32" ? resolved.toLowerCase() === real.toLowerCase() : resolved === real;
  const validType = kind === "file" ? lstat.isFile() : lstat.isDirectory();
  if (!samePath || !validType) {throw new Error("COFF archive input is not a canonical regular " + kind + ": " + file);}
  return real;
}

/** Freeze regular-file identity and timestamps around a read-only audit.
 * @param {fs.BigIntStats} before @param {fs.BigIntStats} after */
function sameFileState(before, after) {
  return before.dev === after.dev && before.ino === after.ino && before.size === after.size && before.mtimeNs === after.mtimeNs && before.ctimeNs === after.ctimeNs;
}

/** @param {string} file @param {number} fd @param {"file" | "directory"} kind */
function verifyOpenedPath(file, fd, kind) {
  const real = canonicalInput(file, kind), opened = fs.fstatSync(fd, {bigint: true}), named = fs.statSync(real, {bigint: true});
  if (!sameFileState(opened, named)) {throw new Error("COFF archive input path changed during audit: " + file);}
}

/** Verify that a COFF archive is exactly the compiler's expected build-relative object set.
 * Names are evidence only: this never extracts archive members or follows their paths.
 * @param {string} archive @param {string} buildRoot @param {string[]} expectedObjects
 * @returns {{members: number, objects: Array<{path: string, sha256: string, bytes: number}>, archiveSha256: string}} */
function auditCoffArchiveInputs(archive, buildRoot, expectedObjects) {
  if (!Array.isArray(expectedObjects) || !expectedObjects.length || expectedObjects.length > 100000) {throw new Error("Invalid expected COFF archive object set");}
  const root = canonicalInput(buildRoot, "directory"), archivePath = canonicalInput(archive, "file"), expected = new Map(), buffer = Buffer.allocUnsafe(64 << 10);
  for (const name of expectedObjects) {
    if (typeof name !== "string" || !name || name.length > 4096 || /[\\:\0]/.test(name) || hasControls(name) ||
        path.posix.isAbsolute(name) || name.split("/").some(part => !part || part === "." || part === "..") || !/\.obj$/i.test(name) ||
        path.posix.normalize(name) !== name || expected.has(name)) {throw new Error("Invalid or duplicate expected COFF archive object path");}
    const file = path.join(root, ...name.split("/")), resolved = canonicalInput(file, "file"), relative = path.relative(root, resolved);
    if (!relative || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) || relative.split(path.sep).join("/") !== name) {
      throw new Error("Expected COFF object is not inside the canonical build root: " + name);
    }
    const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0));
    try {
      const before = fs.fstatSync(fd, {bigint: true});
      if (!before.isFile() || before.size > BigInt(memberBytes)) {throw new Error("Expected bounded regular COFF object: " + name);}
      verifyOpenedPath(file, fd, "file");
      const payloads = coffSectionPayloads(file);
      verifyOpenedPath(file, fd, "file");
      const sha256 = hashRegion(fd, 0, Number(before.size), buffer), after = fs.fstatSync(fd, {bigint: true});
      if (!sameFileState(before, after)) {throw new Error("Expected COFF object changed during archive audit: " + name);}
      expected.set(name, {path: name, sha256, bytes: Number(before.size)});
      // Keep the parser result live through validation; its bounded walk rejects malformed
      // x64 COFF and bigobj section tables before any archive payload can be accepted.
      if (!payloads.length) {throw new Error("Expected COFF object has no sections: " + name);}
    } finally {fs.closeSync(fd);}
  }

  const fd = fs.openSync(archivePath, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0));
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || before.size > BigInt(archiveBytes)) {throw new Error("Expected a bounded regular COFF archive");}
    verifyOpenedPath(archivePath, fd, "file");
    const seen = new Set(); let count = 0;
    for (const member of members(fd)) {
      const name = member.name;
      if (!name || name.length > 4096 || /[\\:\0]/.test(name) || hasControls(name) || path.posix.isAbsolute(name) ||
          name.split("/").some(part => !part || part === "." || part === "..") || path.posix.normalize(name) !== name || !/\.obj$/i.test(name)) {
        throw new Error("Unsafe COFF archive member path: " + name);
      }
      if (seen.has(name)) {throw new Error("Duplicate COFF archive member: " + name);}
      const object = expected.get(name);
      if (!object) {throw new Error("Unexpected COFF archive member: " + name);}
      if (member.bytes !== object.bytes || hashRegion(fd, member.offset, member.bytes, buffer) !== object.sha256) {
        throw new Error("COFF archive member differs from compiled object: " + name);
      }
      seen.add(name); count++;
      if (count > 100000) {throw new Error("COFF archive member budget exceeded");}
    }
    if (seen.size !== expected.size) {throw new Error("COFF archive is missing expected object members");}
    const digest = hashRegion(fd, 0, Number(before.size), buffer), after = fs.fstatSync(fd, {bigint: true});
    verifyOpenedPath(archivePath, fd, "file");
    if (!sameFileState(before, after)) {throw new Error("COFF archive changed during audit");}
    return {members: count, objects: [...expected.values()], archiveSha256: digest};
  } finally {fs.closeSync(fd);}
}

/** Digest actual symbol order/values, auxiliaries, section selection and raw relocations.
 * Relocations retain their symbol-table indices: equality proves that they still resolve to
 * the same definition after applying only the admitted name map. File offsets may change.
 * @param {number} fd @param {Map<string, string>} names */
function metadata(fd, names) {
  const header = read(fd, 0, 56), big = header.readUInt16LE() === 0 && header.readUInt16LE(2) === 0xffff;
  const start = big ? 56 : 20, sections = big ? header.readUInt32LE(44) : header.readUInt16LE(2), symbols = header.readUInt32LE(big ? 48 : 8), count = header.readUInt32LE(big ? 52 : 12), width = big ? 20 : 18;
  if (!sections || sections > 100000 || !count || count > 2000000 || symbols < start + sections * 40 || symbols + count * width + 4 > fs.fstatSync(fd).size) {throw new Error("Invalid COFF member metadata bounds");}
  const stringsAt = symbols + count * width, stringBytes = read(fd, stringsAt, 4).readUInt32LE();
  if (stringBytes < 4) {throw new Error("Invalid COFF string table");}
  const strings = read(fd, stringsAt, stringBytes), hash = createHash("sha256");
  /** @param {number} offset */
  function longName(offset) {
    const end = strings.indexOf(0, offset);
    if (offset < 4 || end < offset || end - offset > 16384) {throw new Error("Invalid COFF symbol string");}
    return strings.toString("utf8", offset, end);
  }
  const table = Array.from({length: sections}, (_, index) => read(fd, start + index * 40, 40));
  const sectionNames = table.map(section => {
    const end = section.subarray(0, 8).indexOf(0), name = section.toString("utf8", 0, end < 0 ? 8 : end);
    return /^\/\d+$/.test(name) ? longName(Number(name.slice(1))) : name;
  });
  const directiveIndex = sectionNames.indexOf(".drectve"), directiveSection = table[directiveIndex];
  if (sectionNames.filter(name => name === ".drectve").length > 1 || (directiveSection && directiveSection.readUInt16LE(32))) {throw new Error("Unsupported archive linker directive section");}
  const directiveText = directiveSection ? read(fd, directiveSection.readUInt32LE(20), directiveSection.readUInt32LE(16)).toString("utf8") : "";
  const directive = {number: directiveIndex + 1, ...rewriteCoffDirectives(directiveText, names)};
  for (let index = 0; index < count;) {
    const record = read(fd, symbols + index * width, width), aux = record[width - 1] || 0, shortEnd = record.subarray(0, 8).indexOf(0);
    const name = record.readUInt32LE() === 0 ? longName(record.readUInt32LE(4)) : record.toString("utf8", 0, shortEnd < 0 ? 8 : shortEnd);
    hash.update(JSON.stringify(names.get(name) || name)); hash.update(record.subarray(8));
    if (index + aux >= count) {throw new Error("Invalid COFF auxiliary symbol count");}
    if (aux) {
      const data = read(fd, symbols + (index + 1) * width, aux * width), section = big ? record.readInt32LE(12) : record.readInt16LE(12);
      // The section-definition auxiliary repeats the directive length. Normalize exactly
      // that field to the admitted replacement; COMDAT/unwind auxiliaries remain literal.
      if (directive.rewritten && name === ".drectve" && section === directive.number && record[width - 2] === 3 && aux === 1) {data.writeUInt32LE(Buffer.byteLength(directive.text));}
      hash.update(data);
    } index += 1 + aux;
  }
  for (let index = 0; index < sections; index++) {
    const section = table[index], name = sectionNames[index];
    if (!section || name === undefined) {throw new Error("Missing COFF section metadata");}
    if (index === directiveIndex && directive.rewritten) {section.writeUInt32LE(Buffer.byteLength(directive.text), 16);}
    hash.update(JSON.stringify(name)); hash.update(section.subarray(8, 20)); hash.update(section.subarray(32));
    const reloc = section.readUInt32LE(24), overflow = section.readUInt32LE(36) & 0x1000000;
    const relocations = overflow ? read(fd, reloc, 4).readUInt32LE() : section.readUInt16LE(32);
    if (relocations > 1000000 || (relocations && !reloc)) {throw new Error("Invalid COFF relocation bounds");}
    if (relocations) {hash.update(read(fd, reloc, relocations * 10));}
  }
  return {sha256: hash.digest("hex"), directive};
}

/** Independently check every transformed member without extracting an archive tree. Only two
 * private scratch files are reused, keeping memory and disk growth bounded by one object.
 * @param {string} input @param {string} output @param {Map<string, string>} names @param {string} scratch */
function verifyCoffArchive(input, output, names, scratch) {
  const descriptors = [], created = [], files = ["before.obj", "after.obj"].map(name => path.join(scratch, name));
  const hash = createHash("sha256"), empty = new Map(); let count = 0;
  try {
    for (const file of [input, output]) {descriptors.push(fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK));}
    const fds = descriptors.slice();
    for (const file of files) {descriptors.push(fs.openSync(file, "wx+")); created.push(file);}
    const iterators = fds.map(fd => members(fd));
    for (;;) {
      const next = iterators.map(iterator => iterator.next()), a = next[0], b = next[1];
      if (!a || !b) {throw new Error("Missing archive iterator");}
      if (a.done && b.done) {break;}
      if (a.done || b.done || a.value.name !== b.value.name) {throw new Error("COFF archive member order or identity changed");}
      for (const [index, member] of [a.value, b.value].entries()) {
        const fd = descriptors[index + 2];
        if (fd === undefined || fds[index] === undefined) {throw new Error("Missing archive descriptor");}
        fs.ftruncateSync(fd, 0); const data = read(fds[index], member.offset, member.bytes);
        if (fs.writeSync(fd, data, 0, data.length, 0) !== data.length) {throw new Error("Short COFF member staging write");}
      }
      const left = coffSectionPayloads(files[0] || ""), right = coffSectionPayloads(files[1] || "");
      const leftFd = descriptors[2], rightFd = descriptors[3];
      if (leftFd === undefined || rightFd === undefined) {throw new Error("Missing staged COFF member");}
      const before = metadata(leftFd, names), after = metadata(rightFd, empty);
      const directive = before.directive;
      if (directive.rewritten) {
        const payload = left[directive.number - 1]; if (!payload) {throw new Error("Missing directive payload");}
        payload.bytes = Buffer.byteLength(directive.text); payload.sha256 = createHash("sha256").update(directive.text).digest("hex");
      }
      if (JSON.stringify(left) !== JSON.stringify(right) || before.sha256 !== after.sha256 || directive.text !== after.directive.text) {throw new Error("COFF archive code/data/symbol/relocation preservation failed: " + a.value.name);}
      hash.update(JSON.stringify({name: a.value.name, payloads: left, metadata: before.sha256})); count++;
    }
  } finally {
    for (const fd of descriptors) {fs.closeSync(fd);}
    for (const file of created) {fs.unlinkSync(file);}
  }
  return {members: count, checkedMemberSha256: hash.digest("hex")};
}

/** Repair mapped linker text using the GPU object's section staging. Stream original member
 * containers (including duplicate names) into an empty-index scratch archive; LLVM regenerates
 * its index. Only member size fields change here, never names, ordering or object encoding.
 * @param {string} archive @param {Map<string, string>} names @param {string} scratch
 * @param {(args: string[], label: string) => string} update */
function rewriteArchiveDirectives(archive, names, scratch, update) {
  const fd = fs.openSync(archive, "r"), staged = path.join(scratch, "directives"); fs.mkdirSync(staged);
  const original = path.join(staged, "original.obj"), fixed = path.join(staged, "fixed.obj"), packed = path.join(staged, "archive.lib");
  const writer = fs.openSync(packed, "wx"); let changed = 0, linkerMembers = 0;
  try {
    fs.writeFileSync(writer, "!<arch>\n");
    for (const member of rawMembers(fd)) {
      const header = read(fd, member.offset - 60, 60); let data = read(fd, member.offset, member.bytes);
      if (["/", "/SYM64/"].includes(member.token)) {
        // Keep the format markers: two '/' members identify COFF's NUL-terminated name
        // table. Empty counts avoid retaining stale offsets; LLVM rebuilds the real index.
        if (++linkerMembers > 2) {throw new Error("Unsupported archive linker member count");}
        data = Buffer.alloc(linkerMembers === 2 || member.token === "/SYM64/" ? 8 : 4);
        header.write(String(data.length).padEnd(10), 48, "ascii");
      } else if (member.token !== "//") {
        fs.writeFileSync(original, data, {flag: "wx"}); const object = fs.openSync(original, "r"); let directive;
        try {directive = metadata(object, names).directive;} finally {fs.closeSync(object);}
        if (directive.rewritten) {
          stageCoffDirectives(original, fixed, directive.number, directive.text); data = fs.readFileSync(fixed); fs.unlinkSync(fixed); changed++;
          if (data.length > memberBytes) {throw new Error("Rewritten archive member exceeds bounds");}
          header.write(String(data.length).padEnd(10), 48, "ascii");
        }
        fs.unlinkSync(original);
      }
      fs.writeFileSync(writer, header); fs.writeFileSync(writer, data);
      if (data.length % 2) {fs.writeFileSync(writer, "\n");}
      if (fs.fstatSync(writer).size > archiveBytes) {throw new Error("Rewritten archive exceeds bounds");}
    }
  } finally {fs.closeSync(fd); fs.closeSync(writer);}
  // 's' alone skips an existing (here deliberately empty) index. A no-input replace writes
  // every original member back in order and forces a fresh, usable symbol index.
  if (changed) {update(["--format=coff", "rs", packed], "directives"); fs.renameSync(packed, archive);} else {fs.unlinkSync(packed);}
  fs.rmdirSync(staged); return changed;
}

/** Namespace only owned C++/LLVM definitions in a complete original compiler archive closure.
 * OS/CRT C imports and the explicit C bridge remain shared. LLVM objcopy rebuilds each archive
 * index; a streamed before/after audit proves that no code, data, COMDAT or unwind state changed.
 * This is build-time transformation, never runtime extraction or a replacement compiler list.
 * A narrow explicit symbol set also handles a private C helper in another static runtime;
 * all unselected names then stay unchanged. This never permits suppressing link collisions.
 * @param {{archives: string[], tools: {llvmObjcopy: string, llvmNm: string, llvmAr: string}, output: string, prefix: string, entryPoints: readonly string[], privateSymbols?: readonly string[]}} options */
function isolateCoffCompilerArchives({archives, tools, output, prefix, entryPoints, privateSymbols}) {
  const absolute = (/** @type {unknown} */ value) => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && !hasControls(value);
  if (!Array.isArray(archives) || !archives.length || archives.length > 256 || archives.some(file => !absolute(file)) ||
      !absolute(output) || fs.existsSync(output) || !fs.statSync(path.dirname(output), {throwIfNoEntry: false})?.isDirectory() ||
      typeof prefix !== "string" || hasControls(prefix) || !/^[A-Za-z_][A-Za-z0-9_]{0,63}_$/.test(prefix) || !Array.isArray(entryPoints) || !entryPoints.length || entryPoints.length > 1024 ||
      entryPoints.some(name => typeof name !== "string" || hasControls(name) || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) || name.startsWith(prefix)) || new Set(entryPoints).size !== entryPoints.length ||
      !tools || Object.keys(tools).sort().join(",") !== "llvmAr,llvmNm,llvmObjcopy" || !absolute(tools.llvmObjcopy) || !absolute(tools.llvmNm) || !absolute(tools.llvmAr)) {throw new Error("Invalid COFF compiler archive isolation options");}
  if (privateSymbols !== undefined && (!Array.isArray(privateSymbols) || !privateSymbols.length || privateSymbols.length > 1000000 ||
      privateSymbols.some(name => typeof name !== "string" || !name || name.length > 4096 || /\s/.test(name) || hasControls(name) || entryPoints.includes(name)) ||
      new Set(privateSymbols).size !== privateSymbols.length)) {throw new Error("Invalid explicit private compiler symbols");}
  const inputs = new Map(), toolsBefore = new Map(); let total = 0;
  for (const file of archives) {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > archiveBytes || !/\.(?:lib|a)$/i.test(file)) {throw new Error("Expected bounded regular compiler archives");}
    const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
    try {if (!fs.fstatSync(fd).isFile() || read(fd, 0, 8).toString() !== "!<arch>\n") {throw new Error("Expected bounded regular compiler archives");}} finally {fs.closeSync(fd);}
    if ((total += stat.size) > 2 ** 31) {throw new Error("COFF compiler archive closure exceeds bounds");}
    inputs.set(file, fileHash(file));
  }
  if (new Set(archives.map(file => fs.realpathSync(file))).size !== archives.length || new Set(archives.map(file => path.basename(file).toLowerCase())).size !== archives.length) {throw new Error("Compiler archives must have distinct paths and names");}
  for (const file of Object.values(tools)) {
    const stat = fs.statSync(file); if (!stat.isFile() || stat.size > archiveBytes) {throw new Error("LLVM tool exceeds bounds");}
    toolsBefore.set(file, fileHash(file));
  }
  const sourceHashes = Object.fromEntries([__filename, require.resolve("./native-compiler-isolation"), require.resolve("./build-source-inputs")].map(file => [file, fileHash(file)]));
  fs.mkdirSync(output); let logs = 0;
  /** @param {string} tool @param {string[]} args @param {string} label */
  function run(tool, args, label) {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({tool, args}) + "\n", {flag: "wx"});
    const result = spawnSync(tool, args, {encoding: "utf8", timeout: 120000, maxBuffer: 64 << 20}), text = (result.stdout || "") + (result.stderr || "");
    if ((logs += Buffer.byteLength(text)) > diagnosticBytes) {throw new Error("COFF compiler archive diagnostics exceed bounds");}
    fs.writeFileSync(path.join(output, label + ".log"), text, {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("COFF compiler archive tool failed: " + label + ": " + text.slice(0, 2048), {cause: result.error});}
    return result.stdout || "";
  }
  const definitions = new Set(), keep = new Set(entryPoints);
  for (const [index, file] of Object.values(tools).entries()) {if (!/\bLLVM\b/i.test(run(file, ["--version"], "tool-" + index))) {throw new Error("Compiler archive isolation requires LLVM tools");}}
  for (const [index, file] of archives.entries()) {
    for (const line of run(tools.llvmNm, ["-P", "-g", "--defined-only", file], "definitions-" + index).split(/\r?\n/)) {
      if (!line.trim() || /:\s*$/.test(line)) {continue;}
      const match = line.match(/^(\S+)\s+[A-Za-z]\s/), name = match?.[1];
      if (!name || name.length > 4096 || hasControls(name)) {throw new Error("Malformed COFF compiler definition");}
      definitions.add(name); if (definitions.size > 1000000) {throw new Error("COFF compiler symbol budget exceeded");}
    }
  }
  if (entryPoints.some(name => !definitions.has(name))) {throw new Error("Missing COFF compiler C bridge definition");}
  if (privateSymbols?.some(name => !definitions.has(name))) {throw new Error("Missing explicit private compiler symbol");}
  // A namespace substring misses MSVC operators, backreferences and LLVM's unnamespaced
  // internal classes. Isolate complete owned C++ definitions, not just '?...llvm...' spellings.
  const names = new Map((privateSymbols ? [...privateSymbols].sort() : [...definitions].sort().filter(name => !keep.has(name) && (name.startsWith("?") || /^LLVM[A-Z_]/.test(name) || name === "HandleAbort"))).map(name => [name, prefix + name]));
  if (!names.size || [...definitions].some(name => name.startsWith(prefix))) {throw new Error("Empty or already isolated COFF compiler graph");}
  const map = path.join(output, "rename.map");
  fs.writeFileSync(map, [...names].map(([from, to]) => from + " " + to).join("\n") + "\n", {flag: "wx"});
  const transformed = [];
  for (const [index, input] of archives.entries()) {
    if (fileHash(input) !== inputs.get(input)) {throw new Error("Compiler archive changed before transformation");}
    const selected = path.join(output, path.basename(input));
    run(tools.llvmObjcopy, ["--redefine-syms=" + map, input, selected], "isolate-" + index);
    const rewrittenMembers = rewriteArchiveDirectives(selected, names, output, (args, label) => run(tools.llvmAr, args, label + "-" + index));
    const sha256 = fileHash(selected), proof = verifyCoffArchive(input, selected, names, output);
    if (fileHash(input) !== inputs.get(input) || fileHash(selected) !== sha256) {throw new Error("Compiler archive changed during transformation audit");}
    transformed.push({input, inputSha256: inputs.get(input), output: selected, sha256, rewrittenMembers, ...proof});
  }
  for (const [file, hash] of [...inputs, ...toolsBefore, ...Object.entries(sourceHashes)]) {if (fileHash(file) !== hash) {throw new Error("Compiler archive/tool/source changed during isolation");}}
  const result = {version: 1, prefix, entryPoints, symbols: names.size, map, mapSha256: fileHash(map), transformed,
    toolHashes: Object.fromEntries(toolsBefore), sourceHashes, checks: {memberOrderPreserved: true, nonDirectivePayloadsPreserved: true, mappedDirectivesPreserved: true, symbolOrderPreserved: true, auxiliaryRecordsPreserved: true, relocationsPreserved: true},
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  return result;
}

module.exports = {isolateCoffCompilerArchives, verifyCoffArchive, auditCoffArchiveInputs};
