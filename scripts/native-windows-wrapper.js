"use strict";

// DLL-backed SYCL initializes before the executable's early image callback. Static linking
// reverses that assumption: run the callback after runtime constructors and route it through
// the existing per-compiler deferred gates. Never rewrite the kernel or registration bodies.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {createHash} = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {fileHash} = require("./build-source-inputs");
const {elfRangeHash: fileRangeHash} = require("./native-elf");
/** @typedef {{number: number, bytes: number, virtualSize: number, characteristics: number, sha256: string}} CoffSectionPayload */
const {coffSectionPayloads} = /** @type {{coffSectionPayloads: (file: string) => CoffSectionPayload[]}} */ (require("./native-compiler-isolation"));
const {deviceLinkArgs} = require("./build-native-gpu");
const INPUT_CRT = ".CRT$XCA00001", OUTPUT_CRT = ".CRT$XCU99999", MAX_BYTES = 256 * 1024 * 1024;
const MAX_STABLE_ID_LENGTH = 4096, MAX_STABLE_ID_COUNT = 65536, MAX_PROPERTY_FILE_COUNT = 4096, MAX_PROPERTY_TEXT_BYTES = 4 * 1024 * 1024;
const IMAGE_SYMBOLS = [".sycl_offloading.0.data", ".sycl_offloading.0.info"];
const PROFILES = new Set(["dpcpp", "oneapi", "dpcpp-opencl"]);
const SPECIALIZATION_SECTION = "[SYCL/specialization constants]";
/** @typedef {{name: string, raw: number, chars: string[], relocs: Array<{type: string, symbol: string, offset: number}>}} Section */
/** @typedef {{gates: {key: string, register: string, unregister: string}, list: Section[], imageSections: Record<string, string>, raw: {readobj: string, undefined: string}}} Inspection */

/** @param {unknown} value @returns {value is string} */
function absolutePath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}
/** @param {unknown} profile @returns {{key: string, register: string, unregister: string}} */
function profileGateNames(profile) {
  if (typeof profile !== "string" || !PROFILES.has(profile)) {throw new Error("Invalid Windows wrapper registration profile");}
  const key = profile.replaceAll("-", "_");
  return {key, register: "mom_native_register_" + key, unregister: "mom_native_unregister_" + key};
}
/** @param {string} file @param {string} label @param {boolean} [required] */
function boundedFile(file, label, required = true) {
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  if (required && (!stat?.isFile() || stat.size > MAX_BYTES)) {throw new Error(label + " must be a bounded regular file");}
}
/** Read one fixed file generation; a replaced FIFO or growing object must not bypass the cap.
 * @param {string} file @param {string} label @param {number} [maximum] @returns {Buffer} */
function readBounded(file, label, maximum = MAX_BYTES) {
  if (!absolutePath(file)) {throw new Error(label + " must be an absolute path");}
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd);
    if (!before.isFile() || before.size > maximum) {throw new Error(label + " must be a bounded regular file");}
    const data = Buffer.allocUnsafe(before.size + 1); let size = 0;
    for (let count; size < data.length && (count = fs.readSync(fd, data, size, data.length - size, null));) {size += count;}
    const after = fs.fstatSync(fd);
    if (size !== before.size || after.size !== before.size || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs) {
      throw new Error(label + " changed during inspection");
    }
    return data.subarray(0, size);
  } finally {fs.closeSync(fd);}
}
const STABLE_ID_RE = /^uid[0-9a-f]{16}____Z[A-Za-z0-9_.$?]+$/;
/** @param {Buffer} data @param {string} label @returns {string[]} */
function stableIds(data, label) {
  const ids = new Set(); let count = 0;
  // COFF padding can contain millions of NULs: splitting the whole object would allocate an
  // enormous array. Visit only candidate string starts, ignoring decorated-symbol fragments.
  for (let index = data.indexOf("uid"); index >= 0; index = data.indexOf("uid", index + 3)) {
    if (index && data[index - 1] !== 0) {continue;}
    let end = data.indexOf(0, index);
    if (end < 0) {end = data.length;}
    const bytes = data.subarray(index, end);
    if (bytes.length > MAX_STABLE_ID_LENGTH || bytes.some(value => value > 0x7f) || !STABLE_ID_RE.test(bytes.toString("ascii"))) {
      throw new Error("Malformed or unsupported specialization ID in " + label);
    }
    if (++count > MAX_STABLE_ID_COUNT) {throw new Error("Too many specialization IDs in " + label);}
    ids.add(bytes.toString("ascii"));
  }
  return [...ids].sort();
}
/** @param {Buffer} data @param {string} label @returns {string[]} */
function propertyIds(data, label) {
  if (data.length > MAX_PROPERTY_TEXT_BYTES) {throw new Error("Specialization property text is too large in " + label);}
  if (data.some(value => value === 0 || value > 0x7f)) {throw new Error("Malformed non-ASCII property text in " + label);}
  const text = data.toString("ascii");
  if (!text || !text.endsWith("\n")) {throw new Error("Truncated specialization property file in " + label);}
  const ids = new Set(), recordsById = new Map(), lines = text.split("\n"), validKey = /^uid[0-9a-f]{16}____Z[A-Za-z0-9_.$?]+$/;
  let section = "", records = 0;
  const finish = () => {if (section === SPECIALIZATION_SECTION && records === 0) {throw new Error("Empty specialization property section in " + label);}};
  for (let index = 0; index < lines.length - 1; index++) {
    let line = lines[index] || ""; if (line.endsWith("\r")) {line = line.slice(0, -1);}
    if (line.includes("\r")) {throw new Error("Malformed specialization property line in " + label);}
    if (line.startsWith("[")) {finish(); if (!/^\[[^\]\r\n]+\]$/.test(line)) {throw new Error("Malformed property section header in " + label);} section = line; records = 0; continue;}
    if (section === SPECIALIZATION_SECTION) {
      const split = line.indexOf("=2|");
      if (!line || split <= 0 || !line.slice(split + 3) || !validKey.test(line.slice(0, split))) {throw new Error("Malformed specialization property record in " + label);}
      const key = line.slice(0, split), payload = line.slice(split + 3), previous = recordsById.get(key);
      if (previous !== undefined && previous !== payload) {throw new Error("Conflicting specialization property record in " + label);}
      recordsById.set(key, payload); ids.add(key); records++;
    } else if (line && /uid[0-9A-Fa-f]{16}____/.test(line)) {
      throw new Error("Specialization ID outside its property section in " + label);
    }
  }
  finish();
  return [...ids].sort();
}
/** @param {string} file @param {string} label @param {(data: Buffer, label: string) => string[]} parse */
function specializationEvidence(file, label, parse = stableIds) {
  const data = readBounded(file, label, parse === propertyIds ? MAX_PROPERTY_TEXT_BYTES : MAX_BYTES), ids = parse(data, file);
  return {path: file, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex"), ids};
}
/** @param {string} file @returns {string[]} */
function extractSpecializationIds(file) {return specializationEvidence(file, "Windows specialization-ID input").ids;}
/** Compare one original combined COFF object with its host-retained derivative. Section numbers
 * may be renumbered after removal, but every kept section's logical payload and flags must stay
 * byte-for-byte equivalent. @param {string} original @param {string} retained @param {number[]} removedSectionNumbers */
function verifyWindowsHostRetention(original, retained, removedSectionNumbers) {
  if (!absolutePath(original) || !absolutePath(retained) || path.resolve(original) === path.resolve(retained)) {
    throw new Error("Windows host retention needs distinct absolute objects");
  }
  if (!Array.isArray(removedSectionNumbers) || removedSectionNumbers.length > 65536 ||
      removedSectionNumbers.some(number => !Number.isSafeInteger(number) || number < 1) ||
      new Set(removedSectionNumbers).size !== removedSectionNumbers.length) {
    throw new Error("Windows host retention section removals must be unique positive integers");
  }
  const originalSha256 = fileHash(original), retainedSha256 = fileHash(retained), originalSections = coffSectionPayloads(original), retainedSections = coffSectionPayloads(retained);
  if (removedSectionNumbers.some(number => number > originalSections.length)) {throw new Error("Windows host retention section removal is out of range");}
  if (removedSectionNumbers.length + retainedSections.length !== originalSections.length) {throw new Error("Windows host retention section count mismatch");}
  const removed = new Set(removedSectionNumbers), kept = originalSections.filter(section => !removed.has(section.number));
  if (kept.length !== retainedSections.length) {throw new Error("Windows host retention section count mismatch");}
  for (const [index, expected] of kept.entries()) {
    const actual = retainedSections[index];
    if (!actual || expected.bytes !== actual.bytes || expected.virtualSize !== actual.virtualSize ||
        expected.characteristics !== actual.characteristics || expected.sha256 !== actual.sha256) {
      throw new Error("Windows host retention changed a kept section payload or flags");
    }
  }
  if (fileHash(original) !== originalSha256 || fileHash(retained) !== retainedSha256) {
    throw new Error("Windows host retention input changed during verification");
  }
  return {version: 1, input: {path: original, bytes: fs.statSync(original).size, sha256: originalSha256},
    output: {path: retained, bytes: fs.statSync(retained).size, sha256: retainedSha256},
    originalSectionCount: originalSections.length, retainedSectionCount: retainedSections.length,
    removedSectionCount: removedSectionNumbers.length, removedSectionNumbers: removedSectionNumbers.slice().sort((a, b) => a - b)};
}
/** @param {string} hostObject @param {string} deviceObject @param {string[]} [propertyPaths] */
function verifySpecializationIds(hostObject, deviceObject, propertyPaths = []) {
  if (!absolutePath(hostObject) || !absolutePath(deviceObject) || path.resolve(hostObject) === path.resolve(deviceObject)) {throw new Error("Windows specialization-ID verifier needs distinct absolute host/device objects");}
  if (!Array.isArray(propertyPaths) || propertyPaths.length > MAX_PROPERTY_FILE_COUNT || propertyPaths.some(file => !absolutePath(file))) {throw new Error("Windows specialization-ID property paths must be bounded absolute files");}
  const host = specializationEvidence(hostObject, "Windows specialization host object"), device = specializationEvidence(deviceObject, "Windows specialization device object");
  const properties = propertyPaths.map(file => specializationEvidence(file, "Windows specialization property file", propertyIds));
  const union = [...new Set(properties.flatMap(property => property.ids))].sort();
  if (JSON.stringify(host.ids) !== JSON.stringify(device.ids)) {throw new Error("Windows specialization-ID mismatch between host and device objects: host=" + host.ids.join(",") + " device=" + device.ids.join(","));}
  if (propertyPaths.length && JSON.stringify(device.ids) !== JSON.stringify(union)) {throw new Error("Windows specialization-ID mismatch between device object and property keys: device=" + device.ids.join(",") + " properties=" + union.join(","));}
  return {version: 1, matched: true, host, device, properties, specializationIds: {host: host.ids, device: device.ids, properties: union}};
}
/** Reuse the maintained DPC++ device-link recipe for a Windows host object. Drop Linux
 * PIC and disable wall-clock COFF stamps in the generated registration object. @param {string} gyp
 * @param {string | undefined} cuda @param {string[]} objects @param {string} output
 * @param {string} [profile] @param {string} [wrapper] @returns {string[]} */
function windowsDeviceLinkArgs(gyp, cuda, objects, output, profile = "dpcpp", wrapper = "") {
  const args = deviceLinkArgs(gyp, cuda, objects, output, profile, wrapper);
  return ["--target=x86_64-pc-windows-msvc", "-mno-incremental-linker-compatible",
    ...args.filter((flag) => flag !== "-fPIC")];
}
/** @param {unknown} tools */
function validateTools(tools) {
  if (!tools || typeof tools !== "object" || Array.isArray(tools)) {throw new Error("Invalid Windows wrapper toolpaths");}
  const value = /** @type {Record<string, unknown>} */ (tools), names = ["llvmObjcopy", "llvmReadobj", "llvmNm"];
  if (Object.keys(value).length !== names.length || names.some(name => typeof value[name] !== "string" || !absolutePath(value[name]))) {
    throw new Error("Windows wrapper toolpaths must name three absolute LLVM tools");
  }
  return /** @type {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} */ (tools);
}
/** Keep tool failures actionable without copying an unbounded compiler diagnostic into errors.
 * @param {string} command @param {string[]} args @param {string} cwd @param {string} label @returns {string} */
function run(command, args, cwd, label) {
  boundedFile(command, "Windows wrapper tool");
  const env = {...process.env}; delete env["MOM_NATIVE_RUNTIME"];
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: 120000, maxBuffer: 32 * 1024 * 1024, env});
  if (result.error || result.signal || result.status !== 0) {
    const detail = (result.stderr || "").trim().slice(-2048);
    throw new Error("Windows wrapper " + label + " failed (exit " + result.status + ")" + (detail ? ": " + detail : ""), {cause: result.error});
  }
  return result.stdout || "";
}
/** @param {string} object @param {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} tools */
function readObject(object, tools) {
  return {readobj: run(tools.llvmReadobj, ["--file-headers", "--sections", "--section-relocations", "--symbols", object], path.dirname(object), "llvm-readobj").replace(/\r\n?/g, "\n"),
    undefined: run(tools.llvmNm, ["--undefined-only", "--format=just-symbols", object], path.dirname(object), "llvm-nm").replace(/\r\n?/g, "\n")};
}
/** @param {string} text @returns {Array<{name: string, raw: number, chars: string[], relocs: Array<{type: string, symbol: string, offset: number}>}>} */
function sections(text) {
  const result = [];
  for (const match of text.matchAll(/(?:^|\n)\s{2}Section \{\n([\s\S]*?)\n\s{2}\}/g)) {
    const block = match[1];
    if (!block) {throw new Error("Malformed COFF section inspection");}
    const name = block.match(/^\s*Name:\s*(.*?)\s*$/m)?.[1]?.replace(/\s+\([0-9A-Fa-f ]+\)$/, "").trim();
    const rawText = block.match(/^\s*RawDataSize:\s*(\S+)/m)?.[1];
    if (!name || !rawText) {throw new Error("Malformed COFF section inspection");}
    const relocs = [...block.matchAll(/^\s*(0x[0-9A-Fa-f]+)\s+(IMAGE_REL_AMD64_\S+)\s+(.+?)(?:\s+\(\d+\))?\s*$/gm)].map(match => {
      const offset = match[1], type = match[2], symbol = match[3];
      if (!offset || !type || !symbol) {throw new Error("Malformed COFF relocation");}
      return {offset: Number(offset), type, symbol: symbol.trim()};
    });
    const chars = [...block.matchAll(/\b(IMAGE_SCN_[A-Z0-9_]+)/g)].map(match => match[1]).filter((value) => value !== undefined);
    const raw = Number(rawText);
    if (!Number.isSafeInteger(raw) || raw < 0) {throw new Error("Malformed COFF section size");}
    result.push({name, raw, chars, relocs});
  }
  if (!result.length || new Set(result.map(section => section.name)).size !== result.length) {throw new Error("Malformed or duplicate COFF sections");}
  return result;
}
/** @param {string} text @returns {Record<string, string>} */
function symbolSections(text) {
  const result = /** @type {Record<string, string>} */ ({});
  for (const match of text.matchAll(/(?:^|\n)\s{2}Symbol \{\n([\s\S]*?)\n\s{2}\}/g)) {
    const block = match[1];
    if (!block) {continue;}
    const name = block.match(/^\s*Name:\s*(.+)$/m)?.[1]?.trim(), section = block.match(/^\s*Section:\s*(\S+)/m)?.[1];
    if (name && section) {result[name] = section;}
  }
  return result;
}
/** @param {string} text @returns {string[]} */
function undefinedSymbols(text) {return text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);}
/** Admit the compiler's exact lifetime shape; unknown layouts need review, not a best guess.
 * @param {string} object @param {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} tools @param {"input"|"output"} phase @param {string} profile @returns {Inspection} */
function inspect(object, tools, phase, profile) {
  boundedFile(object, "Windows wrapper input");
  const gates = profileGateNames(profile), raw = readObject(object, tools), list = sections(raw.readobj), names = undefinedSymbols(raw.undefined);
  if (!/^Format:\s*COFF-x86-64\s*$/m.test(raw.readobj)) {throw new Error("Windows wrapper must be x86-64 COFF");}
  const crtName = phase === "input" ? INPUT_CRT : OUTPUT_CRT, crt = list.filter(section => section.name.startsWith(".CRT$"));
  if (crt.length !== 1 || crt[0]?.name !== crtName) {throw new Error("Unknown or misplaced Windows wrapper CRT section");}
  const crtSection = crt[0];
  if (crtSection.raw !== 8 || crtSection.relocs.length !== 1 || !crtSection.chars.includes("IMAGE_SCN_ALIGN_8BYTES") ||
      !crtSection.chars.includes("IMAGE_SCN_CNT_INITIALIZED_DATA") || !crtSection.chars.includes("IMAGE_SCN_MEM_READ") ||
      crtSection.chars.includes("IMAGE_SCN_MEM_WRITE") || crtSection.relocs[0]?.type !== "IMAGE_REL_AMD64_ADDR64" ||
      crtSection.relocs[0]?.offset !== 0 || crtSection.relocs[0]?.symbol !== "sycl.descriptor_reg") {
    throw new Error("Invalid Windows wrapper CRT section or descriptor relocation");
  }
  const startup = list.find(section => section.name === ".text.startup"), reg = phase === "input" ? "__sycl_register_lib" : gates.register;
  const unreg = phase === "input" ? "__sycl_unregister_lib" : gates.unregister;
  const expected = [".sycl_offloading.descriptor", unreg, ".sycl_offloading.descriptor", reg, "sycl.descriptor_unreg.atexit", "atexit"];
  if (!startup || startup.raw < 1 || !startup.chars.includes("IMAGE_SCN_CNT_CODE") || !startup.chars.includes("IMAGE_SCN_MEM_EXECUTE") ||
      startup.relocs.length !== expected.length || startup.relocs.some((reloc, index) => reloc.type !== "IMAGE_REL_AMD64_REL32" || reloc.symbol !== expected[index])) {
    throw new Error("Windows wrapper startup registration relocations do not match the canonical shape");
  }
  const imports = new Set(names), expectedImports = new Set([reg, unreg, "atexit"]);
  if (imports.size !== expectedImports.size || [...expectedImports].some(name => !imports.has(name))) {throw new Error("Windows wrapper imports do not match its profile gates");}
  const images = symbolSections(raw.readobj), imageSections = /** @type {Record<string, string>} */ ({});
  for (const symbol of IMAGE_SYMBOLS) {
    const section = images[symbol], selected = section && list.find(candidate => candidate.name === section);
    if (!selected || selected.raw < 1) {throw new Error("Windows wrapper is missing image symbol section " + symbol);}
    imageSections[symbol] = section;
  }
  return {gates, list, imageSections, raw};
}
/** LLVM can rewrite COFF even during section dumps. Recopy one scratch object per section,
 * never retain a full object for every section. Hash on disk with bounded storage and compare
 * original payload prefixes so harmless alignment padding cannot hide a kernel change.
 * @param {string} object @param {Array<{name: string, raw: number}>} list @param {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} tools
 * @param {Record<string, number>} [prefixLengths] */
function sectionHashes(object, list, tools, prefixLengths = {}) {
  const selected = list.filter(section => section.raw > 0 && !/^\.(?:llvm_|debug|drectve)/.test(section.name));
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-wrapper-")), hashes = /** @type {Record<string, {bytes: number, sha256: string, prefixSha256: string}>} */ ({});
  try {
    const dump = path.join(temp, "section.bin"), copy = path.join(temp, "object.obj");
    for (const section of selected) {
      fs.copyFileSync(object, copy);
      run(tools.llvmObjcopy, ["--dump-section", section.name + "=" + dump, copy], temp, "llvm-objcopy section dump");
      const bytes = fs.statSync(dump).size;
      if (bytes < section.raw) {throw new Error("COFF section dump is shorter than its raw data: " + section.name);}
      const prefixLength = prefixLengths[section.name] || section.raw;
      if (prefixLength > bytes || prefixLength < 1) {throw new Error("COFF section prefix is unavailable: " + section.name);}
      const sha256 = fileRangeHash(dump, 0, section.raw);
      hashes[section.name] = {bytes: section.raw, sha256, prefixSha256: prefixLength === section.raw ? sha256 : fileRangeHash(dump, 0, prefixLength)};
      fs.unlinkSync(dump);
    }
  } finally {fs.rmSync(temp, {recursive: true, force: true});}
  return hashes;
}
/** Patch only the COFF string-table entry referenced by the one admitted CRT section. */
/** @param {string} file */
function patchCoffSectionName(file) {
  const data = fs.readFileSync(file);
  if (data.length < 20 || data.readUInt16LE(0) !== 0x8664) {throw new Error("Windows wrapper metadata edit needs AMD64 COFF");}
  const count = data.readUInt16LE(2), symbolOffset = data.readUInt32LE(8), symbols = data.readUInt32LE(12), table = symbolOffset + symbols * 18;
  if (!count || count > 96 || !Number.isSafeInteger(table) || table + 4 > data.length || 20 + count * 40 > symbolOffset) {throw new Error("Malformed bounded COFF metadata");}
  const stringBytes = data.readUInt32LE(table), tableEnd = table + stringBytes;
  if (stringBytes < 4 || tableEnd > data.length) {throw new Error("Malformed COFF string table");}
  const oldName = Buffer.from(INPUT_CRT), newName = Buffer.from(OUTPUT_CRT); let target = -1, matches = 0;
  for (let index = 0; index < count; index++) {
    const field = data.subarray(20 + index * 40, 28 + index * 40);
    if (field[0] !== 0x2f) {continue;}
    const token = field.subarray(0, field.indexOf(0) < 0 ? field.length : field.indexOf(0)).toString("ascii");
    if (!/^\/\d+$/.test(token)) {throw new Error("Malformed COFF long section name");}
    const offset = Number(token.slice(1));
    if (!Number.isSafeInteger(offset) || offset < 4 || table + offset + oldName.length + 1 > tableEnd) {throw new Error("COFF section name points outside its string table");}
    const name = data.subarray(table + offset, table + offset + oldName.length);
    if (name.equals(newName)) {throw new Error("Windows wrapper already has the target CRT name");}
    if (name.equals(oldName) && data[table + offset + oldName.length] === 0) {target = table + offset; matches++;}
  }
  if (matches !== 1 || target < 0) {throw new Error("COFF metadata has no unique admitted CRT name");}
  newName.copy(data, target); fs.writeFileSync(file, data);
}
/** @param {unknown} value */
function plan(value) {
  const options = /** @type {{input: string, output: string, profile: string, tools: unknown, manifestPath?: string, sourceFiles?: string[]}} */ (value);
  if (!options || typeof options !== "object" || !absolutePath(options.input) || !absolutePath(options.output) || path.resolve(options.input) === path.resolve(options.output)) {
    throw new Error("Windows wrapper needs distinct absolute input and output paths");
  }
  const tools = validateTools(options.tools), gates = profileGateNames(options.profile), parent = path.dirname(options.output);
  if (fs.existsSync(options.output) || !fs.statSync(parent, {throwIfNoEntry: false})?.isDirectory()) {throw new Error("Windows wrapper output must be a new path in an existing directory");}
  if (options.manifestPath !== undefined && (!absolutePath(options.manifestPath) || fs.existsSync(options.manifestPath))) {throw new Error("Windows wrapper manifest must be a new absolute path");}
  if (options.sourceFiles !== undefined && (!Array.isArray(options.sourceFiles) || options.sourceFiles.some(file => !absolutePath(file)))) {throw new Error("Windows wrapper source files must be absolute");}
  return {input: options.input, output: options.output, profile: options.profile, tools, gates,
    args: ["--redefine-sym", "__sycl_register_lib=" + gates.register,
      "--redefine-sym", "__sycl_unregister_lib=" + gates.unregister], options};
}
/** @param {string[]} files @returns {Record<string, string>} */
function sourceHashes(files) {return Object.fromEntries(files.map(file => {boundedFile(file, "Windows wrapper source"); return [file, fileHash(file)];}));}
/** Change only callback timing and its existing profile gate imports, then prove image and
 * source stability before emitting a reusable receipt. This does not execute or qualify a GPU.
 * @param {{input: string, output: string, profile: string, tools: unknown, manifestPath?: string, sourceFiles?: string[]}} options */
function transformWindowsWrapper(options) {
  const selected = plan(options); boundedFile(selected.input, "Windows wrapper input");
  const inputSha256 = fileHash(selected.input);
  const inputReal = fs.realpathSync(selected.input), outputReal = path.join(fs.realpathSync(path.dirname(selected.output)), path.basename(selected.output));
  if (inputReal === outputReal) {throw new Error("Windows wrapper output aliases its input");}
  const beforeSources = sourceHashes(selected.options.sourceFiles || []), before = inspect(selected.input, selected.tools, "input", selected.profile), beforeHashes = sectionHashes(selected.input, before.list, selected.tools);
  const stageDir = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-wrapper-transform-"));
  try {
    const stagedInput = path.join(stageDir, "input.obj");
    fs.copyFileSync(selected.input, stagedInput); patchCoffSectionName(stagedInput);
    run(selected.tools.llvmObjcopy, [...selected.args, stagedInput, selected.output], stageDir, "llvm-objcopy wrapper transform");
  } finally {fs.rmSync(stageDir, {recursive: true, force: true});}
  boundedFile(selected.output, "Windows wrapper output");
  const after = inspect(selected.output, selected.tools, "output", selected.profile), outputPrefixes = Object.fromEntries(before.list.map(section =>
    [section.name === INPUT_CRT ? OUTPUT_CRT : section.name, section.raw]));
  const afterHashes = sectionHashes(selected.output, after.list, selected.tools, outputPrefixes);
  const beforeNames = new Set(before.list.map(section => section.name === INPUT_CRT ? OUTPUT_CRT : section.name)), afterNames = new Set(after.list.map(section => section.name));
  if (beforeNames.size !== afterNames.size || [...beforeNames].some(name => !afterNames.has(name))) {throw new Error("Windows wrapper changed its section set");}
  const expectedNames = new Set(Object.keys(beforeHashes).map(name => name === INPUT_CRT ? OUTPUT_CRT : name));
  if (expectedNames.size !== Object.keys(afterHashes).length || [...expectedNames].some(name => !Object.hasOwn(afterHashes, name))) {throw new Error("Windows wrapper changed its section set");}
  for (const [name, hash] of Object.entries(beforeHashes)) {
    const outputName = name === INPUT_CRT ? OUTPUT_CRT : name;
    if (!afterHashes[outputName] || hash.sha256 !== afterHashes[outputName].prefixSha256) {throw new Error("Windows wrapper changed section bytes: " + name);}
  }
  const afterSources = sourceHashes(selected.options.sourceFiles || []);
  if (JSON.stringify(beforeSources) !== JSON.stringify(afterSources)) {throw new Error("Windows wrapper source changed during transform");}
  const imagesBefore = /** @type {Record<string, {bytes: number, sha256: string}>} */ ({}), imagesAfter = /** @type {Record<string, {bytes: number, sha256: string}>} */ ({});
  for (const symbol of IMAGE_SYMBOLS) {
    const beforeSection = before.imageSections[symbol], afterSection = after.imageSections[symbol];
    if (!beforeSection || !afterSection || !beforeHashes[beforeSection] || !afterHashes[afterSection]) {throw new Error("Windows wrapper image section hash is missing");}
    imagesBefore[symbol] = {bytes: beforeHashes[beforeSection].bytes, sha256: beforeHashes[beforeSection].sha256};
    imagesAfter[symbol] = {bytes: beforeHashes[beforeSection].bytes, sha256: afterHashes[afterSection].prefixSha256};
  }
  if (JSON.stringify(imagesBefore) !== JSON.stringify(imagesAfter)) {throw new Error("Windows wrapper device image changed");}
  if (fileHash(selected.input) !== inputSha256) {throw new Error("Windows wrapper input changed during transform");}
  const manifest = {version: 1, kind: "windows-sycl-wrapper-transform", profile: selected.profile, gates: selected.gates,
    input: {path: selected.input, bytes: fs.statSync(selected.input).size, sha256: inputSha256}, output: {path: selected.output, bytes: fs.statSync(selected.output).size, sha256: fileHash(selected.output)},
    sourceHashes: {before: beforeSources, after: afterSources}, sectionHashes: {before: beforeHashes, after: afterHashes}, imageHashes: {before: imagesBefore, after: imagesAfter},
    renamedSection: {from: INPUT_CRT, to: OUTPUT_CRT}, plan: {command: selected.tools.llvmObjcopy, args: selected.args,
      sectionMetadataEdit: "COFF string-table entry only"}};
  if (selected.options.manifestPath) {fs.writeFileSync(selected.options.manifestPath, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx"});}
  return manifest;
}

module.exports = {profileGateNames, windowsWrapperPlan: plan, inspectWindowsWrapper: inspect, transformWindowsWrapper, extractSpecializationIds,
  verifyWindowsHostRetention, verifySpecializationIds, windowsDeviceLinkArgs};
