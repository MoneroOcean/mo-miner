"use strict";

const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {hasExactBuildFields, fileHash} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const dpcppEntries = Object.freeze(["addToJITConfiguration", "resetJITConfiguration", "materializeSpecConstants", "calculateHash", "compileSYCL", "destroyBinary"]);
const COFF_TOOL_KEYS = Object.freeze(["llvmObjcopy", "llvmReadobj", "llvmNm"]);
const COFF_OBJECT_LIMIT = 128, COFF_OBJECT_BYTES = 256 << 20, COFF_TOTAL_BYTES = 512 << 20;
const COFF_TOOL_BYTES = 256 << 20, COFF_LOG_BYTES = 64 << 20, COFF_TIMEOUT = 120000;
const COFF_TOTAL_SECTIONS = 100000, COFF_TOTAL_RELOCATIONS = 1000000, COFF_TOTAL_SYMBOLS = 200000;
const COFF_TOTAL_LOG_BYTES = 256 << 20, COFF_RECEIPT_BYTES = 16 << 20;
/** @typedef {{logBytes: number, sections: number, relocations: number, symbols: number}} CoffBudget */

/** Keep complete command/output evidence; arguments are never evaluated as shell text.
 * @param {string} output @param {string} tool @param {string[]} args @param {string} label */
function run(output, tool, args, label) {
  fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({tool, args}) + "\n", {flag: "wx"});
  const result = spawnSync(tool, args, {encoding: "utf8", env: {...process.env, LC_ALL: "C"}, timeout: 180000, maxBuffer: 64 << 20});
  fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
  if (result.error || result.signal || result.status) {throw new Error(`Compiler isolation failed: ${label}`, {cause: result.error});}
  return result.stdout;
}

/** Tokenize the single link command without evaluating shell syntax. CMake emits only simple
 * double-quoted paths; reject escaping and controls instead of guessing their meaning.
 * @param {string} line */
function windowsLinkTokens(line) {
  if (/[;|<>$`'\\]/.test(line) || Array.from(line).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) || line.length > (8 << 20)) {
    throw new Error("Unrecognized original static DPC++ compiler archive graph");
  }
  let text = line.trim();
  if (text.startsWith(": && ")) {text = text.slice(5);}
  if (text.endsWith(" && :")) {text = text.slice(0, -5);}
  if (/[&*?()[\]{}#~]/.test(text)) {throw new Error("Unrecognized original static DPC++ compiler archive graph");}
  /** @type {Array<{value:string, quoted:boolean, wholeQuoted:boolean}>} */ const tokens = [];
  let raw = "", value = "", quoted = false, quoteCount = 0;
  const push = () => {
    if (!raw) {return;}
    if (quoted) {throw new Error("Unrecognized original static DPC++ compiler archive graph");}
    tokens.push({value, quoted: quoteCount !== 0, wholeQuoted: quoteCount === 2 && raw.startsWith('"') && raw.endsWith('"')});
    raw = ""; value = ""; quoteCount = 0;
  };
  for (const char of text) {
    if (char === '"') {raw += char; quoted = !quoted; quoteCount++; continue;}
    if (/\s/.test(char) && !quoted) {push(); continue;}
    raw += char; value += char;
  }
  push();
  if (tokens.some(token => token.value === "&&" || token.value === "||")) {throw new Error("Unrecognized original static DPC++ compiler archive graph");}
  return tokens;
}

/** Use the original CMake/Ninja target's actual archive closure, not a maintained LLVM list.
 * Windows support paths are accepted only as complete quoted tokens from the already-hashed
 * toolchain inputs. The command is parsed conservatively and never evaluated as shell text.
 * @param {string} commands @param {string} build @param {boolean} [windows] @param {string[]} [supportLibraries] */
function compilerArchives(commands, build, windows = false, supportLibraries = []) {
  const target = windows ? /(?:^|\s)\/out:bin\/native-dpcpp-jit\.exe(?:\s|$)/ : /(?:^|\s)-o bin\/native-dpcpp-jit(?:\s|$)/;
  const links = commands.split("\n").filter(line => target.test(line));
  const invalidSupport = !Array.isArray(supportLibraries) || supportLibraries.length > 16384 ||
    new Set(supportLibraries).size !== supportLibraries.length || supportLibraries.some(file => !coffAbsolutePath(file) ||
      path.normalize(file) !== file || !/\.lib$/i.test(file) || /[;|<>$`"'\\]/.test(file));
  if (invalidSupport || (!windows && supportLibraries.length)) {throw new Error("Unrecognized original static DPC++ compiler archive graph");}
  let names;
  if (windows) {
    const admitted = new Set(supportLibraries), seenSupport = new Set();
    names = [];
    for (const token of windowsLinkTokens(links[0] || "")) {
      const word = token.value;
      if (!/\.lib|\.a/i.test(word)) {continue;}
      if (word === "/implib:lib/native-dpcpp-jit.lib" && !token.quoted) {continue;}
      if (/\.a/i.test(word)) {throw new Error("Unrecognized original static DPC++ compiler archive graph");}
      if (/^lib\/[A-Za-z0-9_.-]+\.lib$/i.test(word) && !token.quoted) {names.push(word); continue;}
      // Bare library names are linker imports, not paths into the compiler archive closure.
      if (/^[A-Za-z0-9_.-]+\.lib$/i.test(word) && !token.quoted) {continue;}
      if (path.isAbsolute(word) && token.wholeQuoted && admitted.has(word) && path.normalize(word) === word && !seenSupport.has(word)) {
        seenSupport.add(word); continue;
      }
      throw new Error("Unrecognized original static DPC++ compiler archive graph");
    }
  } else {
    names = links.length === 1 ? (links[0] || "").split(/\s+/).filter(word => /\.a(?:$|[^A-Za-z0-9_.-])/.test(word)) : [];
  }
  const pattern = windows ? /^lib\/[A-Za-z0-9_.-]+\.lib$/ : /^lib\/[A-Za-z0-9_.-]+\.a$/;
  if (!path.isAbsolute(build) || links.length !== 1 || names.length < 50 || names.length > 512 ||
      names[0] !== (windows ? "lib/sycl-jit.lib" : "lib/libsycl-jit.a") ||
      new Set(names).size !== names.length || names.some(name => !pattern.test(name))) {
    throw new Error("Unrecognized original static DPC++ compiler archive graph");
  }
  return names.map((name) => path.join(build, name));
}

/** Compiler constructors/destructors can use local C5/D5 COMDAT signatures. Those keys must
 * be private too: renaming globals alone lets the linker discard another LLVM version's
 * constructor while leaving its weak call unresolved (a null call, not a link error).
 * @param {string} text */
function groupSignatures(text) {
  const names = [...text.matchAll(/^COMDAT group section \[[^\]]+\] .* \[([^\]]+)\] contains /gm)].map((match) => match[1] || "");
  if (names.length !== (text.match(/^.*group section /gm) || []).length || names.some((name) => !name || /\s/.test(name))) {
    throw new Error("Unsupported ELF group signature format");
  }
  return names;
}

/** Isolate an already partially linked original compiler graph, not a second library registry.
 * This is a Linux build step, never runtime extraction. Keep the narrow C bridge unchanged;
 * all other definitions AND COMDAT selection keys belong to this compiler version only.
 * Input is trusted build output. A fresh output directory retains complete failed-job evidence.
 * @param {string} input @param {string} output
 * @param {{prefix: string, entryPoints: readonly string[]}} options */
function isolateCompilerObject(input, output, {prefix, entryPoints}) {
  if (process.platform !== "linux" || !path.isAbsolute(input) || !path.isAbsolute(output) ||
      !/^[A-Za-z_][A-Za-z0-9_]{0,63}_$/.test(prefix) || !entryPoints.length ||
      entryPoints.some((name) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) || new Set(entryPoints).size !== entryPoints.length) {
    throw new Error("ELF compiler isolation needs absolute paths, a private prefix and unique C entries");
  }
  fs.mkdirSync(output);
  if (!/Type:\s+REL \(/.test(run(output, "readelf", ["--file-header", input], "header"))) {
    throw new Error("Compiler isolation requires an ELF relocatable object");
  }
  const symbols = run(output, "nm", ["-P", "-g", "--defined-only", input], "definitions");
  const definitions = symbols.trim().split("\n").filter(Boolean).map((line) => line.split(/\s+/)[0] || "");
  const keep = new Set(entryPoints);
  if (entryPoints.some((name) => !definitions.includes(name))) {throw new Error("Missing compiler C bridge definition");}
  const groups = groupSignatures(run(output, "readelf", ["--wide", "--section-groups", input], "groups"));
  const renamed = [...new Set([...definitions, ...groups])].filter((name) => !keep.has(name));
  if (!renamed.length || renamed.some((name) => !name || /\s/.test(name) || name.startsWith(prefix))) {
    throw new Error("Empty, malformed or already isolated compiler symbols");
  }
  const map = path.join(output, "rename.map"), object = path.join(output, "compiler.o");
  fs.writeFileSync(map, renamed.map((name) => `${name} ${prefix}${name}`).join("\n") + "\n", {flag: "wx"});
  run(output, "objcopy", ["--redefine-syms=" + map, input, object], "isolate");
  const after = run(output, "nm", ["-P", "-g", "--defined-only", object], "isolated-definitions");
  const privateOrBridge = (/** @type {string} */ name) => keep.has(name) || name.startsWith(prefix);
  const afterGroups = groupSignatures(run(output, "readelf", ["--wide", "--section-groups", object], "isolated-groups"));
  const undefinedSymbols = run(output, "nm", ["-P", "-u", object], "undefined");
  if (after.trim().split("\n").some((line) => !privateOrBridge(line.split(/\s+/)[0] || "")) ||
      afterGroups.length !== groups.length || afterGroups.some((name) => !privateOrBridge(name)) ||
      undefinedSymbols.split("\n").some((line) => line.startsWith(prefix) || /^_Z\S*(?:4llvm|5clang)/.test(line))) {
    throw new Error("Compiler graph is not self-contained or fully isolated");
  }
  const receipt = {input, object, prefix, entryPoints, definitions: definitions.length, groups: groups.length,
    renamed: renamed.length, inputSha256: fileHash(input), objectSha256: fileHash(object)};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx"});
  return receipt;
}

/** @param {unknown} value @returns {value is string} */
function coffAbsolutePath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} file @param {string} label @param {number} maximum @returns {fs.Stats} */
function coffRegularFile(file, label, maximum) {
  if (!coffAbsolutePath(file)) {throw new Error(label + " must be an absolute path");}
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.size > maximum) {throw new Error(label + " must be a bounded regular file");}
  return stat;
}

/** @param {unknown} value @returns {value is string} */
function coffSymbolName(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && !/\s/.test(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} tool @param {string[]} args @param {string} cwd @param {string} label @param {CoffBudget} budget */
function invokeCoffTool(tool, args, cwd, label, budget) {
  const result = spawnSync(tool, args, {cwd, encoding: "utf8", env: {...process.env, LC_ALL: "C"},
    timeout: COFF_TIMEOUT, maxBuffer: COFF_LOG_BYTES});
  const stdout = typeof result.stdout === "string" ? result.stdout : "", stderr = typeof result.stderr === "string" ? result.stderr : "";
  const captured = {stdout, stderr, text: stdout + stderr};
  if (result.error || result.signal || result.status !== 0) {
    const error = new Error("COFF " + label + " failed", {cause: result.error});
    Object.assign(error, {coffResult: captured}); throw error;
  }
  if (Buffer.byteLength(captured.text) > COFF_LOG_BYTES) {
    const error = new Error("COFF " + label + " output is too large");
    Object.assign(error, {coffResult: {stdout: "", stderr: "output exceeded evidence cap", text: "output exceeded evidence cap\n"}}); throw error;
  }
  budget.logBytes += Buffer.byteLength(captured.text);
  if (budget.logBytes > COFF_TOTAL_LOG_BYTES) {throw new Error("COFF aggregate diagnostic budget exceeded");}
  return captured;
}

/** @param {string} output @param {string} label @param {string} tool @param {string[]} args @param {{text: string}} result */
function recordCoffResult(output, label, tool, args, result) {
  if (!/^[A-Za-z0-9_.-]+$/.test(label)) {throw new Error("Invalid COFF evidence label");}
  fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({tool, args}) + "\n", {flag: "wx"});
  fs.writeFileSync(path.join(output, label + ".log"), result.text, {flag: "wx"});
}

/** @param {string} output @param {string} label @param {string} tool @param {string[]} args @param {CoffBudget} budget */
function recordCoffTool(output, label, tool, args, budget) {
  try {
    const result = invokeCoffTool(tool, args, output, label, budget); recordCoffResult(output, label, tool, args, result); return result;
  } catch (error) {
    const failed = /** @type {{coffResult?: {text: string}}} */ (error);
    if (failed.coffResult) {recordCoffResult(output, label, tool, args, failed.coffResult);}
    throw error;
  }
}

/** @param {string} tool @param {string} label @param {CoffBudget} budget */
function coffToolRecord(tool, label, budget) {
  const bytes = coffRegularFile(tool, label, COFF_TOOL_BYTES).size, sha256 = fileHash(tool), realpath = fs.realpathSync(tool);
  const version = invokeCoffTool(tool, ["--version"], path.dirname(tool), label + " version", budget);
  verifyCoffStable(tool, bytes, sha256, "COFF tool");
  if (fs.realpathSync(tool) !== realpath) {throw new Error("COFF tool identity changed during admission");}
  if (!/\bLLVM\b/i.test(version.text)) {throw new Error(label + " is not an LLVM tool");}
  const trimmed = version.text.trim();
  return {path: tool, bytes, realpath, sha256, version: trimmed.slice(0, 4096),
    versionSha256: createHash("sha256").update(trimmed).digest("hex")};
}

/** @param {string} text @param {string} label @returns {string[]} */
function coffNmNames(text, label) {
  const names = [];
  for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (!line.trim()) {continue;}
    const fields = line.trim().split(/\s+/), name = fields[0] || "";
    if (!coffSymbolName(name) || fields.length < 2) {throw new Error("Malformed COFF " + label + " output");}
    names.push(name);
  }
  if (new Set(names).size !== names.length) {throw new Error("Duplicate COFF " + label + " symbol");}
  return names;
}

/** @param {string} text @param {string} label @returns {{sections: Array<{number: number, name: string, rawDataSize: number, relocationCount: number, chars: string[]}>, selections: Array<{number: number, selection: number, association: number | undefined}>, relocations: Array<{section: number, offset: string, type: string, symbol: string}>, format: string, sectionCount: number}} */
function parseCoffInspection(text, label) {
  const normalized = text.replace(/\r\n?/g, "\n");
  if (!/^Format:\s+COFF-x86-64\s*$/m.test(normalized) || !/^Arch:\s+x86_64\s*$/m.test(normalized) ||
      !/^AddressSize:\s+64bit\s*$/m.test(normalized)) {throw new Error("Unsupported or unknown COFF format in " + label);}
  const sectionCountMatch = normalized.match(/^\s+SectionCount:\s+(\d+)\s*$/m), sectionStart = normalized.indexOf("\nSections [");
  if (!sectionCountMatch || sectionStart < 0) {throw new Error("Unresolved COFF section format in " + label);}
  const sectionBodyMatch = normalized.slice(sectionStart).match(/^\nSections \[\n([\s\S]*?)\n\]\n(?:Relocations \[|Symbols \[)/);
  if (!sectionBodyMatch || sectionBodyMatch[1] === undefined) {throw new Error("Unresolved COFF sections in " + label);}
  const sections = [], sectionBlocks = [];
  for (const match of sectionBodyMatch[1].matchAll(/(?:^|\n)\x20{2}Section \{\n([\s\S]*?)\n\x20{2}\}/g)) {
    const block = match[1] || "", number = Number(block.match(/^\s*Number:\s+(\d+)\s*$/m)?.[1]),
      rawText = block.match(/^\s*RawDataSize:\s+(\S+)\s*$/m)?.[1], relocationText = block.match(/^\s*RelocationCount:\s+(\d+)\s*$/m)?.[1],
      rawName = block.match(/^\s*Name:\s*(.*?)\s*$/m)?.[1], name = rawName?.replace(/\s+\([0-9A-Fa-f ]+\)$/, "").trim() || "";
    const rawDataSize = rawText === undefined ? NaN : Number(rawText), relocationCount = relocationText === undefined ? NaN : Number(relocationText);
    const chars = [...block.matchAll(/\b(IMAGE_SCN_[A-Z0-9_]+)/g)].map(item => item[1] || "");
    if (!Number.isSafeInteger(number) || number < 1 || !name || !Number.isSafeInteger(rawDataSize) || rawDataSize < 0 ||
        !Number.isSafeInteger(relocationCount) || relocationCount < 0 || !chars.length) {throw new Error("Malformed COFF section in " + label);}
    sections.push({number, name, rawDataSize, relocationCount, chars}); sectionBlocks.push({number, block});
  }
  if (sections.length !== Number(sectionCountMatch[1]) || new Set(sections.map(section => section.number)).size !== sections.length ||
      sections.some((section, index) => section.number !== index + 1)) {
    throw new Error("COFF section count or numbering is unresolved in " + label);
  }
  const relocations = [];
  for (const {number: section, block} of sectionBlocks) {
    const body = block.match(/\n\x20{4}Relocations \[\n([\s\S]*?)\n\x20{4}\]/)?.[1] || "";
    for (const line of body.split("\n")) {
      if (!line.trim()) {continue;}
      const item = line.match(/^\s*(0x[0-9A-Fa-f]+)\s+(\S+)\s+(.+?)(?:\s+\(\d+\))?\s*$/);
      if (!item || !item[1] || !item[2] || !item[3]) {throw new Error("Malformed COFF relocation in " + label);}
      relocations.push({section, offset: item[1], type: item[2], symbol: item[3].trim()});
    }
  }
  const symbolStart = normalized.indexOf("\nSymbols [");
  if (symbolStart < 0) {throw new Error("Unresolved COFF symbol format in " + label);}
  for (const section of sections) {
    if (section.relocationCount !== relocations.filter(relocation => relocation.section === section.number).length) {
      throw new Error("COFF relocation count is unresolved in " + label);
    }
  }
  const symbolBody = normalized.slice(symbolStart), symbols = [...symbolBody.matchAll(/(?:^|\n)\x20{2}Symbol \{\n([\s\S]*?)\n\x20{2}\}/g)];
  if (!symbols.length) {throw new Error("Unresolved COFF symbol format in " + label);}
  const selections = [];
  for (const match of symbols) {
    const block = match[1] || "", aux = block.match(/\n\x20{4}AuxSectionDef \{\n([\s\S]*?)\n\x20{4}\}/), auxBlock = aux?.[1] || "";
    const selectionLine = auxBlock.match(/^\s*Selection:\s*(?:(?:[A-Za-z]+)\s+)?\((0x[0-9A-Fa-f]+)\)\s*$/m) ||
      auxBlock.match(/^\s*Selection:\s*(0x[0-9A-Fa-f]+)\s*$/m);
    if (!auxBlock || !selectionLine || !selectionLine[1]) {continue;}
    const number = Number(block.match(/^\s*Section:\s*.*\((-?\d+)\)\s*$/m)?.[1]), selection = Number.parseInt(selectionLine[1], 16),
      associationText = auxBlock.match(/^\s*AssocSection:\s*.*\((-?\d+)\)\s*$/m)?.[1], association = associationText === undefined ? undefined : Number(associationText);
    if (!Number.isSafeInteger(number) || number < 1 || !Number.isInteger(selection) || selection < 0 || selection > 7 ||
        (selection === 5 && (association === undefined || !Number.isSafeInteger(association) || association < 1)) || (selection !== 5 && association !== undefined)) {
      throw new Error("Unsupported COFF COMDAT selection in " + label);
    }
    selections.push({number, selection, association});
  }
  if (!selections.length) {throw new Error("COFF has no section selection metadata in " + label);}
  const sectionByNumber = new Map(sections.map(section => [section.number, section]));
  const selectedNumbers = selections.filter(item => item.selection !== 0).map(item => item.number), comdatNumbers = sections.filter(section => section.chars.includes("IMAGE_SCN_LNK_COMDAT")).map(section => section.number);
  if (new Set(selections.map(item => item.number)).size !== selections.length ||
      selectedNumbers.some(number => !sectionByNumber.get(number)?.chars.includes("IMAGE_SCN_LNK_COMDAT")) ||
      selections.some(item => item.selection === 5 && (!sectionByNumber.get(item.association || 0)?.chars.includes("IMAGE_SCN_LNK_COMDAT") || item.association === item.number)) ||
      JSON.stringify([...selectedNumbers].sort((a, b) => a - b)) !== JSON.stringify([...comdatNumbers].sort((a, b) => a - b))) {
    throw new Error("COFF COMDAT section keys are unresolved in " + label);
  }
  return {sections, selections, relocations, format: "COFF-x86-64", sectionCount: sections.length};
}

/** @param {ReturnType<typeof parseCoffInspection>} inspection */
function coffMetadata(inspection) {
  return {
    sections: inspection.sections.map(({number, name, rawDataSize, relocationCount, chars}) => ({number, name, rawDataSize, relocationCount, chars})),
    selections: inspection.selections.map(({number, selection, association}) => ({number, selection, association})),
    associativeXdata: inspection.selections.filter(item => item.selection === 5 && inspection.sections.find(section => section.number === item.number)?.name === ".xdata")
      .map(({number, association}) => ({number, association}))
  };
}

/** Raw LLVM reports already retain full structural evidence. Keep the manifest compact without
 * weakening any comparison; separate linker-text changes from immutable code/data metadata.
 * @param {ReturnType<typeof coffMetadata>} metadata @param {ReturnType<typeof coffPayloads>} payloads @param {number} directive */
function coffEvidenceDigests(metadata, payloads, directive) {
  const digest = (/** @type {unknown} */ value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
  return {metadataSha256: digest(metadata), payloadSha256: digest(payloads),
    nonDirectiveMetadataSha256: digest({...metadata, sections: metadata.sections.filter(section => section.number !== directive)}),
    nonDirectivePayloadSha256: digest(payloads.filter(section => section.number !== directive))};
}

/** @param {Array<{section: number, offset: string, type: string, symbol: string}>} relocations @param {Set<string>} definitions @param {Set<string>} publicNames @param {string} prefix */
function coffRelocationDigest(relocations, definitions, publicNames, prefix) {
  const mapped = relocations.map(relocation => ({...relocation, symbol: definitions.has(relocation.symbol) && !publicNames.has(relocation.symbol) ? prefix + relocation.symbol : relocation.symbol}));
  return {count: mapped.length, sha256: createHash("sha256").update(JSON.stringify(mapped)).digest("hex")};
}

/** @param {string} file @param {number} bytes @param {string} sha256 @param {string} label */
function verifyCoffStable(file, bytes, sha256, label) {
  const stat = coffRegularFile(file, label, COFF_OBJECT_BYTES);
  if (stat.size !== bytes || fileHash(file) !== sha256) {throw new Error(label + " changed during isolation");}
}

/** Hash original section data, not file offsets or symbol-table bytes that objcopy may relocate.
 * Stream bounded COFF/bigobj payloads directly: dumping sections through objcopy can rewrite
 * its input, and metadata/relocation equality alone does not prove host code survived.
 * @param {string} file @param {(section: number, data: Buffer) => void} [visit] */
function coffPayloads(file, visit) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const stat = fs.fstatSync(fd), header = Buffer.alloc(56), section = Buffer.alloc(40), chunk = Buffer.allocUnsafe(1 << 20);
    if (!stat.isFile() || stat.size < 20 || stat.size > COFF_OBJECT_BYTES || fs.readSync(fd, header, 0, 20, 0) !== 20) {throw new Error("Invalid bounded COFF payload input");}
    const big = header.readUInt16LE(0) === 0 && header.readUInt16LE(2) === 0xffff;
    if (big && fs.readSync(fd, header, 20, 36, 20) !== 36) {throw new Error("Truncated COFF bigobj header");}
    const start = big ? 56 : 20, count = big ? header.readUInt32LE(44) : header.readUInt16LE(2);
    if ((big ? header.readUInt16LE(6) !== 0x8664 || header.readUInt16LE(4) < 2 : header.readUInt16LE(0) !== 0x8664 || header.readUInt16LE(16) !== 0) ||
        !count || count > COFF_TOTAL_SECTIONS || start + count * 40 > stat.size) {throw new Error("Unsupported COFF payload header");}
    const payloads = []; let total = 0;
    for (let index = 0; index < count; index++) {
      if (fs.readSync(fd, section, 0, 40, start + index * 40) !== 40) {throw new Error("Truncated COFF section table");}
      const bytes = section.readUInt32LE(16), offset = section.readUInt32LE(20), uninitialized = Boolean(section.readUInt32LE(36) & 0x80);
      total += bytes;
      if (total > COFF_OBJECT_BYTES || (bytes && (!offset ? !uninitialized : offset < start + count * 40 || offset + bytes > stat.size))) {throw new Error("Invalid bounded COFF section payload");}
      const hash = createHash("sha256");
      // LLVM can materialize an uninitialized .bss payload as explicit zeros. Its logical
      // bytes, size and flags must match; the presence of raw storage is not semantic.
      if (!offset) {chunk.fill(0, 0, Math.min(bytes, chunk.length));}
      for (let position = 0; position < bytes;) {
        const wanted = Math.min(chunk.length, bytes - position), read = offset ? fs.readSync(fd, chunk, 0, wanted, offset + position) : wanted;
        if (!read) {throw new Error("Truncated COFF section payload");}
        hash.update(chunk.subarray(0, read)); visit?.(index + 1, chunk.subarray(0, read)); position += read;
      }
      payloads.push({number: index + 1, bytes, virtualSize: section.readUInt32LE(8), characteristics: section.readUInt32LE(36), sha256: hash.digest("hex")});
    }
    return payloads;
  } finally {fs.closeSync(fd);}
}

/** Symbol references also occur as linker text, not COFF relocations. Keep CRT/library policy
 * literal and change only mapped INCLUDE/EXPORT/ALTERNATENAME names, preserving quoting.
 * @param {string} text @param {Map<string, string>} names */
function rewriteCoffDirectives(text, names) {
  /** @type {Array<[string, string]>} */
  const alternates = [];
  let rewritten = 0;
  const output = text.replace(/([/-](?:include|export|alternatename):)(?:"([^"\0]*)"|([^\s"\0]+))/gi, (_whole, option, quoted, bare) => {
    const argument = quoted === undefined ? bare : quoted, kind = option.toLowerCase();
    const split = kind.endsWith("export:") ? argument.indexOf(",") : -1;
    const reference = split < 0 ? argument : argument.slice(0, split), suffix = split < 0 ? "" : argument.slice(split);
    const parts = reference.split("=");
    if (parts.some((/** @type {string} */ part) => !coffSymbolName(part)) || parts.length > 2 ||
        (kind.endsWith("alternatename:") ? parts.length !== 2 : kind.endsWith("include:") && parts.length !== 1)) {throw new Error("Unsupported COFF symbol directive");}
    if (kind.endsWith("alternatename:")) {alternates.push([parts[0], parts[1]]);}
    const mapped = parts.map((/** @type {string} */ name) => {const replacement = names.get(name); if (replacement) {rewritten++;} return replacement || name;}).join("=") + suffix;
    return option + (quoted === undefined ? mapped : '"' + mapped + '"');
  });
  return {text: output, rewritten, alternates};
}

/** Read only the bounded linker-text section; hashing still streams all code/data unchanged.
 * @param {string} file @param {number} number */
function coffDirectiveText(file, number) {
  /** @type {Buffer[]} */
  const chunks = []; let size = 0;
  coffPayloads(file, (section, data) => {
    if (section !== number) {return;}
    if ((size += data.length) > 1 << 20) {throw new Error("COFF directives exceed the text budget");}
    chunks.push(Buffer.from(data));
  });
  const data = Buffer.concat(chunks), text = data.toString("utf8");
  if (!Buffer.from(text).equals(data)) {throw new Error("COFF directives are not valid UTF-8");}
  return text;
}

/** LLVM's COFF update-section cannot grow data. Append new linker text to a private copy,
 * update only its raw-data header/section length, then let objcopy rebuild the final object.
 * Original code, relocations, symbol indices and the source file remain untouched.
 * @param {string} input @param {string} output @param {number} number @param {string} text */
function stageCoffDirectives(input, output, number, text) {
  fs.copyFileSync(input, output, fs.constants.COPYFILE_EXCL);
  const fd = fs.openSync(output, "r+");
  try {
    const size = fs.fstatSync(fd).size, header = Buffer.alloc(56), data = Buffer.from(text), patch = Buffer.alloc(8);
    if (fs.readSync(fd, header, 0, 20, 0) !== 20) {throw new Error("Truncated COFF directive staging header");}
    const big = header.readUInt16LE(0) === 0 && header.readUInt16LE(2) === 0xffff;
    if (big && fs.readSync(fd, header, 20, 36, 20) !== 36) {throw new Error("Truncated bigobj directive staging header");}
    const start = big ? 56 : 20, count = big ? header.readUInt32LE(44) : header.readUInt16LE(2), offset = Math.ceil(size / 4) * 4;
    if (number < 1 || number > count || offset + data.length > COFF_OBJECT_BYTES) {throw new Error("COFF directive staging exceeds bounds");}
    fs.writeSync(fd, data, 0, data.length, offset);
    patch.writeUInt32LE(data.length, 0); patch.writeUInt32LE(offset, 4);
    fs.writeSync(fd, patch, 0, 8, start + (number - 1) * 40 + 16);
    // The section-definition auxiliary symbol repeats the raw-data length. Keep that
    // metadata consistent as well; ordinary symbols and COMDAT selection never change.
    const symbols = header.readUInt32LE(big ? 48 : 8), symbolCount = header.readUInt32LE(big ? 52 : 12), symbolBytes = big ? 20 : 18;
    if (symbols + symbolCount * symbolBytes > size) {throw new Error("Invalid staged COFF symbol table");}
    const symbol = Buffer.alloc(symbolBytes);
    for (let index = 0; index < symbolCount;) {
      if (fs.readSync(fd, symbol, 0, symbolBytes, symbols + index * symbolBytes) !== symbolBytes) {throw new Error("Truncated staged COFF symbol");}
      const auxiliaries = symbol[symbolBytes - 1] || 0, section = big ? symbol.readInt32LE(12) : symbol.readInt16LE(12);
      if (index + auxiliaries >= symbolCount) {throw new Error("Invalid staged COFF auxiliary count");}
      if (section === number && symbol[symbolBytes - 2] === 3 && auxiliaries === 1 && symbol.toString("ascii", 0, 8) === ".drectve") {
        fs.writeSync(fd, patch, 0, 4, symbols + (index + 1) * symbolBytes);
      }
      index += 1 + auxiliaries;
    }
  } finally {fs.closeSync(fd);}
}

/** @typedef {{label: string, tool: string, args: string[], result: {text: string}}} CoffEvidence
 * @typedef {{path: string, bytes: number, sha256: string, payloads: ReturnType<typeof coffPayloads>, inspection: ReturnType<typeof parseCoffInspection>, definitions: string[], undefinedSymbols: string[], metadata: ReturnType<typeof coffMetadata>, relocationDigest: {count: number, sha256: string}, directives: {number: number, original: string, text: string, rewritten: number, alternates: Array<[string, string]>}, evidence: CoffEvidence[]}} CoffRecord */

/** @param {unknown} value @returns {{tools: {llvmObjcopy: string, llvmReadobj: string, llvmNm: string}, objects: string[], output: string, prefix: string, entryPoints: string[]}} */
function validateCoffOptions(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("COFF isolation needs an explicit options object");}
  const options = /** @type {{tools: unknown, objects: unknown, output: unknown, prefix: unknown, entryPoints: unknown}} */ (value), keys = Object.keys(options).sort();
  if (keys.join(",") !== "entryPoints,objects,output,prefix,tools") {throw new Error("COFF isolation options are incomplete or contain unsupported fields");}
  const output = /** @type {string} */ (options.output);
  if (!coffAbsolutePath(output) || fs.existsSync(output) || !fs.statSync(path.dirname(output), {throwIfNoEntry: false})?.isDirectory()) {
    throw new Error("COFF isolation output must be a new absolute directory in an existing directory");
  }
  const rawObjects = options.objects;
  if (!Array.isArray(rawObjects) || !rawObjects.length || rawObjects.length > COFF_OBJECT_LIMIT ||
      rawObjects.some(file => !coffAbsolutePath(file))) {throw new Error("COFF isolation objects must be bounded absolute paths");}
  const objects = /** @type {string[]} */ (rawObjects);
  const seen = new Set(); let total = 0;
  for (const [index, file] of objects.entries()) {
    const stat = coffRegularFile(file, "COFF object " + index, COFF_OBJECT_BYTES), real = fs.realpathSync(file);
    if (seen.has(real)) {throw new Error("COFF isolation objects must be distinct");}
    seen.add(real); total += stat.size;
  }
  if (total > COFF_TOTAL_BYTES) {throw new Error("COFF isolation inputs exceed the bounded total size");}
  const prefix = /** @type {string} */ (options.prefix);
  if (typeof options.prefix !== "string" || !/^[A-Za-z_][A-Za-z0-9_]{0,63}_$/.test(prefix)) {
    throw new Error("COFF isolation needs a private symbol prefix");
  }
  const rawEntryPoints = options.entryPoints;
  if (!Array.isArray(rawEntryPoints) || !rawEntryPoints.length || rawEntryPoints.length > 1024 ||
      rawEntryPoints.some(name => typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) ||
      new Set(rawEntryPoints).size !== rawEntryPoints.length || rawEntryPoints.some(name => name.startsWith(prefix))) {
    throw new Error("COFF isolation needs unique public C factory names");
  }
  const valueTools = options.tools;
  if (!hasExactBuildFields(valueTools, COFF_TOOL_KEYS) ||
      COFF_TOOL_KEYS.some(name => !coffAbsolutePath(/** @type {Record<string, unknown>} */ (valueTools)[name]))) {
    throw new Error("COFF isolation needs three absolute LLVM tools");
  }
  const tools = /** @type {{llvmObjcopy: string, llvmReadobj: string, llvmNm: string}} */ (valueTools);
  return {tools, objects, output, prefix, entryPoints: /** @type {string[]} */ (rawEntryPoints)};
}

/** An unresolved alternate is external ABI glue, not a private definition. Preserve its
 * fallback chain so the CRT can still supply a strong override. Shared by object and archive
 * isolation; renaming either end independently can disable overrides or create link conflicts.
 * @param {Set<string>} definitions @param {Array<[string, string]>} alternatives @param {number} [maximum] */
function coffExternalFallbacks(definitions, alternatives, maximum = COFF_TOTAL_SYMBOLS) {
  const aliases = new Map(), shared = new Set();
  if (!Number.isInteger(maximum) || maximum < 0 || maximum > COFF_TOTAL_SYMBOLS) {throw new Error("Invalid COFF fallback budget");}
  for (const [name, fallback] of alternatives) {
    if (aliases.has(name) && aliases.get(name) !== fallback) {throw new Error("Conflicting COFF fallback aliases");}
    aliases.set(name, fallback);
    if (aliases.size > maximum) {throw new Error("COFF aggregate metadata budget exceeded");}
  }
  const pending = [...aliases.keys()].filter(name => !definitions.has(name));
  for (let name; (name = pending.pop()) !== undefined;) {
    if (shared.has(name)) {continue;}
    shared.add(name);
    const fallback = !definitions.has(name) && aliases.get(name);
    if (fallback) {pending.push(fallback);}
  }
  return shared;
}

/** Isolate a complete set of trusted Windows x64 COFF objects with one map. LLVM's COFF
 * inspection is the format boundary: section selection and associated unwind sections are
 * compared before/after, while only complete external definitions and their relocations are
 * renamed. The producer supplies the graph closure; unresolved SDK/runtime imports remain
 * unchanged. No ELF partial link or runtime library loading is involved.
 * @param {{tools: {llvmObjcopy: string, llvmReadobj: string, llvmNm: string}, objects: string[], output: string, prefix: string, entryPoints: string[]}} options */
function isolateCoffObjects(options) {
  const selected = validateCoffOptions(options), budget = {logBytes: 0, sections: 0, relocations: 0, symbols: 0}, toolRecords = {
    llvmObjcopy: coffToolRecord(selected.tools.llvmObjcopy, "llvm-objcopy", budget),
    llvmReadobj: coffToolRecord(selected.tools.llvmReadobj, "llvm-readobj", budget),
    llvmNm: coffToolRecord(selected.tools.llvmNm, "llvm-nm", budget)
  };
  const records = /** @type {CoffRecord[]} */ (selected.objects.map((object, index) => {
    const stat = coffRegularFile(object, "COFF input " + index, COFF_OBJECT_BYTES), sha256 = fileHash(object);
    const readobjArgs = ["--file-headers", "--sections", "--section-relocations", "--symbols", object],
      definitionsArgs = ["-P", "-g", "--defined-only", object], undefinedArgs = ["-P", "--undefined-only", object];
    const readobj = invokeCoffTool(selected.tools.llvmReadobj, readobjArgs, path.dirname(object), "COFF input " + index + " readobj", budget),
      inspection = parseCoffInspection(readobj.stdout, object),
      definitionsOutput = invokeCoffTool(selected.tools.llvmNm, definitionsArgs, path.dirname(object), "COFF input " + index + " definitions", budget),
      undefinedOutput = invokeCoffTool(selected.tools.llvmNm, undefinedArgs, path.dirname(object), "COFF input " + index + " undefined", budget),
      definitions = coffNmNames(definitionsOutput.stdout, "definitions"), undefinedSymbols = coffNmNames(undefinedOutput.stdout, "undefined");
    budget.sections += inspection.sections.length; budget.relocations += inspection.relocations.length; budget.symbols += definitions.length + undefinedSymbols.length;
    if (budget.sections > COFF_TOTAL_SECTIONS || budget.relocations > COFF_TOTAL_RELOCATIONS || budget.symbols > COFF_TOTAL_SYMBOLS) {throw new Error("COFF aggregate metadata budget exceeded");}
    const sections = inspection.sections.filter(section => section.name === ".drectve");
    if (sections.length > 1 || sections.some(section => section.relocationCount)) {throw new Error("Unsupported COFF directive section");}
    const number = sections[0]?.number || 0, original = number ? coffDirectiveText(object, number) : "";
    return {path: object, bytes: stat.size, sha256, payloads: coffPayloads(object), inspection, definitions, undefinedSymbols, metadata: coffMetadata(inspection),
      relocationDigest: {count: 0, sha256: ""}, directives: {number, original, ...rewriteCoffDirectives(original, new Map())}, evidence: [
        {label: "input-" + index + "-readobj", tool: selected.tools.llvmReadobj, args: readobjArgs, result: {text: readobj.text}},
        {label: "input-" + index + "-definitions", tool: selected.tools.llvmNm, args: definitionsArgs, result: {text: definitionsOutput.text}},
        {label: "input-" + index + "-undefined", tool: selected.tools.llvmNm, args: undefinedArgs, result: {text: undefinedOutput.text}},
      ]};
  }));
  const allDefinitions = new Set(records.flatMap(record => record.definitions)), entrySet = new Set(selected.entryPoints), publicCounts = new Map();
  for (const record of records) {
    for (const name of record.definitions) {publicCounts.set(name, (publicCounts.get(name) || 0) + (entrySet.has(name) ? 1 : 0));}
  }
  if (selected.entryPoints.some(name => !allDefinitions.has(name) || publicCounts.get(name) !== 1)) {throw new Error("COFF public factory definitions are missing or duplicated");}
  const sharedFallbackSymbols = coffExternalFallbacks(allDefinitions, records.flatMap(record => record.directives.alternates), COFF_TOTAL_SYMBOLS - budget.symbols);
  for (const name of sharedFallbackSymbols) {entrySet.add(name);}
  const undefinedImports = new Set(records.flatMap(record => record.undefinedSymbols));
  const privateDefinitions = [...allDefinitions].filter(name => !entrySet.has(name)).sort();
  if (!privateDefinitions.length || privateDefinitions.some(name => name.startsWith(selected.prefix))) {throw new Error("COFF graph has no fresh private definitions or is already isolated");}
  const mapEntries = privateDefinitions.map(name => ({from: name, to: selected.prefix + name}));
  const directiveNames = new Map(mapEntries.map(entry => [entry.from, entry.to]));
  const mappedNames = new Set(mapEntries.map(entry => entry.to));
  if (mappedNames.size !== mapEntries.length || [...mappedNames].some(name => !coffSymbolName(name) || allDefinitions.has(name) || undefinedImports.has(name)) ||
      records.some(record => record.undefinedSymbols.some(name => name.startsWith(selected.prefix)))) {
    throw new Error("COFF private symbol mapping collides with an existing definition or import");
  }
  for (const record of records) {
    Object.assign(record.directives, rewriteCoffDirectives(record.directives.original, directiveNames));
    record.relocationDigest = coffRelocationDigest(record.inspection.relocations, allDefinitions, entrySet, selected.prefix);
    record.inspection.relocations = [];
  }
  for (const record of records) {
    verifyCoffStable(record.path, record.bytes, record.sha256, "COFF input");
  }
  const mapPath = path.join(selected.output, "rename.map");
  // All semantic checks above run before creating the output directory. A malformed object,
  // unsupported format, or bad trust boundary therefore cannot create a partial receipt.
  fs.mkdirSync(selected.output);
  for (const record of records) {
    verifyCoffStable(record.path, record.bytes, record.sha256, "COFF input");
    // Capture the original bounded inspection, not a second execution. Rehashing proves these
    // are still the same object bytes; duplicate LLVM dumps waste both the log cap and disk.
    for (const evidence of record.evidence) {recordCoffResult(selected.output, evidence.label, evidence.tool, evidence.args, evidence.result);}
    record.evidence = [];
  }
  const mapText = mapEntries.map(entry => entry.from + " " + entry.to).join("\n") + "\n";
  fs.writeFileSync(mapPath, mapText, {flag: "wx"});
  const outputs = [];
  for (const [index, record] of records.entries()) {
    verifyCoffStable(record.path, record.bytes, record.sha256, "COFF input");
    const outputObject = path.join(selected.output, "object-" + String(index).padStart(4, "0") + ".obj"),
      isolateArgs = ["--redefine-syms=" + mapPath, record.path, outputObject];
    if (record.directives.rewritten) {
      const file = path.join(selected.output, "directives-" + index + ".txt");
      fs.writeFileSync(file, record.directives.text, {flag: "wx"});
      const staged = path.join(selected.output, "directives-" + index + ".obj");
      stageCoffDirectives(record.path, staged, record.directives.number, record.directives.text);
      isolateArgs[1] = staged;
    }
    recordCoffTool(selected.output, "isolate-" + index, selected.tools.llvmObjcopy, isolateArgs, budget);
    if (fs.lstatSync(outputObject).isSymbolicLink()) {throw new Error("COFF isolated output must not be a symlink");}
    coffRegularFile(outputObject, "COFF isolated object " + index, COFF_OBJECT_BYTES);
    const readobjArgs = ["--file-headers", "--sections", "--section-relocations", "--symbols", outputObject], definitionsArgs = ["-P", "-g", "--defined-only", outputObject],
      undefinedArgs = ["-P", "--undefined-only", outputObject], readobj = recordCoffTool(selected.output, "output-" + index + "-readobj", selected.tools.llvmReadobj, readobjArgs, budget),
      inspection = parseCoffInspection(readobj.stdout, outputObject), definitionsOutput = recordCoffTool(selected.output, "output-" + index + "-definitions", selected.tools.llvmNm, definitionsArgs, budget),
      undefinedOutput = recordCoffTool(selected.output, "output-" + index + "-undefined", selected.tools.llvmNm, undefinedArgs, budget), definitions = coffNmNames(definitionsOutput.stdout, "output definitions"),
      undefinedSymbols = coffNmNames(undefinedOutput.stdout, "output undefined");
    const expectedDefinitions = record.definitions.map(name => entrySet.has(name) ? name : selected.prefix + name).sort(), expectedUndefined = record.undefinedSymbols.map(name => allDefinitions.has(name) && !entrySet.has(name) ? selected.prefix + name : name).sort(),
      actualDefinitions = definitions.slice().sort(), actualUndefined = undefinedSymbols.slice().sort(), relocationDigest = coffRelocationDigest(inspection.relocations, allDefinitions, entrySet, selected.prefix);
    const expectedMetadata = {...record.metadata, sections: record.metadata.sections.map(section => section.number === record.directives.number && record.directives.rewritten
      ? {...section, rawDataSize: Buffer.byteLength(record.directives.text)} : section)};
    if (JSON.stringify(actualDefinitions) !== JSON.stringify(expectedDefinitions) || JSON.stringify(actualUndefined) !== JSON.stringify(expectedUndefined) ||
        JSON.stringify(coffMetadata(inspection)) !== JSON.stringify(expectedMetadata) || relocationDigest.count !== record.relocationDigest.count || relocationDigest.sha256 !== record.relocationDigest.sha256) {
      throw new Error("COFF graph isolation changed definitions, imports, relocations or section metadata");
    }
    if (actualDefinitions.some(name => privateDefinitions.includes(name)) || actualUndefined.some(name => privateDefinitions.includes(name)) ||
        inspection.relocations.some(relocation => privateDefinitions.includes(relocation.symbol))) {throw new Error("COFF graph retains an old private collision definition or reference");}
    const payloads = coffPayloads(outputObject);
    const expectedPayloads = record.payloads.map(section => section.number === record.directives.number && record.directives.rewritten
      ? {...section, bytes: Buffer.byteLength(record.directives.text), sha256: createHash("sha256").update(record.directives.text).digest("hex")} : section);
    if (JSON.stringify(payloads) !== JSON.stringify(expectedPayloads)) {throw new Error("COFF graph isolation changed section payload bytes");}
    outputs.push({path: outputObject, bytes: fs.statSync(outputObject).size, sha256: fileHash(outputObject), definitions: definitions.length,
      undefined: undefinedSymbols.length, sections: inspection.sections.length,
      ...coffEvidenceDigests(coffMetadata(inspection), payloads, record.directives.number),
      directives: {rewritten: record.directives.rewritten, inputSha256: createHash("sha256").update(record.directives.original).digest("hex"), outputSha256: createHash("sha256").update(record.directives.text).digest("hex")}});
  }
  for (const record of records) {verifyCoffStable(record.path, record.bytes, record.sha256, "COFF input");}
  for (const tool of Object.values(toolRecords)) {
    verifyCoffStable(tool.path, tool.bytes, tool.sha256, "COFF tool");
    if (fs.realpathSync(tool.path) !== tool.realpath) {throw new Error("COFF tool identity changed during isolation");}
  }
  for (const output of outputs) {verifyCoffStable(output.path, output.bytes, output.sha256, "COFF isolated output");}
  const receipt = {version: 1, kind: "windows-coff-graph-isolation", scope: "Provided COFF private definitions isolated; public C factories and external ABI fallback bindings preserved; SDK host/runtime and SYCL images remain outside this operation",
    output: selected.output, prefix: selected.prefix, entryPoints: selected.entryPoints, tools: toolRecords, budget,
    sharedFallbackSymbols: [...sharedFallbackSymbols].sort(),
    inputs: records.map(record => ({path: record.path, bytes: record.bytes, sha256: record.sha256, definitions: record.definitions.length,
      undefined: record.undefinedSymbols.length, sections: record.metadata.sections.length, comdat: record.metadata.selections.filter(item => item.selection !== 0).length,
      associativeXdata: record.metadata.associativeXdata.length, ...coffEvidenceDigests(record.metadata, record.payloads, record.directives.number)})),
    map: {path: mapPath, sha256: fileHash(mapPath), entries: mapEntries}, outputs,
    checks: {sectionMetadataPreserved: records.every(record => !record.directives.rewritten), sectionPayloadsPreserved: records.every(record => !record.directives.rewritten),
      nonDirectiveMetadataPreserved: true, nonDirectivePayloadsPreserved: true, linkerDirectiveReferencesPreserved: true, externalFallbackAliasesPreserved: true,
      comdatSelectionPreserved: true, associativeXdataPreserved: true,
      oldCollisionDefinitions: 0, oldCollisionReferences: 0, privateUndefinedReferencesRenamed: true, externalUndefinedImportsPreserved: true}};
  const receiptText = JSON.stringify(receipt, null, 2) + "\n";
  if (Buffer.byteLength(receiptText) > COFF_RECEIPT_BYTES) {throw new Error("COFF aggregate receipt budget exceeded");}
  fs.writeFileSync(path.join(selected.output, "result.json"), receiptText, {flag: "wx"});
  return receipt;
}

/** Opt-in build-only prototype. Normal Engineering/build:release never runs this compiler link. */
function main() {
  const [build, output] = process.argv.slice(2);
  if (process.env["MOM_AOT_BUILD_ONLY"] !== "1" || process.platform !== "linux" ||
      !build || !output || process.argv.length !== 4 || !path.isAbsolute(build) || !path.isAbsolute(output)) {
    throw new Error("Usage (Linux, MOM_AOT_BUILD_ONLY=1): native-compiler-isolation.js <absolute original JIT build> <new output directory>");
  }
  fs.mkdirSync(output);
  const commands = run(output, "ninja", ["-C", build, "-t", "commands", "native-dpcpp-jit"], "original-graph");
  const archives = compilerArchives(commands, build);
  for (const file of archives) {
    if (!isStaticArchive(file)) {throw new Error("Compiler requires regular, non-thin static archives");}
  }
  const input = path.join(output, "compiler-raw.o");
  run(output, "ld", ["-r", "--whole-archive", archives[0] || "", "--no-whole-archive", "--start-group", ...archives.slice(1), "--end-group", "-o", input], "partial-link");
  const isolated = isolateCompilerObject(input, path.join(output, "isolated"), {prefix: "mom_dpcpp23_", entryPoints: dpcppEntries});
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({archives, ...isolated,
    scope: "Original static compiler graph isolated; NOT execution, host-JIT bridge, GPU or release qualification"}, null, 2) + "\n", {flag: "wx"});
  console.log(`Static DPC++ compiler graph isolated; NOT executed: ${isolated.object}`);
}

module.exports = {dpcppEntries, isolateCompilerObject, isolateCoffObjects, coffExternalFallbacks, rewriteCoffDirectives, stageCoffDirectives, coffSectionPayloads: coffPayloads, groupSignatures, compilerArchives};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
