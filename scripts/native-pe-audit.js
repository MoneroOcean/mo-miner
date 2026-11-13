"use strict";

const fs = require("node:fs"), {createHash} = require("node:crypto");
const {nativePePages} = require("./native-pe-pages");
const {elfRangeHash: fileRangeHash} = require("./native-elf");

/** @typedef {{rva: number, bytes: number}} MetadataRange */
/** Open only the supported native PE shape. Keep metadata reads bounded and leave large
 * application/JIT payloads on disk; this is an offline audit, not a Windows loader.
 * @param {string} file */
function openImage(file) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const stat = fs.fstatSync(fd, {bigint: true}), header = Buffer.alloc(4096);
    if (!stat.isFile() || fs.readSync(fd, header, 0, header.length, 0) !== header.length) {throw new Error("Missing native PE audit input");}
    const layout = nativePePages(header, Number(stat.size)), pe = header.readUInt32LE(60), optional = pe + 24, table = optional + 240;
    const sections = layout.sections.map((section, i) => ({...section, offset: header.readUInt32LE(table + i * 40 + 20),
      raw: header.readUInt32LE(table + i * 40 + 16), virtual: header.readUInt32LE(table + i * 40 + 8)}));
    /** @param {number} rva @param {number} size */
    const read = (rva, size) => {
      if (!Number.isSafeInteger(rva) || !Number.isSafeInteger(size) || size < 0 || size > 1 << 20) {throw new Error("Invalid native PE metadata range");}
      const section = sections.find(s => rva >= s.rva && rva + size <= s.rva + s.raw);
      if (!section) {throw new Error("Native PE metadata escapes initialized sections");}
      const data = Buffer.alloc(size);
      if (fs.readSync(fd, data, 0, size, section.offset + rva - section.rva) !== size) {throw new Error("Truncated native PE metadata");}
      return data;
    };
    /** Preserve name bytes exactly; only DLL identity comparison is case-insensitive.
     * @param {number} rva */
    const string = rva => {
      const section = sections.find(s => rva >= s.rva && rva < s.rva + s.raw);
      if (!section) {throw new Error("Native PE name escapes initialized sections");}
      const data = read(rva, Math.min(16384, section.rva + section.raw - rva)), end = data.indexOf(0);
      if (end < 0) {throw new Error("Unterminated native PE name");}
      return data.subarray(0, end).toString("latin1");
    };
    const unchanged = () => {
      const after = fs.fstatSync(fd, {bigint: true});
      if (after.size !== stat.size || after.mtimeNs !== stat.mtimeNs || after.ctimeNs !== stat.ctimeNs) {throw new Error("Native PE changed during payload audit");}
    };
    return {file, fd, header, optional, sections, layout, read, string, unchanged};
  } catch (error) {fs.closeSync(fd); throw error;}
}
/** @typedef {ReturnType<typeof openImage>} Image */
/** @param {Image} image @param {number} index */
const directory = (image, index) => ({rva: image.header.readUInt32LE(image.optional + 112 + index * 8), bytes: image.header.readUInt32LE(image.optional + 116 + index * 8)});
/** @param {unknown} value */
const digest = value => createHash("sha256").update(JSON.stringify(value)).digest("hex");

/** UPX may rearrange export tables, but function RVAs, names, ordinals and forwarders must
 * retain their exact meaning. Sorted name pointers are required for Windows binary lookup.
 * @param {Image} image */
function exportsOf(image) {
  const region = directory(image, 0);
  if (!region.rva && !region.bytes) {return {bindings: null, metadata: []};}
  if (!region.rva || region.bytes < 40 || region.bytes > 1 << 20) {throw new Error("Invalid native PE export directory");}
  const data = image.read(region.rva, 40), base = data.readUInt32LE(16), count = data.readUInt32LE(20), named = data.readUInt32LE(24);
  if (!count || count > 16384 || named > count) {throw new Error("Invalid native PE export count");}
  const tables = [{rva: data.readUInt32LE(28), bytes: count * 4}, ...(named ? [
      {rva: data.readUInt32LE(32), bytes: named * 4}, {rva: data.readUInt32LE(36), bytes: named * 2},
    ] : [])], metadata = [{rva: region.rva, bytes: 40}, ...tables];
  const functions = image.read(tables[0]?.rva || 0, count * 4), names = named ? image.read(tables[1]?.rva || 0, named * 4) : Buffer.alloc(0),
    ordinals = named ? image.read(tables[2]?.rva || 0, named * 2) : Buffer.alloc(0);
  let metadataBytes = metadata.reduce((n, row) => n + row.bytes, 0);
  /** Exempt only referenced tables/strings, never unrelated bytes in the export directory.
   * @param {number} rva */
  const nameAt = rva => {
    const name = image.string(rva), bytes = name.length + 1; metadataBytes += bytes;
    if (metadataBytes > 1 << 20 || metadata.length >= 8192) {throw new Error("Excessive native PE export metadata");}
    metadata.push({rva, bytes}); return name;
  };
  const rows = Array.from({length: count}, (_, index) => {
    const rva = functions.readUInt32LE(index * 4);
    return {ordinal: base + index, rva, forwarder: rva >= region.rva && rva < region.rva + region.bytes ? nameAt(rva) : null, names: /** @type {string[]} */ ([])};
  });
  let previous = "";
  for (let i = 0; i < named; i++) {
    const index = ordinals.readUInt16LE(i * 2), name = nameAt(names.readUInt32LE(i * 4));
    if (index >= rows.length || !name || (i && name <= previous)) {throw new Error("Unsorted or invalid native PE export names");}
    rows[index]?.names.push(name); previous = name;
  }
  return {bindings: {base, module: nameAt(data.readUInt32LE(12)), rows}, metadata};
}

/** Compare actual DLL/function/ordinal-to-IAT bindings, not advisory hints or the optional
 * separate lookup table: Windows can resolve the exact same imports from the IAT itself.
 * @param {Image} image */
function importsOf(image) {
  const region = directory(image, 1);
  if (!region.rva || region.bytes < 40 || region.bytes > 4096 || region.bytes % 20) {throw new Error("Invalid native PE import directory");}
  const descriptors = image.read(region.rva, region.bytes), metadata = [region], rows = [];
  let terminated = false, symbolsRead = 0;
  for (let at = 0; at < descriptors.length; at += 20) {
    const lookup = descriptors.readUInt32LE(at), timestamp = descriptors.readUInt32LE(at + 4), chain = descriptors.readUInt32LE(at + 8),
      nameRva = descriptors.readUInt32LE(at + 12), iat = descriptors.readUInt32LE(at + 16);
    if (!lookup && !timestamp && !chain && !nameRva && !iat) {terminated = true; break;}
    const name = image.string(nameRva), symbols = [];
    if (!iat || rows.length >= 64 || !/^[A-Za-z0-9_.-]{1,255}\.dll$/i.test(name)) {throw new Error("Invalid native PE import identity");}
    metadata.push({rva: nameRva, bytes: name.length + 1});
    let count = 0;
    for (; count < 4096; count++) {
      const value = image.read((lookup || iat) + count * 8, 8).readBigUInt64LE();
      if (!value) {break;}
      if (++symbolsRead > 4096) {throw new Error("Excessive native PE import symbols");}
      if (value & (1n << 63n)) {
        if (value & ~((1n << 63n) | 65535n)) {throw new Error("Invalid native PE import ordinal");}
        symbols.push({ordinal: Number(value & 65535n)});
      } else {
        if (value > 0xffffffffn) {throw new Error("Invalid native PE import name RVA");}
        const rva = Number(value), symbol = image.string(rva + 2);
        if (!symbol) {throw new Error("Empty native PE imported symbol");}
        symbols.push({name: symbol}); metadata.push({rva, bytes: 2});
      }
    }
    if (count === 4096) {throw new Error("Unterminated native PE import lookup");}
    if (lookup && lookup !== iat) {metadata.push({rva: lookup, bytes: (count + 1) * 8});}
    rows.push({name: name.toLowerCase(), iat, timestamp, chain, symbols});
  }
  if (!terminated || !rows.length) {throw new Error("Unterminated native PE import descriptors");}
  return {bindings: rows.sort((a, b) => a.iat - b.iat), metadata};
}

/** Prove packing preserved initialized code, GPU images and JIT resources. Exclude only
 * bounded read-only loader metadata whose bindings were independently proved identical,
 * plus file alignment past VirtualSize. This does not qualify runtime/extraction/performance.
 * @param {string} original @param {string} restored */
function auditNativePePayload(original, restored) {
  const a = openImage(original);
  /** @type {Image | undefined} */
  let b;
  try {
    b = openImage(restored);
    if (digest(a.layout) !== digest(b.layout)) {throw new Error("Restored native PE layout differs");}
    const oldIat = directory(a, 12), newIat = directory(b, 12);
    if (digest(oldIat) !== digest(newIat) && (newIat.rva || newIat.bytes)) {throw new Error("Restored native PE optional IAT directory differs");}
    // Ignore only the checksum and the optional IAT directory which UPX-d removes. All
    // execution options, directory RVAs, image base, entrypoint, stack/heap limits stay exact.
    const options = (/** @type {Image} */ image) => {
      const data = Buffer.from(image.header.subarray(image.optional, image.optional + 240));
      data.fill(0, 64, 68); data.fill(0, 112 + 12 * 8, 120 + 12 * 8); return data;
    };
    if (!options(a).equals(options(b))) {throw new Error("Restored native PE execution header differs");}
    const ae = exportsOf(a), be = exportsOf(b), ai = importsOf(a), bi = importsOf(b);
    if (digest(ae.bindings) !== digest(be.bindings) || digest(ai.bindings) !== digest(bi.bindings)) {throw new Error("Restored native PE loader bindings differ");}
    const metadata = [...ae.metadata, ...be.metadata, ...ai.metadata, ...bi.metadata];
    if (metadata.length > 8192 || metadata.reduce((n, row) => n + row.bytes, 0) > 1 << 20) {throw new Error("Excessive native PE loader metadata");}
    for (const row of metadata) {
      const section = a.sections.find(s => row.rva >= s.rva && row.rva + row.bytes <= s.rva + Math.min(s.raw, s.virtual));
      if (!section || (section.flags & 0xe0000000) !== 0x40000000) {throw new Error("Native PE metadata is not bounded read-only loader data");}
    }
    const payloads = a.sections.map((section, index) => {
      const second = b?.sections[index]; if (!second) {throw new Error("Missing restored PE section");}
      const size = Math.min(section.raw, section.virtual), pieces = [], cuts = metadata.filter(row => row.rva >= section.rva && row.rva + row.bytes <= section.rva + size)
        .map(row => ({begin: row.rva - section.rva, end: row.rva + row.bytes - section.rva})).sort((x, y) => x.begin - y.begin);
      let cursor = 0;
      for (const cut of cuts) {if (cut.begin > cursor) {pieces.push([cursor, cut.begin]);} cursor = Math.max(cursor, cut.end);}
      if (cursor < size) {pieces.push([cursor, size]);}
      const checks = pieces.map(([begin = 0, end = 0]) => {
        const hash = fileRangeHash(original, section.offset + begin, end - begin);
        if (fileRangeHash(restored, second.offset + begin, end - begin) !== hash) {throw new Error("Restored native PE code/resource payload differs");}
        return {rva: section.rva + begin, bytes: end - begin, sha256: hash};
      });
      return {rva: section.rva, checks};
    });
    a.unchanged(); b.unchanged();
    return {exportBindings: digest(ae.bindings), importBindings: digest(ai.bindings), metadata, payloads};
  } finally {fs.closeSync(a.fd); if (b) {fs.closeSync(b.fd);}}
}

/** Packing must never admit a legacy/unguarded PE. Match every byte of the sealed fixed
 * map to the original image layout, including reserved zeros, before invoking any tool.
 * @param {string} file */
function verifyNativePeMap(file) {
  const image = openImage(file);
  try {
    const expected = Buffer.alloc(4096), actual = Buffer.alloc(4096), {layout} = image;
    [0x504d4f4d, 0x34393050, 1, layout.sections.length, layout.imageSize].forEach((value, index) => expected.writeUInt32LE(value, index * 4));
    layout.sections.forEach((row, index) => [row.rva, row.size, row.flags].forEach((value, field) => expected.writeUInt32LE(value, 32 + index * 12 + field * 4)));
    if (fs.readSync(image.fd, actual, 0, actual.length, layout.mapOffset) !== actual.length || !actual.equals(expected)) {throw new Error("Native PE packing requires an exact sealed pre-CRT page map");}
    image.unchanged(); return layout;
  } finally {fs.closeSync(image.fd);}
}

module.exports = {auditNativePePayload, verifyNativePeMap};
