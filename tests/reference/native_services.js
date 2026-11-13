"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const zlib = require("node:zlib");
const os = require("node:os");

// Private native hooks have an additional pointer/range boundary absent from Node's JS API.
// Exercise it directly as well as the public adapters, without logging buffers or entropy.
function nativeAdmission() {
  if (typeof globalThis.__momGzip !== "function") {return;}
  /** @param {() => unknown} action */
  function rejects(action) {
    let rejected = false;
    try {action();} catch {rejected = true;}
    if (!rejected) {throw new Error("Native admission accepted malformed input");}
  }
  const buffer = new ArrayBuffer(8);
  for (const [byteOffset, byteLength] of [[-1, 1], [1.5, 1], [0, -1], [0, 1.5],
    [0, 9], [9, 0], [Infinity, 0], [0, NaN], [Number.MAX_VALUE, 1]]) {
    const forged = /** @type {Uint8Array} */ (/** @type {unknown} */ ({buffer, byteOffset, byteLength}));
    rejects(() => globalThis.__momGzip(forged));
    const hash = globalThis.__momHashCreate();
    rejects(() => hash.update(forged));
    hash.update(Buffer.from("abc"));
    if (Buffer.from(hash.digest()).toString("hex") !== "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
      throw new Error("Rejected hash input corrupted native state");
    }
  }
  for (const size of [-1, 1.5, NaN, Infinity, 0x80000000]) {rejects(() => globalThis.__momRandomBytes(size));}
  const hash = globalThis.__momHashCreate();
  const reentrant = /** @type {Uint8Array} */ (/** @type {unknown} */ ({buffer,
    get byteOffset() {hash.digest(); return 0;}, byteLength: 1}));
  rejects(() => hash.update(reentrant));
  for (const storage of [{}, null, [], "bad"]) {
    const view = /** @type {Uint8Array} */ (/** @type {unknown} */ ({buffer: storage, byteOffset: 0, byteLength: 0}));
    rejects(() => globalThis.__momGzip(view));
  }
  for (const fd of [-1, 1.5, NaN, Infinity, 0x80000000]) {rejects(() => globalThis.__momWrite(fd, Buffer.alloc(1)));}
  rejects(() => globalThis.__momReadFile("bad\0path"));
  rejects(() => globalThis.__momReadDirectory("bad\0path"));
}

module.exports = function servicesFixture() {
  nativeAdmission();
  const root = process.env["MOM_AOT_FS_DIRECTORY"];
  if (!root) {throw new Error("Missing native filesystem fixture");}
  const writes = require("./native_filesystem")(root);
  const file = path.join(root, "config-é-💎.json");
  /** @param {() => unknown} action */
  function errorCode(action) {
    try {action(); return "NOT_REJECTED";} catch (error) {
      return error instanceof Error ? String(Reflect.get(error, "code")) : "unknown";
    }
  }
  const random = crypto.randomBytes(32);
  const cpus = os.cpus();
  const parent = Buffer.from("prefix-raw-é-💎-suffix");
  const view = parent.subarray(7, parent.length - 7);
  const hash = crypto.createHash("sha256").update(view).update("tail");
  const digest = hash.digest("hex");
  const compressed = zlib.gzipSync(Buffer.concat([Buffer.from("Pearl proof fixture:"), view, Buffer.alloc(8192, 7)]));
  return {
    filesystem: {
      writes,
      text: fs.readFileSync(file, "utf8"), bytes: fs.readFileSync(path.join(root, "binary")).toString("hex"),
      empty: fs.readFileSync(path.join(root, "empty"), "utf8"), names: fs.readdirSync(root),
      exists: [file, path.join(root, "absent"), file + "\0ignored"].map((name) => fs.existsSync(name)),
      missing: errorCode(() => fs.readFileSync(path.join(root, "absent"), "utf8")),
      invalid: errorCode(() => fs.readFileSync(file + "\0ignored", "utf8")),
      directory: errorCode(() => fs.readFileSync(root)),
    },
    crypto: {
      empty: crypto.createHash("sha256").update("").digest("hex"),
      abc: crypto.createHash("SHA256").update("abc").digest("hex"), digest,
      finalized: errorCode(() => hash.digest()),
      finalizedUpdate: errorCode(() => hash.update("late")),
      random: {buffer: Buffer.isBuffer(random), length: random.length, empty: crypto.randomBytes(0).length},
    },
    gzip: compressed.toString("base64"),
    cpus: {count: cpus.length, models: [...new Set(cpus.map((cpu) => cpu.model))],
      valid: cpus.every((cpu) => typeof cpu.speed === "number" && Object.values(cpu.times).every((value) => value >= 0))},
  };
};
