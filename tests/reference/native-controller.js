"use strict";

// The AOT probe imports maintained modules, not copies or native translations. Its JSON transcript
// is compared byte-for-byte with direct Node execution before any larger controller claim is made.
const {normalizeAlgoName} = require("../../miner/algorithms");
const hash = require("../../helper/hash");
const tuning = require("../../gpu-tuning");
const policy = require("../../compiler-policy");
const nodePath = require("node:path");
const {EventEmitter} = require("node:events");
const {StringDecoder} = require("node:string_decoder");

if (process.env["MOM_AOT_PROCESS_EXIT"] === "forced") {
  setTimeout(() => {throw new Error("Forced exit left the process running");}, 60000);
  console.log("EXPECTED_AOT_PROCESS_EXIT");
  process.exitCode = 41;
  process.exit();
}
console.log(JSON.stringify({process: require("./native-process")()}));
console.log(JSON.stringify({path: require("./native-path")()}));
console.log(JSON.stringify({environment: require("./native-environment")()}));
console.log(JSON.stringify({services: require("./native-services")()}));

/** @param {() => unknown} action */
function failure(action) {
  try {
    action();
    return "NO ERROR";
  } catch (error) {
    return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  }
}

const closures = [];
for (let i = 0; i < 3; i++) {closures.push(() => i);}
console.log(JSON.stringify({
  aliases: ["CUCKAROO", "xel/3", "pearlhash", "", null].map(normalizeAlgoName),
  counts: [0n, 1234567890123456789n, (1n << 128n) - 1n].map(hash.formatHashCount),
  target: hash.pearlhashTarget("000000000000000000000000000000000000000000000000000000000000ffff", 2048, 128),
  shape: tuning.validatePearlHashShape("131072", 131072, 2048, 128),
  devices: tuning.parseDeviceList("gpu1*[intensity=256],gpu2*[intensity=512]", "kawpow"),
  candidates: tuning.autotuneCandidates("c29", {
    device: "gpu1", processes: 1, tuning: {seed_workgroup: 128, seed_blocks: 16},
  }),
  invalidShape: failure(() => tuning.validatePearlHashShape(128, 128, 2048, 127)),
  invalidTarget: failure(() => hash.pearlhashTarget("xyz", 2048, 128)),
  closures: closures.map((get) => get()),
  policy: policy.parse(),
  paths: [[], [""], ["/root", "/child", "..", "file"], ["a", "b/"],
    ["a", "."], ["a", "../"], ["///", "..", ".."], ["..", "a", ".."],
    ["//host//", "share"], ["."], ["./"], ["/a", "../../b"], ["a\\b", "c"]]
    .map((parts) => nodePath.join(...parts)),
  invalidPath: failure(() => nodePath.join(/** @type {string} */ (/** @type {unknown} */ (4))))
    .split(":")[0],
  // Large words are consumed unchanged by pool/connection.js, not rounded through Number.
  exactJson: JSON.parse('{"word":18446744073709551615}',
    (key, value, context = /** @type {{source?: string}} */ ({})) =>
      key === "word" ? context.source : value),
}));

const bytes = Buffer.from("Mining-é-💎", "utf8");
const decoder = new StringDecoder("utf8");
const decoded = [decoder.write(bytes.subarray(0, 8)), decoder.write(bytes.subarray(8, 11)),
  decoder.end(bytes.subarray(11))].join("");
const numbers = Buffer.alloc(16);
numbers.writeBigUInt64LE(0xfedcba9876543210n, 0);
numbers.writeBigUInt64BE(1n << 63n, 8);
const emitter = new EventEmitter();
/** @type {string[]} */
const events = [];
emitter.on("work", (value) => events.push(value));
emitter.once("work", () => events.push("once"));
emitter.emit("work", "a");
emitter.emit("work", "b");
const view = bytes.subarray(0, 1);
view[0] = 0x6d;
console.log(JSON.stringify({
  hex: bytes.toString("hex"), base64: bytes.toString("base64"), decoded,
  numbers: numbers.toString("hex"), littleEndian: numbers.readBigUInt64LE().toString(),
  bigEndian: numbers.readBigUInt64BE(8).toString(), events,
  viewIsBuffer: Buffer.isBuffer(view), viewSharesBytes: bytes[0] === 0x6d,
}));

// Exercise asynchronous language semantics separately from Node networking. Ordering differences
// are a failed parity probe, not permission to silently change application scheduling.
/** @type {string[]} */
const trace = [];
async function asynchronousResult() {
  trace.push("start");
  const value = await Promise.resolve(1n << 80n);
  trace.push("await");
  try {
    await Promise.reject(new Error("expected"));
  } catch (error) {
    trace.push(error instanceof Error ? error.message : "wrong exception");
  }
  return value.toString();
}
Promise.all([asynchronousResult(), Promise.resolve("second")]).then((values) => {
  console.log(JSON.stringify({values, trace}));
});

async function timers() {
  /** @type {string[]} */
  const order = [];
  const cancelled = setTimeout(() => order.push("cancelled"), 1);
  clearTimeout(cancelled);
  const unreferenced = setTimeout(() => order.push("unreferenced"), 60000).unref();
  const unrefWorks = !unreferenced.hasRef();
  await new Promise((resolve) => {
    setTimeout(() => {
      order.push("first");
      Promise.resolve().then(() => order.push("microtask"));
      process.nextTick(() => order.push("nextTick"));
    }, 1);
    setTimeout(() => {order.push("second"); resolve(undefined);}, 1);
  });
  let ticks = 0;
  await new Promise((resolve) => {
    const interval = setInterval(() => {
      if (++ticks === 2) {clearInterval(interval); resolve(undefined);}
    }, 1);
  });
  /** @type {string[]} */
  const immediateOrder = [];
  const cancelledImmediate = setImmediate(() => immediateOrder.push("cancelled"));
  clearImmediate(cancelledImmediate);
  const before = performance.now();
  await new Promise((resolve) => setTimeout(() => {
    setImmediate(() => immediateOrder.push("immediate"));
    setTimeout(() => {immediateOrder.push("timer"); resolve(undefined);}, 0);
  }, 1));
  return {order, ticks, unrefWorks, immediateOrder, monotonic: performance.now() >= before};
}

Promise.all([require("./native-transport")(), timers(), require("./native-tls")(),
  process.env["MOM_AOT_CPU_CORE"] ? require("./native-core")() : null, require("./native-children")()])
  .then(([transport, timers, tls, core, children]) => {
    console.log(JSON.stringify({transport, timers, tls, core, children}));
    if (process.env["MOM_AOT_PROCESS_EXIT"] === "graceful") {process.exitCode = 37;}
  }).catch((error) => {
    // Surface async fixture failures as an uncaught task in either mode, not an ignored rejection.
    setTimeout(() => {throw error;}, 0);
  });

// The driver separately requires failure exit status and bounded native teardown while a TCP
// connection, timers and their JS references are live. These cases must never pass as successes.
if (["timer", "nextTick"].includes(process.env["MOM_AOT_FAILURE"] || "")) {
  const fail = () => {throw new Error("EXPECTED_AOT_ASYNC_FAILURE");};
  if (process.env["MOM_AOT_FAILURE"] === "nextTick") {
    process.nextTick(fail);
  } else {setTimeout(fail, 2);}
}
