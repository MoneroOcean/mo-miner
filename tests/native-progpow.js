"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const {sequence, probe} = require("./reference/native-progpow");

test.describe("Native ProgPoW specialization parity fixture", () => {
  /** @param {string} [fault] */
  function fixture(fault = "") {
    const cases = sequence();
    /** @type {Array<{name: string, values: {backend?: unknown, dev?: unknown, intensity?: unknown} | undefined}>} */
    const commands = [];
    /** @type {string[]} */
    const results = [];
    let starts = 0, index = 0;
    return {commands, results, get starts() {return starts;}, AsyncWorker: class {
      /** @param {(name: string, values: Record<string, string>) => void} progress
       * @param {() => void} complete @param {(error: Error) => void} failed */
      constructor(progress, complete, failed) {
        starts++; this.progress = progress; this.complete = complete; this.failed = failed;
        if (fault === "constructor") {throw new Error("constructor failure");}
      }
      /** @param {string} name @param {object} [values] */
      sendToCpp(name, values) {
        commands.push({name, values});
        if (name === "close") {
          if (fault === "close") {throw new Error("close failure");}
          setImmediate(() => {if (fault === "late") {this.progress("test", {result: "00".repeat(32) + " " + "00".repeat(32)});} this.complete();});
          return;
        }
        if (fault === "send") {throw new Error("send failure");}
        const current = cases[index++];
        const word = createHash("sha256").update(JSON.stringify(current?.job)).digest("hex");
        const result = current?.expected || word + " " + word; results.push(result);
        setImmediate(() => {
          if (fault === "early") {this.complete();} else if (fault === "native") {this.failed(new Error("native failure"));} else {
            this.progress(fault === "event" ? "error" : "test", {result: fault === "missing" ? "" :
              fault === "format" ? "invalid" : fault === "hash" ? "00".repeat(32) + " " + word : result});
          }
        });
      }
    }};
  }
  test("four variants each change period and return, then restore the original KawPow job", () => {
    assert.deepEqual(sequence().map(({job}) => [job.algo, job.height]), [
      ["kawpow", 0], ["kawpow", 3], ["kawpow", 0], ["firopow", 1], ["firopow", 2], ["firopow", 1],
      ["evrprogpow", 0], ["evrprogpow", 3], ["evrprogpow", 0], ["meowpow", 0], ["meowpow", 6], ["meowpow", 0], ["kawpow", 0],
    ]);
  });
  test("reuses the maintained vectors and only omits unavailable adjacent-period references", () => {
    assert.equal(sequence().filter(({expected}) => expected).length, 9);
    assert.ok(sequence().every(({job}) => job.dev === "gpu1*[intensity=1]" && job.nonceoffset === 32 && job.noncebytes === 8));
  });
  test("one worker retains state across all thirteen requests and closes once", async () => {
    const core = fixture(), results = await probe(core);
    assert.equal(core.starts, 1); assert.equal(results.length, 13);
    assert.deepEqual(core.commands.map(({name}) => name), [...Array(13).fill("test"), "close"]);
    assert.deepEqual(results.map(({result}) => result), core.results);
  });
  test("explicit SYCL selection cannot silently route through a CPU miner", async () => {
    const core = fixture(); await probe(core);
    assert.ok(core.commands.filter(({name}) => name === "test").every(({values}) => values?.backend === "sycl" && values.dev === "gpu1" && values.intensity === 1));
  });
  test("real execution requires fresh ownership before loading an addon", async () => {
    const saved = process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"]; delete process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"];
    try {await assert.rejects(probe(), /Verify fresh GPU ownership/);} finally {
      if (saved === undefined) {delete process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"];} else {process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"] = saved;}
    }
  });
  /** @type {Array<[string, RegExp]>} */
  const failures = [["event", /Invalid ProgPoW/], ["missing", /Invalid ProgPoW/],
    ["format", /Invalid ProgPoW/], ["hash", /Invalid ProgPoW/], ["early", /closed before/],
    ["native", /native failure/], ["send", /send failure/], ["close", /close failure/],
    ["late", /after closing/], ["constructor", /constructor failure/]];
  for (const [fault, message] of failures) {
    test(`rejects ${fault} failures instead of treating partial coverage as success`, async () => {
      await assert.rejects(probe(fixture(fault)), message);
    });
  }
});
