"use strict";

const s = require("./support");
const {test, loadMinerWithStubs} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;

const HASH = "AB".repeat(32);
const MIX_HASH = "CD".repeat(32);
const HEADER_HASH = "EF".repeat(32);
const EDGES = "00000001".repeat(42);
const SOLUTION = "12".repeat(104);
const PROOF = Buffer.from("proof").toString("base64");

/** @param {unknown} params @param {string | undefined} [result] @returns {UnknownRecord} */
function nativeSubmit(params, result) {
  return {
    jsonrpc: "2.0", id: 3, method: "mining.submit", params,
    ...(result === undefined ? {} : {result}),
  };
}

/** @typedef {{
 *   name: string, job: UnknownRecord, value?: UnknownRecord, omitHash?: boolean,
 *   mutatePool?: boolean, expected: UnknownRecord
 * }} SubmitCase */

test("native submit-result negotiation changes only supported submit envelopes", async () => {
  /** @type {SubmitCase[]} */
  const cases = [
    {
      name: "Raven native submit",
      job: {submit_mode: "raven", submit_result: true, header_hash: HEADER_HASH},
      value: {header_hash: "0x" + HEADER_HASH, mix_hash: MIX_HASH},
      expected: nativeSubmit([
        "wallet", "job", "0x0000000000000001", "0x" + HEADER_HASH, "0x" + MIX_HASH,
      ], HASH.toLowerCase()),
    },
    {
      name: "Eth native submit",
      job: {submit_mode: "eth", submit_result: true, header_hash: HEADER_HASH},
      value: {header_hash: "0x" + HEADER_HASH, mix_hash: MIX_HASH},
      expected: nativeSubmit([
        "wallet", "job", "0x0000000000000001", "0x" + HEADER_HASH, "0x" + MIX_HASH,
      ], HASH.toLowerCase()),
    },
    {
      name: "Ergo native submit",
      job: {submit_mode: "erg", submit_result: true, extra_nonce2_size: 6, ntime: "00000002"},
      expected: nativeSubmit([
        "wallet", "job", "000000000001", "00000002", "0000000000000001",
      ], HASH.toLowerCase()),
    },
    {
      name: "Kaspa native submit",
      job: {submit_mode: "kaspa", submit_result: true},
      expected: nativeSubmit(["wallet", "job", "0x0000000000000001"], HASH.toLowerCase()),
    },
    {
      name: "FishHash custom submit keeps its envelope",
      job: {submit_mode: "ironfish", submit_result: true},
      expected: {
        id: 2, method: "mining.submit",
        body: {miningRequestId: "job", randomness: "0000000000000001"},
        result: HASH.toLowerCase(),
      },
    },
    {
      name: "pool negotiation mutation does not alter the matched job",
      job: {submit_mode: "raven", submit_result: true, header_hash: HEADER_HASH},
      value: {header_hash: "0x" + HEADER_HASH, mix_hash: MIX_HASH},
      mutatePool: true,
      expected: nativeSubmit([
        "wallet", "job", "0x0000000000000001", "0x" + HEADER_HASH, "0x" + MIX_HASH,
      ], HASH.toLowerCase()),
    },
    {
      name: "unset submit-result flag",
      job: {submit_mode: "raven", header_hash: HEADER_HASH},
      value: {header_hash: "0x" + HEADER_HASH, mix_hash: MIX_HASH},
      expected: nativeSubmit([
        "wallet", "job", "0x0000000000000001", "0x" + HEADER_HASH, "0x" + MIX_HASH,
      ]),
    },
    {
      name: "false submit-result flag",
      job: {submit_mode: "raven", submit_result: false, header_hash: HEADER_HASH},
      value: {header_hash: "0x" + HEADER_HASH, mix_hash: MIX_HASH},
      expected: nativeSubmit([
        "wallet", "job", "0x0000000000000001", "0x" + HEADER_HASH, "0x" + MIX_HASH,
      ]),
    },
    {
      name: "proof-only PearlHash submit",
      job: {submit_mode: "pearlhash", submit_result: true},
      value: {
        plain_proof: PROOF,
        jackpot: "0123456789abcdef".repeat(4),
        adjustment_factor: "524288",
      },
      omitHash: true,
      expected: nativeSubmit({
        job_id: "job",
        plain_proof: PROOF,
        jackpot: "0123456789abcdef".repeat(4),
        adjustment_factor: 524288,
      }),
    },
    {
      name: "C29 generic submit keeps result and edges",
      job: {submit_mode: null, submit_result: true},
      value: {edges: EDGES},
      expected: {
        jsonrpc: "2.0", id: 3, method: "submit",
        params: {
          job_id: "job", nonce: "0000000000000001", id: "worker", result: HASH,
          pow: Array(42).fill(1),
        },
      },
    },
    {
      name: "Cortex submit has no native result member",
      job: {submit_mode: "cortex", submit_result: true, header_hash: HEADER_HASH},
      value: {edges: EDGES},
      expected: {
        jsonrpc: "2.0", id: 73, method: "ctxc_submitWork",
        params: ["0x0000000000000001", "0x" + HEADER_HASH, "0x" + EDGES], worker: "mom",
      },
    },
    {
      name: "Beam submit has no native result member",
      job: {submit_mode: "beam", submit_result: true},
      value: {solution: SOLUTION},
      expected: {
        jsonrpc: "2.0", id: "job", method: "solution",
        nonce: "0100000000000000", output: SOLUTION,
      },
    },
  ];

  for (const scenario of cases) {
    const miner = await loadMinerWithStubs();
    const pool = miner.global.opt.pools[0];
    pool.login = "wallet";
    pool.last_job = {
      job_id: "job", job_token: "token", ...scenario.job,
    };
    if (scenario.mutatePool) {
      pool["extensions"] = ["mo-native"];
    }
    const value = {
      pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
      nonce: "0000000000000001", hash: HASH, ...(scenario.value ?? {}),
    };
    if (scenario.omitHash) {
      Reflect.deleteProperty(value, "hash");
    }

    miner.messageHandler({thread_id: 0, type: "result", value});
    assert.equal(miner.poolWrites.length, 1, scenario.name);
    const write = miner.poolWrites[0];
    assert.ok(write, scenario.name);
    assert.deepEqual(write.json, scenario.expected, scenario.name);
    assert.equal(Object.hasOwn(write.json, "result"), Object.hasOwn(scenario.expected, "result"), scenario.name);
  }
});

test("C29 submissions match the worker edge count to the job proof size", async () => {
  const miner = await loadMinerWithStubs();
  const pool = miner.global.opt.pools[0];
  pool.last_job = {
    job_id: "job", job_token: "token", proofsize: 32,
  };
  const value = {
    pool_id: "0", worker_id: "worker", job_id: "job", job_token: "token",
    nonce: "00000001", hash: HASH,
  };

  miner.messageHandler({thread_id: 0, type: "result", value: {...value, edges: EDGES}});
  assert.equal(miner.poolWrites.length, 0);
  miner.messageHandler({
    thread_id: 0, type: "result", value: {...value, edges: "00000001".repeat(32)},
  });
  assert.equal(miner.poolWrites.length, 1);
  const write = miner.poolWrites[0];
  assert.ok(write);
  const params = /** @type {{pow?: number[]}} */ (write.json.params);
  assert.equal(params.pow?.length, 32);
});
