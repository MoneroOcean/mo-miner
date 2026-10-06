"use strict";

const s = require("./support");
const crypto = require("node:crypto");
const {test, helper, pool, loadMinerWithStubs, withMockPool, unexpectedPoolJob} = s;
/** @type {typeof import("node:assert/strict")} */
const assert = s.assert;
/** @type {(job: PoolJob) => MiningJob} */
const completeMiningJob = s.completeMiningJob;

/** @typedef {PoolJob & {extranonce2?: string}} VerthashJob */

const VECTOR_VERSION = "01020304";
const VECTOR_PREVIOUS_HASH = "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f";
const VECTOR_COINBASE1 = "aa55cc";
const VECTOR_COINBASE2 = "deadc0";
const VECTOR_BRANCH = "ffeeddccbbaa998877665544332211000123456789abcdeffedcba9876543210";
const VECTOR_NTIME = "11223344";
const VECTOR_BITS = "a1b2c3d4";
const VECTOR_EXTRA_NONCE1 = "10203040";
const VECTOR_EXTRA_NONCE2 = "00000000";

/** @param {Buffer} data @returns {Buffer} */
function doubleSha256(data) {
  return crypto.createHash("sha256").update(crypto.createHash("sha256").update(data).digest()).digest();
}

/** @returns {Buffer} */
function referenceVectorHeader() {
  let merkle = doubleSha256(Buffer.from(
    VECTOR_COINBASE1 + VECTOR_EXTRA_NONCE1 + VECTOR_EXTRA_NONCE2 + VECTOR_COINBASE2, "hex"
  ));
  merkle = doubleSha256(Buffer.concat([merkle, Buffer.from(VECTOR_BRANCH, "hex")]));
  /** @param {string} value */
  const reverseWord = (value) => Buffer.from(value, "hex").reverse();
  const words = VECTOR_PREVIOUS_HASH.match(/.{8}/g);
  if (!words) {throw new Error("Invalid Verthash previous-hash vector");}
  const previousHash = Buffer.concat(words.map(reverseWord));
  return Buffer.concat([
    reverseWord(VECTOR_VERSION),
    previousHash,
    merkle,
    reverseWord(VECTOR_NTIME),
    reverseWord(VECTOR_BITS),
    Buffer.alloc(4),
  ]);
}

test("Verthash pools build the standard 80-byte Bitcoin header", async () => {
  /** @type {VerthashJob | undefined} */
  let job;
  await withMockPool({
    pool: {login: "vtc-address.worker", protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
  }, async ({socket, writes}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
      job = next;
      return completeMiningJob(next);
    });
    socket.emit("connect");
    assert.ok(writes[0]);
    assert.equal(writes[0].method, "mining.subscribe");
    socket.emit("data", Buffer.from(
      JSON.stringify({id: 1, result: [["mining.notify", "1"], "01020304", 4], error: null}) + "\n" +
      JSON.stringify({id: 2, result: true, error: null}) + "\n" +
      JSON.stringify({id: null, method: "mining.set_difficulty", params: [2]}) + "\n" +
      JSON.stringify({id: null, method: "mining.notify", params: [
        "job", "01".repeat(32), "00", "00", [], "20000000", "1d00ffff", "65000000", true,
      ]}) + "\n"
    ));
  });
  assert.ok(job);
  assert.equal(job.blob?.length, 160);
  assert.equal(job.nonceoffset, 76);
  assert.equal(job.noncebytes, 4);
  assert.equal(job.difficulty, 2);
  assert.equal(job.extranonce2, "00000000");
});

test("Verthash serializes non-palindromic fields in the upstream word order", async () => {
  /** @type {VerthashJob | undefined} */
  let job;
  await withMockPool({
    pool: {login: "vtc-address.vector", protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
      job = next;
      return completeMiningJob(next);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({id: 1, result: [["mining.notify", "1"], VECTOR_EXTRA_NONCE1, 4], error: null}) + "\n" +
      JSON.stringify({id: 2, result: true, error: null}) + "\n" +
      JSON.stringify({id: null, method: "mining.set_difficulty", params: [2]}) + "\n" +
      JSON.stringify({id: null, method: "mining.notify", params: [
        "vector-job", VECTOR_PREVIOUS_HASH, VECTOR_COINBASE1, VECTOR_COINBASE2,
        [VECTOR_BRANCH], VECTOR_VERSION, VECTOR_BITS, VECTOR_NTIME, true,
      ]}) + "\n"
    ));
  });
  assert.ok(job);
  const actual = Buffer.from(job.blob || "", "hex");
  const expected = referenceVectorHeader();
  assert.equal(actual.length, 80);
  assert.deepEqual(actual.subarray(0, 36), expected.subarray(0, 36));
  assert.deepEqual(actual.subarray(36, 68), expected.subarray(36, 68));
  assert.deepEqual(actual.subarray(68), expected.subarray(68));
  assert.equal(job.extranonce2, VECTOR_EXTRA_NONCE2);
  assert.equal(job.ntime, VECTOR_NTIME);
  assert.equal(job.nonceoffset, 76);
  assert.equal(job.noncebytes, 4);
});

test("Verthash preserves a zero-length extranonce2", async () => {
  /** @type {VerthashJob | undefined} */
  let job;
  await withMockPool({
    pool: {login: "vtc-address.worker", protocol: "verthash"},
    opt: {job: {algo: "verthash"}},
  }, async ({socket}) => {
    pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
      job = next;
      return completeMiningJob(next);
    });
    socket.emit("connect");
    socket.emit("data", Buffer.from(
      JSON.stringify({id: 1, result: [["mining.notify", "1"], "", 0], error: null}) + "\n" +
      JSON.stringify({id: 2, result: true, error: null}) + "\n" +
      JSON.stringify({id: null, method: "mining.set_difficulty", params: [2]}) + "\n" +
      JSON.stringify({id: null, method: "mining.notify", params: [
        "job", "01".repeat(32), "00", "00", [], "20000000", "1d00ffff", "65000000", true,
      ]}) + "\n"
    ));
  });
  assert.ok(job);
  assert.equal(job.extranonce2, "");
  assert.equal(job.nonceoffset, 76);
  assert.equal(job.noncebytes, 4);
});

test("Verthash targets apply the upstream difficulty multiplier", async () => {
  /** @param {number} difficulty @param {string} [explicitTarget] @returns {Promise<string>} */
  async function targetFor(difficulty, explicitTarget) {
    const miner = await loadMinerWithStubs();
    const nativeJob = miner.getSetJob()({
      algo: "verthash", blob_hex: "00".repeat(80), difficulty,
      ...(explicitTarget === undefined ? {} : {target: explicitTarget}),
      job_id: "target-job",
    });
    return nativeJob.target;
  }

  const bitcoinDiff1Target = BigInt("0x00000000ffff0000000000000000000000000000000000000000000000000000");
  const maxTarget = (1n << 256n) - 1n;
  /** @param {bigint} numerator @param {bigint} denominator @param {bigint} multiplier @returns {string} */
  function expectedTarget(numerator, denominator = 1n, multiplier = 256n) {
    const target = bitcoinDiff1Target * multiplier * denominator / numerator;
    return (target > maxTarget ? maxTarget : target).toString(16).padStart(64, "0");
  }
  assert.equal(await targetFor(2), expectedTarget(2n));
  assert.equal(await targetFor(2.5), expectedTarget(5n, 2n));
  assert.equal(helper.ethDiff2Target(2n, 256n), expectedTarget(2n));
  assert.equal(await targetFor(1e-100), "f".repeat(64));
  assert.equal(await targetFor(2, "1234"), "0".repeat(60) + "1234");
  assert.equal(helper.ethDiff2Target(2), expectedTarget(2n, 1n, 1n));
});

test("Verthash merkle branches hash identically with or without a 0x prefix", async () => {
  const branch = "ab".repeat(32);
  /** @param {string} branchValue @returns {Promise<string>} */
  async function buildBlob(branchValue) {
    /** @type {VerthashJob | undefined} */
    let job;
    await withMockPool({
      pool: {login: "vtc-address.worker", protocol: "verthash"},
      opt: {job: {algo: "verthash"}},
    }, async ({socket}) => {
      pool.connect_pool_throttle(0, /** @param {PoolJob} next */ (next) => {
        job = next;
        return completeMiningJob(next);
      });
      socket.emit("connect");
      socket.emit("data", Buffer.from(
        JSON.stringify({id: 1, result: [["mining.notify", "1"], "01020304", 4], error: null}) + "\n" +
        JSON.stringify({id: 2, result: true, error: null}) + "\n" +
        JSON.stringify({id: null, method: "mining.set_difficulty", params: [2]}) + "\n" +
        JSON.stringify({id: null, method: "mining.notify", params: [
          "job", "01".repeat(32), "00", "00", [branchValue],
          "20000000", "1d00ffff", "65000000", true,
        ]}) + "\n"
      ));
    });
    assert.ok(job);
    return job.blob || "";
  }

  assert.equal(await buildBlob(branch), await buildBlob("0x" + branch));
});

test("Verthash pools reject malformed or oversized merkle branches", async () => {
  for (const branches of ["00".repeat(32), ["00"], Array(65).fill("00".repeat(32))]) {
    await withMockPool({
      pool: {protocol: "verthash", logged_in: true},
      opt: {job: {algo: "verthash"}},
    }, async ({socket, switched}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        id: null, method: "mining.notify", params: [
          "job", "01".repeat(32), "00", "00", branches,
          "20000000", "1d00ffff", "65000000", true,
        ],
      }) + "\n"));
      assert.equal(socket.destroyed, true);
      assert.equal(switched(), true);
    });
  }
});

test("Verthash pools reject non-string coinbase fields", async () => {
  for (const [coinbase1, coinbase2] of [[null, "00"], ["00", null], [1, ""]]) {
    await withMockPool({
      pool: {protocol: "verthash", logged_in: true},
      opt: {job: {algo: "verthash"}},
    }, async ({socket, switched}) => {
      pool.connect_pool_throttle(0, unexpectedPoolJob);
      socket.emit("data", Buffer.from(JSON.stringify({
        id: null, method: "mining.notify", params: [
          "job", "01".repeat(32), coinbase1, coinbase2, [],
          "20000000", "1d00ffff", "65000000", true,
        ],
      }) + "\n"));
      assert.equal(socket.destroyed, true);
      assert.equal(switched(), true);
    });
  }
});

test("Verthash submit uses stored metadata for its current job", async () => {
  const miner = await loadMinerWithStubs();
  const poolConfig = miner.global.opt.pools[0];
  poolConfig.inferred_protocol = "echelon";
  poolConfig.last_job = {
    job_id: "old", job_token: "token", submit_mode: "verthash",
    extranonce2: "00000000", ntime: "65000000",
  };
  miner.messageHandler({thread_id: 0, type: "result", value: {
    pool_id: "0", worker_id: "worker", job_id: "old", job_token: "token",
    nonce: "12345678", hash: "00".repeat(32),
  }});
  const write = miner.poolWrites[0];
  assert.ok(write);
  assert.deepEqual(JSON.parse(JSON.stringify(write.json.params)),
    ["user", "old", "00000000", "65000000", "12345678"]);
  assert.ok(Array.isArray(write.json.params));
  assert.equal(write.json.params[3], poolConfig.last_job.ntime);
  assert.equal(write.json.params[4], "12345678");
});
