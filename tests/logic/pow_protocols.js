"use strict";

const s = require("./support");
const { test, assert, loadMinerWithStubs } = s;

test("KawPow pool jobs append the nonce field to a header hash", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, headerHash + "0000000000000000");
  assert.equal(jobMessage.job.noncebytes, 8);
  assert.equal(jobMessage.job.nonceoffset, 32);
  assert.equal(jobMessage.job.worker_id, "worker");
});

test("KawPow stratum jobs can use pool login as worker id", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);

  setJob({
    algo: "kawpow",
    blob: headerHash,
    difficulty: 1,
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.worker_id, "user");
});

test("KawPow pool jobs preserve provided nonce template", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();
  const headerHash = "00".repeat(32);
  const extraNonce = "00000000000081ff";

  setJob({
    algo: "kawpow",
    blob: headerHash + extraNonce,
    header_hash: headerHash,
    nonce: "ff81000000000000",
    nicehash_mask: "ffff000000000000",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  });

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.blob_hex, headerHash + extraNonce);
  assert.equal(jobMessage.job.header_hash, headerHash);
  assert.equal(jobMessage.job.nonce, "ff81000000000000");
  assert.equal(jobMessage.job.nicehash_mask, "ffff000000000000");
});

test("KawPow submit uses the header hash carried by the worker result", async () => {
  const miner = await loadMinerWithStubs();
  const oldHeaderHash = "11".repeat(32);
  const newHeaderHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "raven";
  miner.global.opt.pools[0].last_job = {
    job_id: "new",
    header_hash: newHeaderHash,
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "old",
      nonce: "ff81000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: oldHeaderHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "old",
    "0xff81000000000001",
    "0x" + oldHeaderHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses Eth mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "eth";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("Etchash submit uses ethproxy eth_submitWork format", async () => {
  const miner = await loadMinerWithStubs();
  const headerHash = "22".repeat(32);
  miner.global.opt.pools[0].submit_mode = "ethproxy";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: headerHash,
      nonce: "080c000000000001",
      hash: "00".repeat(32),
      mix_hash: "33".repeat(32),
      header_hash: headerHash,
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(miner.poolWrites[0].json.method, "eth_submitWork");
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "0x080c000000000001",
    "0x" + headerHash,
    "0x" + "33".repeat(32),
  ]));
});

test("PearlHash submit uses its canonical submit mode", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "pearlhash";

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      job_id: "job1",
      plain_proof: "proof",
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(miner.poolWrites[0].json)), {
    jsonrpc: "2.0",
    id: 3,
    method: "mining.submit",
    params: { job_id: "job1", plain_proof: "proof" },
  });
});

test("Autolykos2 submit uses Ergo mining.submit format", async () => {
  const miner = await loadMinerWithStubs();
  miner.global.opt.pools[0].submit_mode = "erg";
  miner.global.opt.pools[0].erg_submit_jobs = {
    "203d": { extra_nonce: "080c", extra_nonce2_size: 6, ntime: "00000002" },
  };

  miner.messageHandler({
    type: "result",
    value: {
      pool_id: 0,
      worker_id: "worker",
      job_id: "203d",
      nonce: "080c000000000001",
      hash: "00".repeat(32),
    },
  });

  assert.equal(miner.poolWrites.length, 1);
  assert.equal(JSON.stringify(miner.poolWrites[0].json.params), JSON.stringify([
    "user",
    "203d",
    "000000000001",
    "00000002",
    "080c000000000001",
  ]));
});

test("nicehash xn prefixes longer than noncebytes are truncated", async () => {
  const miner = await loadMinerWithStubs();
  const setJob = miner.getSetJob();

  assert.doesNotThrow(() => setJob({
    algo: "cn/0",
    blob_hex: "abcd",
    noncebytes: 4,
    xn: "001122334455",
    difficulty: 1,
    id: "worker",
    job_id: "job",
  }));

  const jobMessage = miner.sentMessages.find((msg) => msg.type === "job");
  assert.equal(jobMessage.job.nonce, "00112233");
  assert.equal(jobMessage.job.nicehash_mask, "ffffffff");
});
