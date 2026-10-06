"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const verifierPath = process.env["MOM_NODE_POWHASH_PATH"];
const device = process.env["MOM_PEARLHASH_CLAIM_DEVICE"];
const skip = !verifierPath || !device
  ? "set MOM_NODE_POWHASH_PATH and MOM_PEARLHASH_CLAIM_DEVICE"
  : false;
const m = Number(process.env["MOM_PEARLHASH_CLAIM_M"] || 128);
const n = Number(process.env["MOM_PEARLHASH_CLAIM_N"] || 128);
const k = Number(process.env["MOM_PEARLHASH_CLAIM_K"] || 2048);
const rank = Number(process.env["MOM_PEARLHASH_CLAIM_RANK"] || 128);
if (![m, n, k, rank].every((value) => Number.isSafeInteger(value) && value > 0)) {
  throw new Error("Invalid PearlHash claim-test profile");
}

test("PearlHash claim matches the independent verifier", {skip, timeout: 120000}, async () => {
  if (!verifierPath || !device) {throw new Error("PearlHash claim test configuration is incomplete");}
  const platformBuild = process.platform === "win32" ? "win" : "lin";
  const addonPath = process.env["MOM_NATIVE_PATH"] ||
    path.join(__dirname, "..", "build", platformBuild, "Release", "mom.node");
  const core = require(addonPath);
  const powhash = require(path.resolve(verifierPath));
  const header = Buffer.alloc(76);
  header.writeUInt32LE(3, 0);
  const expectedFactor = 16 * 16 * Math.floor(k / rank) * 128;
  const maximumTarget = (1n << 256n) - 1n;
  const baseTargetValue = maximumTarget / BigInt(expectedFactor);
  const baseTarget = Buffer.from(
    baseTargetValue.toString(16).padStart(64, "0"), "hex"
  ).reverse();
  const workerTarget = Buffer.from(
    (baseTargetValue * BigInt(expectedFactor)).toString(16).padStart(64, "0"), "hex"
  );

  /** @type {Record<string, string> | undefined} */
  let claim;
  await new Promise((resolve, reject) => {
    let stopped = false;
    /** @param {Error} error */
    const fail = (error) => {
      if (stopped) {return;}
      stopped = true;
      clearTimeout(deadline);
      worker.sendToCpp("close");
      reject(error);
    };
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {Record<string, string>} */ values) => {
        try {
          if (name === "error") {
            fail(new Error(values["message"] || "PearlHash worker error"));
          } else if (name === "result" && !stopped) {
            stopped = true;
            claim = values;
            clearTimeout(deadline);
            worker.sendToCpp("close");
          }
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
        }
      },
      () => resolve(undefined),
      reject
    );
    const deadline = setTimeout(() => fail(new Error("PearlHash claim timed out")), 110000);
    worker.sendToCpp("job", {
      algo: "pearlhash",
      backend: process.env["MOM_PEARLHASH_CLAIM_BACKEND"] || "sycl",
      blob_hex: header.toString("hex"),
      dev: device,
      intensity: String(m),
      job_id: "claim-test",
      job_token: "claim-test",
      nonce: "0",
      noncebytes: "8",
      pearlhash_cert_version: "3",
      pearlhash_k: String(k),
      pearlhash_n: String(n),
      pearlhash_rank: String(rank),
      pool_id: "claim-test",
      target: workerTarget.toString("hex"),
      worker_id: "claim-test",
    });
  });

  assert.ok(claim);
  /** @param {string} name @returns {string} */
  const claimField = (name) => {
    const value = claim?.[name];
    if (typeof value !== "string") {throw new Error(`PearlHash claim omitted ${name}`);}
    return value;
  };
  const proof = claimField("plain_proof");
  const jackpot = claimField("jackpot");
  const factor = Number(claimField("adjustment_factor"));
  assert.equal(Buffer.from(proof, "base64").toString("base64"), proof);
  assert.match(jackpot, /^[0-9a-f]{64}$/);
  assert.ok(Number.isInteger(factor) && factor > 0 && factor <= 0xffffffff);

  const verified = powhash.pearl_v3(header, proof, baseTarget);
  assert.equal(verified.valid, true, verified.error || "independent verifier rejected proof");
  assert.equal(verified.candidate, true, verified.error || "proof did not meet the target");
  assert.equal(jackpot, verified.jackpot.toString("hex"));
  assert.equal(factor, expectedFactor);
  assert.equal(factor, verified.config.adjustment_factor);
});
