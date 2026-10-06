"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const policy = require("../compiler-policy");
const {hashTests} = require("./vectors");

/** @typedef {{sendToCpp: (name: string, values?: Record<string, string>) => void}} Worker */
/** @typedef {{AsyncWorker: new(progress: (name: string, values: Record<string, string>) => void,
 * complete: () => void, error: (error: Error) => void) => Worker}} Core */

const vendor = process.env["MOM_GPU_BACKEND"] || "";
const root = path.join(__dirname, "..");

// Opt-in full-DAG test: run through r.sh with one vendor selected. One-nonce vectors cannot
// detect lanes overwriting the leader's nonce with the original header during a mining batch.
/** @param {Core} core @param {HashVectorDefinition} definition */
async function checkBatch(core, definition) {
  const {algo, blob_hex, nonceoffset} = definition.job;
  assert.equal(typeof blob_hex, "string");
  const header = Buffer.from(String(blob_hex), "hex");
  const offset = Number(nonceoffset);
  assert.ok(Number.isInteger(offset) && offset >= 0 && offset + 8 <= header.length);
  header.fill(0, offset, offset + 8);
  const job = Object.fromEntries(Object.entries(definition.job).map(([key, value]) => [key, String(value)]));
  Object.assign(job, {
    dev: "gpu1", intensity: "32", blob_hex: header.toString("hex"), nonce: "0".repeat(16),
    backend: policy.selection(algo, vendor, process.platform)?.backend || "sycl",
    target: "f".repeat(64), pool_id: "offline", worker_id: "offline",
    job_id: "nonce-batch", job_token: "nonce-batch",
  });

  await new Promise((resolve, reject) => {
    let phase = "mining";
    let checked = false;
    let seen = 0;
    let expected = "";
    let closing = false;
    /** @type {Error | null} */
    let failure = null;
    /** @param {Error | null} error */
    const close = (error) => {
      failure ||= error;
      if (!closing) {
        closing = true;
        worker.sendToCpp("close");
      }
    };
    const worker = new core.AsyncWorker((name, values) => {
      try {
        if (name === "error") {throw new Error("Native nonce-batch worker reported an error");}
        if (phase === "mining" && name === "result") {
          const nonceHex = values["nonce"];
          const hash = values["hash"];
          assert.ok(nonceHex && /^[0-9a-f]{16}$/i.test(nonceHex), "Missing or invalid mining nonce");
          assert.ok(hash && /^[0-9a-f]{64}$/i.test(hash), "Missing or invalid mining hash");
          const nonce = BigInt("0x" + nonceHex);
          if (nonce % 32n === 0n) {
            if (++seen >= 8) {throw new Error("No noninitial batch candidate was observed");}
            return;
          }
          expected = hash;
          const candidate = Buffer.from(header);
          if (algo === "fishhash" && header.length === 180) {
            candidate.writeBigUInt64BE(nonce, offset);
          } else {
            candidate.writeBigUInt64LE(nonce, offset);
          }
          phase = "scalar";
          worker.sendToCpp("pause");
          worker.sendToCpp("test", {
            ...job, intensity: "1", blob_hex: candidate.toString("hex"),
            nonce: nonceHex, target: "0".repeat(64),
          });
        } else if (phase === "scalar" && name === "test") {
          if (values["result"] !== expected) {throw new Error("Batch/scalar digest mismatch");}
          checked = true;
          phase = "closing";
          close(null);
        }
      } catch (error) {
        close(error instanceof Error ? error : new Error(String(error)));
      }
    }, () => {
      clearTimeout(timer);
      if (failure) {
        reject(failure);
      } else if (!checked) {
        reject(new Error("Worker closed before scalar verification"));
      } else {
        resolve(null);
      }
    }, (error) => {
      clearTimeout(timer);
      reject(error);
    });
    // Request cooperative shutdown; never use a signal as normal GPU-test teardown.
    const timer = setTimeout(() => close(new Error("Nonce-batch validation timed out")), 12 * 60 * 1000);
    worker.sendToCpp("job", job);
  });
  console.log(`NONCE_BATCH_LAYOUT algo=${algo} header_bytes=${header.length} scalar_match=true`);
}

if (process.argv[2] === "--worker") {
  (async () => {
    assert.equal(process.env["MOM_FISHHASH_FULL_TEST"], undefined);
    assert.equal(process.env["MOM_FISHHASH_COOP"], "1");
    const nativePath = process.env["MOM_NATIVE_PATH"];
    assert.ok(nativePath, "No FishHash worker selected");
    /** @type {Core} */
    const core = require(nativePath);
    const vectors = hashTests.filter(({job}) => ["fishhash", "karlsenhashv2"].includes(job.algo));
    assert.equal(vectors.length, 3, "Expected both FishHash headers and the Karlsen header");
    for (const definition of vectors) {await checkBatch(core, definition);}
    console.log(`NONCE_BATCH_COMPLETE layouts=${vectors.length}`);
  })().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
} else {
  test("FishHash/Karlsen mining batches match the independent scalar path", () => {
    assert.ok(["intel", "nvidia", "amd"].includes(vendor), "Set MOM_GPU_BACKEND to one GPU vendor");
    /** @type {NodeJS.ProcessEnv} */
    const base = {...process.env, MOM_COMPILER_POLICY_STRICT: "1", MOM_FISHHASH_COOP: "1"};
    delete base["MOM_NATIVE_PATH"];
    delete base["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"];
    delete base["MOM_NATIVE_DIR"];
    // Mining uses the full DAG; test mode uses its independent scalar/light-cache oracle.
    // Set flags before loading the addon, including on Windows where CRT getenv is separate.
    delete base["MOM_FISHHASH_FULL_TEST"];
    const env = {...base, ...policy.workerEnv("fishhash", base)};
    assert.ok(env["MOM_NATIVE_PATH"] && fs.existsSync(env["MOM_NATIVE_PATH"]), "FishHash worker is missing");
    const result = spawnSync(process.execPath, [__filename, "--worker"], {cwd: root, env, stdio: "inherit"});
    if (result.error) {throw result.error;}
    assert.equal(result.signal, null, "Nonce-batch child did not exit naturally");
    assert.equal(result.status, 0, "Nonce-batch scalar comparison failed");
  });
}
