"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("PearlHash DG2 paired packing preserves operands and transcripts", () => {
  const matrix = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/matrix.inc"), "utf8");
  const kernel = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dpasw_search.inc"), "utf8");
  /** @param {string} source @param {RegExp} pattern */
  const extract = (source, pattern) => {
    const value = source.match(pattern)?.[1];
    assert.equal(typeof value, "string", `Missing production layout operation: ${pattern}`);
    return /** @type {string} */ (value);
  };
  // Execute the actual scalar offsets from trusted project source, not copied layout formulas.
  /** @param {string[]} names @param {string} body @returns {(...values: number[]) => number} */
  const offset = (names, body) => vm.runInNewContext(`(${names.join(",")}) => ${body}`,
    {Math}, {timeout: 1000, contextCodeGeneration: {strings: false, wasm: false}});
  /** @param {string} body */
  const scalar = (body) => body.replaceAll("static_cast<size_t>(k)", "k")
    .replaceAll("static_cast<size_t>(k / 32)", "(k / 32)")
    .replaceAll("(j / 8)", "Math.floor(j / 8)")
    .replaceAll("(r / 4)", "Math.floor(r / 4)")
    .replaceAll("size_t(", "(").replaceAll("int(", "(");
  const packA = offset(["i", "c", "k"], scalar(extract(matrix, /ap\[([^\]]*\(c & 31\))\] = v;/u)));
  const packB = offset(["r", "j", "k"], scalar(extract(matrix,
    /bp\[(\(j \/ 8\)[\s\S]*?\(r % 4\))\] = v;/u)));
  const loadA = offset(["band0", "k", "p", "aHalfOffset"], scalar(extract(kernel, /Ap \+ ([^;]*?)\);/u)));
  const loadB = offset(["colBase", "c", "half", "k", "p"], scalar(extract(kernel, /Bp \+ ([^;]*?)\);/u)));
  const halfOffset = offset(["localId"], extract(kernel,
    /const size_t aHalfOffset = ([^;]*);/u).replace("id.get_local_id(0)", "localId"));
  assert.equal(halfOffset(0), 0);
  assert.equal(halfOffset(1), 128);
  const localSize = offset(["nWI"], extract(kernel, /const size_t localSize = ([^;]*);/u));
  const blockRows = Number(extract(kernel, /pearlhash_dpasw_block_rows = (\d+);/u));
  const blockColumns = Number(extract(kernel, /pearlhash_dpasw_block_columns = (\d+);/u));
  const mapping = extract(kernel, /( {6}int Rg, Cg;[\s\S]*?) {6}const int rowBase/u)
    .replace("int Rg, Cg;", "let Rg, Cg;")
    .replace("const int blk = wi / (BH * BW)", "const blk = Math.floor(wi / (BH * BW))")
    .replace("Rg = (blk / blocksW)", "Rg = Math.floor(blk / blocksW)")
    .replace("intra / BW", "Math.floor(intra / BW)")
    .replace("Rg = wi / tilesW", "Rg = Math.floor(wi / tilesW)");
  /** @type {(wi: number, tilesH: number, tilesW: number) => [number, number]} */
  const map = vm.runInNewContext(`(wi,tilesH,tilesW) => {
    const BH=${blockRows}, BW=${blockColumns};
    const blocked=${extract(kernel, /const bool blocked = ([^;]*);/u)};
    const blocksW=${extract(kernel, /const int blocksW = ([^;]*);/u)};
    ${mapping} return [Rg,Cg];
  }`, {}, {timeout: 1000, contextCodeGeneration: {strings: false, wasm: false}});
  for (const [height = 0, width = 0] of [[1, 2], [2, 6], [64, 64], [64, 128], [128, 64]]) {
    const count = height * width;
    const local = localSize(count);
    const seen = new Set();
    for (let item = 0; item < count; item++) {
      const [row, column] = map(item, height, width);
      assert(row >= 0 && row < height && column >= 0 && column < width);
      assert(!seen.has(row * width + column));
      seen.add(row * width + column);
      if (item % 2 === 0) {
        const [nextRow, nextColumn] = map(item + 1, height, width);
        assert.equal(nextRow, row);
        assert.equal(nextColumn, column + 1);
        assert.equal(item % local & 1, 0);
        assert.equal((item + 1) % local & 1, 1);
      }
    }
    assert.equal(seen.size, count);
  }
  let wrongHalf = 0;
  let wrongPartner = 0;
  for (const [inner = 0, proofRank = 0] of [[2048, 128], [4096, 256]]) {
    const a = new Int8Array(64 * inner);
    const b = new Int8Array(inner * 64);
    const packedA = new Int8Array(a.length);
    const packedB = new Int8Array(b.length);
    for (let row = 0; row < 64; row++) {
      for (let index = 0; index < inner; index++) {
        const value = (row * 37 + index * 13 + (row ^ index) * 3) % 255 - 127;
        a[row * inner + index] = value;
        packedA[packA(row, index, inner)] = value;
      }
    }
    for (let index = 0; index < inner; index++) {
      for (let column = 0; column < 64; column++) {
        const value = (index * 29 + column * 11 + (index ^ column) * 5) % 255 - 127;
        b[index * 64 + column] = value;
        packedB[packB(index, column, inner)] = value;
      }
    }
    for (let item = 0; item < 8; item++) {
      const [rowGroup, columnGroup] = map(item, 2, 4);
      for (let tile = 0; tile < 2; tile++) {
        const directFolds = new Uint32Array(inner / proofRank);
        const packedFolds = new Uint32Array(directFolds.length);
        for (let row = 0; row < 16; row++) {
          for (let column = 0; column < 16; column++) {
            const globalRow = rowGroup * 32 + tile * 16 + row;
            const globalColumn = columnGroup * 16 + column;
            const band = Math.floor(globalRow / 8);
            let direct = 0;
            let packed = 0;
            let duplicated = 0;
            let unrelated = 0;
            for (let index = 0; index < inner; index += 32) {
              const aBase = loadA(band, inner, index, halfOffset(row % 8 < 4 ? 0 : 1));
              const bBase = loadB(columnGroup, 0, Math.floor(column / 8), inner, index);
              for (let step = 0; step < 32; step++) {
                const left = packedA[aBase + row % 4 * 32 + step];
                const right = packedB[bBase + Math.floor(step / 4) * 32 + column % 8 * 4 + step % 4];
                assert.equal(left, a[globalRow * inner + index + step]);
                assert.equal(right, b[(index + step) * 64 + globalColumn]);
                assert(left !== undefined && right !== undefined);
                direct += /** @type {number} */ (a[globalRow * inner + index + step]) *
                /** @type {number} */ (b[(index + step) * 64 + globalColumn]);
                packed += left * right;
                duplicated += /** @type {number} */ (packedA[loadA(band, inner, index, 0) + row % 4 * 32 + step]) * right;
                unrelated += /** @type {number} */ (packedA[loadA((band + 4) % 8, inner, index, 0) + row % 8 * 32 + step]) * right;
              }
              if ((index + 32) % proofRank === 0) {
                const fold = (index + 32) / proofRank - 1;
                directFolds[fold] = /** @type {number} */ (directFolds[fold]) ^ direct;
                packedFolds[fold] = /** @type {number} */ (packedFolds[fold]) ^ packed;
              }
            }
            assert.equal(packed, direct);
            wrongHalf += Number(duplicated !== direct);
            wrongPartner += Number(unrelated !== direct);
          }
        }
        const directTranscript = new Uint32Array(16);
        const packedTranscript = new Uint32Array(16);
        for (let fold = 0; fold < directFolds.length; fold++) {
          const slot = fold % 16;
          const direct = /** @type {number} */ (directTranscript[slot]);
          const packed = /** @type {number} */ (packedTranscript[slot]);
          directTranscript[slot] = (direct << 13 | direct >>> 19) ^ /** @type {number} */ (directFolds[fold]);
          packedTranscript[slot] = (packed << 13 | packed >>> 19) ^ /** @type {number} */ (packedFolds[fold]);
        }
        assert.deepEqual(packedTranscript, directTranscript);
      }
    }
  }
  assert(wrongHalf > 0 && wrongPartner > 0);
});

test("PearlHash paired dispatch preserves unqualified OpenCL and width16", () => {
  const dispatch = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dispatch.inc"), "utf8");
  const route = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/esimd_route.h"), "utf8");
  const pairedBody = route.match(/pearlhash_dpasw_route\([\s\S]*?return ([^;]*);/u)?.[1];
  const selectedBody = dispatch.match(/const bool use_dpasw = ([\s\S]*?);/u)?.[1];
  assert(pairedBody && selectedBody);
  const paired = vm.runInNewContext(`(esimd_allowed,dpas_width,dg2,rank) => ${pairedBody.replace("8u", "8")}`,
    {}, {timeout: 1000, contextCodeGeneration: {strings: false, wasm: false}});
  const selected = selectedBody.replaceAll("sycl::backend::", "sycl.backend.").replaceAll("8u", "8");
  for (const fixture of [
    {name: "Level Zero P128", backend: 1, portable: false, width: 8, rank: 128, expected: true},
    {name: "Level Zero P256", backend: 1, portable: false, width: 8, rank: 256, expected: true},
    {name: "unqualified OpenCL retains ordinary", backend: 2, portable: false, width: 8, rank: 128, expected: false},
    {name: "B580 width16", backend: 1, portable: false, width: 16, rank: 128, expected: false},
    {name: "non-DG2 width8", backend: 1, portable: false, width: 8, rank: 128, dg2: false, expected: false},
    {name: "portable path", backend: 1, portable: true, width: 8, rank: 128, expected: false},
    {name: "ordinary rank512", backend: 1, portable: false, width: 8, rank: 512, expected: false},
  ]) {
    const actual = vm.runInNewContext(selected, {
      use_portable: fixture.portable, esimd_width: fixture.width, rank: fixture.rank,
      q: {get_backend: () => fixture.backend, get_device: () => true},
      sycl: {backend: {ext_oneapi_level_zero: 1}},
      dpasw_esimd_device: () => fixture.dg2 !== false, pearlhash_dpasw_route: paired,
    }, {timeout: 1000, contextCodeGeneration: {strings: false, wasm: false}});
    assert.equal(actual, fixture.expected, fixture.name);
  }
});

test("PearlHash combined AOT retains ordinary ESIMD without paired entries", () => {
  const kernel = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dpasw_search.inc"), "utf8");
  const dispatch = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dispatch.inc"), "utf8");
  const guard = kernel.match(/^#if (defined\(PEARLHASH_ESIMD\)[^\n]*)$/mu)?.[1];
  assert(guard);
  for (const fixture of [
    {name: "oneAPI main TU", esimd: true, standalone: false, expected: true},
    {name: "combined standalone TU", esimd: true, standalone: true, expected: false},
    {name: "combined main TU", esimd: false, standalone: false, expected: false},
    {name: "portable TU", esimd: false, standalone: false, expected: false},
  ]) {
    /** @type {string} */
    const selected = guard.replaceAll("defined(PEARLHASH_ESIMD)", String(fixture.esimd))
      .replaceAll("defined(MOM_PEARLHASH_ESIMD_TU)", String(fixture.standalone));
    assert.equal(vm.runInNewContext(selected, {}, {
      timeout: 1000, contextCodeGeneration: {strings: false, wasm: false},
    }), fixture.expected, fixture.name);
  }
  assert.doesNotMatch(kernel, /#if defined\(MOM_PEARLHASH_HAS_ESIMD\)/u);
  assert.match(dispatch, /#if defined\(PEARLHASH_ESIMD\)\s*\/\/[^\n]*\n\s*const bool use_dpasw/u);
  assert.match(dispatch, /#if defined\(PEARLHASH_ESIMD\)\s*if \(use_dpasw\)\s*compute_ab<true>/u);
  assert.match(dispatch, /#if defined\(PEARLHASH_ESIMD\)\s*if \(use_dpasw\)\s*\{\s*search_esimd_dpasw/u);
});

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
