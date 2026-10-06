"use strict";

const vectors = [
  {
    // Accepted Conflux mainnet block vector, independently reconstructed from the RLP header and
    // Conflux Rust consensus implementation by tests/reference/octopus_conflux.cpp. Height 100,000,000
    // selects stage 190; test mode deliberately uses the light cache so CPU SYCL remains practical.
    name: "octopus gpu1*[intensity=1] recorded-mainnet h100000000",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "octopus",
      dev: "gpu1*[intensity=1]",
      height: 100000000,
      noncebytes: 8,
      nonceoffset: 32,
      target: "ff".repeat(32),
      blob_hex: "8c03f99c72afad07ad0fe89d28c83a6aad5118558d52a2800fcfaec46bd5d97f" +
        "95071b0600778435",
    },
    expected: "000000000013df09cedb71556c0744d461ab9acb3cce92a4b4ec924be585e704",
  },
  {
    // Recorded Echelon vector, independently checked by the CPU consensus oracle.
    name: "nexapow gpu1*[intensity=1] recorded-vector",
    gpu: true,
    syclCpu: true,
    // Windows oneAPI cold initialization can finish device compilation near 15 minutes and still
    // need to build the fixed-base table. Keep enough margin for a genuinely cold qualification;
    // persistent-cache repeats remain much faster.
    timeoutMs: 25 * 60 * 1000,
    perfTimeoutMs: 15 * 60 * 1000,
    job: {
      algo: "nexapow", dev: "gpu1*[intensity=1]", noncebytes: 8, nonceoffset: 40,
      target: "ff".repeat(32),
      blob_hex: "0a4ac49b2d02e3c8d12c7093255ba7c49624f9c374d9f1c2f8e37c58705e74b0" +
        "10000000000000001182dc5800000000",
    },
    expected: "00000042cbc240375242e14641488a0e2dca7b54458a2cea23dc1d2c178bb188",
  },
  {
    // Independent CPU oracle case: short-zero exercises the generic short-nonce GPU path.
    name: "nexapow gpu1*[intensity=1] short-zero-vector",
    gpu: true,
    syclCpu: false,
    timeoutMs: 25 * 60 * 1000,
    perfTimeoutMs: 15 * 60 * 1000,
    job: {
      algo: "nexapow", dev: "gpu1*[intensity=1]", noncebytes: 8, nonceoffset: 32,
      target: "ff".repeat(32),
      blob_hex: "00000000000000000000000000000000000000000000000000000000000000000000000000000000",
    },
    expected: "b034dd02a3ba15d16b1f0325e44de3e374489182b2305ca13740772184c3bcd9",
  },
  {
    // Independent CPU oracle case: short-pattern exercises the generic short-nonce GPU path.
    name: "nexapow gpu1*[intensity=1] short-pattern-vector",
    gpu: true,
    syclCpu: false,
    timeoutMs: 25 * 60 * 1000,
    perfTimeoutMs: 15 * 60 * 1000,
    job: {
      algo: "nexapow", dev: "gpu1*[intensity=1]", noncebytes: 8, nonceoffset: 32,
      target: "ff".repeat(32),
      blob_hex: "0104070a0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e0102030405060708",
    },
    expected: "a254ce16b11c81a3bebf46569b2598d28e3fd2223a565a9245cfa8bbca03e997",
  },
  {
    name: "etchash gpu1*[intensity=256]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "etchash",
      dev: "gpu1*[intensity=256]",
      height: 0,
      seed_hex: "",
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0100000000000000",
    },
    expected:
      "f31cafe3b6ec655c82ebe64a470f6599f513674420a32490402ad897c827cf7e " +
      "756598185990f2143a94d65787ce5fea2b1feae6bed481e79dd216ef426c3eaa",
  },
  {
    // REAL pool-ACCEPTED share captured live from the MoneroOcean pool (gulf.moneroocean.stream:20001
    // TLS), 2026-06-18. MO sends a seed (not a height) -> the epoch resolves from seed_hex (a real ETC
    // mainnet epoch, ~5 GiB DAG). The pool ACCEPTED this share (network consensus-validated) AND it
    // reproduces offline. blob = 32-byte header + 8-byte winning nonce LE (nonce 0xff2b000015b0c4dc).
    name: "etchash gpu1*[intensity=1] recorded-share MO",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "etchash",
      dev: "gpu1*[intensity=1]",
      height: 0,
      seed_hex: "6c81497f04471e1f108bbef0c523cbd56e9c42f5bd589208601eeb88c1460cc6",
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "e3c03345c43176e9af64c85c6a9c7bace62f964aa6b467a43c732d9f442eee51dcc4b01500002bff",
    },
    expected:
      "0000000011ba1869669c2433f9b264b8faf1b8c991ec175d64f0b934c63d78a6 " +
      "2352df3a7c7911c9bdafb65bf2665f30938faaee96259a88f18d88e4e6a7b577",
  },
  {
    name: "autolykos2 gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 20 * 60 * 1000,
    job: {
      algo: "autolykos2",
      dev: "gpu1*[intensity=1]",
      height: 614400,
      target: "0003fffffffffffffffffffffffffffffffaeabb739abd2280eeff497a3340d9",
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f0531000000000000",
    },
    expected: "0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a",
  },
  {
    // FishHash (Iron Fish / Karlsen): ASIC-resistant memory-hard (Ethash-derived + BLAKE3). 8-byte LE
    // nonce at offset 32. Derived offline from the iron-fish/fish-hash C++ reference (light cache + lazy
    // lookup). dev gpu1*[intensity=1]: the lazy kernel recomputes dataset items, so one nonce only for the vector.
    name: "fishhash gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "fishhash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 32,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "abababababababababababababababababababababababababababababababab9a78563412000000",
    },
    expected: "d30e3afb6f50be1bbb8544ad6ad2a303169c5192409a42c85a73706953f04d57",
  },
  {
    // FishHash (Iron Fish) LIVE-POOL vector: a real mining.notify from ironfish.herominers.com:1145 (TLS),
    // miningRequestId 0. The Iron Fish custom OBJECT stratum (pool.js handleIronfishMessage) delivered this
    // 180-byte block header in body.header; the fishhash job builder carries it through verbatim with the
    // 8-byte nonce ("randomness") at offset 0. Hash captured here is for nonce 0 (gid 0), bit-exact from
    // the GPU solver AND the SYCL-CPU device (light cache, no 4.6 GiB DAG) -- the same value the live share
    // submit hashes against the pool target. (Live mining connected/logged in/jobs+set_target parsed and a
    // structurally-valid mining.submit was accepted+mapped by the pool; shares only rejected "Job expired"
    // because herominers' 1 GH/share minimum static-difficulty floor exceeds the per-job window.)
    name: "fishhash recorded-job gpu1*[intensity=1]",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "fishhash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 0,
      target: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
      blob_hex:
        "4d696e6564206279206865726f6d696e6572732e636f6d20313231303332363187811900000000000004" +
        "f9b7ccc6451ac540edba4270b85229935561e0f15ff66cfeb1f30ae9c776536abc9ccaf5ad6bde50d7d28" +
        "7522fefae5d9183f4c2e195f232be657bd40a70f2aa4c246662195b7860036594090543f74014f8d1f3af" +
        "0c196869c700000000000bd8c1067d7a9d5be59a30ad1470fe654b7fd8a54ce41a83c282ecc65f32e29e0" +
        "100000000000000000000",
    },
    expected: "b555131f962db9c12e402d8ae792f1428af2b42353b155e1a6d4831f83dc9954",
  },
  {
    // PearlHash certificate-v3 native-profile regression: reuse the deterministic 76-byte header
    // and expected test marker from the portable vector in tests/vectors/equihash.js.
    name: "pearlhash v3 gpu1*[m=256;k=4096;rank=256]",
    gpu: true,
    syclCpu: false,
    timeoutMs: 600000,
    job: {
      algo: "pearlhash",
      dev: "gpu1*[m=256;k=4096;rank=256]",
      pearlhash_cert_version: /** @type {3} */ (3),
      blob_hex:
        "000040205d1cd9b9049d9f594cd0d05697f99a8a6770bbd59a2aefcf669be71e3b3eb253866b" +
        "c496a00224b6bdf05ed1983a52622fc90bc3ef86969c27bc0d6686afacdfa9db2c6a21000118",
    },
    expected: "ok",
  },
  {
    // Exercise four transcript folds per slot (K/rank=64) without the full comparison profile's
    // memory footprint. GPU tests check commitment seeds; all tuned GPU paths, including Intel ESIMD,
    // also cross-check the search checksum against portable SYCL.
    name: "pearlhash v3 ratio64 gpu1*[m=256;n=256;k=8192;rank=128]",
    gpu: true,
    syclCpu: false,
    timeoutMs: 600000,
    job: {
      algo: "pearlhash",
      dev: "gpu1*[m=256;n=256;k=8192;rank=128]",
      pearlhash_cert_version: /** @type {3} */ (3),
      blob_hex:
        "000040205d1cd9b9049d9f594cd0d05697f99a8a6770bbd59a2aefcf669be71e3b3eb253866b" +
        "c496a00224b6bdf05ed1983a52622fc90bc3ef86969c27bc0d6686afacdfa9db2c6a21000118",
    },
    expected: "ok",
  },
  {
    // KarlsenHashV2 (FishHashPlus): same 4.6 GB FishHash DAG, folded index derivation + plain-BLAKE3
    // wrapping. 80-byte Kaspa blob, 8-byte LE nonce at offset 72. From
    // the authoritative rusty-karlsen test_khashv2 vector (prePow=0x2a*32, ts=5435345234, nonce=432432432).
    // dev gpu1*[intensity=1]: lazy DAG recompute, one nonce.
    name: "karlsenhashv2 gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "karlsenhashv2",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 72,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a52c9f8430100000000000000000000000000000000000000000000000000000000000000000000003065c61900000000",
    },
    expected: "71e8a7ff50f4eba67fbf00af449c12e6e74b1edfc1577b59c41c77922e546f87",
  },
  {
    // Four endpoint pairs from the official block key exercise the portable device SipHash path
    // without allocating the full 14-GiB C30 graph.
    name: "c30 gpu1 portable edge generation",
    gpu: true,
    syclCpu: true,
    portableOnly: true,
    env: {MOM_C30_TEST_EDGE: "1"},
    job: {
      algo: "c30", dev: "gpu1", proofsize: 42, noncebytes: 8, nonceoffset: 32,
      blob_hex: "8bbb8897a7967634e15bae662ee23e16e8c85669f3ca0a9e6584f8f4aa41f220" +
        "d42008d90000001e",
    },
    expected: "c48b882d2d166f29779c36235d4c8e140833a20f387ad915e26cae3fae4be53c",
  },
  {
    // Cortex block 0xee4093: seal hash followed by the winning nonce in little-endian byte order.
    name: "c30 gpu1 recorded-mainnet block 0xee4093",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "c30", dev: "gpu1", proofsize: 42, noncebytes: 8, nonceoffset: 32,
      blob_hex: "8bbb8897a7967634e15bae662ee23e16e8c85669f3ca0a9e6584f8f4aa41f220" +
        "d42008d90000001e",
    },
    expected: "2028e22aa12ab3b07ec6f0c574b28298565d47d8b02625fbfb85b37da0e9b4e4",
  },
  {
    // Authoritative HTND HooHash v1.1.0 vector. Blob = prePow||timestamp LE||zero(32)||nonce LE.
    name: "hoohash gpu1*[intensity=1]",
    gpu: true,
    job: {
      algo: "hoohash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 72,
      target: "00".repeat(32),
      blob_hex: "a49dbc7d44ae83253823592fd388f219f3cb83639d54c9e4c3154db36f2b5157" +
        "079856b891010000" + "00".repeat(32) + "a73d6b8876c27369",
    },
    expected: "ef799688a9be14c60f0e62bba2e86bac14427a0b9d48ae16f1a937217b167987",
  },
  {
    // HTND reference digest for a nonce where the accelerated search filter differs. This keeps
    // test mode and candidate verification tied to the exact consensus path.
    name: "hoohash exact-candidate gpu1*[intensity=1]",
    gpu: true,
    job: {
      algo: "hoohash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 72,
      target: "00".repeat(32),
      blob_hex: "a49dbc7d44ae83253823592fd388f219f3cb83639d54c9e4c3154db36f2b5157" +
        "079856b891010000" + "00".repeat(32) + "83436b8876c27369",
    },
    expected: "65eba5343cf2adffa5db81d59b4ddf91d24caaf45a6a038242a76197a4da50a9",
  },
  {
    // Captured Intel candidate independently reproduced by the HTND C verifier. Device
    // transcendental math produced a different digest, so this guards the host canonical check.
    name: "hoohash canonical-host gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    job: {
      algo: "hoohash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 72,
      target: "00".repeat(32),
      blob_hex: "976a8b2b4b4982bfec000f68719cf11ddad42f69aa2d5aa2f6b1a6fa868f3a88" +
        "228983aea0010000" + "00".repeat(32) + "fa6b70180000e939",
    },
    expected: "03abbe817ab99fabd4be8bd6e6d6731fcf39f6f73528bfb68d5d2cd7f9d590ea",
  },
];

vectors.push({
    // Authoritative waglaylad-rusty v0.14.5 vector, independently evaluated with its PowHash,
    // SHA3-256, and Matrix implementations. Blob = prePow||timestamp LE||zero(32)||nonce LE.
    name: "walahash gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    job: {
      algo: "walahash",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 72,
      target: "00".repeat(32),
      blob_hex: "63e71d5599e1ebcf24ed03376a15dd7a1c33f94cbe8099f4bd681ab2aa04b167" +
        "e2860c6d8f010000" + "00".repeat(32) + "f2b5fd5e8b4d0a9b",
    },
    expected: "c72d158213c8337ef62e2e364858255ab36606264ce214053f8247e7ff29b683",
  });

vectors.push({
    // Authoritative XELIS xelis-hash V3 Rust reference vector: src/v3.rs::test_verify_output
    // (https://github.com/xelis-project/xelis-hash/blob/master/src/v3.rs). The 112 input bytes are
    // copied verbatim; bytes 40..47 are 19865a6b746eec35 (big-endian serialization of nonce
    // 0x19865a6b746eec35).
    name: "xelishashv3 gpu1*[intensity=1]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 5 * 60 * 1000,
    job: {
      algo: "xelishashv3",
      dev: "gpu1*[intensity=1]",
      noncebytes: 8,
      nonceoffset: 40,
      target: "00".repeat(32),
      blob_hex: "acec6cd4b51f6d2d2cf236e18f85592cb36c27bf2074e5213f822178b959928d" +
        "0a4fb76bee7a5cde19865a6b746eec35ff05d67e18d861c794effd66c7b8e8fd" +
        "9e9156bb70514e46506e21259fe9c601b26cd2646d9b6a7c7c535932c573e720" +
        "4a025c2f19dc87f97aacdc898fea44bc",
    },
    expected: "f208b0decb1b68bb162844494f4f41538a650a74c22999155ca30ccee79c4653",
  });

const miscIntensity2Algos = new Set(["hoohash", "walahash", "xelishashv3", "nexapow"]);
for (const vector of vectors.filter(({job}) => miscIntensity2Algos.has(job.algo))) {
  const regression = {...vector};
  delete regression.syclCpu;
  regression.name = vector.name.replace("intensity=1", "intensity=2");
  regression.job = {...vector.job, dev: vector.job.dev.replace("intensity=1", "intensity=2")};
  vectors.push(regression);
}

vectors.push({
  name: "verthash official dataset gpu1*[intensity=16]",
  gpu: true,
  syclCpu: true,
  // First-use dataset generation can take 30 minutes; retain five minutes for the proof.
  timeoutMs: 35 * 60 * 1000,
  perfTimeoutMs: 5 * 60 * 1000,
  job: {
    algo: "verthash", dev: "gpu1*[intensity=16]", noncebytes: 4, nonceoffset: 76,
    target: "00".repeat(32),
    blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f" +
      "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f" +
      "404142434445464748494a4b00000000",
  },
  expected: "26e66d827ac59889008c30f2d9cc2fee6544f5e43bcdd4f1b85928ecb63f9412",
});

module.exports = vectors;
