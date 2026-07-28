"use strict";

module.exports = [
  {
    name: "kawpow gpu1*[intensity=256]",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kawpow",
      dev: "gpu1*[intensity=256]",
      height: 0,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0100000000000000",
    },
    expected:
      "fd91ea3ed018d9d823fa219f7d6bce93ba920f318f8e9b934ebb19290aa112c8 " +
      "60abafe4148f34284b2c9e2e4a222ddcba272cb0669a8673cc1d5934ba5ecbfc",
  },
  {
    // REAL pool-ACCEPTED share captured live from RavenMiner (stratum.ravenminer.com TLS), 2026-06-18,
    // height 4415577 = epoch 588 (mainnet-scale ~5.4 GiB DAG). The pool ACCEPTED this share (network
    // consensus-validated) AND it reproduces offline -> end-to-end real-mainnet vector. blob = 32-byte
    // header + 8-byte winning nonce LITTLE-ENDIAN (nonce 0xef00000002249aaf; ef000000 = pool extranonce).
    name: "kawpow gpu1*[intensity=1] recorded-share h4415577",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kawpow",
      dev: "gpu1*[intensity=1]",
      height: 4415577,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "c3d504e2946989b90767a7a98b59e0770483d256a0178497b0278908ef0edac6af9a2402000000ef",
    },
    expected:
      "0000004eac670a2c1362781f7a9eaa299df135b7bbe4dd276a8eba9587733255 " +
      "2ed045ae6a3828170e292b9a105cb79586d6a74090c772ea3eb438154ae485be",
  },
  {
    // FiroPow (ProgPoW-0.9.4 variant of KawPoW): EPOCH_LENGTH=1300, PERIOD_LENGTH=1, and a padding-
    // constant keccak seal (no magic array). Vectors are firoorg/firo's own firopow_test_vectors.hpp.
    // dev gpu1*[intensity=1] runs a single hash at gid 0; blob_hex = 32-byte header + 8-byte nonce LITTLE-ENDIAN
    // (the firo vector lists nonce big-endian: 85f22c9b3cd2f123 -> stored 23f1d23c9b2cf285).
    // expected = "<final_hash> <mix_hash>". height 1 = epoch 0 (tiny DAG): clears seal + fill_mix.
    name: "firopow gpu1*[intensity=1] height 1",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "firopow",
      dev: "gpu1*[intensity=1]",
      height: 1,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "2d794e900dcad779e658de9078d9a88eee87d75f7b09a8fdd270d3a8e76650c723f1d23c9b2cf285",
    },
    expected:
      "00017c7de1fa499314f9e3dd3537546982073624f7d478592cf28a6d13929f2d " +
      "cfab3766331d6c4e6913e6688a71e4c26b7f36c1581cdbec0f5b19db8956eb50",
  },
  {
    // REAL share captured live from WoolyPooly FIRO (pool.woolypooly.com:3104 TLS), 2026-06-18, at
    // mainnet height 1326124 = epoch 1020 (~10 GiB DAG). The pool's vardiff (4.29 GH) outruns a single
    // GPU so it was submitted stale, but the hash is a genuine mainnet-job result and reproduces offline
    // (correctness anchored by firo's own height 1/2/1300 reference vectors above). blob = 32-byte header
    // + 8-byte winning nonce LE (nonce 0x2e6f000020c42945).
    name: "firopow gpu1*[intensity=1] recorded-mainnet h1326124",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "firopow",
      dev: "gpu1*[intensity=1]",
      height: 1326124,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "308c3193f94225113edb4a8727a753c10b97dff393eda9b227a880208768f1814529c42000006f2e",
    },
    expected:
      "00000000ebdedafd5e17a6ccd9ef312dd5322363a7afb02415340446b741ebe5 " +
      "51839afd7148b3121aac84b0bb4ef0b081a2ab3283a41b15ea59821bab64e381",
  },
  {
    // EvrProgPow (Evrmore): KawPoW with epoch=12000, period=3, "EVRMORE-PROGPOW" seal magic, and
    // chfast/EIP-1057 DAG sizing with a 3 GiB init. Vectors generated from the EvrmoreOrg/cpp-evrprogpow
    // reference (the repo's committed vectors are STALE classic-ProgPoW copies -- do not use those).
    // header_hash = 000102..1f, nonce u64 0x0102030405060708 -> blob LE "0807060504030201".
    // expected = "<final_hash> <mix_hash>". height 0 = epoch 0 (3 GiB DAG): clears seal + sizing.
    name: "evrprogpow gpu1*[intensity=1] height 0",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "evrprogpow",
      dev: "gpu1*[intensity=1]",
      height: 0,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0807060504030201",
    },
    expected:
      "d812833c51da91c0e217e2d02b01cb37f4361f7293fe3abc018a5a39ad0c037f " +
      "0135e22005ad373005c518a9f68099ffff8f698af61580495b620ef2fbe6380f",
  },
  {
    // REAL accepted share captured live from Mining4People (us-east.mining4people.com:24173 TLS),
    // 2026-06-17, height 1896108 = epoch 158 (mainnet-scale ~4 GiB DAG). The pool ACCEPTED this share
    // (consensus-validated) AND it reproduces offline -> the strongest end-to-end vector: real job
    // header, real winning nonce, mainnet DAG. Submitted nonce 0xb8c700003165dfca stored little-endian
    // in the blob (b8c70000 = pool extranonce high bytes). See logs/evrprogpow_live_captured_shares.log.
    name: "evrprogpow gpu1*[intensity=1] recorded-share h1896108",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "evrprogpow",
      dev: "gpu1*[intensity=1]",
      height: 1896108,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "bb9c2f5035d8ad57359cab200094105e233e1c0e42ae1baf9568b710faba0c6ccadf65310000c7b8",
    },
    expected:
      "000000044f87561c07bf42dca0c22211319d29fffcf95988e94c4791f4628f6d " +
      "fec20ef703aa7e02ec069d1f392c2e73236ed4116f44650e108405b53382916b",
  },
  {
    // MeowPow (Meowcoin): KawPoW with a SHORTER ProgPoW inner loop (REGS 16, CNT_CACHE 6, CNT_MATH 9),
    // period 6, epoch 7500, the "MEOWCOINMEOWPOW" seal magic, AND a one-time epoch-110 "dag change"
    // (block 960000): at epoch >= 110 the DATASET+CACHE SIZES use a 4x-scaled epoch (epoch*4) while the
    // keccak seed keeps the real epoch -- so this block sits exactly on the fork and is the strongest
    // guard for it. REAL mainnet block 825000 = epoch 110 -> meow_epoch 440 -> ~4.4 GiB DAG (network-
    // consensus-validated). blob = 32-byte header_hash + 8-byte nNonce64 0xdcc1030548373115 stored
    // LITTLE-ENDIAN (153137480503c1dc). expected = "<final_hash/powhash> <mix_hash>" (display/BE).
    name: "meowpow gpu1*[intensity=1] height 825000",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "meowpow",
      dev: "gpu1*[intensity=1]",
      height: 825000,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "ae601a830bf62cd859802ad2a4a0b3748d5ca37cbd06722759ee3eb27776316a153137480503c1dc",
    },
    expected:
      "00000000002ae4fd9e99671d5af0d2a22aa65a3b5efb1febc0b838b22af798f5 " +
      "fddd5a68df54c35c14e5a2eacaf123ee561d01eb6257bd618316265b0b52665b",
  },
  {
    // MeowPow epoch-0 sanity vector. This keeps the SYCL CPU suite off the live/mainnet DAGs while still
    // exercising the MeowPow-specific reduced ProgPoW shape and seal.
    name: "meowpow gpu1*[intensity=1] height 0",
    gpu: true,
    syclCpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "meowpow",
      dev: "gpu1*[intensity=1]",
      height: 0,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0100000000000000",
    },
    expected:
      "9cdf1d91320fd51404860aa692cbcca0ecf4a40e3f8b7f4eeff325612f344c76 " +
      "80e1899bacd477a2efe2ce5994f784206612802626675b9287bc045a0c7eb425",
  },
];
