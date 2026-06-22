"use strict";

function dup(str, n) {
  return Array(n).fill(str).join(" ");
}

const hashTests = [
  {
    name: "rx/0 cpu*2",
    job: { algo: "rx/0", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("38f638606c730dd6f271d037556b83988c71acc6980e22e25271b22389ecfce6", 2),
  },
  {
    name: "rx/2 cpu*2",
    job: { algo: "rx/2", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("ad6eff4f6d8a301b40183174edb4cf72b85caa65e8e5616354c92a2607022712", 2),
  },
  {
    name: "rx/wow cpu*2",
    job: { algo: "rx/wow", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("15c9bd99b3180ab256e89beecaf7b693abb7cdb0d1dfe30020c72f0c70b904ce", 2),
  },
  {
    name: "rx/arq cpu*2",
    job: { algo: "rx/arq", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("8b9937420651309742f833333371a8ab4d04e5f06d4b3d0a2dbec1b381e9a0e5", 2),
  },
  {
    name: "rx/graft cpu*2",
    job: { algo: "rx/graft", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("08135aaeb098d86f269d0a037aba423093e4c48e9e31f16c13d2ed3dc4489ba7", 2),
  },
  {
    name: "rx/sfx cpu*2",
    job: { algo: "rx/sfx", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("19e786570a6d959f023cb77b0be5bede76cc4a5c61c090b3d24c5ebd2c9ecb27", 2),
  },
  {
    name: "rx/yada cpu*2",
    job: { algo: "rx/yada", dev: "cpu*2", blob_hex: "5468697320697320612074657374" },
    expected: dup("c6cc14fe859f917013223aa7a1959a169162df14510d462b681c70895ea6f874", 2),
  },
  {
    name: "panthera cpu*2",
    job: {
      algo: "panthera",
      dev: "cpu*2",
      seed_hex: "786d7269672d6d6f2d636865636b2d6861736865732073656564",
      blob_hex: "786d7269672d6d6f2d636865636b2d68617368657320696e7075742041",
    },
    expected: dup("1805aa4fd26f5686dada26d9e16ccb290811b2a79404fdd1064f1ca092902212", 2),
  },
  {
    name: "ghostrider cpu*8",
    job: {
      algo: "ghostrider",
      dev: "cpu*8",
      blob_hex:
        "000000208c246d0b90c3b389c4086e8b672ee040" +
        "d64db5b9648527133e217fbfa48da64c0f3c0a0b" +
        "0e8350800568b40fbb323ac3ccdf2965de51b9aa" +
        "eb939b4f11ff81c49b74a16156ff251c00000000",
    },
    expected: dup("84402e62b6bedafcd65f6ba13b59ff19ad7f273900c59fa49bfbb5f67e10030f", 8),
  },
  {
    name: "argon2/chukwa",
    job: { algo: "argon2/chukwa" },
    expected: "c158a105ae75c7561cfd029083a47a87653d51f914128e21c1971d8b10c49034",
  },
  {
    name: "argon2/chukwav2",
    job: { algo: "argon2/chukwav2" },
    expected: "77cf6958b3536e1f9f0d1ea165f22811ca7bc487ea9f52030b5050c17fcdd8f5",
  },
  {
    name: "argon2/wrkz",
    job: { algo: "argon2/wrkz" },
    expected: "35e083d4b9c64c2a68820a431f61311998a8cd1864dba4077e25b7f121d54bd1",
  },
  {
    name: "cn/0",
    job: { algo: "cn/0" },
    expected: "1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100",
  },
  {
    name: "cn/1",
    job: { algo: "cn/1" },
    expected: "f22d3d6203d2a08b41d9027278d8bcc983acada9b68e52e3c689692a50e921d9",
  },
  {
    name: "cn/2",
    job: { algo: "cn/2" },
    expected: "97378282cf10e7ad033f7b8074c40e14d06e7f609dddda787680b58c05f43d21",
  },
  {
    name: "cn/r height 1806260",
    job: {
      algo: "cn/r",
      height: 1806260,
      blob_hex:
        "54686973206973206120746573742054686973206973" +
        "20612074657374205468697320697320612074657374",
    },
    expected: "f759588ad57e758467295443a9bd71490abff8e9dad1b95b6bf2f5d0d78387bc",
  },
  {
    name: "cn/fast",
    job: { algo: "cn/fast" },
    expected: "3c7a61084c5eb865b498ab2f5a1ac52c49c177c2d0133442d65ed514335c82c5",
  },
  {
    name: "cn/half",
    job: { algo: "cn/half" },
    expected: "5d4fbc356097ea6440b0888edeb635ddc84a0e397c868456895c3f29be7312a7",
  },
  {
    name: "cn/xao",
    job: { algo: "cn/xao" },
    expected: "9a29d0c4afdc639b6553b1c83735114c5d77162142975cb850c0a51f6407bd33",
  },
  {
    name: "cn/rto",
    job: { algo: "cn/rto" },
    expected: "82661e1c6e6436668406327a9bb11319a5561615dfec1c9ee3884a6c1ceb76a5",
  },
  {
    name: "cn/rwz",
    job: { algo: "cn/rwz" },
    expected: "5f56c6b0996ba23e0bba0729c99074855a10e3087fdbfe947533547376f075b8",
  },
  {
    name: "cn/zls",
    job: { algo: "cn/zls" },
    expected: "516e33c6e446abbccdad18c04cd9a25e64102853b20a42dfdeaa8b599ecf40e2",
  },
  {
    name: "cn/double",
    job: { algo: "cn/double" },
    expected: "aefbb3f0cc88046d119f6c54b96d90c9e884ea3b5983a60d50a42d7d3ebe4821",
  },
  {
    name: "cn/ccx",
    job: { algo: "cn/ccx" },
    expected: "b3a16786d2c985ecadc45f910527c7a196f0e1e97c8709381d7d419335f81672",
  },
  {
    name: "cn/upx2",
    job: { algo: "cn/upx2" },
    expected: "aabbb8ed14a835fa22cfb1b5dea872b0a1d6cbd846f4391c0f01f3875e3a3761",
  },
  {
    name: "cn-pico/0",
    job: { algo: "cn-pico/0" },
    expected: "08f421d7833117300eda66e98f4a2569093df300500173944efc401e9a4a17af",
  },
  {
    name: "cn-pico/tlo",
    job: { algo: "cn-pico/tlo" },
    expected: "9975f2c1b3b45434a49386213097f31bb4b9a6586a7e81f4429f6d5f65c38d1a",
  },
  {
    name: "cn-lite/0",
    job: { algo: "cn-lite/0" },
    expected: "3695b4b53bb00358b0ad38dc160feb9e004eece09b83a72ef6ba9864d3510c88",
  },
  {
    name: "cn-lite/1",
    job: { algo: "cn-lite/1" },
    expected: "6d8cdc444e9bbbfd68fc43fcd4855b228c8a1bd91d9d00285bec02b7ca2d6741",
  },
  {
    name: "cn-heavy/0",
    job: { algo: "cn-heavy/0" },
    expected: "9983f21bdf2010a8d707bb2f14d78664bbe1187f55014b39e5f3d69328e48fc2",
  },
  {
    name: "cn-heavy/xhv",
    job: { algo: "cn-heavy/xhv" },
    expected: "5ac3f785c490c58550ec95d2726563577e7c1c212d0cde591273201e44fdd5b6",
  },
  {
    name: "cn-heavy/tube",
    job: { algo: "cn-heavy/tube" },
    expected: "fe53352076eae689fa3b4fda614634cfc312ee0c387df2b8b74da2a159741235",
  },
  {
    name: "cn/gpu gpu1*8",
    gpu: true,
    job: { algo: "cn/gpu", dev: "gpu1*8" },
    expected: dup("e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df", 8),
  },
  {
    name: "kawpow gpu1*256",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kawpow",
      dev: "gpu1*256",
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
    name: "kawpow gpu1*1 live-share h4415577",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kawpow",
      dev: "gpu1*1",
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
    // dev gpu1*1 runs a single hash at gid 0; blob_hex = 32-byte header + 8-byte nonce LITTLE-ENDIAN
    // (the firo vector lists nonce big-endian: 85f22c9b3cd2f123 -> stored 23f1d23c9b2cf285).
    // expected = "<final_hash> <mix_hash>". height 1 = epoch 0 (tiny DAG): clears seal + fill_mix.
    name: "firopow gpu1*1 height 1",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "firopow",
      dev: "gpu1*1",
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
    name: "firopow gpu1*1 live-share h1326124",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "firopow",
      dev: "gpu1*1",
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
    name: "evrprogpow gpu1*1 height 0",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "evrprogpow",
      dev: "gpu1*1",
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
    name: "evrprogpow gpu1*1 live-share h1896108",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "evrprogpow",
      dev: "gpu1*1",
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
    name: "meowpow gpu1*1 height 825000",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "meowpow",
      dev: "gpu1*1",
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
    name: "meowpow gpu1*1 height 0",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "meowpow",
      dev: "gpu1*1",
      height: 0,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0100000000000000",
    },
    expected:
      "9cdf1d91320fd51404860aa692cbcca0ecf4a40e3f8b7f4eeff325612f344c76 " +
      "80e1899bacd477a2efe2ce5994f784206612802626675b9287bc045a0c7eb425",
  },
  {
    // REAL pool-ACCEPTED share captured live from RPlant MeowCoin (stratum-eu.rplant.xyz:17120 TLS),
    // 2026-06-19. RPlant accepted this share. blob = 32-byte header + 8-byte winning nonce LE
    // (native/pool nonce 0x193400002ebab394; 0x1934 is the subscribe extranonce prefix).
    name: "meowpow gpu1*1 live-share h1960554",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "meowpow",
      dev: "gpu1*1",
      height: 1960554,
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "5af87965c0fa82fcb208977be5e3cecae9974b81a907be1436bf55490c3ee74b94b3ba2e00003419",
    },
    expected:
      "000000018175d1e821428d61145bcc14859f6741e2311446159a48c7d220e38b " +
      "90b603f366e690aa8b6d35bd018d1ae7384339ad43597c0702fe0f8ddc47073d",
  },
  {
    name: "etchash gpu1*256",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "etchash",
      dev: "gpu1*256",
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
    name: "etchash gpu1*1 live-share MO",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "etchash",
      dev: "gpu1*1",
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
    name: "autolykos2 gpu1*1",
    gpu: true,
    timeoutMs: 20 * 60 * 1000,
    job: {
      algo: "autolykos2",
      dev: "gpu1*1",
      height: 614400,
      target: "0003fffffffffffffffffffffffffffffffaeabb739abd2280eeff497a3340d9",
      noncebytes: 8,
      nonceoffset: 32,
      blob_hex: "548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f0531000000000000",
    },
    expected: "0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a",
  },
  {
    // kHeavyHash (Kaspa): compute-bound 64x64-matrix + double-Keccak. 80-byte header, 8-byte LE nonce
    // at offset 72. Derived offline from the rusty-kaspa reference (validated bit-exact vs js-sha3
    // cshake256): pre_pow_hash=0x2a*32, timestamp=5435345234, nonce=432432432. No DAG/epoch.
    name: "kheavyhash gpu1*256",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kheavyhash",
      dev: "gpu1*256",
      noncebytes: 8,
      nonceoffset: 72,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a52c9f8430100000000000000000000000000000000000000000000000000000000000000000000003065c61900000000",
    },
    expected: "5a5bcd6e352eb8c87c80d0f0574a45a5fcc3d5755660ac120dc9893684c19be6",
  },
  {
    // kHeavyHash (Kaspa) LIVE-POOL vector: a real mining.notify from kaspa.herominers.com:1206 (TLS),
    // job_id "73". The pool sent the pre-pow as 4 LE uint64 words [14311454634314100036,
    // 12026711064689961364, 15283992124271141330, 4909777105915057546] + timestamp 1781909733171; the
    // kaspa stratum dialect (pool.js kaspaNotifyJob) builds this 80-byte header (LE words || ts || 32
    // zero pad || nonce) with the 8-byte nonce at offset 72. Hash captured here is for nonce 0 (gid 0),
    // bit-exact from the GPU solver -- the same value the live share submit hashes against the pool target.
    name: "kheavyhash live gpu1*256",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "kheavyhash",
      dev: "gpu1*256",
      noncebytes: 8,
      nonceoffset: 72,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "44ed68212a809cc694e91a3bfe75e7a6d2a10ce5dba51bd48a9569c44308234433bf18e29e01000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    },
    expected: "139108d4e9d63e8fac47cff8f98f2b2c6c757eb08d46ff13d1ba65daddfe69c7",
  },
  {
    // FishHash (Iron Fish / Karlsen): ASIC-resistant memory-hard (Ethash-derived + BLAKE3). 8-byte LE
    // nonce at offset 32. Derived offline from the iron-fish/fish-hash C++ reference (light cache + lazy
    // lookup). dev gpu1*1: the lazy kernel recomputes dataset items, so one nonce only for the vector.
    name: "fishhash gpu1*1",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "fishhash",
      dev: "gpu1*1",
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
    // because herominers' 1 GH/share min static-diff floor exceeds the per-job window at the L4 rate.)
    name: "fishhash live gpu1*1",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "fishhash",
      dev: "gpu1*1",
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
    // KarlsenHashV2 (FishHashPlus): same 4.6 GB FishHash DAG, folded index derivation + plain-BLAKE3
    // wrapping. 80-byte Kaspa blob, 8-byte LE nonce at offset 72 (same blob shape as kHeavyHash). From
    // the authoritative rusty-karlsen test_khashv2 vector (prePow=0x2a*32, ts=5435345234, nonce=432432432).
    // dev gpu1*1: lazy DAG recompute, one nonce.
    name: "karlsenhashv2 gpu1*1",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "karlsenhashv2",
      dev: "gpu1*1",
      noncebytes: 8,
      nonceoffset: 72,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a52c9f8430100000000000000000000000000000000000000000000000000000000000000000000003065c61900000000",
    },
    expected: "71e8a7ff50f4eba67fbf00af449c12e6e74b1edfc1577b59c41c77922e546f87",
  },
  {
    // PyrinHashV2 (Pyrin PYI): kHeavyHash-family (no DAG) -- 64x64 matrix matvec, plain-BLAKE3 powHash/
    // final + V2 nibble-XOR reduction. 80-byte Kaspa blob, 8-byte LE nonce at offset 72. From the
    // authoritative Pyrinpyi/pyrin source (prePow=0x2a*32, ts=1715521488610, nonce=11171827086635415026;
    // powHash e369d9a9...117626 is the repo's published intermediate).
    name: "pyrinhashv2 gpu1*256",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    job: {
      algo: "pyrinhashv2",
      dev: "gpu1*256",
      noncebytes: 8,
      nonceoffset: 72,
      target: "0000000000000000000000000000000000000000000000000000000000000000",
      blob_hex: "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2ae2860c6d8f0100000000000000000000000000000000000000000000000000000000000000000000f2b5fd5e8b4d0a9b",
    },
    expected: "646bfd6050195afc4ce933216c28681954cfb3c2354a0f0ed556ba4d96f7f254",
  },
  {
    name: "c29 proofsize 32 gpu1*1",
    gpu: true,
    timeoutMs: 10 * 60 * 1000,
    job: {
      algo: "c29",
      dev: "gpu1*1",
      blob_hex:
        "0001000000000000202e000000005c2e43ce014ca55dc4e0dffe987ee3eef9ca78e517f5ae7383c40797a4e8a9dd75ddf57eafac5471135202aa6054a2cc66aa5510ebdd58edcda0662a9e02d8232a4c90e90b7bddec1f32031d2894d76e3c390fc12b2dcc7a6f12b52be1d7aea70eac7b8ae0dc3f0ffb267e39b95a77e44e66d523399312a812d538afd00c7fd87275f4be7ef2f447ca918435d537c3db3c1d3e5d4f3b830432e5a283fab48917a5695324a319860a329cb1f6d1520ad0078c0f1dd9147f347f4c34e26d3063f117858d75000000000000babd0000000000007f23000000001ac67b3b0000015545f385f2",
      proofsize: 32,
    },
    expected:
      "9f89402d614224adc4da5bd7c98f70e9e8b72841cfaa28fe61420af6ef1514ca " +
      "793a07f3e629809f7a0d06287fbfb138e1f6266946610d0279e2f8e04ade52f8 EOL",
  },
  {
    name: "c29 proofsize 42 gpu1*1",
    gpu: true,
    timeoutMs: 10 * 60 * 1000,
    job: {
      algo: "c29",
      dev: "gpu1*1",
      blob_hex: "000000000000001e3695b4b53bb00358b0ad38dc160feb9e004eece09b83a72ef6ba9864d3510c88",
      proofsize: 42,
    },
    expected: "68005b0465cf34675f804de6ef37b3ea2fed0f4796236fc55d1ecd996b54db2d EOL",
  },
  {
    // Equihash 125,4 (ZelHash / Flux): the GPU Wagner solver run end-to-end on Flux mainnet block
    // 400000. With MOM_EQUIHASH_SOLVE the native is_test path runs the full solve (M2 rounds + M3
    // recovery) and dumps the whole SMALL_BLOB_SOL_LEN(=5120 B) out-of-band buffer as hex:
    // [count:u8][count * 52-byte compressed solution][zero pad]. The solver finds the 2 distinct
    // proofs for this header (sorted), the first being the known block-400000 solution 0898985c...199b.
    // Heavy: a full solve is ~0.5 s + ~7.3 GiB of Wagner arenas (progressive per-level slot shrink),
    // so this needs a >=8 GiB GPU (B580/L4) and is gated behind MOM_EQUIHASH_SOLVE -- the default GPU
    // suite without it skips straight past.
    name: "equihash125_4 gpu1 (block 400000 solve)",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    env: { MOM_EQUIHASH_SOLVE: "1" },
    job: {
      algo: "equihash125_4",
      dev: "gpu1",
      noncebytes: 8,
      nonceoffset: 108,
      target: "ff".repeat(32),
      height: 400000,
      // 140-byte Flux block-400000 header (version || prevBlock || merkleRoot || finalSapling ||
      // nTime || nBits || 32-byte nonce); the solver's 32-byte nonce lives at offset 108.
      blob_hex:
        "04000000a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000" +
        "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3" +
        "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69" +
        "90e44f5dce28421d" +
        "600000160000000000000000000000000000000000000000000000009cfd1100",
    },
    // [02][known sol 0898985c...199b][second sol 08abd63b...88aa][zero pad to 5120 bytes].
    expected:
      "02" +
      "0898985c54d7c66d04fbe9509445c19e664aca4b8d16a701f9f90eda443a556f37cc4627335201339b001ea2ed29eeeec7da199b" +
      "08abd63ba97c17abd02ba77b782f5a810c84fcd6edb26fad50cc1a690d930ec62ed67183c146d4943590b475c7fba048ff8888aa" +
      "0".repeat(10240 - 2 - 104 * 2),
  },
  {
    // pearl is a NoisyGEMM search, not a fixed-hash algo: in test mode pearl() forces m=n=256, sets
    // the target to all-0xFF (first tile wins) and runs one attempt, so the core returns "ok". This
    // exercises the GPU kernel end-to-end deterministically and also seeds the pearl perf entry.
    name: "pearl gpu1*1",
    gpu: true,
    timeoutMs: 10 * 60 * 1000,
    job: {
      algo: "pearl",
      dev: "gpu1*1",
      blob_hex: "000040205d1cd9b9049d9f594cd0d05697f99a8a6770bbd59a2aefcf669be71e3b3eb253866bc496a00224b6bdf05ed1983a52622fc90bc3ef86969c27bc0d6686afacdfa9db2c6a21000118",
    },
    expected: "ok",
  },
  {
    // Equihash 125,4 (ZelHash / Flux) GEN sub-step ONLY -- a LIGHT, CPU-runnable regression guard for
    // the BLAKE2b "ZelProof" gen kernel. The default is_test path (no MOM_EQUIHASH_SOLVE) runs ONLY the
    // gen kernel and dumps the first EQUIHASH_TEST_ROWS(=256) expanded 20-byte collision rows (5120 B =
    // SMALL_BLOB_SOL_LEN) as hex. The FULL Wagner solve is far too heavy for the CPU SYCL device (~7 GiB
    // arenas) and stays in the gpu1-only solve vector above; this gen-only vector is cheap enough to run
    // on the SYCL CPU device. Header = Flux block-400000 (matches the solve vector). Expected derived
    // from reference indexHashRow(makeBaseState(buildHeader400000()), i) for i in 0..255, and cross-checked
    // bit-exact against the GPU + SYCL-CPU gen dump.
    name: "equihash125_4 gpu1 gen",
    gpu: true,
    timeoutMs: 5 * 60 * 1000,
    job: {
      algo: "equihash125_4",
      dev: "gpu1",
      noncebytes: 8,
      nonceoffset: 108,
      target: "ff".repeat(32),
      height: 400000,
      blob_hex:
        "04000000a8675c842f7a1342fadd00cd9b4e4909526b1c0ab5a747c5529b4deb13000000" +
        "ce7d6ea2452245925fc70c3a08a3c3dd2ca4beab7481f237a19751666bfd25c3" +
        "0fd282d94b1e1a7f2c57eb3fb9e2853d990753fa137e13c99bd43f220d4fce69" +
        "90e44f5dce28421d" +
        "600000160000000000000000000000000000000000000000000000009cfd1100",
    },
    expected:
      "00c3ba2c00db9935009896a301ae2ec200e3383200a862b6016fcbcb0076b9f6007c345f00349b8d00d92f4c01d7ebaf019af3e40039dc920095b04a01cd8530016bb88901275f7600297b4100b55e8901a0106900f7c7a301de090c01d0306c008f333c00ade85700a3204a0038330001c1b96b00fc22ba00f41cbc00f9aff400c12074016b49d90107454d014f21db013a363801bfff6c00c73fd401789f5001c12a0e01dfb9b0011bd1170059adc8012b491400973e830055b10501c21ef6006294eb013b028d009a18b6007947d9002928b2002ab07000d76c71015de1260101bcd0000e75a901c97f4c0063326a01c876a700c8cba001363aa7012f6a5301cee25201373c8e01de3ce901f4082f01b8bd9e011463f3002b79c801d548aa017a978301b4ae2900e8dcb8013e16f0008dc574011820e60023f2d401bd278a01710f960159bccc00dd224a00d01c950096a75f0158980200fbf6fe008af27701534a2500b12a7501bff6cc00002a28012d07890053042a018bc87600a05a8b0198347a01d9ce22009fbb6d008b636e00ecd3fb0129814f00d0c699007ea7d4013866da01afd24e01fcbaea01e9ce5a009b0ffe00d4fb1b00dd8ceb00e34150008f773401c3296f017e313b0191dc1c00f539e300b57ddf01c348f801ba33910086f5ef002831bf01748f0900555cd7017dfc9d0185589701d0d8de00405b3c0097217b0176d1df01497f4200c8ee09017f666501a16b2901843b900079d5f50190ea310167fe8500ca680a015c66d2001f4dff01c4460a0106af990105337101f250d2001b6e760100232700c8b9f700de1f3900d6159b00246fa600d1735901d99fcd00bf7d4200fd798e00b5755c00130403010b681f00590e6001e9254e01be58c200952522011e8b2501b0d6ae019856f0007801fe00efb61100a0eade01e76c7801bc0aee01ee00ce0183042201b908eb010006d60130365f0008b75300d0269000eaba4e0077d852018cf0430092d91e004e6dcd01693eb50193eaad01c08fa700d4fe130066cd1301d164e1007f3c7901f7e8c5006c7a440102987701e5b2ba018d68f901a73a0d015c0bbe00ad518e01956f3d015ac4010092c74f0135187b01fa2a3501ec17b9018fe43c01b7e21f01ed7a9200ab80a1014fa5f10103f1b901592cf50024774d01e89fcb004e3aa001a456c301087e75016434f00078225f0071af0601998054008e20f80039b7ca01d2063800616312014ca89f015ebd6d00c75dff001013730136c53301b9f6fb003825ba00933a6800ba3cf701070cc9003faa92001d6fe101684a7900bf599a010aab6a00b9f6cc014f827f006f530c00e4eed300fa58a901780980006877f20135ddc1010013b201027f1c0177db6701f5e8de0128eca500332b8d01779caf00fd4d5601fa5683003d843e01d00607003662b801fe72ce001f3aec001ee1f600479c7801cbb6dd005b4d0200eb477300658a17010e7e2d01a1111d016353e101c12a2100aa9a9701b84e5700a4b84800ba465800409bbe015db7f801c9e4ae00123bf4004fcb4e0178703c008e57f7001aae130035bb5600a65106009894270183d05401a2761e01a0ed8601a9e19801386d9501581a7000531c7d00785b3b012e874a019e286e01984bd700e51be30091180600d03c4c004f88ce008ccabe012b08fe00f28bcc01516c9701b5b8630037953c00d40ef101cdb293008eb678015b1912018c9d1e003b838b004e8f3700664b5e01151eb7019d7597014002a100db730f0154a51f004af2a300a021c9008503f1018fb991017a966701948c0701e46a4a01deb109007eba8401db91720063e9c600a48daa0194848c00105faf002c0e6701824cdb0017128700bd526200c3072a00a257530157e7ac001dfc040022f99100859fde01be8a9d00f748200191450c00a7fd4401ad81e8000ac87d00243284000137d10026061b0125c969005ded1001f21fdb01d4e4fb01a4746701cff30e001c49c5012f10c20017abd700a89997006089910002566d00c7d6a2018dd38e00ff054801bacf80009405be01b57c6000ca11a50024b4bc007106450033e41801a79c9401f6e936017b6d4401b8e01300beb629004682b0003fa72101f718c200ac8eb0019cd26900c69e09012e5d3700114c8e00af7506006a0129001c682d0179bd5601c28650015cbb130085c52e01e83d77004a3da40010a0b7017ba5790194c3ff00a50ab600c91af000758c3d018ac3fd007bd8730130a84301bf0002013e0b4501bd50e6018af17401ef6903006e577200ed743500c052a00195e63000cce2fe01155371005dd315017739dd0188d30a011a6155003f90f8015c443300bcd6fa012e582400b2369401abb84a00ff5b9f003839960153b42e0136573400c57fec00b186bc00011820012f89a5011fb1bb01994fab01f40cef00b518cb0153d895019f9f5e0095aff6008a03e800b6b33701fc8ecc00d446f0014ad12400bcec130101e82000910615017232ec017ccc1c00b9531f003725a101c73e2e004d2987005555ce015411db00f36941018645f50125b01a006bc9be000b3fd100c798cc01c2316301d3c66e00c32cc60022b7090162d914003d48620060d66500b187a401d523c50103bd1201714312005ec1120181f7b000c9579001786ead01b75454011d343f0175017701b459580052166201f9556701fa461c00d00ab20155cd9c00fa1c9900811f850077250a015d760201f45f7f002279c00081a155008f3a460136535201d184c5017923e3010c9a450067fd400150950f018502ab01106159007eb87c002754c401c0104a001f405c013e170d0184b32501ec19c0014579900187f2a800bc612600aba04100a9181b01f849680052f41f008b68020147210101b4d909000681de0056318201f6d8b801343a79006b257400fc3d76018d41bc010e45a80100a25c015ddd9e01574f0201f40dae0143736f0165efe0012e64ef0118779501b839a001138b3600bc8b340121fffc01ad2aad01a44afa010e66ff00cc016101f5764d004fe90d009a349c01a3a04f0076dcdd01602ce30077d6830026ea4800532b03004a48cc00f7776e00f2a16101cab98b000fdc8e0126356f00a88de70040e3590036237100cd6380003a983100fe7f11003f151901bda880009ea13501ad394600062fcc00ebadc8006f9f710061b81500fd92740094027500cbb45c0000ceae01b5f9fd015e915d01c9b3c20143c0e601fb4a50009cfc16003c708d00b4745a00ba298b00fa9d2e019325cb01148cfd00c1bd07011a89ca01a23bde016650dd00dd564901aef863018e0d2401b3189500db38fa00d5d23601943ffc00973f79004b254e009e528e01689e1100ffcf6000f8d18400850af601b4766101845f3e009a7bda014b388701ab8843001f44e800c4cda70021af9f01e4deb7007bdfcc00de0fdd018842dc011a80a8017818c9014560d101c7a7ed0016a70900a1e13d00813ac20155f79700223b8c0095b30100963abe00f1872b01180b4201b2b0de0086d1130049bee600975a6300df9b2a01260bb30171003500adb99f01259287019f5fab0056b57700a4ffdf00b9efbd01f2908701572d0701ef50c50159e6b7004a4e8800b82f80017abb67013b1fcd01f2eaec005e687900bced7e00f8a3d00173094f0145e55a00d6a31100880b0f01676afc003b1e5100e22cd901ecbdcd009a582e016117f101409547016c0bb9015863f701a6e0e2016c0a0e0177ae1401d084e6013e50ab011cf6be00a6eb12008bed75015d82e1016f279800dfb2e80161c280019faca20052198b008d8e0001174e5e00ec45030169fa9600ed84e7008da8bb01dd1cd5009578c5008052d101684e87010bb22d0177f07d00e34eff004533e70159a11a01e3b7b300c11112009ee53b01a83647014322b6016d1c6000eeb1f401d65e3d018be19601e205b100621cec01615e64010a440501a35b8101f00c5e00cc992e017177290011cc6c00116ce30146eb93006ec90801c9c50f00e5543b00af2d9c016d825a004f0236011845e60037f1080116d80901526684017af15401d1ba6e007d07d90074e49001a508a70135f92901d3050500047fb80165cb2c00ec3a8300aabd0d00c96bc4010022940199139c0019c5bf00275a2a013e29640152d7290044b4070000a86300f141d901f30a0300fa330201678fff01ae8895000364e9017314c40046366f005fc23100b33c9f006b45ce0026b1a9019e5e37011fac9e007d174c0166729d007cfae60044b2070183e04a01b85199011db78e000296710005fba300681a5c0055def9003dd3af00b929cf00fe6c58014e574701bee91d011125d70093210d0088078000a8b827006e537c01f18d4b01d95bc800cbb9dc00b5af8a00a0833500b514c301ff1150016f2fd100cff7dd0051697d01d24d090066242601fc11100069da0b00656cca00b657e000e8497f007996d900ec0a2000ae92a2008aea5301803f2f018aa68d008ba27000424c64009340eb007c17ba008ec53d003435b8004bf89700cdaec4006185e1012eb22d0060669a017bce5b00a646a200cf2dfd01742c7200549e9f0073f976006b1d5901a21edd001f577e008272ba003710c700e576a30049cfbb006f7abe003fd0a100371ca700876db600e5b77001c2923b0014226f0111ae0c00baa7360033e3d6007d551d01d716b900cb51cb008e83cc0017b2e70199bbe500d2f4fa00f5e7d00015d67800ce809301266a5b01ea7eb801302f7d01ad8171015a17430063f59a0151119e00140a6d017c8ff8008731e301f7117d018b61e0009bf5800055539d0185f57f009326b6003b539101add89b01d400f301631a5e011043e60067f6ba0060335701bf2bde0157cefd00eabffc00ac91eb00e725e600bb6d4600d9c80101660a000118f8950072ba7a00c6657a00a696e5018b359c003b46c3014eb4df015487f2004d97ff001c47c7013ea8d200f354b80178d4f8016929f101edff5301c370d301f2b498002967fd0081c7ae004de5020168a1aa0054308e00dadc8b00b68e0b010b03c200072d9300767d56018d6b3b009b1db400ed78370017046401db2671001f13f5011e74e801f8b5d501207e520183f3eb013cf6bf00423ec200c2112c00a1729e01514c1d00016b5f0036a5a800efa15701044f650136f72700cc9e4501c1b4fe0170983700681f8801043b21007f51fb010870ce0186f8310157672b01cc2ef60176040b0039bf19006228ff001a051b00677f7a0192f3d80102f714005cbdf90138dae7007221550140b6600158688701afcacc004793d2003da228015deb6a0052d97500075b820037c7a20057244e01f58e0b0069155201469d3e0018432800d95934010752c40071fd3d00c8003301c80d6001bd4347003a03f100b13c4900beee0e01e48569012d0e7200c691df01344a3a01b6a40201804ef30093178e0049369f012c162a00d1521901ecafcd01968961010a66670001899001689c22003195c7000850de0147934200a682f6000228e6013f2b98004ee5f000c9f41a01ba5ce001cb0851008cc1bf0008757e01a4d90f00bc276300a613ee01774b1a01fbf15a01e6a73c01b5862101cb047c006ce24900d1c84001e6ddf20017d51c004ac7980117ed4501d0d3bc009b317501b47bd6003f494b011c363a00be749e00cef83001a3c403012402d20053e67d0072190f0178ce1801d8e62f002e7abd005edcd401f425020057ecef016ae8a801cbbc3b000c788201623590005fefdf01000709013f53c901f5152b007fac4e0060babe01fe62ba01972524005eaacb0001cc1e00a8bed30115c4cb0077b652004566640165e36b0092ec0e00da0499007abf940112ad86009a6069018cff630034c66301615ec7004eee87012f0661004273a5012d7c3801bcff2f009d13720126ee4400be73540047d64d008ed323002f328700c287530145a35d00d2b36300b40b1000869608005a3b0901e3e08a00f0e7d801d3379800812bd3004546f801f7794201970ffb01fe0202002be47900ad6c6a00005c160094e4b001b3294600eff94201f5d30b00cf8680003ec48b0120d70e0141355f00a665cd01c8672400d0b9f801ce713e012f380d0050497101ec06b1010217d301d98be50126e4710074d7050120351e002b98bc01e770ce0041bebb01ce7d9300d627dc019998160103edb801ebf1f3016817930098aa4c00025416005530c800c7b3cd00dd53a4011e187b00db362e004cd4d600dacc1401ff1f1a0160e5ea01f38bc801ae88a001d17a2a0077efa900daa7ef00851e640196792a017ec97b007e100b01260ab2000f2ef500ca5a9f00727c1700512cc501f7795800283ad601b5b64d00525b2d00dad1150123c8310061663b01a4bd2c003de00501f01c7b00869c45017fc47201d6540e00d208a100fa8090019f8d9701651b0a00003c8501c7310901ab1f0e017cdd6d01b93297000316d40035ebbc0139223000bada7601bb06af0150dce300d712d1014aceee018b7f6b01a9ae6200a13c8700cf7a600018f197005668080020300600f19605001a4475015d692300beb68c0019c637004a8d9b01aa65cd0198897b0070a17601315ed70130a83e00dae324008b7206011fdeb100231be7019c93a200169b2f013192e30148576a0045d5ba01cccb580170e0390064bffe01f5012301216a9a014fa000013436b30005c333017e61a90005865b01d8307101b335c301935e7600e49e62001fa17d01e38b05006704100158f86e00c261f2014c0f85008f548f00ed3350001adb6c018e9e30015dab2f000b4cec00d27a4a017447d00096ed3100228dec00f3cb91018841df0188ec3600a4637a00ea3cf4016d309401e2afe700c1f1af006bd5ae00d75da100fd542b0022e14500111ef10135860e00a664a001576b020164d56b0061c80b006cb53e0068b1040012c1bf017a23010024c9c6017b486001bf2040006ad23d00f2cd8300c2d5a50085c33800785a7a009de8fb006c8a3c00236048015a4a4b0121790b00a94013001322750185335100a0ff360170fc4c00d23c9200c8a3220104b87600fe83f3002689740052f0f10105093500ef19cf0161bdc901e6eecd0125f3ed00c806ab008bffcd00a0b1b1",
  },
  {
    // BeamHash III GEN sub-step ONLY -- a LIGHT, CPU-runnable regression guard for the SipHash-2-4 leaf
    // gen (make_leaf). The default is_test path (no MOM_BEAMHASH3_SOLVE) runs ONLY the gen kernel and
    // dumps the first BEAMHASH3_TEST_ROWS(=64) raw 448-bit leaf rows (7 LE u64 words = 56 B each, 3584 B)
    // into the SMALL_BLOB_SOL_LEN(=5120 B) out-of-band buffer, zero-padded to the full 5120 B so the hex
    // dump is deterministic. The full Wagner solve is too heavy for the CPU SYCL device; this gen-only
    // vector is cheap. Blob = the M4 keystone (prework||nonce||extranonce). Expected derived from
    // reference makeLeaf(prePow, e) for e in 0..63 (prePow = blake2bFull("Beam-PoW"..., blob)),
    // serialized 7 LE u64 + zero pad, cross-checked bit-exact against the GPU + SYCL-CPU gen dump.
    name: "beamhash3 gpu1 gen",
    gpu: true,
    timeoutMs: 5 * 60 * 1000,
    job: {
      algo: "beamhash3",
      dev: "gpu1",
      noncebytes: 8,
      nonceoffset: 32,
      target: "ff".repeat(32),
      blob_hex: "fc40996a518c221384c9f2542ca811cd66c4ccddb001ef40b9f9ba059c20352eb32c7d4f07a3001c511122a7",
    },
    // [64 gen rows: 56 bytes each] then zero pad to SMALL_BLOB_SOL_LEN(=5120 B).
    expected:
      "77f3ab2588544cf01546763eea8b954be9b0cb49a3b7f479bccb42ff13ce98dbc9f16d9ce8541e47bb4940a3f2a5123e0491d7eab01fc7c33c9ed8a5e900dbbb05fcb53901a46a0e568d2c79275306d8aa464e7fc0720a937607bd06648f1dd66513013b2584aea3c0cdd263dc26a9fecfb711104c7284d488d0372911108634a1386ca42fd92c3a7055b3a5ecdc1f937a709e68429cff675ead9a35b5003eaec917427717471bcaea1ec6a17292c3600fe10cf71f777203926be59fc7294d06dee9cd4a68ce6c3259ebca433c8117f29ed107bb8111b297b3330925090481d9e3deecd02b48b8562daf45e8f21c87abc3ddf134b35d92844123615234c6bdac33ab3941d48168241fe0eba00edd977ef7b9ba92c81db01786e257b8f023c0d29c2828a7ea05e8f87d733230dbf1e9a0b6b9e9fa7b219ef72b177c9a735e5276380e5d1675643efbefb36f4940e7fb227728d29413ef17c42a1591c2639316e60b032e8d29d275a4f1ff9fa79d3456051e7c35585b91aedace059242a4ff04c4747d81b621faa9d6eddabbcece3ef8d92ddee3ecb44b84aa720129c43cfc276c521920ad89e517b1e6189dcbc3b922a6017b88843c881d8df9d8a949ab697f1a007ad84f582574acc31b3c11cdc81e4c65bd74d7fff8d9154121dfd5ef8c5cd6367e0946dfe25daed603a3a3bae436e6cc72ea0a4a3b58b57b786ffd9ca12e0cc53d6114a098dfea723fb3de40c0780d3a3138cb247cc5247fc95a9692261762ba6c5d635fa4146ae0832501bee4a1868b4de601c0d9f124260712586d9cd10b257c952357b8b85fc462bab75b02185854b74f7bc932ec5903b6270f44ec658f5c3bb4635b33a024457d619e02d88d49c0c5ed3a74bc6ddb7fee64007e5a12249abde329fc488de35bfb38bc977300e0cb6c27a4817cb4e2652724e9fd78a2b90fd1b69f52d843be6839e2417bd59c260aadb23e2bb7a645a9dce75f8987aab1fed57ce289f4ec111d46aee993f9b18675be0ab30ef8b1a7d7aaebb67a50dfc26cf08afccb4f87d3da1fd9add879cd30be5e2eb395d61e9109c41194b09b663cbc5f5a3c4f7458186c6ac96ddd140763dea6466f3633715d3908969106c62d5d70cd67a474731649a55b93017372e07b1fad4cdf87f8164acd0fc76422c938b6df9832231472253f657227e73e31e22c3d22598895e39e75eeb9208062c807e5bb047d76d371da16b8d4dd03e387d66516c5d7950797719eb2f382ec186c17c28867c6d32126fc8afd3d7255398851c757b0ed744732f737a28ad87d32ded99b660832a137593de50634b99b2a2ff8ab5222bd29ec6f1126a7d582325d90c78f22c8c0ec9d872df241041d8969771fe37ee291f2545d7b32c5af09215d48e8334a3a24a8692fc009bf3510e3e3f392a31ba5f4f1778cf65131f768a35c8fceb5b615088a1e16f55239046340e0f368f22b0e7f2d95b43080497956dfb164a4c3974749e75eec58a012fa4f4d7df7731ba525d010c359cf923e2ce5aa195513ccf01cbcf3eea880ccfa49e320dfd1d00d73f99ca9f0f7a281528deada5264046c91d28e81260a321300445453602e939d3bd9ddfe80127a17d5757593c7db9e042439f103a07561ffcf9add8f4642d5009b8062e97b88995c1e451f2cfa75280b519abd0c760882c0f9f9d395718bc10f45d73ed073fdf42d689ceaf3e4be22a469dd114a7d72e94199f9cde78753dd016a11ca141703606429f4576bba1798c3097637ad2b9c143773accd1fefc2289494357d15034260950f23cef21962c5b8228ab1e5a5bfc98ecf5094a4748a128e04836dae1f2b04280f6d0c6f981abd4363cbce5cc48fa20e5dde90f7ea7e4c5ef86919a51dde471e72a836ae897384243bf58a4287600024b193641bc050699a35975c64810b1c1f484c1ff2bec8d7b3c04a09ff3efccf255d0f26abf8ff190182d742be1fb3736be489614abc38dc7a5427de9296f3ec5e65b67737f11849d6a686cef794399bc86746ca9f166068d69e4264889256b2bad538f08da4aacad8696cd89b90a7be68f100ee19b1bd448ed18d53593ea79a40b9f365f41398ae4e45d69a4ed555a925fb01fea342037c5cc1f57f804b981d362883acae8b0285ea09d0ef5860d05ce7c8c1ca26a28bfcde39366c55dd50931de84fb3453201d15e8eae40f6a24e80e8b6c725832443b8824b9253c24ec2170b8eea31a7f1982b3d7e11a5b6269bb2b1aaf9a7a73a4629990e23896b4f3a320ef5c4e1d4ecf71589bb6fc531114579607811890f28874a0439df072aeb7467077b68b2daa7e9060122eb758ac5b562cf7c5688957e4d11a9ce2815b8ff174c0254ea4e530e202362d745aef5ccd897380555dd6996dbeeab87bbff1ce60893dd034ae576e96064977d58d69fb07d0fba1c1b4c2fdbf1fe060f6513ab335874fe0ffb2e3f4aced5667f42b734be929bce92d78b52f1ec7ef14c8941d08693d831b89d45dd5f779eda854da3bb668b4f00b08876878aa5236a9a3063899ff7e1976630f7b816cc652e06b8cdddba83e6f3db907ce5e2bddfe7fa9450fd1f845153c6f14c45cc40157cffd656e54ce7b3cfe35f11f4d46f8521b01670ceafe5efc296590d53ce8e744945d2eadb9adea4c616e6192b6b1e13d05aef111f6699cd61c98011c39743542dd9188005a1bf4a5cfa2d9af0a2200b460b7ff7352031f26fe4f94447336581de0df66b2e1ddcaad7461eaa92f5c47598496cc10fb45e2ad70672f91c9086674b1e3bf8db7821a24968662573f4ec82963596f92d2e0c52e37e35fbc92e8e9b8daf1e908770eb77b35f4b11b65c45dcd4219145a932dbbcb853350152407cc50edb2c5c92b159728a356d4d9c47d314c2e1321450ddf549956b98f6e08144f8b5a41892c98c92f6ecd357efe85f48ef887d67a091a76abf45ee157898d8b90dba32474487d18581fa83ef884a7f2b8ac95da66d0eb936621425a6a002710012e3da18c7e997ec05634a26a0aeab9383a7914995b77446d62309528e63ee6587baceebc24db4e41b9e61e570bffae63b7bb00036bcfc62654172a99f12a7a656da2f07faa87314edff151f9daf27e5c06b12daaf57ea8746f5b642c6e6f09070298575584b1ba0418650daa7c5972326ad9f3fb06c88f86fc7e1b1cce350a489d02aa6e8870910d9a8eee37e2ea9d32016eb39f29a128a377a0ae0bc7ca78cee730d4d6df0b4cd80725e5402b12c284869b111180d0565c470466721c15555a6b013b40ba94ce6fe054fdfe109bda534bd9fe87708e819cbffe9f38807487691ee4b9fd0514230c40ca279c846253508ecdffb7d996c0e873fc196ee9949c12299cb2fccd6d782e13b964ab0bdf24c4f50b5f506fb411451a4f3d695873d84bc94fea701a09de63715ef5f3337f9de0f69759c5654753d52977a4ad596ee2015799067542b6b0ab6d003ab44bca4c531b8dc9ea071fbdc315fcb5852cb703778247f25406ffa76bbcfbe1aa7140daa39282ba7b231598002cfd911ba4df71ea7a669738b4ae55168f506e5e24322e4c08f2edd09a4b44314c904e39860e134a3e6618a34eb3dfd9e80b72ef529d5f46991b745d1f3ed6b5365fbd7051964b75293e6e761dbfca0f1769ec0c7bc6bae7e404d56d4a5c8b62435b631a6077b90bac1ccc76d2dc31afc6fd13663ed7d5657d60b3581aa08adbc6e977fa8ca225b0513de7f591185e0d77f27ca6273c6552e37d32eddb4bc7b463188833526851f51761064e1ab3accc19f811fc4329f274bda62fd0cdccba2d6348a994c38efaa5e4ea1a679afd2c24ac17e43682e3de9b6b8484968b88d76cff992efa718923972a59c7fbeb18a4c5d20f9a603938537728deb96180f25f23b91ef9f7f64a5ccf77fc8c33914c87b76d86c0e1bd4e131895c1e7701e666cd6fb8284d3bde6c656617bb996313fc6707b185f8dc2fdde1995218f5753840deb0b768ee62bece9af1278c5aaa188c3d556e54e1925576aba51e40e1a996658f1fd76b48e872d50a135229ea427658833688dd7c99e156cf7e122ebdf5083b69af823aea889fe8aa3c1c4385223c499ba6612c0330f77562f12d768693997c17e219daff6c540cc68e679a39e92df4d6e1ac777c8c876028f9fda932b245f865821f0224c80a430101bc30925581522658e977f8d1adc8fa17a987db549d99333768f89f31dd8f4561388e62bd312f838744344bb982258ea79d8d4002ced0522868c83502c078a9ca9aba032c6c369a50ab2e888ca84e87d7e931394f60967f9a6c885d097f7445502a506f8651e65b78b95afe39ce97ff6a6e9cdd121be831799a8bb108397040e082eeda5fdcb86505fd2c6e80573a81a3d138d0f91cb4bb587ac77f64ebad3818ed8bdcb76eba1dfe20ae494104bbb29640c3580269fde5eef3b008f6c9b4dff1003fd62f0343c0eb723e07da131f7edc89607b7f6a8ce4dd657ef3d19c4e5a175dab4ecf8cbf9a0d93975361ef0781219609cbf88b84c8d63420ee716b6d51bf167b6135bf67b98bb677943db3918292a4cde55f86acd2e0226a6cc35153fd84dfd00af4c540965fcf0aae0119b43bffb0faa1469cf7c2e0548237950bfb34e2abdbd50cddd1e920bb52a1a0fd1e5a2bd02aa9b262c9f739feae3666e6f2939e14e60c03f47d83db21e594cdecfc6e5ac0d59171db68e96615fa2a3dfc9539058bff4d9303b5db8568a1a1d6cf5512b063c122173d232b7eda693695d39c6b2b2076384995710ebd0e04fcf4fc0728b12c33ce7c28af30dff67c8303cee52f256aa302ae6d8d81879d0ada8242e604ab982d370ea69bc3bb5a6c1945a8ae365ac5bf7adba09da51afceb9304293105c7cdb79a902d5df0a16e3246e83a6ac920d49bd58346a04702e66c174ca5250027f5277ae6b5e5b0750971ddbbc41e7db69cced6babf61f83a7928bad564d6a1377affd5d92fb5ba0cae3028d50c6294aa57f06d6b52234518a7cf0544a365abab06bbe92fcb1972546eff983a4b7a92361ad76edbbe96c40227af36806b18a51bb5" +
      "0".repeat(10240 - 7168),
  },
  {
    // BeamHash III full Wagner solve over the M4 keystone. MOM_BEAMHASH3_SOLVE switches the test path
    // from gen-only to solve and dumps [count:u8][count * 104-byte solution][zero pad]. The 3 emitted
    // proofs are cross-checked against the BeamHash III verify3 flow; the first is the known keystone solution.
    name: "beamhash3 gpu1 (M4 keystone solve)",
    gpu: true,
    timeoutMs: 15 * 60 * 1000,
    env: { MOM_BEAMHASH3_SOLVE: "1" },
    job: {
      algo: "beamhash3",
      dev: "gpu1",
      noncebytes: 8,
      nonceoffset: 32,
      target: "ff".repeat(32),
      blob_hex: "fc40996a518c221384c9f2542ca811cd66c4ccddb001ef40b9f9ba059c20352eb32c7d4f07a3001c00000000",
    },
    expected:
      "03" +
      "0fc81c684be229c36b844ef8299a9744dbb8727276bff8cbd610fa7414fb6cfd67b92586f84f8bffaeeb99266994d79da3fb026a24128b84901f244b08ee6b6b954372fcb0a7d33318da6bf1854ae48f94fe8af2d3147bdc7302cc12daa1a306511122a700000000" +
      "929901b03918d93f53013be378a69eac442e64dda1c06d3deceef104940d4bd45df4990c2f5782158baeb9f4fe55b7915f9fa6f6146e6bb0671375fcd0149c3af5e16c8eec930bd011eaea15d16666bce61545cc52d75b8ce31c6a789c565efd5118e9b500000000" +
      "99d00156409438c98000163ced617c007c9aeba740c1524ca571ba20a274af37011cd20e6538e1170261cc902a7b168306cc01c224146d99d7f883e4b3bccc4e592320b569bffb9c6cee3fea1f27d09cb869264a49dd99256655657c263318966b87b0fe00000000" +
      "0".repeat(10240 - 2 - 208 * 3),
  },
];


const nonceAt32Algos = new Set(["kawpow", "firopow", "evrprogpow", "meowpow", "etchash", "autolykos2"]);
// Heights sampled from coin mainnets so perf DAG/table sizes match live pool jobs
// (ETC 2026-06-04, RVN and ERG 2026-06-12). Keep in sync with benchHeightByAlgo in mom.js.
const benchHeightByAlgo = {
  etchash:    24689903,
  kawpow:     4407982,
  firopow:    600000,
  evrprogpow: 1800000,
  meowpow:    825000,
  autolykos2: 1806198,
};

// Build a perf job from a hash vector's source job. Nonce-at-32 algos (see nonceAt32Algos above)
// carry a blob and need a live-sized DAG, so we keep the source job (clearing its dev for autoDev)
// and stamp the sampled height; all other algos only need the algo name.
function perfJob(sourceJob) {
  const algo = sourceJob.algo;
  if (!nonceAt32Algos.has(algo)) {return { algo };}

  const job = { ...sourceJob, dev: undefined };
  if (benchHeightByAlgo[algo]) {job.height = benchHeightByAlgo[algo];}
  return job;
}

// One perf entry per distinct algo, taken from its first hash vector.
const perfTests = [];
const seenAlgos = new Set();
for (const definition of hashTests) {
  const algo = definition.job.algo;
  if (seenAlgos.has(algo)) {continue;}
  seenAlgos.add(algo);

  perfTests.push({
    algo,
    gpu: definition.gpu,
    autoDev: true,
    name: algo,
    timeoutMs: definition.timeoutMs || 3 * 60 * 1000,
    job: perfJob(definition.job),
  });
}

module.exports = {
  hashTests,
  perfTests,
};
