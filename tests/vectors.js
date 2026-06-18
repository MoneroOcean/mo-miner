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
];

const nonceAt32Algos = new Set(["kawpow", "firopow", "evrprogpow", "etchash", "autolykos2"]);
// Heights sampled from coin mainnets so perf DAG/table sizes match live pool jobs
// (ETC 2026-06-04, RVN and ERG 2026-06-12). Keep in sync with benchHeightByAlgo in mom.js.
const benchHeightByAlgo = {
  etchash:    24689903,
  kawpow:     4407982,
  firopow:    600000,
  evrprogpow: 1800000,
  autolykos2: 1806198,
};

// Build a perf job from a hash vector's source job. Nonce-at-32 algos (kawpow/etchash/autolykos2)
// carry a blob and need a live-sized DAG, so we keep the source job (clearing its dev for autoDev)
// and stamp the sampled height; all other algos only need the algo name.
function perfJob(sourceJob) {
  const algo = sourceJob.algo;
  if (!nonceAt32Algos.has(algo)) return { algo };

  const job = { ...sourceJob, dev: undefined };
  if (benchHeightByAlgo[algo]) job.height = benchHeightByAlgo[algo];
  return job;
}

// One perf entry per distinct algo, taken from its first hash vector.
const perfTests = [];
const seenAlgos = new Set();
for (const definition of hashTests) {
  const algo = definition.job.algo;
  if (seenAlgos.has(algo)) continue;
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
