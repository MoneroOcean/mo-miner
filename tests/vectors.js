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
];

const perfTests = [];
const perfAlgos = new Set();
const nonceAt32Algos = new Set(["kawpow", "etchash", "autolykos2"]);
const currentEtcBenchHeight = 24689903; // Sampled from ETC mainnet on 2026-06-04 for live-size Etchash DAG benchmarks.

function perfJob(definition) {
  const algo = definition.job.algo;
  if (!nonceAt32Algos.has(algo)) return { algo };

  const job = { ...definition.job, dev: undefined };
  if (algo === "etchash") job.height = currentEtcBenchHeight;
  return job;
}

for (const definition of hashTests) {
  const algo = definition.job.algo;
  if (perfAlgos.has(algo)) continue;

  perfAlgos.add(algo);
  perfTests.push({
    algo,
    gpu: definition.gpu,
    autoDev: true,
    name: algo,
    timeoutMs: definition.timeoutMs || 3 * 60 * 1000,
    job: perfJob(definition),
  });
}

module.exports = {
  hashTests,
  perfTests,
};
