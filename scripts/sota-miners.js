"use strict";

// Official release archives are downloaded to a cache, checksum-verified, and only executed in
// an offline disposable container. Pool-only miners remain metadata; any live-pool fallback is
// manual and explicitly containerized, never part of the default harness.
/** @typedef {{url: string, sha256: string}} Asset */
/** @typedef {{platform?: string, vendor?: string, gpu?: string, samples?: number[],
 * referenceRate?: number, unit?: string, status?: string, reason?: string, momRate?: number,
 * sotaRate?: number, minRatio?: number, reference?: string}} Evidence */
/** @typedef {{runner?: string, args: string[], rate?: string, unit?: string, runs?: number,
 * tailSamples?: number, longStats?: number, vendors?: string[]}} BenchmarkSpec */
/** @typedef {{command?: string[], network?: string, seconds?: number, vendors?: string[],
 * evidence?: Evidence | Evidence[], reason: string}} PoolSpec */
/** @typedef {{platform?: string, vendor?: string, gpu?: string, momRate: number, sotaRate?: number,
 * unit: string, minRatio: number, status?: string, reason?: string, reference?: string}} Qualification */
/** @typedef {{accepted: number[], evidence: string}} ExitPolicy */
/** @typedef {{name: string, url?: string, sha256?: string, strip?: number, exe?: string,
 * archive?: string, windowsAsset?: Asset, linuxEvidence?: Record<string, Evidence | Evidence[]>,
 * windowsEvidence?: Record<string, Evidence | Evidence[]>, benchmark?: Record<string, BenchmarkSpec | string>,
 * poolOnly?: Record<string, PoolSpec>, qualification?: Record<string, Qualification>,
 * exitPolicy?: Record<string, ExitPolicy>}} MinerSpec */

/** @satisfies {Record<string, MinerSpec>} */
module.exports = {
  lolminer: {
    name: "lolMiner 1.98a",
    url: "https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.98a/lolMiner_v1.98a_Lin64.tar.gz",
    sha256: "0b8078299654a12846e4967f1db3506409cfb8b1031687a910965d1a99c6f270",
    strip: 1,
    exe: "lolMiner",
    windowsAsset: {
      url: "https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.98a/lolMiner_v1.98a_Win64.zip",
      sha256: "f2bbda2d2255155d50935967b8c55105b9aeefbd27cda3e8d01beaf535a16762",
    },
    linuxEvidence: {
      equihash192_7: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        samples: [20.2, 20.2, 20.2], referenceRate: 20.2, unit: "I/s",
        reason: "median of three offline benchmark input-iteration rows; " +
          "the solution headline counts multiple outputs per input",
      },
      octopus: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        samples: [38.28, 38.27, 38.27], referenceRate: 38.27, unit: "MH/s",
        reason: "median of three steady 10-second real-pool rates; a 119.81G share was accepted",
      },
      zhash: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        samples: [83.2, 83.1, 83.1], referenceRate: 83.1, unit: "Sol/s",
        reason: "median of three isolated offline benchmark runs",
      },
      c29: {
        platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
        samples: [7.40, 7.40, 7.40], referenceRate: 7.40, unit: "g/s",
        reason: "three official lolMiner 1.98a offline 10-second runs at 150 W; " +
          "host nvidia-smi captured compute-active windows",
      },
    },
    windowsEvidence: {
      octopus: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [32.18, 32.21, 32.18], referenceRate: 32.18, unit: "MH/s",
        reason: "median of the final three warmed 10-second rates from a bounded real-pool run",
      },
      equihash192_7: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [20.3, 20.2, 20.2], referenceRate: 20.2, unit: "I/s",
        reason: "Use lolMiner's input-iteration column; its solution headline counts multiple outputs per input.",
      },
      nexapow: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [0, 0, 0], status: "unsupported",
        reason: "Every run failed to load the device kernels with error -11.",
      },
      zhash: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [83.0, 82.9, 83.0], referenceRate: 83.0, unit: "Sol/s",
      },
    },
    benchmark: {
      c30: "C30CTX",
      c29: {
        args: ["--benchmark", "CR29", "--no-oc-reset", "1", "--no-cl", "1"],
        rate: "average-speed-runs", unit: "g/s", runs: 3, vendors: ["nvidia"],
      },
      equihash192_7: {
        args: ["--benchmark", "EQUI192_7"], rate: "lolminer-input-iterations", unit: "I/s", runs: 3,
      },
      nexapow: {args: ["--benchmark", "NEXA"], rate: "gpu-speed", unit: "Mh/s", runs: 12},
      // The AMD offline benchmark reports roughly twice its accepted-share live-pool throughput.
      // Keep this reproducible offline path only for the NVIDIA comparator it matches.
      octopus: {args: ["--benchmark", "OCTOPUS"], rate: "average-speed-runs", unit: "Mh/s",
        runs: 3, longStats: 60, tailSamples: 3, vendors: ["nvidia"]},
      zhash: "EQUI144_5",
    },
    poolOnly: {
      octopus: {
        vendors: ["amd"], network: "bridge", seconds: 90,
        command: ["lolMiner", "--algo", "OCTOPUS", "--pool",
          "us.conflux.herominers.com:1170", "--user",
          "WALLET.WORKER", "--devices", "0"],
        evidence: [
          {
            platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
            samples: [38.28, 38.27, 38.27], referenceRate: 38.27, unit: "MH/s",
            momRate: 29.18, sotaRate: 38.27, minRatio: 0.8, reference: "lolMiner 1.98a",
            reason: "A 119.81G share was accepted during the controlled direct-pool run.",
          },
          {
            platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
            samples: [32.18, 32.21, 32.18], referenceRate: 32.18, unit: "MH/s",
            momRate: 26.49, sotaRate: 32.18, minRatio: 0.8, reference: "lolMiner 1.98a",
            reason: "The bounded 175-second run authorized and received current jobs without errors.",
          },
        ],
        reason: "The AMD offline benchmark overstates accepted-share throughput by roughly 2x; " +
          "use direct pool work in a constrained container.",
      },
    },
    qualification: {
      nexapow: {
        gpu: "RTX 5060 Ti", vendor: "nvidia", momRate: 76.10, sotaRate: 94.59,
        unit: "MH/s", minRatio: 0.8, status: "qualified",
        reason: "five-sample median is 80.5% of lolMiner's last-six sustained GPU-speed median",
      },
      octopus: {
        gpu: "RTX 5060 Ti", vendor: "nvidia", momRate: 45.62, sotaRate: 51.65,
        unit: "MH/s", minRatio: 0.8, status: "qualified",
        reason: "five-sample median is 88.3% of lolMiner 1.98a's local real-pool median",
      },
    },
    exitPolicy: {
      nexapow: {accepted: [15], evidence: "positive Average speed line"},
    },
  },
  bzminer10036: {
    name: "BZMiner 100.36",
    url: "https://github.com/bzminer/bzminer/releases/download/v100.36/" +
      "bzminer_v100.36_linux.tar.gz",
    sha256: "8a665ddd7138a87b94cabba851cc6897e86c65bfcd8f0639ebcb99c1b7c3f889",
    windowsAsset: {
      url: "https://github.com/bzminer/bzminer/releases/download/v100.36/" +
        "bzminer_v100.36_windows.zip",
      sha256: "85b07dafea1e61a76711fcbd5ca06a4d3671698963283bd829a67f54060fdd49",
    },
    strip: 1,
    exe: "bzminer",
    linuxEvidence: {
      c29: [
        {
          platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [7.64, 7.68, 7.68, 7.64, 7.68], referenceRate: 7.68, unit: "g/s",
          reason: "median of the final five rates from an isolated 100-second offline benchmark",
        },
        {
          platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
          samples: [6.67, 6.64, 6.64, 6.61, 6.61], referenceRate: 6.64, unit: "g/s",
          reason: "median of the final five rates from an isolated offline benchmark",
        },
      ],
      autolykos2: [
        {
          platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [119.13, 119.29, 119.21, 119.28, 119.28],
          referenceRate: 119.28, unit: "MH/s",
          reason: "median of the final five rates from an isolated 95-second offline benchmark",
        },
        {
          platform: "linux", vendor: "intel", gpu: "Intel B580",
          samples: [40.44, 40.51, 40.51], referenceRate: 40.51, unit: "MH/s",
          reason: "median of the final three warmed rates from an isolated offline benchmark",
        },
      ],
      pearlhash: [
        {
          platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [92.03, 92.05, 91.46, 91.46, 90.85],
          referenceRate: 91.46, unit: "TH/s",
          reason: "median of the final five rates at M=131072, N=524288, K=8192, rank=128",
        },
        {
          platform: "linux", vendor: "intel", gpu: "Intel B580",
          samples: [37.41, 37.32, 37.39, 37.39, 37.39],
          referenceRate: 37.39, unit: "TH/s",
          reason: "median of the final five rates at M=N=131072, K=2048, rank=128",
        },
        {
          platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
          samples: [35.87, 35.84, 35.79, 35.70, 35.70],
          referenceRate: 35.79, unit: "TH/s",
          status: "candidate",
          reason: "median of the final five rates at M=N=131072, K=2048, rank=128; " +
            "does not replace the qualified faster KRig reference",
        },
      ],
    },
    windowsEvidence: {
      autolykos2: [
        {
          platform: "windows", vendor: "intel", gpu: "Intel B580",
          samples: [37.84, 37.99, 37.94], referenceRate: 37.94, unit: "MH/s",
          reason: "median of three independent 110-second offline benchmarks",
        },
        {
          platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [119.34, 119.25, 119.20], referenceRate: 119.25, unit: "MH/s",
          reason: "median of three independent 110-second offline benchmarks at 150 W",
        },
        {
          platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
          samples: [36.57, 36.59, 36.52], referenceRate: 36.57, unit: "MH/s",
          reason: "median of three independent 110-second offline benchmarks at 145 W",
        },
      ],
      "cn/gpu": {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [2.63, 2.63, 2.63], referenceRate: 2.63, unit: "KH/s",
        reason: "median of three independent 60-second offline benchmarks at 145 W",
      },
      c29: {
        platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
        samples: [6.21, 6.14, 6.24], referenceRate: 6.21, unit: "g/s",
        reason: "median of three independent 80-second offline benchmarks at 150 W",
      },
      etchash: {
        platform: "windows", vendor: "intel", gpu: "Intel B580",
        samples: [13.81, 13.86, 13.85], referenceRate: 13.85, unit: "MH/s",
        reason: "median of three independent 100-second offline benchmarks",
      },
      pearlhash: [
        {
          platform: "windows", vendor: "intel", gpu: "Intel B580",
          samples: [37.59, 37.60, 37.79], referenceRate: 37.60, unit: "TH/s",
          reason: "median of three runs at M=N=131072, K=2048, rank=128",
        },
        {
          platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [90.93, 91.24, 91.19], referenceRate: 91.19, unit: "TH/s",
          reason: "median of three runs at M=131072, N=524288, K=8192, rank=128 at 150 W",
        },
        {
          platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
          samples: [35.27, 35.33, 35.06], referenceRate: 35.27, unit: "TH/s",
          status: "candidate",
          reason: "median of three runs at M=N=131072, K=2048, rank=128 at 145 W; " +
            "does not replace the qualified faster KRig Linux reference",
        },
      ],
      xelishashv3: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [2.34, 2.37, 2.37], referenceRate: 2.37, unit: "KH/s",
        reason: "median of three independent 70-second offline benchmarks at 145 W",
      },
    },
  },
  krig: {
    name: "KRig 1.5.2",
    linuxEvidence: {
      pearlhash: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        referenceRate: 46.29, unit: "TH/s",
        reason: "verified final-five median at M=8192, N=32768, K=2048, rank=128; " +
          "six accepted shares, no rejects, and 1170 seconds of warm samples at 145 W",
      },
    },
  },
  wildrig: {
    name: "WildRig Multi 0.50.3",
    url: "https://github.com/andru-kun/wildrig-multi/releases/download/0.50.3/" +
      "wildrig-multi-linux-0.50.3.tar.gz",
    sha256: "88186d34019600a82297237932bfaa2a5b4d88a3a818d21fd79e4a67c8b73ac3",
    windowsAsset: {
      url: "https://github.com/andru-kun/wildrig-multi/releases/download/0.50.3/" +
        "wildrig-multi-windows-0.50.3.zip",
      sha256: "a2634e061e4e20b4826903a28a53bc5add8d7b1e6e1427a8ff8537d9f9231e27",
    },
    windowsEvidence: {
      nexapow: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [31.29, 31.22, 31.18], referenceRate: 31.22, unit: "MH/s",
        reason: "median of three final 60-second rates from 90-second offline benchmarks",
      },
    },
    strip: 0,
    exe: "wildrig-multi",
    benchmark: {
      nexapow: {
        runner: "wildrig",
        args: ["--algo", "nexapow", "--benchmark"],
        rate: "wildrig-60s",
        unit: "MH/s",
        runs: 3,
      },
    },
    qualification: {
      nexapow: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        momRate: 27.34, sotaRate: 32.41, unit: "MH/s", minRatio: 0.8, status: "qualified",
        reason: "three-run median of the final 60-second averages from 90-second offline benchmarks",
      },
    },
    exitPolicy: {
      nexapow: {accepted: [1], evidence: "Benchmark finished plus a positive final 60-second rate"},
    },
  },
  cortexpoolminer: {
    name: "Cortex PoolMiner 1.0.0",
    url: "https://github.com/CortexFoundation/PoolMiner/releases/download/" +
      "cortex_miner_1.0.0/cortex_miner.tar.gz",
    sha256: "4ec195a9060956c490038fec578bc6fb61fd22034dc8ea8b9f9565c5bc61d4d4",
    strip: 0,
    exe: "cortex_miner/cortex_miner",
    linuxEvidence: {
      c30: {
        platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
        samples: [0], status: "unsupported",
        reason: "The unchanged release crashed in CuckooInitialize before producing a rate.",
      },
    },
    poolOnly: {
      c30: {
        command: ["cortex_miner", "-pool_uri=POOL:PORT", "-worker=WORKER", "-devices=0",
          "-account=ADDRESS"],
        reason: "The official miner has no offline benchmark; its documented pool is now unavailable.",
      },
    },
  },
  gminer: {
    name: "GMiner 3.44",
    windowsAsset: {
      url: "https://github.com/develsoftware/GMinerRelease/releases/download/3.44/" +
        "gminer_3_44_windows64.zip",
      sha256: "dfa02b289f555d7055342967b1d1c390f5c93108c2b73beee1853ce0281bbe04",
    },
    windowsEvidence: {
      c30: {
        platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
        samples: [0], status: "unsupported",
        reason: "The unchanged release identified the GPU, then reported no compatible kernel image.",
      },
    },
  },
  miniz: {
    name: "miniZ 2.5e3",
    url: "https://github.com/miniZ-miner/miniZ/releases/download/v2.5e3/miniZ_v2.5e3_linux-x64.tar.gz",
    sha256: "873f011442df76b9aae541f3c1a385677f5ea75d94a41da08647a7e51f92ae30",
    strip: 0,
    exe: "miniZ",
    poolOnly: {
      equihash192_7: {
        command: ["miniZ", "--url=stratum+tcp://WALLET.WORKER@POOL:PORT", "--par=192,7",
          "--pers=ZcashPoW"],
        evidence: {platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti", momRate: 26.85,
          sotaRate: 30.92, unit: "I/s", minRatio: 0.8, reference: "miniZ 2.5e3"},
        reason: "miniZ requires Stratum; both qualified rates count input headers per second.",
      },
      zhash: {
        command: ["miniZ", "--url=stratum+tcp://WALLET.WORKER@POOL:PORT", "--par=144,5"],
        evidence: {platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti", momRate: 87.63,
          sotaRate: 107.68, unit: "Sol/s", minRatio: 0.8, reference: "miniZ 2.5e3"},
        reason: "Both rates include every valid solution; MoM found exactly 216 solutions per 100 inputs.",
      },
    },
  },
  srbminer: {
    name: "SRBMiner-Multi 2.9.3",
    url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/2.9.3/SRBMiner-Multi-2-9-3-Linux.tar.gz",
    sha256: "2ad54cbf943bfb6d7d09ec52eef086813963020b1d48484f18a1b5dd1969e54b",
    windowsAsset: {
      url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/2.9.3/SRBMiner-Multi-2-9-3-win64.zip",
      sha256: "0b62e1e6447c456ca04e8bf4ccf3cea23c3077a92185d15bada06afdc9797398",
    },
    strip: 1,
    exe: "SRBMiner-MULTI",
    linuxEvidence: {
      walahash: [
        {
          platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [1034.62, 1030.15, 1014.56],
          referenceRate: 1030.15, unit: "MH/s",
          reason: "median of three final pool-rate samples",
        },
        {
          platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
          samples: [784.37, 782.93, 769.28],
          referenceRate: 782.93, unit: "MH/s",
          reason: "median of three final pool-rate samples",
        },
      ],
    },
    windowsEvidence: {
      walahash: [
        {
          platform: "windows", vendor: "nvidia", gpu: "RTX 5060 Ti",
          samples: [974.89, 1025.77, 1027.06],
          referenceRate: 1025.77, unit: "MH/s",
          reason: "median of three final pool-rate samples",
        },
        {
          platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
          samples: [782.3517920891048, 782.4291863086698, 780.9717604646505],
          referenceRate: 782.3517920891048, unit: "MH/s",
          reason: "median of three final pool-rate samples",
        },
      ],
    },
    poolOnly: {
      walahash: {
        command: ["SRBMiner-MULTI", "--algorithm", "walahash", "--pool", "POOL:PORT",
          "--wallet", "WALLET.WORKER", "--password", "x"],
        evidence: [
          {platform: "linux", vendor: "intel", gpu: "Intel B580", momRate: 623.38, sotaRate: 362.76,
            unit: "MH/s", minRatio: 0.8, reference: "SRBMiner-Multi 2.9.3"},
          {platform: "windows", vendor: "intel", gpu: "Intel B580", momRate: 532.93, sotaRate: 404.39,
            unit: "MH/s", minRatio: 0.8, reference: "SRBMiner-Multi 2.9.3"},
          {platform: "linux", vendor: "amd", gpu: "RX 9060 XT", momRate: 679.50, sotaRate: 782.93,
            unit: "MH/s", minRatio: 0.8, reference: "SRBMiner-Multi 2.9.3"},
          {platform: "windows", vendor: "amd", gpu: "RX 9060 XT", momRate: 667.68,
            sotaRate: 782.3517920891048, unit: "MH/s", minRatio: 0.8,
            reference: "SRBMiner-Multi 2.9.3"},
        ],
        reason: "SRBMiner-Multi has no offline benchmark; qualification uses pinned 2.9.3 " +
          "loopback-Stratum baselines because later releases removed WalaHash.",
      },
    },
  },
  srbminer356: {
    name: "SRBMiner-Multi 3.5.6",
    url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.5.6/SRBMiner-Multi-3-5-6-Linux.tar.gz",
    sha256: "9e28c43dc16e687a97e666643bd453f6f94432d132d3d8d21388ffda835f63fc",
    windowsAsset: {
      url: "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.5.6/SRBMiner-Multi-3-5-6-win64.zip",
      sha256: "de2c450019b7e10bd7a824e42d1ff28e3e87d559adca3c1f010ce9066470c6c1",
    },
    strip: 1,
    exe: "SRBMiner-MULTI",
    linuxEvidence: {
      xelishashv3: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        samples: [2.63524, 2.63534, 2.63487], referenceRate: 2.63524, unit: "KH/s",
        reason: "median of three final pool-rate samples",
      },
      verthash: {
        platform: "linux", vendor: "intel", gpu: "Intel B580",
        samples: [316.98933, 316.6901, 316.62288], referenceRate: 316.6901, unit: "KH/s",
        reason: "median of three final pool-rate samples",
      },
    },
    windowsEvidence: {
      xelishashv3: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [2.27814, 2.26737, 2.26749], referenceRate: 2.26749, unit: "KH/s",
        reason: "median of three final pool-rate samples",
      },
      verthash: {
        platform: "windows", vendor: "intel", gpu: "Intel B580",
        samples: [317.35542, 317.24746, 317.17253], referenceRate: 317.24746, unit: "KH/s",
        reason: "median of three final pool-rate samples",
      },
    },
    poolOnly: {
      xelishashv3: {
        command: ["SRBMiner-MULTI", "--algorithm", "xelishashv3", "--pool",
          "de.xelis.herominers.com:1225", "--wallet", "WALLET.WORKER", "--password", "x"],
        reason: "SRBMiner-Multi requires Stratum; retain the command as metadata only.",
      },
      verthash: {
        command: ["SRBMiner-MULTI", "--algorithm", "verthash", "--pool",
          "vertcoin.cedric-crispin.com:3334", "--wallet", "WALLET.WORKER", "--password", "x",
          "--verthash-dat-path", "/data/verthash.dat"],
        reason: "SRBMiner-Multi requires Stratum and the external verthash.dat dataset; retain the command as metadata only.",
      },
    },
  },
  rigel: {
    name: "Rigel 1.23.2",
    url: "https://github.com/rigelminer/rigel/releases/download/1.23.2/rigel-1.23.2-linux.tar.gz",
    sha256: "eae492ffb64aeb4ab4ba7e66631567984a31d5adb1ef547bda6601aee1793f0d",
    strip: 1,
    exe: "rigel",
    poolOnly: {
      xelishashv3: {
        command: ["rigel", "--algorithm", "xelishashv3", "--url", "stratum+tcp://POOL:PORT",
          "--username", "WALLET.WORKER", "--password", "x"],
        evidence: {platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti", momRate: 5.88, sotaRate: 6.95,
          unit: "KH/s", minRatio: 0.8, reference: "Rigel 1.23.2"},
        reason: "Rigel's published CLI documents pool mining, not an offline benchmark.",
      },
    },
  },
  teamblackminer: {
    name: "TeamBlackMiner 2.44",
    url: "https://github.com/sp-hash/TeamBlackMiner/releases/download/v2.44/TeamBlackMiner_2_44_Ubuntu_22_04_Cuda_12_9.tar.xz",
    sha256: "284bef135566135d67439a3b5bc9abfd95e14c62169a3937ebf558434a474d11",
    strip: 1,
    exe: "TBMiner",
    archive: "tar.xz",
    poolOnly: {
      verthash: {
        command: ["TBMiner", "--algo", "verthash", "--hostname", "POOL", "--port", "PORT",
          "--wallet", "WALLET", "--worker-name", "WORKER", "--verthash-data", "/data/verthash.dat"],
        evidence: [
          {platform: "linux", vendor: "nvidia", gpu: "RTX 5060 Ti", momRate: 908.06,
            sotaRate: 925.88, unit: "KH/s", minRatio: 0.8, reference: "TeamBlackMiner 2.44"},
          {platform: "linux", vendor: "amd", gpu: "RX 9060 XT", momRate: 293.48,
            sotaRate: 291.85, unit: "KH/s", minRatio: 0.8, reference: "TeamBlackMiner 2.44"},
        ],
        reason: "TeamBlackMiner documents pool mining only; Verthash also needs the external verthash.dat dataset.",
      },
    },
  },
  hoominer: {
    name: "hoo_gpu 1.4.23",
    url: "https://htn.foztor.net/hoo_gpu-1.4.23.tar.gz",
    sha256: "1be57fe82cd6bcc1e5c48e93f3f86c9b08f7ff3964951d361b477761dbdcec6f",
    strip: 1,
    exe: "hoo_gpu",
    poolOnly: {
      hoohash: {
        command: ["hoo_gpu", "-o", "stratum+tcp://mainnet-node-1.hoosat.fi:5555",
          "-u", "hoosat:README_DONATION_ADDRESS", "-p", "x", "--same-stratum",
          "--no-share-hashrate", "--gpu-id", "0"],
        network: "bridge",
        seconds: 120,
        evidence: {gpu: "RTX 5060 Ti", momRate: 1440, sotaRate: 1740, unit: "KH/s", minRatio: 0.8},
        reason: "hoo_gpu 1.4.23 requires Stratum; use only as a short, containerized fallback.",
      },
    },
  },
  hoosathoominer: {
    name: "Hoominer 0.4.1",
    windowsAsset: {
      url: "https://github.com/HoosatNetwork/hoominer/releases/download/0.4.1/" +
        "hoominer-0.4.1-windows.zip",
      sha256: "0db7fb5e59a8b115460a11e8f4143bdca5c89fce748432fd6a713afa1933fc3d",
    },
    windowsEvidence: {
      hoohash: {
        platform: "windows", vendor: "amd", gpu: "RX 9060 XT",
        samples: [0, 0], status: "unsupported",
        reason: "Both bundled- and system-OpenCL-loader runs crashed in the packaged " +
          "pthreadVC3.dll before GPU enumeration.",
      },
    },
  },
  hoogpuamd: {
    name: "hoo_gpu_amd 1.4.22",
    url: "https://htn.foztor.net/hoo_gpu_amd-1.4.22.tar.gz",
    sha256: "d1e72920b06ff4577460bb2c9022ee40c8aa8d889c6851f163adc94ed834ff7f",
    strip: 1,
    exe: "hoo_gpu_amd",
    linuxEvidence: {
      hoohash: {
        platform: "linux", vendor: "amd", gpu: "RX 9060 XT",
        samples: [1.49, 1.49, 1.49], referenceRate: 1.49, unit: "MH/s",
        reason: "median of three per-run final-five 30-second pool-rate medians",
      },
    },
    poolOnly: {
      hoohash: {
        command: ["hoo_gpu_amd", "-o", "stratum+tcp://mainnet-node-1.hoosat.fi:5555",
          "-u", "hoosat:README_DONATION_ADDRESS", "-p", "x", "--same-stratum",
          "--no-share-hashrate", "--gpu-id", "0"],
        network: "bridge",
        seconds: 240,
        evidence: {
          platform: "linux", vendor: "amd", gpu: "RX 9060 XT", momRate: 1.71,
          sotaRate: 1.49, unit: "MH/s", minRatio: 0.8, reference: "hoo_gpu_amd 1.4.22",
        },
        reason: "The AMD reference requires live HooHash Stratum work and was replicated with accepted shares.",
      },
    },
  },
};
