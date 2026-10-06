export {};

declare global {
  type TestEnvironment = Record<string, string | undefined>;

  type HashJob = Omit<Partial<MiningJob>, "algo"> & {algo: string};

  interface HashDefinition {
    autoDev?: boolean;
    benchSamples?: number;
    env?: TestEnvironment;
    expected?: string | string[];
    gpu?: boolean;
    job: HashJob;
    name: string;
    perfTimeoutMs?: number;
    portableOnly?: boolean;
    syclCpu?: boolean;
    timeoutMs?: number;
  }

  interface HashVectorDefinition extends HashDefinition {
    expected: string | string[];
    // Only the bounded CPU dataset differs; GPU/iGPU clones retain the consensus vector.
    syclCpuExpected?: string | string[];
  }

  interface PerfDefinition extends HashDefinition {
    algo: string;
    autoDev: true;
    timeoutMs: number;
  }
}
