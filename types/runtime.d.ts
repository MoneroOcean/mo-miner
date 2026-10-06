export {};

declare global {
  type UnknownRecord = Record<string, unknown>;

  interface GpuTuning {
    cache_block?: number;
    compact_workgroup?: number;
    dag_chunk?: number;
    dag_workgroup?: number;
    intensity?: number;
    k?: number;
    layout?: "auto" | "compact" | "full";
    m?: number;
    n?: number;
    prehash_workgroup?: number;
    rank?: number;
    scatter_workgroup?: number;
    search_mode?: "auto" | "cooperative" | "scalar";
    seed_blocks?: number;
    seed_workgroup?: number;
    slots?: number;
    table_chunk?: number;
    tile?: "auto" | "1x1" | "2x2" | "2x4" | "4x2" | "4x4" | "8x2";
    workgroup?: number;
  }

  interface DeviceEntry {
    device: string;
    processes: number;
    tuning: GpuTuning;
  }

  interface AlgoParam {
    backend: string;
    dev: string;
    perf: number | null;
    tuning: GpuTuning;
  }

  interface MsrSetting {
    mask: string;
    value: string;
  }

  interface PoolSocket extends NodeJS.EventEmitter {
    destroyed?: boolean;
    destroy(): unknown;
    write(data: string): unknown;
  }

  interface PoolMessage {
    algo?: unknown;
    body?: unknown;
    code?: unknown;
    description?: unknown;
    difficulty?: unknown;
    error?: unknown;
    id?: unknown;
    input?: unknown;
    method?: unknown;
    nonceprefix?: unknown;
    params?: unknown;
    result?: unknown;
    __kaspa_timestamp?: unknown;
    __kaspa_words?: unknown;
  }

  type ArrayPoolMessage = PoolMessage & {params: unknown[]};
  type BodyPoolMessage = PoolMessage & {body: UnknownRecord};
  type ObjectParamsPoolMessage = PoolMessage & {params: UnknownRecord};
  type ResultArrayPoolMessage = PoolMessage & {result: unknown[]};

  interface MiningJob {
    algo: string;
    backend?: string;
    backend_request?: string;
    blob?: string;
    blob_hex?: string;
    dev: string;
    difficulty?: number;
    extra_nonce?: string;
    extra_nonce2_size?: number;
    extranonce2?: string;
    header_hash?: string;
    id?: string;
    height?: number;
    intensity?: number;
    job_token?: string;
    job_id?: string | number;
    nicehash_mask?: string;
    nonce?: number | string;
    nonce_slot?: number | string;
    nonce_stride?: number | string;
    nonce1_len?: number;
    noncebytes?: number;
    nonceoffset?: number;
    ntime?: string;
    pearlhash_base_target?: string;
    pearlhash_cert_version?: 3;
    pearlhash_k?: number;
    pearlhash_n?: number;
    pearlhash_rank?: number;
    pre_pow?: string;
    pool_id?: number | string;
    proofsize?: number;
    seed_hex?: string;
    seed_hash?: string;
    solution?: string;
    submit_mode?: string;
    target?: string;
    thread_id?: number;
    thread_num?: number;
    worker_id?: string | number;
    xn?: string;
  }

  type PoolJob = Omit<Partial<MiningJob>, "algo" | "submit_mode"> & {
    algo?: string | null;
    nbits?: string;
    submit_mode?: string | null;
    submit_result?: boolean;
    protocol?: string;
  };

  interface PoolConfig {
    algo_params?: Record<string, AlgoParam>;
    bad_shares: number;
    beam_difficulty?: number;
    beam_nonceprefix?: string;
    cortex_nonce?: string;
    donation_until?: number;
    eth_difficulty?: number;
    eth_target?: string;
    extra_nonce?: string;
    extra_nonce2_size?: number;
    good_shares: number;
    extensions?: string[];
    requested_extensions?: string[];
    requested_algos?: string[];
    job_algo?: string;
    pending_job?: PoolMessage;
    pending_controls?: PoolMessage[];
    pending_cortex_login?: boolean;
    pending_cortex_submit_ids?: Set<number>;
    pending_cortex_work?: boolean;
    inferred_protocol?: string;
    ironfish_target?: string;
    ironfish_xn?: string;
    is_keepalive: boolean;
    is_nicehash: boolean;
    is_tls: boolean;
    kaspa_difficulty?: number;
    kaspa_target?: string;
    keepalive: NodeJS.Timeout | null;
    last_connect_time: number;
    last_job: PoolJob | null;
    logged_in: boolean;
    login: string;
    nexa_difficulty?: number;
    nexa_target?: string;
    negotiated_keepalive?: boolean;
    negotiated_nicehash?: boolean;
    pass: string;
    pearlhash_difficulty?: number;
    pearlhash_proof_encodings?: Array<"none" | "gzip">;
    pearlhash_target_format: "base" | "jackpot";
    pending_subscribe: boolean;
    pending_authorize: boolean;
    pending_submit_count: number;
    port: number;
    protocol: string | null;
    raven_target?: string;
    socket: PoolSocket | null;
    stratum_target?: string;
    tls_verify: boolean;
    url: string;
    use_subscribe: boolean;
    verthash_difficulty?: number;
    worker: string;
    worker_id?: string | number;
    xelis_difficulty?: number | string;
    xelis_extra_nonce?: string;
    xelis_public_key?: string;
    zelhash_target?: string;
  }

  interface JobOptions {
    algo: string | null;
    backend: string;
    backend_request?: string;
    blob_hex: string;
    dev: string;
    dev_request?: string;
    height: number;
    pearlhash_cert_version?: 3;
    nicehash_mask?: string;
    nonce?: number | string;
    noncebytes?: number;
    nonceoffset?: number;
    proofsize?: number;
    seed_hex: string;
    target?: string;
  }

  interface PoolTimes {
    close_wait: number;
    connect_throttle: number;
    donate_interval: number;
    donate_length: number;
    first_job_wait: number;
    keepalive: number;
    primary_reconnect: number;
    stats: number;
  }

  interface MinerOptions {
    algo_params: Record<string, AlgoParam>;
    bench_algo_params: number;
    default_msrs: Record<string, MsrSetting>;
    gpu_tune: number;
    job: JobOptions;
    log_level: number;
    pool_ids: {active: number; donate: number | null; primary: number | null};
    pool_time: PoolTimes;
    pools: PoolConfig[];
    save_config: string;
  }

  interface ComputeCore {
    emit_to(type: string, value?: object): void;
    from: import("node:events").EventEmitter;
  }

  interface NativeCoreWorker {
    sendToCpp(name: string, payload: Record<string, string>): void;
  }

  interface NativeCoreModule {
    AsyncWorker: new(
      message: (name: string, value: unknown) => void,
      close: () => void,
      error: (error: Error) => void,
    ) => object;
    exitNow?(code: number): void;
  }

  type NativeJob = MiningJob & {blob_hex: string};
  type LiveNativeJob = NativeJob & {
    job_id: string | number;
    pool_id: string | number;
    target: string;
    worker_id: string | number;
    job_token: string;
  };
  type WorkerJobCommand =
    | {job: LiveNativeJob; type: "job"}
    | {job: NativeJob; type: "bench" | "test"};
  type WorkerCommand = WorkerJobCommand | {type: "close" | "pause"};

  interface WorkerEvent {
    thread_id: number;
    type: string;
    value: UnknownRecord;
  }

  type WorkerMessageHandler = (message: WorkerEvent) => void;

  interface MinerTestState {
    result: string;
    result_hash_hex: string | null;
    thread_tested: number;
  }

  interface ShareResult {
    job_id: string | number;
    nonce: string;
  }

  interface WorkerResult {
    commitment?: string;
    edges?: string;
    hash?: string;
    header_hash?: string;
    jackpot?: string;
    job_token: string;
    job_id: string | number;
    mix_hash?: string;
    nonce: string;
    plain_proof?: string;
    adjustment_factor?: string;
    pool_id: string;
    solution?: string;
    worker_id: string;
  }

  interface SubmitParams {
    commitment?: string;
    header_hash?: string;
    id?: string | number;
    job_id: string | number;
    mixhash?: string;
    nonce: string | number;
    pow?: number[];
    result?: string;
  }

  var opt: MinerOptions;
}
