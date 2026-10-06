"use strict";

const {pearlhashTarget} = require("./helper/hash");

const progpowAlgos = ["kawpow", "firopow", "evrprogpow", "meowpow"];
const fishhashAlgos = ["fishhash", "karlsenhashv2"];
const fixedIntensityAlgos = new Set(["c30", "equihash192_7", "zhash"]);
/** @type {Set<keyof GpuTuning>} */
const progpowFields = new Set(["intensity", "workgroup", "dag_workgroup", "dag_chunk"]);
/** @type {Set<keyof GpuTuning>} */
const fishhashFields = new Set(["intensity", "workgroup", "search_mode"]);
/** @type {Array<[string, Set<keyof GpuTuning>]>} */
const algoEntries = [
  ["cn/gpu", new Set(["intensity"])],
  ["c29", new Set(["seed_workgroup", "seed_blocks"])],
  ["c30", new Set(["intensity"])],
  ["etchash", new Set(["intensity", "dag_workgroup", "dag_chunk"])],
  ["octopus", new Set(["intensity"])],
  ["autolykos2", new Set([
    "intensity", "workgroup", "prehash_workgroup", "table_chunk", "search_mode",
  ])],
  ["hoohash", new Set(["intensity", "workgroup"])],
  ["walahash", new Set(["intensity"])],
  ["xelishashv3", new Set(["intensity"])],
  ["nexapow", new Set(["intensity"])],
  ["equihash192_7", new Set(["intensity"])],
  ["zhash", new Set(["intensity"])],
  ["verthash", new Set(["intensity"])],
  ["pearlhash", new Set(["m", "n", "k", "rank", "cache_block", "tile"])],
  ["zelhash", new Set(["slots"])],
  ["beamhash3", new Set(["workgroup", "compact_workgroup", "scatter_workgroup", "layout"])],
];
const algoFields = new Map(algoEntries);
for (const algo of progpowAlgos) {algoFields.set(algo, progpowFields);}
for (const algo of fishhashAlgos) {algoFields.set(algo, fishhashFields);}
/** @type {Set<keyof GpuTuning>} */
const allAlgoFields = new Set([...algoFields.values()].flatMap((fields) => [...fields]));
/** @type {Set<string>} */
const tuningFieldNames = new Set(allAlgoFields);

/** @type {Set<keyof GpuTuning>} */
const integerFields = new Set([
  "intensity", "workgroup", "seed_workgroup", "seed_blocks",
  "dag_workgroup", "dag_chunk", "prehash_workgroup", "table_chunk",
  "m", "n", "k", "rank", "slots", "cache_block",
  "compact_workgroup", "scatter_workgroup",
]);
const zeroAllowedFields = new Set(["dag_chunk", "table_chunk", "cache_block"]);
/** @type {Partial<Record<keyof GpuTuning, Set<string>>>} */
const enumFields = {
  search_mode: new Set(["auto", "scalar", "cooperative"]),
  layout: new Set(["auto", "compact", "full"]),
  tile: new Set(["auto", "1x1", "2x2", "2x4", "4x2", "4x4", "8x2"]),
};
const INT32_MAX = 0x7fffffff;
const PEARLHASH_MAX_DIMENSION = 1 << 24;
const PEARLHASH_MAX_K = 1 << 16;
const PEARLHASH_MAX_RANK = 1024;
const UINT64_MAX = (1n << 64n) - 1n;

/** @param {string} field @returns {field is keyof GpuTuning} */
function isTuningField(field) {
  return tuningFieldNames.has(field);
}

/** @type {Map<string, keyof GpuTuning>} */
const mainFieldByAlgo = new Map([
  ["c29", "seed_workgroup"],
  ["pearlhash", "m"],
  ["zelhash", "slots"],
  ["beamhash3", "workgroup"],
]);
const progpowWorkgroups = [64, 128, 256, 512];
const fishhashWorkgroups = [64, 128, 256, 512];
const workgroupsByAlgo = new Map([
  ["autolykos2", [32, 64, 128, 256]],
  ["hoohash", [64, 128, 256]],
  ["pearlhash", [32, 64, 128, 256]],
]);
for (const algo of progpowAlgos) {workgroupsByAlgo.set(algo, progpowWorkgroups);}
for (const algo of fishhashAlgos) {workgroupsByAlgo.set(algo, fishhashWorkgroups);}
const tuningFieldOrder = [
  "intensity", "seed_workgroup", "m", "slots", "workgroup",
  "seed_blocks", "dag_workgroup", "dag_chunk", "prehash_workgroup",
  "table_chunk", "search_mode", "n", "k", "rank", "cache_block", "tile",
  "compact_workgroup", "scatter_workgroup", "layout",
];
const tuningFieldPosition = new Map(tuningFieldOrder.map((field, index) => [field, index]));

/** @param {string} algo */
function allowedFields(algo) {
  if (algo) {return algoFields.get(algo) || new Set();}
  return allAlgoFields;
}

/**
 * @param {keyof GpuTuning} field
 * @param {unknown} value
 * @param {string} label
 */
function validateInteger(field, value, label) {
  if (typeof value !== "number" && (typeof value !== "string" || !/^\d+$/.test(value))) {
    throw new Error(`${label} must be a base-10 integer`);
  }
  const number = Number(value);
  const minimum = zeroAllowedFields.has(field) ? 0 : 1;
  if (!Number.isSafeInteger(number) || number < minimum || number > 0xffffffff) {
    throw new Error(`${label} must be an integer between ${minimum} and 4294967295`);
  }
  if (field === "seed_workgroup" && ![64, 128, 256].includes(number)) {
    throw new Error(`${label} must be 64, 128, or 256`);
  }
  if (field === "seed_blocks" && ![4, 8, 16, 32].includes(number)) {
    throw new Error(`${label} must be 4, 8, 16, or 32`);
  }
  if (field === "slots" && number % 16 !== 0) {
    throw new Error(`${label} must be a multiple of 16`);
  }
  return number;
}

/**
 * Validate and normalize a complete PearlHash matrix shape.
 * @param {unknown} m
 * @param {unknown} n
 * @param {unknown} k
 * @param {unknown} rank
 * @param {string} [context]
 * @returns {{m: number, n: number, k: number, rank: number}}
 */
function validatePearlHashShape(m, n, k, rank, context = "PearlHash shape") {
  /** @param {unknown} value @param {string} field */
  function normalize(value, field) {
    if (typeof value !== "number" &&
        (typeof value !== "string" || !/^\d+$/.test(value))) {
      throw new Error(`${context}.${field} must be a decimal integer`);
    }
    const number = typeof value === "number" ? value : Number(value);
    if (!Number.isSafeInteger(number) || number < 1 || number > INT32_MAX) {
      throw new Error(`${context}.${field} must be a positive integer at most ${INT32_MAX}`);
    }
    return number;
  }

  const shape = {
    m: normalize(m, "m"), n: normalize(n, "n"),
    k: normalize(k, "k"), rank: normalize(rank, "rank"),
  };
  /** @type {("m" | "n")[]} */
  const matrixFields = ["m", "n"];
  for (const field of matrixFields) {
    const value = shape[field];
    if (value < 128 || value > PEARLHASH_MAX_DIMENSION || value % 32 !== 0) {
      throw new Error(`${context}.${field} must be a multiple of 32 between 128 and ` +
        PEARLHASH_MAX_DIMENSION);
    }
  }
  if (shape.k < 1024 || shape.k > PEARLHASH_MAX_K || shape.k % 64 !== 0) {
    throw new Error(`${context}.k must be a multiple of 64 between 1024 and ` +
      PEARLHASH_MAX_K);
  }
  if (shape.rank < 128 || shape.rank > PEARLHASH_MAX_RANK ||
      (shape.rank & (shape.rank - 1)) !== 0) {
    throw new Error(`${context}.rank must be a power of two between 128 and ` +
      PEARLHASH_MAX_RANK);
  }
  if (shape.k < 16 * shape.rank || shape.k > 4 * shape.rank * shape.rank) {
    throw new Error(`${context}.k must be between 16*rank and 4*rank^2`);
  }
  if (shape.m * shape.rank > INT32_MAX || shape.n * shape.rank > INT32_MAX ||
      shape.m * shape.k > INT32_MAX) {
    throw new Error(`${context} exceeds the supported signed-index range`);
  }
  if ((shape.m / 16) * (shape.n / 16) > INT32_MAX) {
    throw new Error(`${context} exceeds the supported tile range`);
  }
  if (BigInt(shape.m) * BigInt(shape.n) * BigInt(shape.k) > UINT64_MAX) {
    throw new Error(`${context} exceeds the supported matrix-product range`);
  }
  return shape;
}

/**
 * Validate an untrusted tuning object and return its normalized form.
 * @param {string} algo
 * @param {unknown} tuning
 * @param {string} [context]
 * @returns {GpuTuning}
 */
function validateTuning(algo, tuning, context = "tuning") {
  if (!tuning || typeof tuning !== "object" || Array.isArray(tuning)) {
    throw new Error(`${context} must be an object`);
  }
  const allowed = allowedFields(algo);
  /** @type {GpuTuning} */
  const result = {};
  for (const [field, value] of Object.entries(tuning)) {
    if (!isTuningField(field) || !allowed.has(field)) {
      throw new Error(`${context}.${field} is not supported for ${algo}`);
    }
    if (integerFields.has(field)) {
      Object.assign(result, {[field]: validateInteger(field, value, `${context}.${field}`)});
    } else {
      const choices = enumFields[field];
      if (!choices) {throw new Error(`${context}.${field} has no validator`);}
      if (typeof value !== "string") {
        throw new Error(`${context}.${field} must be one of ${[...choices].join(", ")}`);
      }
      const normalized = value.toLowerCase();
      if (!choices.has(normalized)) {
        throw new Error(`${context}.${field} must be one of ${[...choices].join(", ")}`);
      }
      Object.assign(result, {[field]: normalized});
    }
  }
  if (fixedIntensityAlgos.has(algo) && result.intensity !== undefined && result.intensity !== 1) {
    throw new Error(`${context}.intensity must be 1 for ${algo}`);
  }
  /** @type {Array<[keyof GpuTuning, number[] | undefined]>} */
  const workgroupChoices = [
    ["workgroup", workgroupsByAlgo.get(algo)],
    ["dag_workgroup", [32, 64, 128, 256, 512]],
    ["prehash_workgroup", [32, 64, 128, 256]],
  ];
  for (const [field, choices] of workgroupChoices) {
    const value = result[field];
    if (choices && typeof value === "number" && !choices.includes(value)) {
      throw new Error(`${context}.${field} must be one of ${choices.join(", ")}`);
    }
  }
  /** @type {(keyof GpuTuning)[]} */
  const beamWorkgroupFields = ["workgroup", "compact_workgroup", "scatter_workgroup"];
  for (const field of beamWorkgroupFields) {
    const value = result[field];
    if (algo !== "beamhash3" || typeof value !== "number") {continue;}
    const maximum = field === "compact_workgroup" ? 512 : 1024;
    if (value < 16 || value > maximum || value % 16 !== 0) {
      throw new Error(`${context}.${field} must be a multiple of 16 between 16 and ${maximum}`);
    }
  }
  if (algo === "pearlhash") {
    /** @type {("m" | "n")[]} */
    const matrixFields = ["m", "n"];
    for (const field of matrixFields) {
      const value = result[field];
      if (value !== undefined && value > PEARLHASH_MAX_DIMENSION) {
        throw new Error(`${context}.${field} must be at most ${PEARLHASH_MAX_DIMENSION}`);
      }
      if (value !== undefined && (value < 128 || value % 32 !== 0)) {
        throw new Error(`${context}.${field} must be at least 128 and a multiple of 32`);
      }
    }
    if (result.k !== undefined && result.k > PEARLHASH_MAX_K) {
      throw new Error(`${context}.k must be at most ${PEARLHASH_MAX_K}`);
    }
    if (result.k !== undefined && (result.k < 1024 || result.k % 64 !== 0)) {
      throw new Error(`${context}.k must be at least 1024 and a multiple of 64`);
    }
    if (result.rank !== undefined &&
        (result.rank < 128 || result.rank > PEARLHASH_MAX_RANK ||
         (result.rank & (result.rank - 1)) !== 0)) {
      throw new Error(`${context}.rank must be a power of two between 128 and ` +
        PEARLHASH_MAX_RANK);
    }
    if (result.k !== undefined && result.rank !== undefined &&
        (result.k < 16 * result.rank || result.k > 4 * result.rank * result.rank)) {
      throw new Error(`${context}.k must be between 16*rank and 4*rank^2`);
    }
  }
  return result;
}

/**
 * @param {string} algo
 * @param {string | undefined} text
 * @param {string} context
 */
function parseExpandedTuning(algo, text, context) {
  if (text === undefined) {return {};}
  if (!text) {throw new Error(`${context} tuning list must not be empty`);}
  /** @type {Record<string, string>} */
  const raw = {};
  for (const part of text.split(";")) {
    const separator = part.indexOf("=");
    if (separator <= 0 || separator === part.length - 1) {
      throw new Error(`${context} tuning values must use name=value`);
    }
    const key = part.slice(0, separator).trim().toLowerCase();
    if (!/^[a-z][a-z0-9_]*$/.test(key) || key in raw) {
      throw new Error(`${context} has an invalid or duplicate tuning key: ${key}`);
    }
    raw[key] = part.slice(separator + 1).trim();
  }
  return validateTuning(algo, raw, context);
}

/**
 * @param {unknown} entry
 * @param {string} [algo]
 * @returns {DeviceEntry}
 */
function parseDeviceEntry(entry, algo = "") {
  if (typeof entry !== "string") {throw new Error("device entry must be a string");}
  const text = entry.trim();
  const match = text.match(
    /^(cpu\d*|gpu\d+)(?:\*([1-9]\d*)|\*\[([^\]]*)\])?(?:\^([1-9]\d*))?$/i
  );
  if (!match) {throw new Error(`invalid device entry: ${entry}`);}
  const matchedDevice = match[1];
  if (!matchedDevice) {throw new Error(`invalid device entry: ${entry}`);}
  const device = matchedDevice.toLowerCase();
  const deviceIndex = Number((device.match(/\d+$/) || [0])[0]);
  if (!Number.isSafeInteger(deviceIndex) || deviceIndex > 1023) {
    throw new Error(`${device} index must be at most 1023`);
  }
  const tuning = parseExpandedTuning(algo, match[3], text);
  if (match[2]) {
    const mainValue = Number.parseInt(match[2], 10);
    const field = device.startsWith("cpu") ? "intensity" : (mainFieldByAlgo.get(algo) || "intensity");
    if (fixedIntensityAlgos.has(algo) && field === "intensity" && mainValue !== 1) {
      throw new Error(`${text}.intensity must be 1 for ${algo}`);
    }
    if (tuning[field] && tuning[field] !== mainValue) {
      throw new Error(`${text} specifies conflicting ${field} values`);
    }
    Object.assign(tuning, {[field]: validateInteger(field, mainValue, `${text}.${field}`)});
  }
  const processes = match[4] ? Number(match[4]) : 1;
  if (!Number.isSafeInteger(processes) || processes > 1024) {
    throw new Error(`${text} process count must be at most 1024`);
  }
  return {
    device,
    tuning,
    processes,
  };
}

/**
 * @param {unknown} dev
 * @param {string} [algo]
 * @returns {DeviceEntry[]}
 */
function parseDeviceList(dev, algo = "") {
  if (typeof dev !== "string" || !dev.trim()) {throw new Error("device list must be a string");}
  return dev.split(",").map((entry) => parseDeviceEntry(entry, algo));
}

/** @param {string} algo @returns {keyof GpuTuning} */
function primaryTuningField(algo) {
  return mainFieldByAlgo.get(algo) || "intensity";
}

/** @param {string} dev @param {string} algo */
function needsPrimaryTuning(dev, algo) {
  const field = primaryTuningField(algo);
  return parseDeviceList(dev, algo).some(
    (entry) => entry.device.startsWith("gpu") && entry.tuning[field] === undefined
  );
}

/** @param {DeviceEntry} entry */
function formatDeviceEntry(entry) {
  const tuningValues = entry.tuning || {};
  const isCpu = entry.device.startsWith("cpu");
  const cpuIntensity = isCpu ? tuningValues.intensity : undefined;
  const fields = Object.entries(tuningValues).filter(
    ([key]) => !(isCpu && key === "intensity")
  ).sort(([left], [right]) =>
    (tuningFieldPosition.get(left) ?? tuningFieldOrder.length) -
    (tuningFieldPosition.get(right) ?? tuningFieldOrder.length));
  const tuning = fields.length
    ? `*[${fields.map(([key, value]) => `${key}=${value}`).join(";")}]`
    : "";
  const processes = entry.processes > 1 ? `^${entry.processes}` : "";
  return `${entry.device}${cpuIntensity ? `*${cpuIntensity}` : ""}${tuning}${processes}`;
}

/** @param {DeviceEntry[]} entries */
function formatDeviceList(entries) {
  return entries.map((entry) => formatDeviceEntry(entry)).join(",");
}

/** @param {DeviceEntry} entry */
function nativeJobDevice(entry) {
  if (!entry.device.startsWith("cpu")) {return entry.device;}
  const intensity = entry.tuning && entry.tuning.intensity;
  return `${entry.device}${intensity ? `*${intensity}` : ""}`;
}

/** @param {DeviceEntry} entry @param {string} [algo] */
function nativeJobIntensity(entry, algo = "") {
  if (!entry.device.startsWith("gpu")) {return 0;}
  const value = (entry.tuning || {})[algo === "pearlhash" ? "m" : "intensity"];
  if (value === undefined) {return 1;}
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("GPU tuning intensity must be a finite number");
  }
  return value;
}

/**
 * @param {MiningJob} job
 * @param {DeviceEntry} entry
 * @param {string} [algo]
 */
function applyNativeJobTuning(job, entry, algo = "") {
  const inheritedPearlM = algo === "pearlhash" ? job.intensity : undefined;
  const pearlM = entry.tuning.m ?? inheritedPearlM;
  job.dev = nativeJobDevice(entry);
  job.intensity = algo === "pearlhash" && pearlM !== undefined
    ? pearlM : nativeJobIntensity(entry, algo);
  if (algo !== "pearlhash") {return job;}
  const tuning = entry.tuning || {};
  // Pearl workers are independent processes, so each may use its own matrix
  // shape. When only m is specified, retain the established square-matrix
  // behavior by using it as n as well.
  if (tuning.m !== undefined && tuning.n === undefined) {job.pearlhash_n = tuning.m;}
  if (tuning.n !== undefined) {job.pearlhash_n = tuning.n;}
  if (tuning.k !== undefined) {job.pearlhash_k = tuning.k;}
  if (tuning.rank !== undefined) {job.pearlhash_rank = tuning.rank;}
  const hasShape = Boolean(job.pearlhash_base_target) ||
    tuning.m !== undefined || tuning.n !== undefined ||
    tuning.k !== undefined || tuning.rank !== undefined ||
    job.pearlhash_n !== undefined || job.pearlhash_k !== undefined ||
    job.pearlhash_rank !== undefined;
  if (hasShape) {
    if (job.pearlhash_n === undefined || job.pearlhash_k === undefined ||
        job.pearlhash_rank === undefined) {
      throw new Error("PearlHash job shape is incomplete");
    }
    const shape = validatePearlHashShape(
      job.intensity, job.pearlhash_n, job.pearlhash_k, job.pearlhash_rank, "PearlHash job");
    job.intensity = shape.m;
    job.pearlhash_n = shape.n;
    job.pearlhash_k = shape.k;
    job.pearlhash_rank = shape.rank;
  }
  if (job.pearlhash_base_target) {
    if (job.pearlhash_k === undefined || job.pearlhash_rank === undefined) {
      throw new Error("PearlHash job shape is incomplete");
    }
    // Retuning a worker's matrix must preserve the job's certificate scaling.
    job.target = pearlhashTarget(
      job.pearlhash_base_target, job.pearlhash_k, job.pearlhash_rank,
      job.pearlhash_cert_version);
  }
  return job;
}

const progpowEnv = {
  workgroup: "MOM_KAWPOW_WORKGROUP",
  dag_workgroup: "MOM_KAWPOW_DAG_WORKGROUP",
  dag_chunk: "MOM_KAWPOW_DAG_CHUNK_NODES",
};
const fishhashEnv = {workgroup: "MOM_FISHHASH_WORKGROUP"};
/** @type {Record<string, Partial<Record<keyof GpuTuning, string | string[]>>>} */
const envByAlgo = {
  "c29": {
    seed_workgroup: "MOM_C29_SEED_LOCAL_SIZE",
    seed_blocks: "MOM_C29_SEED_BLOCKS",
  },
  ...Object.fromEntries(progpowAlgos.map((algo) => [algo, progpowEnv])),
  "etchash": {
    dag_workgroup: "MOM_ETCHASH_DAG_WORKGROUP",
    dag_chunk: "MOM_ETCHASH_DAG_CHUNK_NODES",
  },
  "autolykos2": {
    workgroup: "MOM_AUTOLYKOS2_WORKGROUP",
    prehash_workgroup: "MOM_AUTOLYKOS2_PREHASH_WORKGROUP",
    table_chunk: "MOM_AUTOLYKOS2_TABLE_CHUNK",
  },
  ...Object.fromEntries(fishhashAlgos.map((algo) => [algo, fishhashEnv])),
  "hoohash": {intensity: "MOM_HOOHASH_INTENSITY", workgroup: "MOM_HOOHASH_WORKGROUP"},
  "walahash": {intensity: "MOM_WALAHASH_INTENSITY"},
  "xelishashv3": {intensity: "MOM_XELISHASHV3_INTENSITY"},
  "nexapow": {intensity: "MOM_NEXAPOW_INTENSITY"},
  "pearlhash": {
    cache_block: [
      "MOM_PEARLHASH_AMD_DP4A_CACHE_BLOCK",
      "MOM_PEARLHASH_CU_BLK",
    ],
  },
  "zelhash": {slots: "MOM_ZELHASH_SLOTS"},
  "beamhash3": {
    workgroup: ["MOM_BEAMHASH3_WORKGROUP", "MOM_BEAMHASH3_COMPACT_WG"],
    compact_workgroup: "MOM_BEAMHASH3_COMPACT_WG",
    scatter_workgroup: "MOM_BEAMHASH3_SCATTER_WG",
  },
};

/** @param {string} algo @param {unknown} tuning @returns {NodeJS.ProcessEnv} */
function tuningEnvironment(algo, tuning) {
  const normalized = validateTuning(algo, tuning, `${algo} tuning`);
  /** @type {NodeJS.ProcessEnv} */
  const env = {};
  for (const [field, envNames] of Object.entries(envByAlgo[algo] || {})) {
    if (!isTuningField(field)) {throw new Error(`Invalid ${algo} tuning environment field: ${field}`);}
    if (normalized[field] === undefined) {continue;}
    for (const envName of Array.isArray(envNames) ? envNames : [envNames]) {
      env[envName] = String(normalized[field]);
    }
  }
  if (algo === "autolykos2" && normalized.search_mode !== undefined) {
    if (normalized.search_mode !== "auto") {
      env["MOM_AUTOLYKOS2_SUBGROUP_COOP"] =
        normalized.search_mode === "cooperative" ? "1" : "0";
    }
  }
  if (fishhashAlgos.includes(algo) &&
      normalized.search_mode !== undefined && normalized.search_mode !== "auto") {
    env["MOM_FISHHASH_COOP"] = normalized.search_mode === "cooperative" ? "1" : "0";
  }
  if (algo === "beamhash3" && normalized.layout !== undefined && normalized.layout !== "auto") {
    env["MOM_BEAMHASH3_COMPACT"] = normalized.layout === "compact" ? "1" : "0";
  }
  if (algo === "pearlhash" && normalized.tile !== undefined && normalized.tile !== "auto") {
    env["MOM_PEARLHASH_AMD_DP4A_TILE"] = normalized.tile;
  }
  return env;
}

/** @param {DeviceEntry} entry @param {Partial<GpuTuning>} changes @returns {DeviceEntry} */
function tuningCandidate(entry, changes) {
  return {
    device: entry.device,
    processes: entry.processes,
    tuning: {...(entry.tuning || {}), ...changes},
  };
}

/** @param {number} value @param {number} numerator @param {number} denominator @param {number} [alignment] */
function alignedScale(value, numerator, denominator, alignment = 256) {
  const scaled = Math.floor(value * numerator / denominator / alignment) * alignment;
  const maximum = Math.floor(0xffffffff / alignment) * alignment;
  return Math.min(maximum, Math.max(alignment, scaled));
}

// Return a deliberately bounded empirical-search set around the device heuristic. Variants change
// one launch dimension at a time: this keeps the optional first-run tuner useful without turning it
// into a combinatorial multi-hour search for every algorithm. Dataset-construction-only controls
// remain on their stability-oriented heuristics because steady-state hashrate cannot rank them.
/** @param {string} algo @param {DeviceEntry} entry @returns {DeviceEntry[]} */
function autotuneCandidates(algo, entry) {
  if (!entry.device.startsWith("gpu")) {return [entry];}
  const base = entry.tuning || {};
  const candidates = [tuningCandidate(entry, {})];
  /** @param {keyof GpuTuning} field @param {(number | string)[]} values */
  const add = (field, values) => {
    for (const value of values) {
      if (base[field] === value) {continue;}
      /** @type {Partial<GpuTuning>} */
      const changes = {};
      Object.assign(changes, {[field]: value});
      candidates.push(tuningCandidate(entry, changes));
    }
  };
  const intensity = Number(base.intensity || 0);
  if (intensity && !fixedIntensityAlgos.has(algo)) {
    if (algo === "cn/gpu") {
      add("intensity", [
        alignedScale(intensity, 1, 2, 8),
        alignedScale(intensity, 3, 4, 8),
      ]);
    } else {
      add("intensity", [
        alignedScale(intensity, 1, 2),
        alignedScale(intensity, 3, 4),
        alignedScale(intensity, 5, 4),
      ]);
    }
  }
  if (algo === "c29") {
    add("seed_workgroup", [64, 128, 256]);
    add("seed_blocks", [8, 16, 32]);
  } else if (progpowAlgos.includes(algo)) {
    add("workgroup", [64, 128, 256]);
  } else if (algo === "autolykos2") {
    add("workgroup", [32, 64, 128, 256]);
  } else if (fishhashAlgos.includes(algo)) {
    add("workgroup", [64, 128, 256]);
    add("search_mode", ["scalar", "cooperative"]);
  } else if (algo === "hoohash") {
    add("workgroup", [64, 128, 256]);
  } else if (algo === "pearlhash" && base.m) {
    add("m", [
      alignedScale(base.m, 1, 4, 64),
      alignedScale(base.m, 1, 2, 64),
    ]);
  } else if (algo === "beamhash3" && base.workgroup) {
    const maximum = base.workgroup;
    add("workgroup", [384, 512, 640, 768, 1024].filter((value) => value <= maximum));
    add("scatter_workgroup", [64, 128, 256]);
  }
  const unique = new Map();
  for (const candidate of candidates) {
    candidate.tuning = validateTuning(algo, candidate.tuning, `${algo} auto-tune candidate`);
    unique.set(formatDeviceEntry(candidate), candidate);
  }
  return [...unique.values()];
}

module.exports = {
  applyNativeJobTuning,
  autotuneCandidates,
  formatDeviceEntry,
  formatDeviceList,
  needsPrimaryTuning,
  parseDeviceEntry,
  parseDeviceList,
  primaryTuningField,
  tuningEnvironment,
  validatePearlHashShape,
  validateTuning,
};
