"use strict";

const algoFields = new Map([
  ["cn/gpu", new Set(["intensity"])],
  ["c29", new Set(["seed_workgroup", "seed_blocks"])],
  ["kawpow", new Set(["intensity", "workgroup", "dag_workgroup", "dag_chunk"])],
  ["firopow", new Set(["intensity", "workgroup", "dag_workgroup", "dag_chunk"])],
  ["evrprogpow", new Set(["intensity", "workgroup", "dag_workgroup", "dag_chunk"])],
  ["meowpow", new Set(["intensity", "workgroup", "dag_workgroup", "dag_chunk"])],
  ["etchash", new Set(["intensity", "dag_workgroup", "dag_chunk"])],
  ["autolykos2", new Set([
    "intensity", "workgroup", "prehash_workgroup", "table_chunk", "search_mode",
  ])],
  ["fishhash", new Set(["intensity", "workgroup", "search_mode"])],
  ["karlsenhashv2", new Set(["intensity", "workgroup", "search_mode"])],
  ["pearlhash", new Set(["m", "n", "k", "rank", "workgroup", "cache_block", "tile"])],
  ["zelhash", new Set(["slots"])],
  ["beamhash3", new Set(["workgroup", "compact_workgroup", "scatter_workgroup", "layout"])],
]);

const integerFields = new Set([
  "intensity", "workgroup", "seed_workgroup", "seed_blocks",
  "dag_workgroup", "dag_chunk", "prehash_workgroup", "table_chunk",
  "m", "n", "k", "rank", "slots", "cache_block",
  "compact_workgroup", "scatter_workgroup",
]);
const zeroAllowedFields = new Set(["dag_chunk", "table_chunk", "cache_block"]);
const enumFields = {
  search_mode: new Set(["auto", "scalar", "cooperative"]),
  layout: new Set(["auto", "compact", "full"]),
  tile: new Set(["auto", "1x1", "2x2", "2x4", "4x2", "4x4", "8x2"]),
};
const mainFieldByAlgo = new Map([
  ["c29", "seed_workgroup"],
  ["pearlhash", "m"],
  ["zelhash", "slots"],
  ["beamhash3", "workgroup"],
]);
const workgroupsByAlgo = new Map([
  ["kawpow", [64, 128, 256, 512]],
  ["firopow", [64, 128, 256, 512]],
  ["evrprogpow", [64, 128, 256, 512]],
  ["meowpow", [64, 128, 256, 512]],
  ["autolykos2", [32, 64, 128, 256]],
  ["fishhash", [64, 128, 256, 512]],
  ["karlsenhashv2", [64, 128, 256, 512]],
  ["pearlhash", [32, 64, 128, 256]],
]);
const tuningFieldOrder = [
  "intensity", "seed_workgroup", "m", "slots", "workgroup",
  "seed_blocks", "dag_workgroup", "dag_chunk", "prehash_workgroup",
  "table_chunk", "search_mode", "n", "k", "rank", "cache_block", "tile",
  "compact_workgroup", "scatter_workgroup", "layout",
];
const tuningFieldPosition = new Map(tuningFieldOrder.map((field, index) => [field, index]));

function allowedFields(algo) {
  if (algo) {return algoFields.get(algo) || new Set();}
  return new Set([...algoFields.values()].flatMap((fields) => [...fields]));
}

function validateInteger(field, value, label) {
  if (typeof value === "string" && !/^\d+$/.test(value)) {
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

function validateTuning(algo, tuning, context = "tuning") {
  if (!tuning || typeof tuning !== "object" || Array.isArray(tuning)) {
    throw new Error(`${context} must be an object`);
  }
  const allowed = allowedFields(algo);
  const result = {};
  for (const [field, value] of Object.entries(tuning)) {
    if (!allowed.has(field)) {throw new Error(`${context}.${field} is not supported for ${algo}`);}
    if (integerFields.has(field)) {
      result[field] = validateInteger(field, value, `${context}.${field}`);
    } else if (enumFields[field]) {
      const normalized = String(value).toLowerCase();
      if (!enumFields[field].has(normalized)) {
        throw new Error(`${context}.${field} must be one of ${[...enumFields[field]].join(", ")}`);
      }
      result[field] = normalized;
    }
  }
  const workgroupChoices = {
    workgroup: workgroupsByAlgo.get(algo),
    dag_workgroup: [32, 64, 128, 256, 512],
    prehash_workgroup: [32, 64, 128, 256],
  };
  for (const [field, choices] of Object.entries(workgroupChoices)) {
    if (choices && result[field] !== undefined && !choices.includes(result[field])) {
      throw new Error(`${context}.${field} must be one of ${choices.join(", ")}`);
    }
  }
  for (const field of ["workgroup", "compact_workgroup", "scatter_workgroup"]) {
    if (algo !== "beamhash3" || result[field] === undefined) {continue;}
    const maximum = field === "compact_workgroup" ? 512 : 1024;
    if (result[field] < 16 || result[field] > maximum || result[field] % 16 !== 0) {
      throw new Error(`${context}.${field} must be a multiple of 16 between 16 and ${maximum}`);
    }
  }
  return result;
}

function parseExpandedTuning(algo, text, context) {
  if (text === undefined) {return {};}
  if (!text) {throw new Error(`${context} tuning list must not be empty`);}
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

function parseDeviceEntry(entry, algo = "") {
  const text = String(entry).trim();
  const match = text.match(
    /^(cpu\d*|gpu\d+)(?:\*([1-9]\d*)|\*\[([^\]]*)\])?(?:\^([1-9]\d*))?$/i
  );
  if (!match) {throw new Error(`invalid device entry: ${entry}`);}
  const device = match[1].toLowerCase();
  const tuning = parseExpandedTuning(algo, match[3], text);
  if (match[2]) {
    const mainValue = Number.parseInt(match[2], 10);
    const field = device.startsWith("cpu") ? "intensity" : (mainFieldByAlgo.get(algo) || "intensity");
    if (tuning[field] && tuning[field] !== mainValue) {
      throw new Error(`${text} specifies conflicting ${field} values`);
    }
    tuning[field] = validateInteger(field, mainValue, `${text}.${field}`);
  }
  return {
    device,
    tuning,
    processes: match[4] ? Number.parseInt(match[4], 10) : 1,
  };
}

function parseDeviceList(dev, algo = "") {
  if (typeof dev !== "string" || !dev.trim()) {throw new Error("device list must be a string");}
  return dev.split(",").map((entry) => parseDeviceEntry(entry, algo));
}

function primaryTuningField(algo) {
  return mainFieldByAlgo.get(algo) || "intensity";
}

function needsPrimaryTuning(dev, algo) {
  const field = primaryTuningField(algo);
  return parseDeviceList(dev, algo).some(
    (entry) => entry.device.startsWith("gpu") && entry.tuning[field] === undefined
  );
}

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

function formatDeviceList(entries) {
  return entries.map((entry) => formatDeviceEntry(entry)).join(",");
}

function nativeJobDevice(entry) {
  if (!entry.device.startsWith("cpu")) {return entry.device;}
  const intensity = entry.tuning && entry.tuning.intensity;
  return `${entry.device}${intensity ? `*${intensity}` : ""}`;
}

function nativeJobIntensity(entry, algo = "") {
  if (!entry.device.startsWith("gpu")) {return 0;}
  return Number((entry.tuning || {})[algo === "pearlhash" ? "m" : "intensity"] || 1);
}

function applyNativeJobTuning(job, entry, algo = "") {
  job.dev = nativeJobDevice(entry);
  job.intensity = nativeJobIntensity(entry, algo);
  if (algo !== "pearlhash") {return job;}
  const tuning = entry.tuning || {};
  // Pearl workers are independent processes, so each may use its own matrix
  // shape. When only m is specified, retain the established square-matrix
  // behavior by using it as n as well.
  if (tuning.m !== undefined && tuning.n === undefined) {job.pearlhash_n = tuning.m;}
  for (const field of ["n", "k", "rank"]) {
    if (tuning[field] !== undefined) {job[`pearlhash_${field}`] = tuning[field];}
  }
  return job;
}

const envByAlgo = {
  "c29": {
    seed_workgroup: "MOM_C29_SEED_LOCAL_SIZE",
    seed_blocks: "MOM_C29_SEED_BLOCKS",
  },
  "kawpow": {
    workgroup: "MOM_KAWPOW_WORKGROUP",
    dag_workgroup: "MOM_KAWPOW_DAG_WORKGROUP",
    dag_chunk: "MOM_KAWPOW_DAG_CHUNK_NODES",
  },
  "firopow": {
    workgroup: "MOM_KAWPOW_WORKGROUP",
    dag_workgroup: "MOM_KAWPOW_DAG_WORKGROUP",
    dag_chunk: "MOM_KAWPOW_DAG_CHUNK_NODES",
  },
  "evrprogpow": {
    workgroup: "MOM_KAWPOW_WORKGROUP",
    dag_workgroup: "MOM_KAWPOW_DAG_WORKGROUP",
    dag_chunk: "MOM_KAWPOW_DAG_CHUNK_NODES",
  },
  "meowpow": {
    workgroup: "MOM_KAWPOW_WORKGROUP",
    dag_workgroup: "MOM_KAWPOW_DAG_WORKGROUP",
    dag_chunk: "MOM_KAWPOW_DAG_CHUNK_NODES",
  },
  "etchash": {
    dag_workgroup: "MOM_ETCHASH_DAG_WORKGROUP",
    dag_chunk: "MOM_ETCHASH_DAG_CHUNK_NODES",
  },
  "autolykos2": {
    workgroup: "MOM_AUTOLYKOS2_WORKGROUP",
    prehash_workgroup: "MOM_AUTOLYKOS2_PREHASH_WORKGROUP",
    table_chunk: "MOM_AUTOLYKOS2_TABLE_CHUNK",
  },
  "fishhash": {workgroup: "MOM_FISHHASH_WORKGROUP"},
  "karlsenhashv2": {workgroup: "MOM_FISHHASH_WORKGROUP"},
  "pearlhash": {
    workgroup: "MOM_PEARLHASH_AMD_WMMA_THREADS",
    cache_block: [
      "MOM_PEARLHASH_AMD_WMMA_CACHE_BLOCK",
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

function tuningEnvironment(algo, tuning) {
  const env = {};
  for (const [field, envNames] of Object.entries(envByAlgo[algo] || {})) {
    if (tuning[field] === undefined) {continue;}
    for (const envName of Array.isArray(envNames) ? envNames : [envNames]) {
      env[envName] = String(tuning[field]);
    }
  }
  if (algo === "autolykos2" && tuning.search_mode !== undefined) {
    if (tuning.search_mode !== "auto") {
      env.MOM_AUTOLYKOS2_SUBGROUP_COOP = tuning.search_mode === "cooperative" ? "1" : "0";
    }
  }
  if ((algo === "fishhash" || algo === "karlsenhashv2") &&
      tuning.search_mode !== undefined && tuning.search_mode !== "auto") {
    env.MOM_FISHHASH_COOP = tuning.search_mode === "cooperative" ? "1" : "0";
  }
  if (algo === "beamhash3" && tuning.layout !== undefined && tuning.layout !== "auto") {
    env.MOM_BEAMHASH3_COMPACT = tuning.layout === "compact" ? "1" : "0";
  }
  if (algo === "pearlhash" && tuning.tile !== undefined && tuning.tile !== "auto") {
    env.MOM_PEARLHASH_AMD_DP4A_TILE = tuning.tile;
  }
  return env;
}

module.exports = {
  applyNativeJobTuning,
  formatDeviceList,
  needsPrimaryTuning,
  parseDeviceEntry,
  parseDeviceList,
  primaryTuningField,
  tuningEnvironment,
  validateTuning,
};
