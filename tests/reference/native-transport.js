"use strict";

const net = require("node:net");
const tls = require("node:tls");
const connection = require("../../pool/connection");

// Run the maintained connection/login/framing code, not a second protocol implementation. The
// surrounding callbacks only record work; no hashing, credentials or external pool is involved.
/** @param {boolean} encrypted @param {string} [fingerprint] */
async function transport(encrypted, fingerprint = "") {
  const port = Number(process.env[encrypted ? "MOM_AOT_TLS_PORT" : "MOM_AOT_POOL_PORT"]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {throw new Error("Missing loopback fixture port");}
  const pool = /** @type {PoolConfig} */ (/** @type {unknown} */ ({
    url: "127.0.0.1", port, is_tls: encrypted, tls_verify: false, tls_fingerprint: fingerprint,
    login: process.env["MOM_AOT_FAILURE"] === "oversized" ? "oversized" : "fixture", pass: "fixture",
    use_subscribe: false, last_connect_time: 0, socket: null,
  }));
  global.opt = /** @type {MinerOptions} */ (/** @type {unknown} */ ({
    pools: [pool], pool_ids: {primary: 0, donate: 1}, job: {algo: "pearlhash"},
    algo_params: {pearlhash: {perf: 123}},
    pool_time: {connect_throttle: 0, first_job_wait: 0.5},
  }));
  /** @type {unknown[]} */
  const received = [];
  /** @type {UnknownRecord[]} */
  const login = [];
  await new Promise((resolve, reject) => {
    const deadline = setTimeout(() => reject(new Error("Native pool transport deadline exceeded")), 3000);
    const api = connection({
      h: {log_err: (message) => reject(new Error(message))}, o: {agent_str: "mom-native-fixture"},
      net, tls,
      systemNetConnect: net.connect,
      systemTlsConnect: tls.connect,
      max_pool_data_buffer: 65536,
      clear_pool_connection: (_id, socket) => {
        if (pool.socket !== socket) {return false;}
        pool.socket = null; socket.destroy(); return true;
      },
      isCurrentPoolSocket: (_id, socket) => pool.socket === socket,
      pearlhashUsesSubscribe: () => false, poolProtocol: () => "json",
      pool_log: () => undefined, pool_log1: () => undefined, pool_log_str: (_id, text) => text,
      poolErrorText: (_id, error) => String(error), pool_log_json: () => undefined,
      pool_message: (_id, json) => {
        if (json.method === "mining.notify") {
          received.push({words: json.__kaspa_words, timestamp: json.__kaspa_timestamp});
          pool.socket?.destroy();
          pool.socket = null;
        } else {received.push(json);}
      },
      pool_str: () => "loopback", usesCortex: () => false,
      usesIronfish: () => false, usesMiningSubscribe: () => false,
      poolWrite: (_id, json) => {
        login.push(json);
        pool.socket?.write(JSON.stringify(json) + "\n");
      },
      switchPool: () => reject(new Error("Unexpected fixture reconnect")),
    });
    api.connectPoolThrottle(0, () => {throw new Error("Unexpected GPU job dispatch");});
    if (!pool.socket) {throw new Error("Loopback connection was not created");}
    pool.socket.once("close", () => {
      clearTimeout(deadline);
      if (received.length !== 2) {
        reject(new Error("Incomplete loopback response"));
      } else {resolve(undefined);}
    });
  });
  return {received, login};
}

module.exports = async function transports() {
  const pin = process.env["MOM_AOT_TLS_FINGERPRINT"];
  if (!pin) {throw new Error("Missing loopback TLS fingerprint");}
  const tcp = await transport(false), tls = await transport(true), pinnedTls = await transport(true, pin);
  const wrong = pin.slice(0, -1) + (pin.endsWith("0") ? "1" : "0");
  try {await transport(true, wrong);} catch (error) {
    if (error instanceof Error && error.message === "TLS certificate fingerprint mismatch") {
      return {tcp, tls, pinnedTls, wrongPinRejected: true};
    }
    throw error;
  }
  throw new Error("Loopback TLS accepted the wrong certificate pin");
};
