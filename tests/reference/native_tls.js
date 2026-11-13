"use strict";

const tls = require("node:tls");
const fs = require("node:fs");

/** @param {string} name @param {import("node:tls").ConnectionOptions} options
 * @param {"success" | "error" | "cancel" | "writeError"} expectation @param {boolean} [large]
 * @param {boolean} [endEarly] */
async function attempt(name, options, expectation, large = false, endEarly = false) {
  return new Promise((resolve, reject) => {
    const socket = tls.connect({host: "127.0.0.1", ...options}).setEncoding("utf8");
    let response = "";
    let connected = 0;
    let secured = 0;
    let fingerprint256 = "";
    let code = "";
    let acknowledged = 0;
    const deadline = setTimeout(() => {socket.destroy(); reject(new Error("TLS fixture deadline exceeded"));}, 3000);
    const request = JSON.stringify({method: "login", params: {
      case: name, credential: "native-parity-sensitive-credential", ...(large ? {payload: "x".repeat(1024 * 1024)} : {}),
    }}) + "\n";
    const split = request.length >> 1;
    // Queue part before TCP connect and part in its callback. Both must remain ordered and secret
    // until verification completes. Pools separately gate credentials on their optional pin.
    socket.write(request.slice(0, split), (error) => {if (!error) {acknowledged++;}});
    socket.on("connect", () => {
      connected++;
      if (endEarly) {
        socket.end(request.slice(split));
      } else {socket.write(request.slice(split), (error) => {if (!error) {acknowledged++;}});}
      if (expectation === "cancel") {setTimeout(() => socket.destroy(), 30);}
    });
    socket.on("secureConnect", () => {
      secured++;
      fingerprint256 = socket.getPeerCertificate().fingerprint256 || "";
    });
    socket.on("error", (error) => {code = /** @type {NodeJS.ErrnoException} */ (error).code || "UNKNOWN";});
    socket.on("data", (chunk) => {response += chunk;});
    socket.on("close", () => {
      clearTimeout(deadline);
      if (expectation === "success" && (!response.includes('"accepted":true') || secured !== 1 || acknowledged !== (endEarly ? 1 : 2))) {
        reject(new Error(`TLS ${name} did not complete verified I/O: ${code}`)); return;
      }
      if (expectation === "error" && (!code || response || secured || acknowledged)) {
        reject(new Error(`TLS ${name} failed to reject before application I/O`)); return;
      }
      if (expectation === "cancel" && (code || response || secured || acknowledged)) {
        reject(new Error(`TLS ${name} sent data before a stalled handshake completed`)); return;
      }
      if (expectation === "writeError" && (code !== "EPIPE" || secured !== 1 || response || acknowledged)) {
        reject(new Error(`TLS ${name} did not report its native write failure safely: ${code}`)); return;
      }
      resolve({name, connected, secured, code, acknowledged, fingerprint256, authorized: socket.authorized,
        authorizationError: socket.authorizationError ? String(socket.authorizationError) : null,
        response: response ? JSON.parse(response) : null});
    });
  });
}

module.exports = async function tlsChecks() {
  const port = Number(process.env["MOM_AOT_TLS_PORT"]);
  const expiredPort = Number(process.env["MOM_AOT_TLS_EXPIRED_PORT"]);
  const spyPort = Number(process.env["MOM_AOT_TLS_SPY_PORT"]);
  if (![port, expiredPort, spyPort].every((value) => Number.isInteger(value) && value > 0 && value <= 65535)) {
    throw new Error("Missing TLS fixture ports");
  }
  const ca = fs.readFileSync(process.env["MOM_AOT_TLS_CA"] || "", "utf8");
  const expiredCa = fs.readFileSync(process.env["MOM_AOT_TLS_EXPIRED_CA"] || "", "utf8");
  if (process.env["MOM_AOT_FAILURE"] === "tlsWrite") {
    return [await attempt("large", {port, ca}, "writeError", true),
      await attempt("recovered", {port, ca}, "success")];
  }
  const results = [];
  results.push(await attempt("trusted", {port, ca}, "success"));
  results.push(await attempt("defaultReject", {port}, "error"));
  results.push(await attempt("wrongIdentity", {port, ca, servername: "wrong.invalid"}, "error"));
  results.push(await attempt("insecure", {port, rejectUnauthorized: false}, "success"));
  results.push(await attempt("caBuffer", {port, ca: [Buffer.from(ca)], servername: "localhost"}, "success"));
  results.push(await attempt("large", {port, ca}, "success", true));
  results.push(await attempt("halfClose", {port, ca}, "success", false, true));
  results.push(await attempt("expired", {port: expiredPort, ca: expiredCa}, "error"));
  results.push(await attempt("cancelHandshake", {port: spyPort}, "cancel"));
  return results;
};
