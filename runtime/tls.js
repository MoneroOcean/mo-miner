"use strict";

const {Socket} = require("./net");
const bytes = require("./bytes");

/** @typedef {{host: string, port: number, servername?: string, rejectUnauthorized?: boolean,
 * ca?: string | Buffer | ArrayBufferView | Array<string | Buffer | ArrayBufferView>}} TlsOptions */

// Match Node's pre-handshake buffering. Pools additionally wait for secureConnect to check
// their optional certificate pin before enqueueing any credentials.
class TLSSocket extends Socket {
  /** @param {TlsOptions} options */
  constructor(options) {
    super(options);
    this.encrypted = true;
    this.authorized = false;
    this.authorizationError = "";
    this.peerCertificate = {};
    this.established = false;
    try {
      const ca = options.ca === undefined ? null :
        (Array.isArray(options.ca) ? options.ca : [options.ca]).map((value) => {
          if (typeof value === "string") {return value;}
          if (!ArrayBuffer.isView(value)) {throw new TypeError("TLS CA must be PEM text or a Buffer");}
          return bytes(value).toString();
        }).join("\n");
      globalThis.__momTLSConfigure(this.id, options.servername || options.host,
        options.servername || "", options.rejectUnauthorized !== false, ca);
    } catch (error) {
      this.destroy();
      throw error;
    }
  }

  /** @override */
  ready() {return this.established;}

  /** @returns {{fingerprint256?: string}} */
  getPeerCertificate() {return {...this.peerCertificate};}

  /** @override @param {string} event @param {unknown} value */
  receive(event, value) {
    if (this.destroyed && event !== "close" && event !== "error") {return;}
    if (event === "data" && !this.established) {throw new Error("TLS data arrived before authentication completed");}
    if (event === "connect") {
      this.connecting = false;
      this.emit("connect");
    } else if (event === "secureConnect") {
      const result = /** @type {{authorized: boolean, authorizationError: string, fingerprint256?: string}} */ (value);
      this.authorized = result.authorized;
      this.authorizationError = result.authorizationError;
      this.peerCertificate = typeof result.fingerprint256 === "string" ? {fingerprint256: result.fingerprint256} : {};
      this.established = true;
      this.flushPending();
      this.emit("secureConnect");
    } else {
      if (event === "error" && value && typeof value === "object") {
        const authorization = Reflect.get(value, "authorizationError");
        if (typeof authorization === "string" && authorization) {this.authorizationError = authorization;}
      }
      super.receive(event, value);
    }
  }
}

/** @param {TlsOptions} options @param {() => void} [callback] */
function connect(options, callback) {
  const socket = new TLSSocket(options);
  if (callback) {socket.once("secureConnect", callback);}
  return socket;
}

module.exports = {TLSSocket, connect};
