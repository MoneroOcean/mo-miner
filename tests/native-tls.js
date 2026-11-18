"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const nodeTls = require("node:tls");
const {TLSSocket, connect} = require("../runtime/tls");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {extractCppBlock} = require("./common/cpp-source");
const compiler = process.env["CXX"] || "c++";

test.describe("Native certificate verification diagnostics", {skip: process.platform !== "linux"}, () => {
  test("production error mapping retains certificate failures and rejects unknown errors generically", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-tls-errors-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = fs.readFileSync(path.join(__dirname, "../runtime/tls.cpp"), "utf8");
    const names = ["INVALID_PURPOSE", "INVALID_CA", "CERT_REVOKED", "CERT_REJECTED", "CERT_UNTRUSTED",
      "CERT_SIGNATURE_FAILURE", "UNABLE_TO_GET_ISSUER_CERT", "PATH_LENGTH_EXCEEDED", "CERT_CHAIN_TOO_LONG",
      "CERT_HAS_EXPIRED", "CERT_NOT_YET_VALID", "DEPTH_ZERO_SELF_SIGNED_CERT", "SELF_SIGNED_CERT_IN_CHAIN"];
    const file = path.join(directory, "errors.cpp"), executable = path.join(directory, "errors");
    fs.writeFileSync(file, "#include <openssl/x509_vfy.h>\n#include <cassert>\n#include <cstring>\n" +
      extractCppBlock(source, "const char *verificationCode(") + "int main(){\n" +
      names.map(name => `assert(std::strcmp(verificationCode(X509_V_ERR_${name}),"${name}")==0);`).join("\n") +
      '\nassert(std::strcmp(verificationCode(X509_V_OK),"")==0);\n' +
      'assert(std::strcmp(verificationCode(X509_V_ERR_HOSTNAME_MISMATCH),"ERR_TLS_CERT_ALTNAME_INVALID")==0);\n' +
      'assert(std::strcmp(verificationCode(X509_V_ERR_IP_ADDRESS_MISMATCH),"ERR_TLS_CERT_ALTNAME_INVALID")==0);\n' +
      'assert(std::strcmp(verificationCode(-1),"CERTIFICATE_VERIFY_FAILED")==0);\n}\n');
    /** @type {Array<[string, string[]]>} */
    const commands = [[compiler, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", file, "-o", executable]], [executable, []]];
    for (const [command, args] of commands) {
      const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });

  test("scoped IPv6 strips the zone only for IP SAN verification", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-tls-scoped-ip-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = fs.readFileSync(path.join(__dirname, "../runtime/tls.cpp"), "utf8");
    assert.match(source, /const auto ip = verificationIp\(hostname\);/);
    const loopbackCertificate = /** @type {import("node:tls").PeerCertificate} */ (/** @type {unknown} */ ({
      subjectaltname: "IP Address:0:0:0:0:0:0:0:1",
    }));
    const linkLocalCertificate = /** @type {import("node:tls").PeerCertificate} */ (/** @type {unknown} */ ({
      subjectaltname: "IP Address:fe80::1",
    }));
    const scopedIpv4Certificate = /** @type {import("node:tls").PeerCertificate} */ (/** @type {unknown} */ ({
      subjectaltname: "IP Address:127.0.0.1",
    }));
    assert.equal(nodeTls.checkServerIdentity("::1%lo", loopbackCertificate), undefined);
    assert.equal(nodeTls.checkServerIdentity("fe80::1%eth0", linkLocalCertificate), undefined);
    const scopedIpv4Error = nodeTls.checkServerIdentity("127.0.0.1%eth0", scopedIpv4Certificate);
    assert.ok(scopedIpv4Error);
    assert.equal(/** @type {Error & {code?: string}} */ (/** @type {unknown} */ (scopedIpv4Error)).code,
      "ERR_TLS_CERT_ALTNAME_INVALID");
    const file = path.join(directory, "scoped-ip.cpp"), executable = path.join(directory, "scoped-ip");
    fs.writeFileSync(file, [
      "#include <openssl/x509v3.h>", "#include <cassert>", "#include <cstring>", "#include <memory>", "#include <string>",
      extractCppBlock(source, "bool validIpv6Scope("),
      extractCppBlock(source, "std::unique_ptr<ASN1_OCTET_STRING, decltype(&ASN1_OCTET_STRING_free)> verificationIp("),
      "int main() {",
      "const unsigned char loopback[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};",
      "const unsigned char linkLocal[16] = {0xfe,0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,1};",
      "auto plain = verificationIp(\"::1\");",
      "assert(plain && ASN1_STRING_length(plain.get()) == 16 && std::memcmp(ASN1_STRING_get0_data(plain.get()), loopback, 16) == 0);",
      "auto scoped = verificationIp(\"::1%lo\");",
      "assert(scoped && ASN1_STRING_length(scoped.get()) == 16 && std::memcmp(ASN1_STRING_get0_data(scoped.get()), loopback, 16) == 0);",
      "auto namedScope = verificationIp(\"fe80::1%eth0\");",
      "assert(namedScope && ASN1_STRING_length(namedScope.get()) == 16 && std::memcmp(ASN1_STRING_get0_data(namedScope.get()), linkLocal, 16) == 0);",
      "auto extendedScope = verificationIp(\"::1%zone.with:chars\");",
      "assert(extendedScope && ASN1_STRING_length(extendedScope.get()) == 16);",
      "auto ipv4 = verificationIp(\"127.0.0.1\");",
      "assert(ipv4 && ASN1_STRING_length(ipv4.get()) == 4);",
      "assert(!verificationIp(\"127.0.0.1%eth0\"));",
      "assert(!verificationIp(\"pool.example\"));",
      "assert(!verificationIp(\"pool.example%eth0\"));",
      "assert(!verificationIp(\"::1%\"));",
      "assert(!verificationIp(\"::1%bad_zone\"));",
      "assert(!verificationIp(\"::1%lo%eth0\"));",
      "}", "",
    ].join("\n"));
    const result = spawnSync(compiler, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", file, "-lcrypto", "-o", executable],
      {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    const run = spawnSync(executable, [], {encoding: "utf8", timeout: 30000});
    assert.equal(run.status, 0, run.error?.message || run.stderr);
  });

  test("peer certificate fingerprint is Node-compatible uppercase SHA-256", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-tls-fingerprint-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = fs.readFileSync(path.join(__dirname, "../runtime/tls.cpp"), "utf8"),
      certificateFile = path.join(__dirname, "fixtures/tls-localhost-cert.pem"),
      expected = new (require("node:crypto").X509Certificate)(fs.readFileSync(certificateFile)).fingerprint256;
    const file = path.join(directory, "fingerprint.cpp"), executable = path.join(directory, "fingerprint");
    fs.writeFileSync(file, [
      "#include <openssl/err.h>", "#include <openssl/evp.h>", "#include <openssl/pem.h>", "#include <openssl/x509.h>",
      "#include <cassert>", "#include <string>",
      extractCppBlock(source, "std::string certificateFingerprint256(X509 *certificate)"),
      "int main(int argc, char **argv) {",
      "assert(argc == 2); BIO *input = BIO_new_file(argv[1], \"r\"); assert(input);",
      "X509 *certificate = PEM_read_bio_X509(input, nullptr, nullptr, nullptr); BIO_free(input); assert(certificate);",
      `assert(certificateFingerprint256(certificate) == "${expected}");`,
      "assert(certificateFingerprint256(nullptr).empty()); X509_free(certificate);",
      "}", "",
    ].join("\n"));
    const result = spawnSync(compiler, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", file, "-lcrypto", "-o", executable],
      {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    const run = spawnSync(executable, [certificateFile], {encoding: "utf8", timeout: 30000});
    assert.equal(run.status, 0, run.error?.message || run.stderr);
  });
});

test.describe("Native TLS adapter", {concurrency: false}, () => {
  /** @type {(event: string, value?: unknown) => void} */
  let dispatch;
  /** @type {unknown[]} */
  let configuration;
  /** @type {string[]} */
  let writes;
  /** @type {number[]} */
  let closes;
  /** @type {number[]} */
  let endings;
  /** @type {Record<string, unknown>} */
  const saved = {};
  const names = ["__momTCPConnect", "__momTCPWrite", "__momTLSConfigure", "__momHandleClose", "__momTCPEnd"];
  test.beforeEach(() => {
    names.forEach((name) => {saved[name] = Reflect.get(globalThis, name);});
    configuration = []; writes = []; closes = []; endings = [];
    globalThis.__momTCPConnect = (_host, _port, callback) => {dispatch = callback; return 9;};
    globalThis.__momTLSConfigure = (...args) => {configuration = args;};
    globalThis.__momTCPWrite = (_id, data) => {writes.push(Buffer.from(data).toString()); return true;};
    globalThis.__momHandleClose = (id) => {closes.push(id);};
    globalThis.__momTCPEnd = (id) => {endings.push(id);};
  });
  test.afterEach(() => {
    names.forEach((name) => {
      if (saved[name] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[name]);}
    });
  });
  function socket() {return new TLSSocket({host: "127.0.0.1", port: 1234});}
  function secure(authorized = true, authorizationError = "", fingerprint256 = "") {
    dispatch("secureConnect", {authorized, authorizationError, fingerprint256});
  }

  test("default certificate verification is enabled without implicit SNI", () => {
    socket(); assert.deepEqual(configuration, [9, "127.0.0.1", "", true, null]);
  });
  test("verification can only be disabled explicitly", () => {
    const client = new TLSSocket({host: "127.0.0.1", port: 1234, rejectUnauthorized: false});
    assert.equal(client.encrypted, true);
    assert.equal(configuration[3], false);
  });
  test("explicit servername controls both SNI and verified peer identity", () => {
    const client = new TLSSocket({host: "127.0.0.1", port: 1234, servername: "pool.example"});
    assert.equal(client.encrypted, true);
    assert.deepEqual(configuration.slice(1, 3), ["pool.example", "pool.example"]);
  });
  test("custom CA text and Buffer arrays preserve PEM contents", () => {
    const client = new TLSSocket({host: "127.0.0.1", port: 1234, ca: ["first", Buffer.from("second")]});
    assert.equal(client.encrypted, true);
    assert.equal(configuration[4], "first\nsecond");
  });
  test("typed-array and DataView CAs preserve their exact byte ranges", () => {
    const first = Uint8Array.from([120, 102, 105, 114, 115, 116, 121]);
    const typed = new Uint8Array(first.buffer, 1, 5);
    const firstClient = new TLSSocket({host: "127.0.0.1", port: 1234, ca: typed});
    assert.equal(firstClient.encrypted, true);
    assert.equal(configuration[4], "first");
    const second = Uint8Array.from([120, 115, 101, 99, 111, 110, 100, 121]);
    const view = new DataView(second.buffer, 1, 6);
    const secondClient = new TLSSocket({host: "127.0.0.1", port: 1234, ca: view});
    assert.equal(secondClient.encrypted, true);
    assert.equal(configuration[4], "second");
  });
  test("invalid CA types fail and release the pending TCP handle", () => {
    assert.throws(() => new TLSSocket({host: "127.0.0.1", port: 1234,
      ca: /** @type {string} */ (/** @type {unknown} */ (42))}), TypeError);
    assert.deepEqual(closes, [9]);
  });
  test("native TLS initialization failures release pending TCP ownership", () => {
    globalThis.__momTLSConfigure = () => {throw new Error("configuration failed");};
    assert.throws(socket, /configuration failed/); assert.deepEqual(closes, [9]);
  });
  test("writes queued before and during TCP connect cannot precede secureConnect", () => {
    const client = socket(); client.write("before");
    client.on("connect", () => client.write("during"));
    dispatch("connect"); assert.deepEqual(writes, []); assert.equal(client.connecting, false);
    secure(); assert.deepEqual(writes, ["before", "during"]); assert.equal(client.authorized, true);
  });
  test("getPeerCertificate is empty until secureConnect and exposes the SHA-256 pin before listeners", () => {
    const client = socket(), fingerprint256 = "AA:BB";
    let observed;
    assert.deepEqual(client.getPeerCertificate(), {});
    client.once("secureConnect", () => {observed = client.getPeerCertificate();});
    secure(true, "", fingerprint256);
    assert.deepEqual(observed, {fingerprint256});
    assert.deepEqual(client.getPeerCertificate(), {fingerprint256});
  });
  test("missing peer fingerprint remains visibly absent for fail-closed pin checks", () => {
    const client = socket();
    secure(true, "");
    assert.deepEqual(client.getPeerCertificate(), {fingerprint256: ""});
  });
  test("TLS connect callback waits for authentication, not raw TCP", () => {
    let count = 0; connect({host: "127.0.0.1", port: 1234}, () => count++);
    dispatch("connect"); assert.equal(count, 0); secure(); assert.equal(count, 1);
  });
  test("unauthorized but explicitly permitted connections retain failure metadata", () => {
    const client = socket(); secure(false, "DEPTH_ZERO_SELF_SIGNED_CERT");
    assert.equal(client.authorized, false); assert.equal(client.ready(), true);
    assert.equal(client.authorizationError, "DEPTH_ZERO_SELF_SIGNED_CERT");
  });
  test("certificate rejection metadata is observable on the failed socket", () => {
    const client = socket(); client.on("error", () => undefined);
    dispatch("error", Object.assign(new Error("certificate failed"), {authorizationError: "CERT_HAS_EXPIRED"}));
    assert.equal(client.authorizationError, "CERT_HAS_EXPIRED"); assert.equal(client.ready(), false);
  });
  test("end waits for authentication and all pre-handshake writes", () => {
    const client = socket(); client.end("final"); dispatch("connect");
    assert.deepEqual(writes, []); assert.deepEqual(endings, []);
    secure(); assert.deepEqual(writes, ["final"]); assert.deepEqual(endings, [9]);
  });
  test("cancelling a handshake cannot resurrect it through a stale secure event", () => {
    const client = socket(); client.write("secret"); client.destroy(); secure();
    assert.deepEqual(writes, []); assert.equal(client.ready(), false); assert.equal(client.destroyed, true);
  });
  test("application data before authentication is an explicit fatal bridge error", () => {
    socket(); assert.throws(() => dispatch("data", new ArrayBuffer(1)), /before authentication/);
  });
});
