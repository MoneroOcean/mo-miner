"use strict";

const bytes = require("./bytes");

// The application uses default gzipSync only. Gzip state and framing live in static zlib, not
// a subprocess, JavaScript compressor, extracted library or separately maintained proof encoder.
module.exports = {gzipSync: (/** @type {unknown} */ data) => Buffer.from(globalThis.__momGzip(bytes(data)))};
