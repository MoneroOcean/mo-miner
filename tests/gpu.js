"use strict";

const { runHashSuite } = require("./common/hash_suite");

runHashSuite("GPU proof-of-work hash vectors", (entry) => entry.gpu, 10 * 60 * 1000);
