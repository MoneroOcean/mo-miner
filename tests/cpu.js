"use strict";

const {runHashSuite} = require("./common/hash-suite");

runHashSuite("CPU proof-of-work hash vectors", (entry) => !entry.gpu, 5 * 60 * 1000);
