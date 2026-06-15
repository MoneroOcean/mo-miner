"use strict";

const { describe, it } = require("node:test");

const { runMinerTest } = require("./miner_command");
const { hashTests } = require("../vectors");

function runHashSuite(title, predicate, defaultTimeoutMs) {
  describe(title, () => {
    for (const definition of hashTests.filter(predicate)) {
      // Per-vector timeout overrides the suite default (GPU vectors are slower).
      it(definition.name, { timeout: definition.timeoutMs || defaultTimeoutMs }, async (t) => {
        // runMinerTest throws on any failure, so a returned result is either a
        // skip or a clean pass; the test passes simply by not throwing.
        const result = await runMinerTest(definition);
        if (result.skipped) t.skip(result.reason);
      });
    }
  });
}

module.exports = { runHashSuite };
