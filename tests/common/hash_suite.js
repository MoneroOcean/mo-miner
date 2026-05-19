"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { runMinerTest } = require("./miner_command");
const { hashTests } = require("../vectors");

function runHashSuite(title, predicate, defaultTimeoutMs) {
  describe(title, () => {
    for (const definition of hashTests.filter(predicate)) {
      it(definition.name, { timeout: definition.timeoutMs || defaultTimeoutMs }, async (t) => {
        const result = await runMinerTest(definition);
        if (result.skipped) {
          t.skip(result.reason);
          return;
        }
        assert.equal(result.skipped, false);
      });
    }
  });
}

module.exports = { runHashSuite };
