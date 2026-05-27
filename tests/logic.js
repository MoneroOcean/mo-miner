"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");

const opts = require("../opts.js");

test("saved config omits job without mutating live options", () => {
  const opt = {
    job: { algo: "rx/0", dev: "cpu" },
    pools: [{ url: "pool.example", port: 443 }],
    pool_ids: { primary: 0 },
  };

  const saved = opts.saved_config(opt);

  assert.deepEqual(saved, {
    pools: [{ url: "pool.example", port: 443 }],
    pool_ids: { primary: 0 },
  });
  assert.deepEqual(opt.job, { algo: "rx/0", dev: "cpu" });
});

test("config file detection requires a .json extension", () => {
  assert.equal(opts.is_config_file("config.json"), true);
  assert.equal(opts.is_config_file("CONFIG.JSON"), true);
  assert.equal(opts.is_config_file("pooljson"), false);
  assert.equal(opts.is_config_file("config-json"), false);
});
