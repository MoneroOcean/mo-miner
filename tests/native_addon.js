"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const path = require("node:path");
const test = require("node:test");

const root = path.join(__dirname, "..");
const platformBuild = process.platform === "win32" ? "win" : "lin";
const addonPath = process.env["MOM_NATIVE_PATH"] ||
  path.join(root, "build", platformBuild, "Release", "mom.node");
const core = require(addonPath);
const noop = () => undefined;

/** @typedef {{message?: string | undefined, nonce?: string | undefined, result?: string | undefined, job_id?: string | undefined, hash?: string | undefined, [key: string]: string | undefined}} NativeValues */

/** @param {unknown} [values] */
function closeWorker(values) {
  return new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(noop, resolve, reject);
    worker.sendToCpp("close", values);
  });
}

test("native addon validates callback arguments", () => {
  assert.throws(
    () => new core.AsyncWorker(noop, noop),
    /requires progress, complete, and error callbacks/
  );
  assert.throws(
    () => new core.AsyncWorker(1, noop, noop),
    /callbacks must be functions/
  );
});

test("native addon accepts omitted, undefined, and null message values", {timeout: 5000}, async () => {
  await closeWorker();
  await closeWorker(undefined);
  await closeWorker(null);
});

test("native addon rejects invalid message argument types without aborting", {timeout: 5000}, async () => {
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(noop, resolve, reject);
    assert.throws(() => worker.sendToCpp(1, {}), /message name must be a string/);
    assert.throws(() => worker.sendToCpp("pause", 1), /values must be an object/);
    worker.sendToCpp("close");
  });
});

test("native worker rejects odd-length target hex", {timeout: 5000}, async () => {
  let message = "";
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        if (name !== "error") {
          return;
        }
        if (typeof values.message !== "string") {throw new Error("Native error omitted its message");}
        message = values.message;
        worker.sendToCpp("close");
      },
      resolve,
      reject
    );
    worker.sendToCpp("job", {
      job_id: "odd-target",
      job_token: "token",
      pool_id: "pool",
      target: "abc",
      worker_id: "worker",
    });
  });
  assert.match(message, /Bad target hex/);
});

test("native worker rejects unknown commands", {timeout: 5000}, async () => {
  let message = "";
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        if (name !== "error") {
          return;
        }
        if (typeof values.message !== "string") {throw new Error("Native error omitted its message");}
        message = values.message;
        worker.sendToCpp("close");
      },
      resolve,
      reject
    );
    worker.sendToCpp("future_command");
  });
  assert.equal(message, "Message processing exception: Unknown native command: future_command");
});

test("native worker delivers progress before completion and rejects posts after close", {
  timeout: 5000,
}, async () => {
  /** @type {string[]} */
  const events = [];
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name) => events.push(name),
      resolve,
      reject
    );
    worker.sendToCpp("job", {});
    worker.sendToCpp("close");
    assert.throws(() => worker.sendToCpp("pause"), /AsyncWorker is stopped/);
  });
  assert.deepEqual(events, ["error"]);
});

test("native addon does not exit after an invalid exitNow code", () => {
  assert.throws(() => core.exitNow("bad"), /code must be a number/);
});

test("native worker rejects batching solvers that process one header per dispatch", {
  timeout: 5000,
}, async () => {
  let message = "";
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        if (name !== "error") {
          return;
        }
        if (typeof values.message !== "string") {throw new Error("Native error omitted its message");}
        message = values.message;
        worker.sendToCpp("close");
      },
      resolve,
      reject
    );
    worker.sendToCpp("test", {
      algo: "beamhash3",
      blob_hex: "00".repeat(44),
      dev: "gpu1",
      intensity: 2,
      noncebytes: 8,
      nonceoffset: 32,
    });
  });
  assert.match(message, /Single-header solvers require intensity 1/);
});

test("native worker lifecycle neither hangs idle nor exits before completion", () => {
  const load = `const core = require(${JSON.stringify(addonPath)}); const noop = () => undefined;`;
  const idle = childProcess.spawnSync(
    process.execPath,
    ["-e", `${load} new core.AsyncWorker(noop, noop, noop);`],
    {timeout: 5000}
  );
  assert.equal(idle.error, undefined);
  assert.equal(idle.status, 0);

  const started = childProcess.spawnSync(
    process.execPath,
    ["-e", `${load}
      const worker = new core.AsyncWorker(
        noop,
        () => console.log("complete"),
        (error) => { throw error; }
      );
      worker.sendToCpp("close");`],
    {encoding: "utf8", timeout: 5000}
  );
  assert.equal(started.error, undefined);
  assert.equal(started.status, 0);
  assert.equal(started.stdout.trim(), "complete");
});

test("native worker joins RandomX threads when collected without close", () => {
  const load = `const core = require(${JSON.stringify(addonPath)});`;
  const result = childProcess.spawnSync(
    process.execPath,
    ["--expose-gc", "-e", `${load}
      const events = [];
      let worker = new core.AsyncWorker(
        (name) => {
          if (name !== "test") throw new Error("Unexpected native event: " + name);
          events.push(name);
          worker = null;
          global.gc();
        },
        () => {
          events.push("complete");
          console.log(events.join(","));
        },
        (error) => { throw error; }
      );
      worker.sendToCpp("test", {
        algo: "rx/0",
        blob_hex: "00".repeat(43),
        dev: "cpu",
        noncebytes: 4,
        nonceoffset: 39,
        seed_hex: "00".repeat(32),
      });`],
    {encoding: "utf8", timeout: 15000}
  );
  assert.equal(result.error, undefined);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "test,complete");
});

test("native worker rejects unavailable CPU batches and remains retryable", {timeout: 10000}, () => {
  const job = {
    algo: "argon2/chukwa",
    blob_hex: "0305A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B" +
      "00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601",
    dev: "cpu",
    noncebytes: 4,
    nonceoffset: 39,
  };
  const expected = "c158a105ae75c7561cfd029083a47a87653d51f914128e21c1971d8b10c49034";
  const result = childProcess.spawnSync(process.execPath, ["-e", `
    const assert = require("node:assert/strict");
    const core = require(${JSON.stringify(addonPath)});
    const job = ${JSON.stringify(job)};
    const events = [];
    let phase = "unsupported";
    const worker = new core.AsyncWorker((name, values) => {
      if (name === "error") {
        assert.equal(phase, "unsupported");
        assert.equal(values.message,
          "Message processing exception: Unsupported CPU algorithm/batch combination");
        events.push("error");
        phase = "valid";
        worker.sendToCpp("test", job);
      } else {
        assert.equal(name, "test");
        assert.equal(phase, "valid");
        assert.equal(values.result, ${JSON.stringify(expected)});
        events.push("test");
        phase = "done";
        worker.sendToCpp("close");
      }
    }, () => {
      clearTimeout(deadline);
      assert.equal(phase, "done");
      events.push("complete");
      console.log(JSON.stringify(events));
    }, (error) => { throw error; });
    const deadline = setTimeout(() => worker.sendToCpp("close"), 5000);
    worker.sendToCpp("test", {...job, dev: "cpu*2"});
  `], {encoding: "utf8", timeout: 8000});
  assert.equal(result.error, undefined);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), ["error", "test", "complete"]);
});

test("native mining resets benchmark state and processes the final nonce batch", {
  timeout: 5000,
}, async () => {
  /** @type {string[]} */
  const results = [];
  /** @type {string[]} */
  const errors = [];
  const job = {
    algo: "cn/0",
    blob_hex: "54686973206973206120746573742054686973206973" +
      "20612074657374205468697320697320612074657374",
    dev: "cpu*2",
    job_id: "final-batch",
    job_token: "token",
    nonce: "fffffffe",
    noncebytes: "4",
    nonceoffset: "39",
    pool_id: "pool",
    target: "ffffffff",
    worker_id: "worker",
  };
  await new Promise((resolve, reject) => {
    let closeRequested = false;
    const close = () => {
      if (closeRequested) {
        return;
      }
      closeRequested = true;
      worker.sendToCpp("close");
    };
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        if (name === "result") {
          if (typeof values.nonce !== "string") {throw new Error("Native result omitted its nonce");}
          results.push(values.nonce);
          // The old wrapping path keeps mining at zero instead of reporting exhaustion.
          if (results.length === 3) {
            close();
          }
        } else if (name === "error") {
          if (typeof values.message !== "string") {throw new Error("Native error omitted its message");}
          errors.push(values.message);
          close();
        }
      },
      () => {
        clearTimeout(timeout);
        resolve(undefined);
      },
      reject
    );
    // Queue both modes together so same-input reuse cannot inherit an untracked bench cursor.
    worker.sendToCpp("bench", job);
    worker.sendToCpp("job", job);
    const timeout = setTimeout(close, 1000);
  });
  assert.deepEqual(results, ["fffffffe", "ffffffff"]);
  assert.deepEqual(errors, ["Nonce overflow"]);
});

test("native benchmark keeps hashrate window across changed input", {timeout: 20000}, async () => {
  const job = {
    algo: "cn/0",
    blob_hex: "00".repeat(43),
    dev: "cpu*2",
    nonce: "0",
    noncebytes: 4,
    nonceoffset: 39,
  };
  const replacement = {...job, blob_hex: "01".repeat(43)};
  /** @type {number | undefined} */
  let hashrate;
  await new Promise((resolve, reject) => {
    let closed = false;
    let failed = false;
    const close = () => {
      if (closed) {return;}
      closed = true;
      clearTimeout(replacementTimer);
      clearTimeout(deadline);
      worker.sendToCpp("close");
    };
    /** @param {Error} error */
    const fail = (error) => {
      if (failed) {return;}
      failed = true;
      close();
      reject(error);
    };
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        try {
          if (name === "hashrate") {
            const rate = Number(values["hashrate"]);
            if (Number.isFinite(rate) && rate > 0) {
              hashrate = rate;
              close();
            }
          } else if (name === "error") {
            const message = typeof values.message === "string"
              ? values.message : "Native error omitted its message";
            fail(new Error(message));
          }
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
        }
      },
      () => {
        clearTimeout(replacementTimer);
        clearTimeout(deadline);
        if (failed) {
          return;
        }
        if (hashrate === undefined) {
          failed = true;
          reject(new Error("Native benchmark closed before reporting hashrate"));
        } else {
          resolve(undefined);
        }
      },
      (/** @type {Error} */ error) => fail(error)
    );
    const replacementTimer = setTimeout(() => {
      if (!closed) {
        worker.sendToCpp("bench", replacement);
      }
    }, 5000);
    const deadline = setTimeout(() => {
      fail(new Error("Native benchmark hashrate timed out"));
    }, 14000);
    worker.sendToCpp("bench", job);
  });
  if (hashrate === undefined) {throw new Error("Native benchmark omitted its hashrate");}
  assert.ok(Number.isFinite(hashrate));
  assert.ok(hashrate > 0);
});

test("native allocation failures leave CPU jobs retryable", {timeout: 15000}, async () => {
  for (const retry of [false, true]) {
    /** @type {string[]} */
    const results = [];
    /** @type {string[]} */
    const errors = [];
    const cpuJob = {
      algo: "cn/0",
      blob_hex: "54686973206973206120746573742054686973206973" +
        "20612074657374205468697320697320612074657374",
      dev: "cpu*2",
      job_id: "allocation-recovery",
      job_token: "token",
      nonce: "fffffffe",
      noncebytes: "4",
      nonceoffset: "39",
      pool_id: "pool",
      target: "ffffffff",
      worker_id: "worker",
    };
    const gpuJob = {...cpuJob, algo: "cn/gpu", dev: "gpu1", intensity: 16777216};
    const expectedError = "Message processing exception: Requested allocation is too large";

    await new Promise((resolve, reject) => {
      let phase = "cpu";
      let closeRequested = false;
      let failed = false;
      const close = () => {
        if (closeRequested) {
          return;
        }
        closeRequested = true;
        worker.sendToCpp("close");
      };
      /**
       * @param {Error} error
       * @param {boolean} shouldClose
       */
      const fail = (error, shouldClose) => {
        if (failed) {
          return;
        }
        failed = true;
        clearTimeout(timeout);
        if (shouldClose) {
          close();
        }
        reject(error);
      };
      const worker = new core.AsyncWorker(
        (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
          try {
            if (name === "test") {
              if (typeof values.result !== "string" || values.result.length === 0) {
                throw new Error("Native test result was empty");
              }
              if (phase === "cpu") {
                results.push(values.result);
                phase = "gpu";
                worker.sendToCpp("test", gpuJob);
              } else {
                assert.equal(phase, "retry");
                results.push(values.result);
                phase = "done";
                close();
              }
            } else if (name === "error") {
              assert.equal(phase, "gpu");
              assert.equal(values.message, expectedError);
              errors.push(values.message);
              if (retry) {
                phase = "retry";
                worker.sendToCpp("test", cpuJob);
              } else {
                phase = "done";
                close();
              }
            } else {
              throw new Error("Unexpected native event: " + name);
            }
          } catch (error) {
            fail(error instanceof Error ? error : new Error(String(error)), true);
          }
        },
        () => {
          clearTimeout(timeout);
          if (!failed && phase !== "done") {
            fail(new Error("Native allocation recovery completed during " + phase), false);
          } else if (!failed) {
            resolve(undefined);
          }
        },
        (/** @type {Error} */ error) => fail(error, false)
      );
      const timeout = setTimeout(() => {
        fail(new Error("Native allocation recovery timed out during " + phase), true);
      }, 5000);
      worker.sendToCpp("test", cpuJob);
    });

    assert.deepEqual(errors, [expectedError]);
    assert.equal(results.length, retry ? 2 : 1);
    assert.ok(results[0]);
    if (retry) {
      assert.equal(results[0], results[1]);
    }
  }
});

for (const {noncebytes, nonces} of [
  {noncebytes: 4, nonces: ["abfffffc", "abfffffd", "abfffffe", "abffffff"]},
  {noncebytes: 8, nonces: [
    "abfffffffffffffc", "abfffffffffffffd", "abfffffffffffffe", "abffffffffffffff",
  ]},
]) {
  test(`native CPU/CN ${noncebytes}-byte protected-prefix nonce batches reach overflow`, {
    timeout: 5000,
  }, async () => {
    /** @type {string[]} */
    const results = [];
    /** @type {string[]} */
    const errors = [];
    const job = {
      algo: "cn/0",
      blob_hex: "00".repeat(39 + noncebytes),
      dev: "cpu*2",
      job_id: `protected-prefix-${noncebytes}`,
      job_token: "token",
      nicehash_mask: "ff" + "00".repeat(noncebytes - 1),
      nonce: nonces[0],
      noncebytes,
      nonceoffset: "39",
      pool_id: "pool",
      target: "ffffffff",
      worker_id: "worker",
    };
    await new Promise((resolve, reject) => {
      let overflow = false;
      let timedOut = false;
      const worker = new core.AsyncWorker(
        (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
          if (name === "result") {
            if (typeof values.nonce !== "string") {throw new Error("Native result omitted its nonce");}
            results.push(values.nonce);
          } else if (name === "error") {
            const message = typeof values.message === "string"
              ? values.message : "Native error omitted its message";
            errors.push(message);
            if (message === "Nonce overflow") {
              overflow = true;
              worker.sendToCpp("close");
            }
          }
        },
        () => {
          clearTimeout(timeout);
          if (timedOut) {
            reject(new Error("Protected-prefix nonce test timed out"));
          } else if (!overflow) {
            reject(new Error("Protected-prefix nonce test completed before overflow"));
          } else {
            resolve(undefined);
          }
        },
        (/** @type {Error} */ error) => {
          clearTimeout(timeout);
          reject(error);
        }
      );
      const timeout = setTimeout(() => {
        if (!overflow) {
          timedOut = true;
          worker.sendToCpp("close");
        }
      }, 1000);
      worker.sendToCpp("job", job);
    });
    assert.deepEqual(results, nonces);
    assert.deepEqual(errors, ["Nonce overflow"]);
  });
}

test("native NiceHash generic four-byte lanes preserve their trailing nonce slot", {
  timeout: 10000,
}, async () => {
  for (const {thread_id, nonces} of [
    {thread_id: 0, nonces: ["fffff8ab", "fffffaab", "fffffcab", "fffffeab"]},
    {thread_id: 1, nonces: ["fffff9ab", "fffffbab", "fffffdab", "ffffffab"]},
  ]) {
    /** @type {string[]} */
    const results = [];
    /** @type {string[]} */
    const errors = [];
    /** @type {string | undefined} */
    let lastNonce;
    const job = {
      algo: "cn/0",
      blob_hex: "00".repeat(39) + "000000ab",
      dev: "cpu*2",
      job_id: `trailing-slot-${thread_id}`,
      job_token: "token",
      nicehash_mask: "000000ff",
      nonce: "fffff800",
      noncebytes: 4,
      nonceoffset: 39,
      pool_id: "pool",
      target: "ffffffff",
      thread_id,
      thread_num: 2,
      worker_id: "worker",
    };
    await new Promise((resolve, reject) => {
      let closed = false;
      let expectedHash = "";
      let phase = "hash";
      const expectedLastNonce = nonces.at(-1);
      const close = () => {
        if (closed) {return;}
        closed = true;
        clearTimeout(timeout);
        worker.sendToCpp("close");
      };
      const worker = new core.AsyncWorker(
        (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
          try {
            if (phase === "hash" && name === "test") {
              if (typeof values.result !== "string" || values.result.length === 0) {
                throw new Error("Native test result was empty");
              }
              expectedHash = values.result;
              phase = "mining";
              worker.sendToCpp("job", job);
            } else if (phase === "mining" && name === "result") {
              if (typeof values.nonce !== "string") {
                throw new Error("Native result omitted its nonce");
              }
              if (values.nonce === nonces[0] && values.hash !== expectedHash) {
                throw new Error("Native result hash did not match its nonce-bearing blob");
              }
              results.push(values.nonce);
            } else if (phase === "mining" && name === "error") {
              if (typeof values.message !== "string") {
                throw new Error("Native error omitted its message");
              }
              errors.push(values.message);
              if (values.message === "Nonce overflow") {close();}
            } else if (closed && name === "last_nonce") {
              if (typeof values.nonce !== "string") {
                throw new Error("Native last_nonce omitted its nonce");
              }
              assert.equal(values.nonce, expectedLastNonce);
              if (values["pool_id"] !== undefined) {assert.equal(values["pool_id"], job.pool_id);}
              if (values.job_id !== undefined) {assert.equal(values.job_id, job.job_id);}
              if (values["job_token"] !== undefined) {assert.equal(values["job_token"], job.job_token);}
              if (lastNonce !== undefined) {
                throw new Error("Native emitted duplicate last_nonce");
              }
              lastNonce = values.nonce;
            } else {
              throw new Error("Unexpected native event: " + name);
            }
          } catch (error) {
            if (!closed) {close();}
            reject(error instanceof Error ? error : new Error(String(error)));
          }
        },
        () => resolve(undefined),
        reject
      );
      const timeout = setTimeout(() => {
        if (!closed) {close();}
        reject(new Error(`Trailing-slot worker ${thread_id} timed out`));
      }, 5000);
      worker.sendToCpp("test", {
        algo: "cn/0",
        blob_hex: "00".repeat(39) + nonces[0],
        dev: "cpu",
        noncebytes: 4,
        nonceoffset: 39,
      });
    });
    assert.deepEqual(results, nonces);
    assert.deepEqual(errors, ["Nonce overflow"]);
    assert.equal(lastNonce, nonces.at(-1));
  }
});

test("native NiceHash generic four-byte replacement ignores a stale supplied slot", {
  timeout: 5000,
}, async () => {
  /** @type {string[]} */
  const results = [];
  /** @type {string[]} */
  const errors = [];
  /** @type {string | undefined} */
  let lastNonce;
  const job = {
    algo: "cn/0",
    blob_hex: "00".repeat(39) + "000000ab",
    dev: "cpu",
    job_id: "trailing-slot-stale",
    job_token: "token",
    nicehash_mask: "000000ff",
    nonce: "ffffffcd",
    noncebytes: 4,
    nonceoffset: 39,
    pool_id: "pool",
    target: "ffffffff",
    worker_id: "worker",
  };
  const expectedLastNonce = "ffffffab";
  await new Promise((resolve, reject) => {
    let closed = false;
    const close = () => {
      if (closed) {return;}
      closed = true;
      clearTimeout(timeout);
      worker.sendToCpp("close");
    };
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        try {
          if (name === "result") {
            if (typeof values.nonce !== "string") {
              throw new Error("Native result omitted its nonce");
            }
            results.push(values.nonce);
          } else if (name === "error") {
            if (typeof values.message !== "string") {
              throw new Error("Native error omitted its message");
            }
            errors.push(values.message);
            if (values.message === "Nonce overflow") {close();}
          } else if (closed && name === "last_nonce") {
            if (typeof values.nonce !== "string") {
              throw new Error("Native last_nonce omitted its nonce");
            }
            assert.equal(values.nonce, expectedLastNonce);
            if (values["pool_id"] !== undefined) {assert.equal(values["pool_id"], job.pool_id);}
            if (values.job_id !== undefined) {assert.equal(values.job_id, job.job_id);}
            if (values["job_token"] !== undefined) {assert.equal(values["job_token"], job.job_token);}
            if (lastNonce !== undefined) {
              throw new Error("Native emitted duplicate last_nonce");
            }
            lastNonce = values.nonce;
          } else {
            throw new Error("Unexpected native event: " + name);
          }
        } catch (error) {
          if (!closed) {close();}
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      },
      () => resolve(undefined),
      reject
    );
    const timeout = setTimeout(() => {
      if (!closed) {close();}
      reject(new Error("Stale-slot worker timed out"));
    }, 2500);
    worker.sendToCpp("job", job);
  });
  assert.deepEqual(results, ["ffffffab"]);
  assert.deepEqual(errors, ["Nonce overflow"]);
  assert.equal(lastNonce, expectedLastNonce);
});

test("native RandomX trailing nonce slots finish two counters", {timeout: 30000}, async () => {
  /** @type {string[]} */
  const results = [];
  /** @type {string[]} */
  const errors = [];
  const job = {
    algo: "rx/0",
    blob_hex: "00".repeat(39) + "000000ab",
    dev: "cpu*2",
    job_id: "rx-trailing-slot",
    job_token: "token",
    nicehash_mask: "000000ff",
    nonce: "fffff800",
    noncebytes: 4,
    nonceoffset: 39,
    pool_id: "pool",
    seed_hex: "00".repeat(32),
    target: "ffffffff",
    thread_id: 0,
    thread_num: 2,
    worker_id: "worker",
  };
  await new Promise((resolve, reject) => {
    let closed = false;
    const close = () => {
      if (closed) {return;}
      closed = true;
      clearTimeout(timeout);
      worker.sendToCpp("close");
    };
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        try {
          if (name === "result") {
            if (typeof values.nonce !== "string") {
              throw new Error("Native result omitted its nonce");
            }
            results.push(values.nonce);
            if (results.length === 4 && errors.length === 2) {close();}
          } else if (name === "error") {
            if (typeof values.message !== "string") {
              throw new Error("Native error omitted its message");
            }
            errors.push(values.message);
            if (results.length === 4 && errors.length === 2) {close();}
          } else {
            throw new Error("Unexpected native event: " + name);
          }
        } catch (error) {
          if (!closed) {close();}
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      },
      () => resolve(undefined),
      reject
    );
    const timeout = setTimeout(() => {
      if (!closed) {close();}
      reject(new Error("RandomX trailing-slot worker timed out"));
    }, 20000);
    worker.sendToCpp("job", job);
  });
  assert.deepEqual(results.sort(), ["fffff8ab", "fffff9ab", "fffffcab", "fffffdab"]);
  assert.deepEqual(errors, ["Nonce overflow", "Nonce overflow"]);
});

test("native RandomX rejects scaled initial and step overflow", {timeout: 5000}, async () => {
  const baseJob = {
    algo: "rx/0",
    blob_hex: "00".repeat(39) + "000000ab",
    job_id: "rx-invalid-range",
    job_token: "token",
    nicehash_mask: "000000ff",
    noncebytes: 4,
    nonceoffset: 39,
    pool_id: "pool",
    seed_hex: "00".repeat(32),
    target: "ffffffff",
    worker_id: "worker",
  };
  for (const {job, message} of [
    {job: {...baseJob, dev: "cpu*2", nonce: "ffffff00", thread_num: 1},
      message: /Initial RandomX worker nonce overflows/},
    {job: {...baseJob, dev: "cpu*16777216", nonce: "0", thread_num: 1},
      message: /Worker nonce step is too large/},
  ]) {
    let received = "";
    await new Promise((resolve, reject) => {
      const worker = new core.AsyncWorker(
        (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
          if (name !== "error") {
            reject(new Error("Expected RandomX validation error"));
            return;
          }
          if (typeof values.message !== "string") {
            reject(new Error("Native error omitted its message"));
            return;
          }
          received = values.message;
          clearTimeout(timeout);
          worker.sendToCpp("close");
        },
        () => {
          clearTimeout(timeout);
          resolve(undefined);
        },
        reject
      );
      const timeout = setTimeout(() => {
        worker.sendToCpp("close");
        reject(new Error("RandomX validation timed out"));
      }, 2500);
      worker.sendToCpp("job", job);
    });
    assert.match(received, message);
  }
});

test("native RandomX labels its first result and resumes the same job", {timeout: 15000}, async () => {
  const job = {
    algo: "rx/0",
    blob_hex: "00".repeat(43),
    dev: "cpu",
    job_token: "token",
    nonce: "0",
    noncebytes: "4",
    nonceoffset: "39",
    pool_id: "pool",
    seed_hex: "00".repeat(32),
    target: "ffffffff",
    worker_id: "worker",
  };

  await new Promise((resolve, reject) => {
    let phase = "test";
    /** @type {string | undefined} */
    let expectedHash;
    const worker = new core.AsyncWorker(
      (/** @type {string} */ name, /** @type {NativeValues} */ values) => {
        if (phase === "test" && name === "test") {
          expectedHash = values.result;
          phase = "first";
          worker.sendToCpp("job", {...job, job_id: "first"});
        } else if (phase === "first" && name === "result" && values.job_id === "first") {
          assert.equal(values.nonce, "00000000");
          assert.equal(values.hash, expectedHash);
          phase = "resumed";
          worker.sendToCpp("pause");
          worker.sendToCpp("job", {...job, job_id: "resumed"});
        } else if (phase === "resumed" && name === "result" && values.job_id === "resumed") {
          phase = "done";
          clearTimeout(timeout);
          worker.sendToCpp("close");
        } else if (name === "error") {
          const message = typeof values.message === "string"
            ? values.message : "Native error omitted its message";
          phase = "error";
          clearTimeout(timeout);
          worker.sendToCpp("close");
          reject(new Error(message));
        }
      },
      () => {
        clearTimeout(timeout);
        if (phase === "done") {
          resolve(undefined);
        } else {
          reject(new Error(`RandomX resume stopped in phase ${phase}`));
        }
      },
      reject
    );
    const timeout = setTimeout(() => {
      phase = "timeout";
      worker.sendToCpp("close");
    }, 10000);
    worker.sendToCpp("test", job);
  });
});
