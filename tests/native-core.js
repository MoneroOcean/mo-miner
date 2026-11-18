"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {AsyncWorker} = require("../runtime/core");

test.describe("Native compute bridge adapter", {concurrency: false}, () => {
  /** @type {typeof globalThis.__momWorkerCreate | undefined} */
  let saved;
  /** @type {Array<[string, Array<[string, string]>]>} */
  let messages;
  const noop = () => undefined;
  test.beforeEach(() => {
    saved = globalThis.__momWorkerCreate;
    messages = [];
    globalThis.__momWorkerCreate = () => ({post: (name, values) => {messages.push([name, values]);}});
  });
  test.afterEach(() => {
    if (saved) {globalThis.__momWorkerCreate = saved;} else {Reflect.deleteProperty(globalThis, "__momWorkerCreate");}
  });
  function worker() {return new AsyncWorker(noop, noop, noop);}
  test("constructor requires three callbacks", () => {
    assert.throws(() => Reflect.construct(AsyncWorker, [noop, noop]), /requires progress, complete, and error/);
  });
  test("constructor rejects non-function callbacks before creating native state", () => {
    let created = false; globalThis.__momWorkerCreate = () => {created = true; throw new Error("unexpected");};
    assert.throws(() => Reflect.construct(AsyncWorker, [noop, 1, noop]), /callbacks must be functions/);
    assert.equal(created, false);
  });
  test("callbacks pass directly to native ownership without capturing the JS wrapper", () => {
    /** @type {unknown[]} */
    let callbacks = [];
    globalThis.__momWorkerCreate = (...args) => {callbacks = args; return {post: noop};};
    worker(); assert.deepEqual(callbacks, [noop, noop, noop]);
  });
  test("message name is required and must be a string", () => {
    const instance = worker();
    assert.throws(() => Reflect.apply(instance.sendToCpp, instance, []), /requires a message name/);
    assert.throws(() => Reflect.apply(instance.sendToCpp, instance, [1]), /message name must be a string/);
    assert.deepEqual(messages, []);
  });
  test("omitted, undefined and null payloads produce empty message maps", () => {
    const instance = worker(); instance.sendToCpp("pause"); instance.sendToCpp("pause", undefined); instance.sendToCpp("pause", null);
    assert.deepEqual(messages, [["pause", []], ["pause", []], ["pause", []]]);
  });
  test("primitive and function payloads cannot reach native code", () => {
    const instance = worker();
    for (const value of [1, "bad", true, noop]) {
      assert.throws(() => Reflect.apply(instance.sendToCpp, instance, ["job", value]), /values must be an object/);
    }
    assert.deepEqual(messages, []);
  });
  test("message fields preserve the Node addon's string coercion", () => {
    worker().sendToCpp("job", {height: 42, enabled: false, absent: undefined, empty: null, big: 1n << 80n});
    assert.deepEqual(messages[0]?.[1], [["height", "42"], ["enabled", "false"], ["absent", "undefined"],
      ["empty", "null"], ["big", (1n << 80n).toString()]]);
  });
  test("enumerable inherited fields match the development message map", () => {
    const values = Object.assign(Object.create({parent: "inherited"}), {own: "value"});
    worker().sendToCpp("job", values);
    assert.deepEqual(messages[0]?.[1], [["own", "value"], ["parent", "inherited"]]);
  });
  test("symbol values fail before submitting a partial native message", () => {
    assert.throws(() => worker().sendToCpp("job", {valid: 1, bad: Symbol("bad")}), TypeError);
    assert.deepEqual(messages, []);
  });
  test("throwing property accessors cannot submit a partial message", () => {
    assert.throws(() => worker().sendToCpp("job", {get bad() {throw new Error("getter");}}), /getter/);
    assert.deepEqual(messages, []);
  });
  test("native stopped/errors propagate without creating replacement workers", () => {
    let starts = 0;
    globalThis.__momWorkerCreate = () => {starts++; return {post() {throw new Error("AsyncWorker is stopped");}};};
    assert.throws(() => worker().sendToCpp("job"), /is stopped/); assert.equal(starts, 1);
  });
  test("Unicode and embedded NUL field values are preserved", () => {
    worker().sendToCpp("job", {value: "é-💎\0tail"}); assert.equal(messages[0]?.[1][0]?.[1], "é-💎\0tail");
  });
});
