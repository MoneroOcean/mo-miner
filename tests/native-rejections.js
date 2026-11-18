"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {EventEmitter} = require("node:events");
const rejectionTracker = require("../runtime/rejections");

test.describe("Native Promise host rejection policy", () => {
  test("an empty checkpoint is inert", () => {assert.equal(rejectionTracker(new EventEmitter()).drain(), false);});
  test("a rejected Error fails with the original identity", () => {
    const tracker = rejectionTracker(new EventEmitter()), error = new Error("fixture");
    tracker.track({}, error, false); assert.throws(() => tracker.drain(), (actual) => actual === error);
  });
  test("a primitive rejection has the default Node failure code", () => {
    const tracker = rejectionTracker(new EventEmitter()); tracker.track({}, "fixture", false);
    assert.throws(() => tracker.drain(), {code: "ERR_UNHANDLED_REJECTION", message: "Unhandled promise rejection: fixture"});
  });
  test("a catch attached before the checkpoint suppresses notification", () => {
    const tracker = rejectionTracker(new EventEmitter()), promise = {};
    tracker.track(promise, "fixture", false); tracker.track(promise, undefined, true); assert.equal(tracker.drain(), false);
  });
  test("an unhandledRejection listener receives exact reason and Promise identities", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), promise = {}, reason = {};
    let count = 0; events.on("unhandledRejection", (error, rejected) => {
      assert.equal(error, reason); assert.equal(rejected, promise); count++;
    });
    tracker.track(promise, reason, false); assert.equal(tracker.drain(), true); assert.equal(tracker.drain(), false); assert.equal(count, 1);
  });
  test("late handling reports the original Promise exactly once", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), promise = {};
    events.on("unhandledRejection", () => undefined);
    /** @type {object[]} */
    const late = []; events.on("rejectionHandled", (value) => late.push(value));
    tracker.track(promise, "fixture", false); tracker.drain();
    tracker.track(promise, undefined, true); tracker.track(promise, undefined, true);
    assert.equal(tracker.drain(), true); assert.deepEqual(late, [promise]); assert.equal(tracker.drain(), false);
  });
  test("a listener can handle another pending rejection before it is reported", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), first = {}, second = {};
    let count = 0; events.on("unhandledRejection", () => {count++; tracker.track(second, undefined, true);});
    tracker.track(first, 1, false); tracker.track(second, 2, false); tracker.drain(); assert.equal(count, 1);
  });
  test("reentrant rejections wait for the next microtask checkpoint", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), next = {};
    let count = 0; events.on("unhandledRejection", () => {if (++count === 1) {tracker.track(next, 2, false);}});
    tracker.track({}, 1, false); tracker.drain(); assert.equal(count, 1); tracker.drain(); assert.equal(count, 2);
  });
  test("reentrant late handlers are delivered in the next batch", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), promise = {};
    let count = 0; events.on("unhandledRejection", () => tracker.track(promise, undefined, true));
    events.on("rejectionHandled", () => count++);
    tracker.track(promise, 1, false); tracker.drain(); assert.equal(count, 0); tracker.drain(); assert.equal(count, 1);
  });
  test("a thrown rejection listener escapes instead of silently resolving", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), error = new Error("listener");
    events.on("unhandledRejection", () => {throw error;}); tracker.track({}, 1, false);
    assert.throws(() => tracker.drain(), (actual) => actual === error);
  });
  test("uncaughtException recovery receives the correct origin", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), error = new Error("fixture");
    let received; events.on("uncaughtException", (value, origin) => {received = value; assert.equal(origin, "unhandledRejection");});
    tracker.track({}, error, false); tracker.drain(); assert.equal(received, error);
  });
  test("the uncaughtException monitor observes but does not suppress failure", () => {
    const events = new EventEmitter(), tracker = rejectionTracker(events), error = new Error("fixture");
    let observed; events.on("uncaughtExceptionMonitor", (value) => {observed = value;}); tracker.track({}, error, false);
    assert.throws(() => tracker.drain(), (actual) => actual === error); assert.equal(observed, error);
  });
});
