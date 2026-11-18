"use strict";

const {spawn} = require("node:child_process");

// Native-only teardown injection: the parent fails while an actual child ignores stdin. The
// driver confirms its exact PID has been reaped, not just that the controller printed an error.
/** @param {boolean} [promiseFailure] */
module.exports = function childFailure(promiseFailure = false) {
  const child = spawn(process.execPath, [], {env: {...process.env, MOM_AOT_CHILD_TASK: "hang"}});
  let stdout = "";
  child.on("error", (error) => {throw error;});
  child.stdin.on("error", (error) => {throw error;});
  child.stderr.on("data", () => undefined);
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (text) => {
    stdout += text;
    if (stdout.includes("ready\n")) {
      console.log("AOT_CHILD_TEARDOWN_PID " + child.pid);
      setTimeout(() => {
        const error = new Error("EXPECTED_CHILD_TEARDOWN_FAILURE");
        if (promiseFailure) {Promise.reject(error);} else {throw error;}
      }, 1);
    }
  });
  child.stdin.end();
  setTimeout(() => {throw new Error("Child teardown fixture did not become ready");}, 5000);
};
