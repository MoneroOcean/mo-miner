"use strict";

const rocrSignalPoolWarning =
  /^Warning: Resource leak detected by SharedSignalPool, \d+ Signals leaked\.\r?$/;
const ANSI_SGR_PREFIX = String.fromCharCode(27) + "[";
const acppBufferAdvisory =
  /^\[AdaptiveCpp Warning\] This application uses SYCL buffers; the SYCL buffer-accessor model is well-known to introduce unnecessary overheads\. Please consider migrating to the SYCL2020 USM model, in particular device USM \(sycl::malloc_device\) combined with in-order queues for more performance\. See the AdaptiveCpp performance guide for more information: ?\r?$/;
const acppBufferAdvisoryUrl =
  /^https:\/\/github\.com\/AdaptiveCpp\/AdaptiveCpp\/blob\/develop\/doc\/performance\.md\r?$/;
const acppJitAdvisory =
  /^\[AdaptiveCpp Warning\] kernel_cache: This application run has resulted in new binaries being JIT-compiled\. This indicates that the runtime optimization process has not yet reached peak performance\. You may want to run the application again until this warning no longer appears to achieve optimal performance\.\r?$/;
const llvmPtx88FallbackAdvisory =
  /^'\+ptx88' is not a recognized feature for this target \(ignoring feature\)\r?$/;
const windowsHipLibraryPath =
  /^HIP Library Path: [A-Za-z]:\\.*\\amdhip64(?:_\d+)?\.dll\r?$/i;
const MAX_PENDING_STDERR = 64 * 1024;

/** @param {string} line */
function hideWorkerStderrLine(line) {
  const plainLine = line.split(ANSI_SGR_PREFIX).map(function(part, index) {
    if (index === 0) {return part;}
    const end = part.indexOf("m");
    if (end >= 0 && /^[0-9;]*$/.test(part.slice(0, end))) {return part.slice(end + 1);}
    return ANSI_SGR_PREFIX + part;
  }).join("");
  return rocrSignalPoolWarning.test(plainLine) ||
         acppBufferAdvisory.test(plainLine) ||
         acppBufferAdvisoryUrl.test(plainLine) ||
         acppJitAdvisory.test(plainLine) ||
         llvmPtx88FallbackAdvisory.test(plainLine);
}

// Ubuntu ships ROCr with its debug-only SharedSignalPool accounting enabled. hsa_shut_down() prints
// this line while tearing down short-lived HIP workers, immediately before freeing the complete
// pool; it is not a persistent host/GPU leak. AdaptiveCpp's buffer text currently comes from C29's
// buffer/accessor implementation; a measured USM conversion is performance work, not an actionable
// runtime failure. The exact kernel-cache summary likewise only says that this process used a
// not-yet-cached kernel; the AppDB still persists that object for later processes. Keep arbitrary
// stderr byte-for-byte while suppressing only these exact, understood diagnostics (including their
// optional ANSI coloring), even when a stream chunk splits one across writes. AdaptiveCpp develop
// requests PTX 8.8 for sm_120, while its Windows LLVM 20.1.8 runtime supports through PTX 8.7.
// LLVM's exact advisory is harmless here: the emitted JIT object is still `.version 8.7` with
// `.target sm_120` (the PTX version that introduced sm_120). Keep every other target-feature line.
/** @param {string} pending @param {string | Buffer} chunk @param {boolean} [flush] */
function filterWorkerStderr(pending, chunk, flush = false) {
  let input = pending + chunk.toString();
  let visible = "";
  let eol;
  while ((eol = input.indexOf("\n")) !== -1) {
    const line = input.slice(0, eol);
    input = input.slice(eol + 1);
    if (!hideWorkerStderrLine(line)) {visible += line + "\n";}
  }
  if (flush) {
    if (input && !hideWorkerStderrLine(input)) {visible += input;}
    input = "";
  } else if (input.length > MAX_PENDING_STDERR) {
    visible += input.slice(0, input.length - MAX_PENDING_STDERR);
    input = input.slice(-MAX_PENDING_STDERR);
  }
  return {pending: input, visible};
}

module.exports.filterWorkerStderr = filterWorkerStderr;
/** @param {string} line */
module.exports.filterWorkerStdoutLine = function(line) {
  return windowsHipLibraryPath.test(line) ? "" : line;
};
