"use strict";

const rocr_signal_pool_warning =
  /^Warning: Resource leak detected by SharedSignalPool, \d+ Signals leaked\.\r?$/;
const ansi_sgr_prefix = String.fromCharCode(27) + "[";
const acpp_buffer_advisory =
  /^\[AdaptiveCpp Warning\] This application uses SYCL buffers; the SYCL buffer-accessor model is well-known to introduce unnecessary overheads\. Please consider migrating to the SYCL2020 USM model, in particular device USM \(sycl::malloc_device\) combined with in-order queues for more performance\. See the AdaptiveCpp performance guide for more information: ?\r?$/;
const acpp_buffer_advisory_url =
  /^https:\/\/github\.com\/AdaptiveCpp\/AdaptiveCpp\/blob\/develop\/doc\/performance\.md\r?$/;
const acpp_jit_advisory =
  /^\[AdaptiveCpp Warning\] kernel_cache: This application run has resulted in new binaries being JIT-compiled\. This indicates that the runtime optimization process has not yet reached peak performance\. You may want to run the application again until this warning no longer appears to achieve optimal performance\.\r?$/;
const llvm_ptx88_fallback_advisory =
  /^'\+ptx88' is not a recognized feature for this target \(ignoring feature\)\r?$/;
const windows_hip_library_path =
  /^HIP Library Path: [A-Za-z]:\\.*\\amdhip64(?:_\d+)?\.dll\r?$/i;

function hideWorkerStderrLine(line) {
  const plainLine = line.split(ansi_sgr_prefix).map(function(part, index) {
    if (index === 0) {return part;}
    const end = part.indexOf("m");
    if (end >= 0 && /^[0-9;]*$/.test(part.slice(0, end))) {return part.slice(end + 1);}
    return ansi_sgr_prefix + part;
  }).join("");
  return rocr_signal_pool_warning.test(plainLine) ||
         acpp_buffer_advisory.test(plainLine) ||
         acpp_buffer_advisory_url.test(plainLine) ||
         acpp_jit_advisory.test(plainLine) ||
         llvm_ptx88_fallback_advisory.test(plainLine);
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
function filterWorkerStderr(pending, chunk, flush = false) {
  let input = pending + chunk;
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
  }
  return {pending: input, visible};
}

module.exports.filterWorkerStderr = filterWorkerStderr;
module.exports.filterWorkerStdoutLine = function(line) {
  return windows_hip_library_path.test(line) ? "" : line;
};
