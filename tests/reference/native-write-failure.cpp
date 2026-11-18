// Linker-only fault injection; never compiled into a release or the normal parity executable.
// A large encrypted write fails synchronously once, exercising TLS's deferred OS-error boundary.
#include <uv.h>
#include <cstring>
#include <cstdlib>

extern "C" int __real_uv_write(uv_write_t *, uv_stream_t *, const uv_buf_t[], unsigned, uv_write_cb);
extern "C" int __real_uv_spawn(uv_loop_t *, uv_process_t *, const uv_process_options_t *);

// Exercise libuv's real failed-spawn initialization/close contract, not an uninitialized fake
// handle. Only this test binary substitutes a nonexistent image; no process is launched there.
extern "C" int __wrap_uv_spawn(uv_loop_t *loop, uv_process_t *process, const uv_process_options_t *options) {
  const char *mode = std::getenv("MOM_AOT_FAILURE");
  if (mode && std::strcmp(mode, "spawn") == 0) {
    auto missing = *options;
    missing.file = "/__mom_missing_spawn_fixture__";
    return __real_uv_spawn(loop, process, &missing);
  }
  return __real_uv_spawn(loop, process, options);
}
extern "C" int __wrap_uv_write(uv_write_t *request, uv_stream_t *stream, const uv_buf_t buffers[],
                              unsigned count, uv_write_cb callback) {
  static bool failed = false;
  const char *mode = std::getenv("MOM_AOT_FAILURE");
  if (!failed && stream->type == UV_TCP && mode && std::strcmp(mode, "tlsWrite") == 0 && count && buffers[0].len >= 65536) {
    failed = true;
    return UV_EPIPE;
  }
  return __real_uv_write(request, stream, buffers, count, callback);
}
