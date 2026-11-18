// Link-only test interposer for our synchronous libuv file API. No injected shared library.
#include <uv.h>
#include <fcntl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int __real_uv_fs_open(uv_loop_t *, uv_fs_t *, const char *, int, int, uv_fs_cb);
extern "C" int __real_uv_fs_close(uv_loop_t *, uv_fs_t *, uv_file, uv_fs_cb);
extern "C" int __real_uv_fs_write(uv_loop_t *, uv_fs_t *, uv_file, const uv_buf_t[], unsigned, int64_t, uv_fs_cb);
static int target_fd = -1;
static unsigned opened = 0, writes = 0, closed = 0;
static bool mode(const char *value) {
  const char *actual = std::getenv("MOM_AOT_FILE_FAULT");
  return actual && std::strcmp(actual, value) == 0;
}

extern "C" int __wrap_uv_fs_open(uv_loop_t *loop, uv_fs_t *req, const char *path, int flags, int permissions, uv_fs_cb cb) {
  const char *target = std::getenv("MOM_AOT_FILE_FAULT_PATH");
  const bool selected = target && std::strcmp(target, path) == 0 && (flags & O_WRONLY);
  if (selected && mode("open")) {req->result = UV_EACCES; return UV_EACCES;}
  const int result = __real_uv_fs_open(loop, req, path, flags, permissions, cb);
  if (selected && result >= 0) {target_fd = result; ++opened;}
  return result;
}

extern "C" int __wrap_uv_fs_write(uv_loop_t *loop, uv_fs_t *req, uv_file fd, const uv_buf_t buffers[], unsigned count, int64_t offset, uv_fs_cb cb) {
  if (fd == target_fd) {
    ++writes;
    if (mode("zero") || mode("space")) {
      req->result = mode("zero") ? 0 : UV_ENOSPC; return static_cast<int>(req->result);
    }
    if (mode("partial") && count == 1) {
      uv_buf_t part = buffers[0]; part.len = std::min<size_t>(part.len, 3);
      return __real_uv_fs_write(loop, req, fd, &part, 1, offset, cb);
    }
  }
  return __real_uv_fs_write(loop, req, fd, buffers, count, offset, cb);
}

extern "C" int __wrap_uv_fs_close(uv_loop_t *loop, uv_fs_t *req, uv_file fd, uv_fs_cb cb) {
  const bool selected = fd == target_fd;
  if (selected) {++closed; target_fd = -1;}
  const int result = __real_uv_fs_close(loop, req, fd, cb);
  if (selected && mode("close") && !result) {req->result = UV_EIO; return UV_EIO;}
  return result;
}

// Observe leak/double-close evidence after the actual language/runtime teardown.
[[gnu::destructor]] static void report() {
  if (std::getenv("MOM_AOT_FILE_FAULT")) {
    std::fprintf(stderr, "FILESYSTEM_FAULT opened=%u writes=%u closed=%u live=%d\n", opened, writes, closed, target_fd);
  }
}
