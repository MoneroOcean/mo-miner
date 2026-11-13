#include <uv.h>

#include <cstdlib>
#include <cstring>
#include <unordered_map>

extern "C" int __real_uv_getaddrinfo(uv_loop_t *, uv_getaddrinfo_t *, uv_getaddrinfo_cb,
    const char *, const char *, const struct addrinfo *);
extern "C" int __real_uv_cancel(uv_req_t *);
extern "C" int __real_uv_tcp_connect(uv_connect_t *, uv_tcp_t *, const struct sockaddr *,
    uv_connect_cb);
extern "C" void __real_uv_close(uv_handle_t *, uv_close_cb);

namespace {
bool enabled(const char *name) {return std::getenv(name) != nullptr;}

void append(struct addrinfo *&head, struct addrinfo *&tail, struct addrinfo *entry) {
  entry->ai_next = nullptr;
  if (tail) {tail->ai_next = entry;} else {head = entry;}
  tail = entry;
}

struct addrinfo *reorder(struct addrinfo *addresses) {
  struct addrinfo *ipv4 = nullptr, *ipv4Tail = nullptr;
  struct addrinfo *ipv6 = nullptr, *ipv6Tail = nullptr;
  struct addrinfo *other = nullptr, *otherTail = nullptr;
  for (auto *entry = addresses; entry;) {
    auto *next = entry->ai_next;
    if (entry->ai_family == AF_INET) {append(ipv4, ipv4Tail, entry);}
    else if (entry->ai_family == AF_INET6) {append(ipv6, ipv6Tail, entry);}
    else {append(other, otherTail, entry);}
    entry = next;
  }
  struct addrinfo *head = nullptr, *tail = nullptr;
  for (auto *entry = ipv4; entry;) {
    auto *next = entry->ai_next; append(head, tail, entry); entry = next;
  }
  for (auto *entry = ipv6; entry;) {
    auto *next = entry->ai_next; append(head, tail, entry); entry = next;
  }
  for (auto *entry = other; entry;) {
    auto *next = entry->ai_next; append(head, tail, entry); entry = next;
  }
  return head;
}

struct ResolveProxy {
  uv_getaddrinfo_t localhost{};
  uv_getaddrinfo_t ipv6Localhost{};
  uv_getaddrinfo_t *caller = nullptr;
  uv_getaddrinfo_cb callback = nullptr;
  struct addrinfo *localhostAddresses = nullptr;
  struct addrinfo *ipv6LocalhostAddresses = nullptr;
  int localhostStatus = 0;
  int ipv6LocalhostStatus = 0;
  unsigned pending = 0;
  bool localhostSubmitted = false;
  bool ipv6LocalhostSubmitted = false;
};
std::unordered_map<uv_req_t *, ResolveProxy *> activeResolveProxies;

// This Linux-only audit links the two glibc addrinfo chains into one list, which the production
// callback releases with its single uv_freeaddrinfo call. It is deliberately not a Windows wrapper.
void resolveProxyCallback(uv_getaddrinfo_t *request, int status, struct addrinfo *addresses) {
  auto *proxy = static_cast<ResolveProxy *>(request->data);
  if (request == &proxy->localhost) {
    proxy->localhostSubmitted = false;
    proxy->localhostStatus = status;
    proxy->localhostAddresses = addresses;
  } else {
    proxy->ipv6LocalhostSubmitted = false;
    proxy->ipv6LocalhostStatus = status;
    proxy->ipv6LocalhostAddresses = addresses;
  }
  if (--proxy->pending) {return;}
  if (proxy->localhostAddresses) {
    auto *tail = proxy->localhostAddresses;
    while (tail->ai_next) {tail = tail->ai_next;}
    tail->ai_next = proxy->ipv6LocalhostAddresses;
  } else {
    proxy->localhostAddresses = proxy->ipv6LocalhostAddresses;
  }
  if (proxy->localhostAddresses) {proxy->localhostAddresses = reorder(proxy->localhostAddresses);}
  const int finalStatus = proxy->localhostStatus == 0 || proxy->ipv6LocalhostStatus == 0
      ? 0 : proxy->localhostStatus;
  activeResolveProxies.erase(reinterpret_cast<uv_req_t *>(proxy->caller));
  proxy->callback(proxy->caller, finalStatus, proxy->localhostAddresses);
  delete proxy;
}

bool timeoutInjected = false;
struct PendingConnect {
  uv_connect_t *request = nullptr;
  uv_tcp_t *tcp = nullptr;
  uv_connect_cb callback = nullptr;
  uv_timer_t timer{};
};
PendingConnect *pendingConnect = nullptr;

void cancelPendingConnect(uv_tcp_t *tcp) {
  if (!pendingConnect || pendingConnect->tcp != tcp) {return;}
  auto *state = pendingConnect;
  pendingConnect = nullptr;
  uv_timer_stop(&state->timer);
  __real_uv_close(reinterpret_cast<uv_handle_t *>(&state->timer), [](uv_handle_t *handle) {
    delete static_cast<PendingConnect *>(handle->data);
  });
}

void releasePendingConnect(uv_timer_t *timer) {
  auto *state = static_cast<PendingConnect *>(timer->data);
  if (pendingConnect == state) {pendingConnect = nullptr;}
  // Make the callback observe the same request data that libuv would have supplied, while the
  // wrapper-owned timer keeps the historical first attempt alive long enough to expose the bug.
  state->callback(state->request, UV_ETIMEDOUT);
  __real_uv_close(reinterpret_cast<uv_handle_t *>(timer), [](uv_handle_t *handle) {
    delete static_cast<PendingConnect *>(handle->data);
  });
}

struct DelayedClose {
  uv_handle_t *target = nullptr;
  uv_close_cb callback = nullptr;
  uv_timer_t timer{};
  uint64_t delay = 0;
};
DelayedClose *delayedClose = nullptr;

void releaseDelayedClose(uv_timer_t *timer) {
  auto *state = static_cast<DelayedClose *>(timer->data);
  state->callback(state->target);
  __real_uv_close(reinterpret_cast<uv_handle_t *>(timer), [](uv_handle_t *handle) {
    auto *state = static_cast<DelayedClose *>(handle->data);
    delayedClose = nullptr;
    delete state;
  });
}

void holdClosedHandle(uv_handle_t *target) {
  auto *state = delayedClose;
  if (!state) {return;}
  const int status = uv_timer_init(target->loop, &state->timer);
  if (status) {
    state->callback(target);
    delete state;
    delayedClose = nullptr;
    return;
  }
  state->timer.data = state;
  const auto requested = std::strtoull(std::getenv("MOM_DUALSTACK_DELAY_FIRST_CLOSE_MS"), nullptr, 10);
  state->delay = requested > 0 && requested < 1000 ? requested : 100;
  uv_timer_start(&state->timer, releaseDelayedClose, state->delay, 0);
}

} // namespace

extern "C" int __wrap_uv_getaddrinfo(uv_loop_t *loop, uv_getaddrinfo_t *caller,
    uv_getaddrinfo_cb callback, const char *node, const char *service, const struct addrinfo *hints) {
  if (!enabled("MOM_DUALSTACK_REORDER") || !node || std::strcmp(node, "dualstack.test") != 0) {
    return __real_uv_getaddrinfo(loop, caller, callback, node, service, hints);
  }
  auto *proxy = new ResolveProxy;
  proxy->caller = caller;
  proxy->callback = callback;
  proxy->localhost.data = proxy;
  proxy->ipv6Localhost.data = proxy;
  proxy->pending = 2;
  activeResolveProxies.emplace(reinterpret_cast<uv_req_t *>(caller), proxy);
  const int localhostStatus = __real_uv_getaddrinfo(loop, &proxy->localhost, resolveProxyCallback,
      "localhost", service, hints);
  if (localhostStatus) {
    proxy->localhostStatus = localhostStatus;
    --proxy->pending;
  } else {proxy->localhostSubmitted = true;}
  const int ipv6Status = __real_uv_getaddrinfo(loop, &proxy->ipv6Localhost, resolveProxyCallback,
      "ip6-localhost", service, hints);
  if (ipv6Status) {
    proxy->ipv6LocalhostStatus = ipv6Status;
    --proxy->pending;
  } else {proxy->ipv6LocalhostSubmitted = true;}
  if (proxy->pending) {return 0;}
  const int status = proxy->localhostStatus ? proxy->localhostStatus : proxy->ipv6LocalhostStatus;
  activeResolveProxies.erase(reinterpret_cast<uv_req_t *>(caller));
  delete proxy;
  return status;
}

extern "C" int __wrap_uv_cancel(uv_req_t *request) {
  const auto found = activeResolveProxies.find(request);
  if (found != activeResolveProxies.end()) {
    auto *proxy = found->second;
    // The production request is a caller-owned token; this wrapper submitted two proxy requests
    // instead. Only cancel requests that actually returned success from uv_getaddrinfo and have
    // not delivered their callback—uv_cancel on an unsubmitted/reused request is invalid.
    int result = 0;
    bool attempted = false;
    if (proxy->localhostSubmitted) {
      attempted = true;
      const int status = __real_uv_cancel(reinterpret_cast<uv_req_t *>(&proxy->localhost));
      if (status && !result) {result = status;}
    }
    if (proxy->ipv6LocalhostSubmitted) {
      attempted = true;
      const int status = __real_uv_cancel(reinterpret_cast<uv_req_t *>(&proxy->ipv6Localhost));
      if (status && !result) {result = status;}
    }
    return attempted ? result : 0;
  }
  return __real_uv_cancel(request);
}

extern "C" int __wrap_uv_tcp_connect(uv_connect_t *request, uv_tcp_t *tcp,
    const struct sockaddr *address, uv_connect_cb callback) {
  if (enabled("MOM_DUALSTACK_TIMEOUT_FIRST") && !timeoutInjected && address &&
      address->sa_family == AF_INET) {
    timeoutInjected = true;
    // Deliberately leave the first IPv4 connect pending. This is a test-only fault injection,
    // labelled as simulated pending-connect; production still calls libuv directly.
    auto *state = new PendingConnect{request, tcp, callback};
    const int status = uv_timer_init(tcp->loop, &state->timer);
    if (status) {delete state; return status;}
    state->timer.data = state;
    const int start = uv_timer_start(&state->timer, releasePendingConnect, 100, 0);
    if (start) {
      __real_uv_close(reinterpret_cast<uv_handle_t *>(&state->timer), [](uv_handle_t *handle) {
        delete static_cast<PendingConnect *>(handle->data);
      });
      return start;
    }
    pendingConnect = state;
    return 0;
  }
  return __real_uv_tcp_connect(request, tcp, address, callback);
}

extern "C" void __wrap_uv_close(uv_handle_t *handle, uv_close_cb callback) {
  if (pendingConnect && handle == reinterpret_cast<uv_handle_t *>(pendingConnect->tcp)) {
    cancelPendingConnect(reinterpret_cast<uv_tcp_t *>(handle));
  }
  if (enabled("MOM_DUALSTACK_DELAY_FIRST_CLOSE_MS") && handle && handle->type == UV_TCP && !delayedClose) {
    delayedClose = new DelayedClose{handle, callback};
    __real_uv_close(handle, [](uv_handle_t *target) {holdClosedHandle(target);});
    return;
  }
  if (delayedClose && handle == delayedClose->target) {return;}
  __real_uv_close(handle, callback);
}
