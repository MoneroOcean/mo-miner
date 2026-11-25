#pragma once

#include <openssl/ssl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace mom::runtime {
// Shared initialization must precede RAND/EVP as well as TLS; never load host config/providers.
void initializeCrypto();
// Node's IPv6 zone syntax after hostname[zone] == '%': one nonempty run of ASCII alnum, '-', '.'
// or ':'. Both net.isIP and TLS identity use it, so a malformed scope can never turn a DNS/IPv4
// name into an IP literal or certificate IP identity.
bool validIpv6Scope(const std::string &hostname, size_t zone);
class TlsError final : public std::runtime_error {
 public:
  std::string code;
  TlsError(std::string code, std::string message)
      : std::runtime_error(std::move(message)), code(std::move(code)) {}
};

// Memory BIOs keep TLS independent of OS sockets. The event loop alone owns encryption state;
// ciphertext and plaintext must never share an output path or escape before peer verification.
class TlsSession {
  std::unique_ptr<SSL_CTX, decltype(&SSL_CTX_free)> context_{nullptr, SSL_CTX_free};
  std::unique_ptr<SSL, decltype(&SSL_free)> ssl_{nullptr, SSL_free};
  bool rejectUnauthorized_;
  long verificationError_ = X509_V_OK;
  [[noreturn]] void failure(int error, bool checkingCertificate = false);
  static int verify(int valid, X509_STORE_CTX *context);
 public:
  TlsSession(const std::string &hostname, const std::string &servername,
             bool rejectUnauthorized, const std::string *customCa);
  TlsSession(const TlsSession &) = delete;
  TlsSession &operator=(const TlsSession &) = delete;
  void feed(const char *bytes, size_t size);
  bool handshake();
  // 0 means more ciphertext is needed; -1 means authenticated close_notify, not raw TCP EOF.
  int read(char *bytes, size_t capacity);
  bool write(const char *bytes, size_t size);
  void shutdown();
  std::vector<char> output();
  std::string authorizationError() const;
  std::string fingerprint256() const;
};
} // namespace mom::runtime
