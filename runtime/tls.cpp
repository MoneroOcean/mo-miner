#include "tls.h"
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include "tls-roots.inc"

namespace mom::runtime {
namespace {
const char *verificationCode(long error) {
  switch (error) {
    case X509_V_OK: return "";
    case X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT: return "DEPTH_ZERO_SELF_SIGNED_CERT";
    case X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN: return "SELF_SIGNED_CERT_IN_CHAIN";
    case X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY: return "UNABLE_TO_GET_ISSUER_CERT_LOCALLY";
    case X509_V_ERR_UNABLE_TO_VERIFY_LEAF_SIGNATURE: return "UNABLE_TO_VERIFY_LEAF_SIGNATURE";
    case X509_V_ERR_CERT_HAS_EXPIRED: return "CERT_HAS_EXPIRED";
    case X509_V_ERR_CERT_NOT_YET_VALID: return "CERT_NOT_YET_VALID";
    // Preserve actionable OpenSSL/Node authorization codes without relaxing verification.
#define MOM_TLS_VERIFY_ERROR(name) case X509_V_ERR_##name: return #name
    MOM_TLS_VERIFY_ERROR(UNABLE_TO_GET_ISSUER_CERT);
    MOM_TLS_VERIFY_ERROR(UNABLE_TO_DECRYPT_CERT_SIGNATURE);
    MOM_TLS_VERIFY_ERROR(UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY);
    MOM_TLS_VERIFY_ERROR(CERT_SIGNATURE_FAILURE);
    MOM_TLS_VERIFY_ERROR(ERROR_IN_CERT_NOT_BEFORE_FIELD);
    MOM_TLS_VERIFY_ERROR(ERROR_IN_CERT_NOT_AFTER_FIELD);
    MOM_TLS_VERIFY_ERROR(CERT_REVOKED);
    MOM_TLS_VERIFY_ERROR(INVALID_PURPOSE);
    MOM_TLS_VERIFY_ERROR(INVALID_CA);
    MOM_TLS_VERIFY_ERROR(CERT_UNTRUSTED);
    MOM_TLS_VERIFY_ERROR(CERT_REJECTED);
    MOM_TLS_VERIFY_ERROR(PATH_LENGTH_EXCEEDED);
    MOM_TLS_VERIFY_ERROR(CERT_CHAIN_TOO_LONG);
#undef MOM_TLS_VERIFY_ERROR
    case X509_V_ERR_HOSTNAME_MISMATCH:
    case X509_V_ERR_IP_ADDRESS_MISMATCH: return "ERR_TLS_CERT_ALTNAME_INVALID";
    default: return "CERTIFICATE_VERIFY_FAILED";
  }
}

// Trust anchors are generated from the build Node's bundled roots, never extracted at startup.
// A custom CA replaces defaults; NODE_EXTRA_CA_CERTS augments defaults just as it does in Node.
void addCertificates(SSL_CTX *context, const std::string &pem) {
  if (pem.size() > 8 * 1024 * 1024) {throw TlsError("ERR_TLS_INVALID_CA", "CA bundle exceeds 8 MiB");}
  std::unique_ptr<BIO, decltype(&BIO_free)> input(
      BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size())), BIO_free);
  if (!input) {throw std::bad_alloc();}
  while (auto *certificate = PEM_read_bio_X509_AUX(input.get(), nullptr, nullptr, nullptr)) {
    std::unique_ptr<X509, decltype(&X509_free)> owned(certificate, X509_free);
    if (!X509_STORE_add_cert(SSL_CTX_get_cert_store(context), certificate) &&
        ERR_GET_REASON(ERR_peek_last_error()) != X509_R_CERT_ALREADY_IN_HASH_TABLE) {
      throw TlsError("ERR_TLS_INVALID_CA", "Cannot install TLS trust anchor");
    }
    ERR_clear_error();
  }
  // EOF is reported as PEM_R_NO_START_LINE. Never let that expected parse status poison I/O errors.
  ERR_clear_error();
}

// Match Node's PeerCertificate.fingerprint256 representation (uppercase, colon-separated SHA-256).
std::string certificateFingerprint256(X509 *certificate) {
  if (!certificate) {return {};}
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (X509_digest(certificate, EVP_sha256(), digest, &length) != 1 || length != 32) {
    ERR_clear_error();
    return {};
  }
  static constexpr char hex[] = "0123456789ABCDEF";
  std::string result;
  result.reserve(95);
  for (unsigned int index = 0; index < length; ++index) {
    if (index) {result.push_back(':');}
    result.push_back(hex[digest[index] >> 4]);
    result.push_back(hex[digest[index] & 0x0f]);
  }
  return result;
}
}

int TlsSession::verify(int valid, X509_STORE_CTX *context) {
  auto *ssl = static_cast<SSL *>(X509_STORE_CTX_get_ex_data(context, SSL_get_ex_data_X509_STORE_CTX_idx()));
  if (!ssl) {return 0;}
  auto *session = static_cast<TlsSession *>(SSL_get_app_data(ssl));
  if (!session) {return 0;}
  if (!valid && session->verificationError_ == X509_V_OK) {
    session->verificationError_ = X509_STORE_CTX_get_error(context);
  }
  // Disabling rejection is explicit, but authorization metadata must still record verification
  // failures. The default always rejects invalid chains and hostname/IP mismatches.
  return valid || !session->rejectUnauthorized_;
}

namespace {
// Match Node's IPv6 zone syntax; malformed scopes must not turn a DNS/IPv4 name into an IP identity.
bool validIpv6Scope(const std::string &hostname, size_t zone) {
  if (!zone || zone + 1 >= hostname.size() || hostname.find('%', zone + 1) != std::string::npos) {return false;}
  for (size_t index = zone + 1; index < hostname.size(); ++index) {
    const auto c = static_cast<unsigned char>(hostname[index]);
    const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    const bool digit = c >= '0' && c <= '9';
    if (!alpha && !digit && c != '-' && c != '.' && c != ':') {return false;}
  }
  return true;
}

// Certificate IP SANs contain address bytes, not the routing scope retained by the socket layer.
std::unique_ptr<ASN1_OCTET_STRING, decltype(&ASN1_OCTET_STRING_free)> verificationIp(
    const std::string &hostname) {
  std::unique_ptr<ASN1_OCTET_STRING, decltype(&ASN1_OCTET_STRING_free)> ip(
      a2i_IPADDRESS(hostname.c_str()), ASN1_OCTET_STRING_free);
  if (ip) {return ip;}
  const auto zone = hostname.find('%');
  if (zone == std::string::npos || !validIpv6Scope(hostname, zone)) {return ip;}
  const auto literal = hostname.substr(0, zone);
  ip.reset(a2i_IPADDRESS(literal.c_str()));
  // A scope ID belongs only to IPv6 routing; it is not part of the certificate IP SAN.
  if (ip && ASN1_STRING_length(ip.get()) != 16) {ip.reset();}
  return ip;
}
} // namespace

void initializeCrypto() {
  static const bool initialized = OPENSSL_init_ssl(OPENSSL_INIT_NO_LOAD_CONFIG, nullptr) == 1;
  if (!initialized) {throw TlsError("ERR_TLS_INITIALIZATION", "Cannot initialize crypto/TLS");}
}

TlsSession::TlsSession(const std::string &hostname, const std::string &servername,
                       bool rejectUnauthorized, const std::string *customCa)
    : rejectUnauthorized_(rejectUnauthorized) {
  initializeCrypto();
  context_.reset(SSL_CTX_new(TLS_client_method()));
  if (!context_ || !SSL_CTX_set_min_proto_version(context_.get(), TLS1_2_VERSION)) {
    throw TlsError("ERR_TLS_INITIALIZATION", "Cannot initialize TLS context");
  }
  SSL_CTX_set_options(context_.get(), SSL_OP_NO_RENEGOTIATION);
  SSL_CTX_set_verify(context_.get(), SSL_VERIFY_PEER, verify);
  if (customCa) {addCertificates(context_.get(), *customCa);}
  else {
    addCertificates(context_.get(), kTlsRoots);
    if (const char *extra = std::getenv("NODE_EXTRA_CA_CERTS"); extra && *extra) {
      std::ifstream input(extra, std::ios::binary);
      if (input) {
        std::string pem;
        char block[8192];
        while (input) {
          input.read(block, sizeof(block));
          const auto size = static_cast<size_t>(input.gcount());
          if (size > 8 * 1024 * 1024 - pem.size()) {
            throw TlsError("ERR_TLS_INVALID_CA", "Extra CA bundle exceeds 8 MiB");
          }
          pem.append(block, size);
        }
        if (input.bad()) {throw TlsError("ERR_TLS_INVALID_CA", "Cannot read extra CA bundle");}
        addCertificates(context_.get(), pem);
      } else {std::fputs("Warning: cannot read NODE_EXTRA_CA_CERTS\n", stderr);}
    }
  }
  ssl_.reset(SSL_new(context_.get()));
  if (!ssl_) {throw std::bad_alloc();}
  SSL_set_app_data(ssl_.get(), this);
  std::unique_ptr<BIO, decltype(&BIO_free)> input(BIO_new(BIO_s_mem()), BIO_free);
  std::unique_ptr<BIO, decltype(&BIO_free)> output(BIO_new(BIO_s_mem()), BIO_free);
  if (!input || !output) {throw std::bad_alloc();}
  BIO_set_mem_eof_return(input.get(), -1);
  SSL_set_bio(ssl_.get(), input.release(), output.release());
  SSL_set_connect_state(ssl_.get());
  if (!servername.empty() && !SSL_set_tlsext_host_name(ssl_.get(), servername.c_str())) {
    throw TlsError("ERR_TLS_INVALID_SERVERNAME", "Invalid TLS server name");
  }
  const auto ip = verificationIp(hostname);
  const bool identity = ip ? X509_VERIFY_PARAM_set1_ip(SSL_get0_param(ssl_.get()),
      ASN1_STRING_get0_data(ip.get()), ASN1_STRING_length(ip.get())) == 1
      : SSL_set1_host(ssl_.get(), hostname.c_str()) == 1;
  if (!identity) {throw TlsError("ERR_TLS_INVALID_SERVERNAME", "Cannot configure TLS peer identity");}
  ERR_clear_error();
}

[[noreturn]] void TlsSession::failure(int error, bool checkingCertificate) {
  const auto authorization = authorizationError();
  if (checkingCertificate && !authorization.empty()) {
    const long status = verificationError_ == X509_V_OK ? SSL_get_verify_result(ssl_.get()) : verificationError_;
    throw TlsError(authorization, X509_verify_cert_error_string(status));
  }
  const auto reason = ERR_peek_last_error();
  const char *text = reason ? ERR_reason_error_string(reason) : nullptr;
  throw TlsError(error == SSL_ERROR_ZERO_RETURN ? "EPIPE" : "ERR_TLS_HANDSHAKE_FAILURE",
      text ? text : "TLS peer closed unexpectedly");
}

void TlsSession::feed(const char *bytes, size_t size) {
  auto *input = SSL_get_rbio(ssl_.get());
  constexpr size_t maximumInput = 4 * 1024 * 1024;
  const size_t pending = BIO_ctrl_pending(input);
  if (pending > maximumInput || size > maximumInput - pending) {
    throw TlsError("ERR_TLS_BUFFER_LIMIT", "TLS input exceeds 4 MiB");
  }
  if (size && BIO_write(input, bytes, static_cast<int>(size)) != static_cast<int>(size)) {
    throw TlsError("ERR_TLS_BUFFER", "Cannot buffer TLS input");
  }
}

bool TlsSession::handshake() {
  ERR_clear_error();
  const int result = SSL_do_handshake(ssl_.get());
  const int error = SSL_get_error(ssl_.get(), result);
  if (result == 1) {return true;}
  if (error == SSL_ERROR_WANT_READ || error == SSL_ERROR_WANT_WRITE) {return false;}
  failure(error, true);
}

int TlsSession::read(char *bytes, size_t capacity) {
  size_t size = 0;
  ERR_clear_error();
  const int result = SSL_read_ex(ssl_.get(), bytes, capacity, &size);
  const int error = SSL_get_error(ssl_.get(), result);
  if (result == 1) {return static_cast<int>(size);}
  if (error == SSL_ERROR_WANT_READ || error == SSL_ERROR_WANT_WRITE) {return 0;}
  if (error == SSL_ERROR_ZERO_RETURN) {return -1;}
  failure(error);
}

// A blocked write must be retried with the same buffer/length. Its caller retains that storage
// until completion; enabling partial writes or advancing the caller's buffer would break TLS.
bool TlsSession::write(const char *bytes, size_t size) {
  size_t written = 0;
  ERR_clear_error();
  const int result = SSL_write_ex(ssl_.get(), bytes, size, &written);
  const int error = SSL_get_error(ssl_.get(), result);
  if (result == 1) {return true;}
  if (error == SSL_ERROR_WANT_READ || error == SSL_ERROR_WANT_WRITE) {return false;}
  failure(error);
}

void TlsSession::shutdown() {
  ERR_clear_error();
  const int result = SSL_shutdown(ssl_.get());
  if (result >= 0) {return;} // zero means close_notify was sent; the peer need not reply yet.
  const int error = SSL_get_error(ssl_.get(), result);
  if (error != SSL_ERROR_WANT_READ && error != SSL_ERROR_WANT_WRITE) {failure(error);}
}

std::vector<char> TlsSession::output() {
  auto *output = SSL_get_wbio(ssl_.get());
  const size_t size = BIO_ctrl_pending(output);
  if (size > 64 * 1024 * 1024) {throw TlsError("ERR_TLS_BUFFER_LIMIT", "TLS output exceeds 64 MiB");}
  std::vector<char> bytes(size);
  if (size && BIO_read(output, bytes.data(), static_cast<int>(size)) != static_cast<int>(size)) {
    throw TlsError("ERR_TLS_BUFFER", "Cannot drain TLS output");
  }
  return bytes;
}

std::string TlsSession::authorizationError() const {
  return verificationCode(verificationError_ == X509_V_OK ? SSL_get_verify_result(ssl_.get()) : verificationError_);
}

std::string TlsSession::fingerprint256() const {
  std::unique_ptr<X509, decltype(&X509_free)> certificate(SSL_get1_peer_certificate(ssl_.get()), X509_free);
  return certificateFingerprint256(certificate.get());
}
} // namespace mom::runtime
