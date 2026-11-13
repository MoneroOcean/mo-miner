#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Runtime {};
struct Owner {Runtime runtime;};

struct Value {
  std::string text;
};

struct JSError final : std::runtime_error {
  JSError(Runtime &, Value) : std::runtime_error("native environment enumeration error") {}
};

Value ioError(Runtime &, int) {
  return {"native environment enumeration error"};
}

struct String {
  static Value createFromUtf8(Runtime &, const char *value) {return {value};}
};

struct Array {
  explicit Array(Runtime &, size_t length) : values(length) {}

  void setValueAtIndex(Runtime &, size_t index, Value value) {
    if (index >= values.size()) {throw std::runtime_error("JSI Array index out of bounds");}
    values[index] = std::move(value.text);
  }

  std::vector<std::string> values;
};

struct uv_env_item_t {
  char *name;
  char *value;
};

static std::vector<std::string> environment;
static std::vector<uv_env_item_t> entries;
static size_t freeCalls = 0;
static size_t freedItems = 0;

int uv_os_environ(uv_env_item_t **items, int *count) {
  *items = nullptr;
  *count = 0;
  if (environment.empty()) {return 0;}
  entries.clear();
  for (auto &name : environment) {entries.push_back({name.data(), nullptr});}
  *items = entries.data();
  *count = static_cast<int>(environment.size());
  return 0;
}

void uv_os_free_environ(uv_env_item_t *items, int count) {
  freeCalls++;
  freedItems += static_cast<size_t>(count);
  if (items && items != entries.data()) {std::abort();}
  entries.clear();
}

/* MOM_ENV_ENUMERATION_BODY */

void setEnvironment(std::initializer_list<const char *> names) {
  environment.clear();
  for (const auto *name : names) {environment.emplace_back(name);}
}

void require(bool condition, const char *message) {
  if (!condition) {throw std::runtime_error(message);}
}

void runCase(Owner &owner, std::initializer_list<const char *> input,
    std::initializer_list<const char *> expected) {
  setEnvironment(input);
  const auto calls = freeCalls, items = freedItems;
  const auto result = enumerate(owner);
  require(result.values == std::vector<std::string>(expected.begin(), expected.end()),
      "__momEnvKeys returned an unexpected ordered key list");
  require(freeCalls == calls + 1, "uv_os_free_environ was not called exactly once");
  require(freedItems == items + input.size(), "uv_os_free_environ received the wrong count");
}

int main() {
  static constexpr const char *unicodeName = "UNICODE_\xC3\xA9_\xF0\x9F\x92\x8E";
  try {
    Owner owner;
    runCase(owner, {}, {});
    runCase(owner, {"ordinary", unicodeName, "__proto__", "toString", "=C:", "=ExitCode"},
#ifdef _WIN32
      {"ordinary", unicodeName, "__proto__", "toString"}
#else
      {"ordinary", unicodeName, "__proto__", "toString", "=C:", "=ExitCode"}
#endif
    );
    runCase(owner, {"first", "=C:", "second", "=ExitCode", "third"},
#ifdef _WIN32
      {"first", "second", "third"}
#else
      {"first", "=C:", "second", "=ExitCode", "third"}
#endif
    );
    runCase(owner, {"=C:", "=ExitCode"},
#ifdef _WIN32
      {}
#else
      {"=C:", "=ExitCode"}
#endif
    );
    std::puts("native environment enumeration passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "native environment enumeration failed: %s\n", error.what());
    return 1;
  }
}
