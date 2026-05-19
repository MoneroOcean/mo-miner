// Copyright GNU GPLv3 (c) 2026-2026 MoneroOcean <support@moneroocean.stream>

#include "xmrig/base/io/log/Log.h"
#include "xmrig/base/tools/Chrono.h"

#include <chrono>

namespace xmrig {

// Minimal XMRig runtime stubs required by upstream object files that are linked
// into the Node addon without XMRig's full application/logging layer.
bool Log::m_background = false;
bool Log::m_colors = false;
LogPrivate* Log::d = nullptr;
uint32_t Log::m_verbose = 0;

void Log::add(ILogBackend*) {}
void Log::destroy() {}
void Log::init() {}
void Log::print(const char*, ...) {}
void Log::print(Level, const char*, ...) {}

double Chrono::highResolutionMSecs()
{
    using namespace std::chrono;

    if (high_resolution_clock::is_steady) {
        return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
    }

    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace xmrig
