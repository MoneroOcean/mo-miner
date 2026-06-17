/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "crypto/rx/RxFix.h"
#include "base/io/log/Log.h"


#include <windows.h>
#include <cstdio>
#include <cstdlib>


namespace xmrig {


static thread_local std::pair<const void*, const void*> mainLoopBounds = { nullptr, nullptr };


static LONG WINAPI MainLoopHandler(_EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == 0xC0000005) {
        const char* accessType = nullptr;
        switch (ExceptionInfo->ExceptionRecord->ExceptionInformation[0]) {
        case 0: accessType = "read"; break;
        case 1: accessType = "write"; break;
        case 8: accessType = "DEP violation"; break;
        default: accessType = "unknown"; break;
        }
        LOG_VERBOSE(YELLOW_BOLD("[THREAD %u] Access violation at 0x%p: %s at address 0x%p"), GetCurrentThreadId(), ExceptionInfo->ExceptionRecord->ExceptionAddress, accessType, ExceptionInfo->ExceptionRecord->ExceptionInformation[1]);
    }
    else {
        LOG_VERBOSE(YELLOW_BOLD("[THREAD %u] Exception 0x%08X at 0x%p"), GetCurrentThreadId(), ExceptionInfo->ExceptionRecord->ExceptionCode, ExceptionInfo->ExceptionRecord->ExceptionAddress);
    }

    void* p = reinterpret_cast<void*>(ExceptionInfo->ContextRecord->Rip); // NOLINT(performance-no-int-to-ptr)
    const std::pair<const void*, const void*>& loopBounds = mainLoopBounds;

    // MOM TEMP DIAG (gated on MOM_DEBUG_STARTUP, set by the release test's failure rerun): surface the
    // fault code, faulting RIP, the registered main-loop bounds, and whether recovery applies. Remove
    // once the AMD-Windows rx crash is confirmed fixed.
    if (std::getenv("MOM_DEBUG_STARTUP")) {
        std::fprintf(stderr, "MOM_RXFIX code=0x%08lX rip=%p bounds=[%p,%p) inBounds=%d\n",
                     (unsigned long)ExceptionInfo->ExceptionRecord->ExceptionCode, p,
                     loopBounds.first, loopBounds.second,
                     (int)((loopBounds.first <= p) && (p < loopBounds.second)));
        std::fflush(stderr);
    }

    if ((loopBounds.first <= p) && (p < loopBounds.second)) {
        ExceptionInfo->ContextRecord->Rip = reinterpret_cast<DWORD64>(loopBounds.second);
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    return EXCEPTION_CONTINUE_SEARCH;
}


} // namespace xmrig


void xmrig::RxFix::setMainLoopBounds(const std::pair<const void *, const void *> &bounds)
{
    mainLoopBounds = bounds;
}


void xmrig::RxFix::setupMainLoopExceptionFrame()
{
    AddVectoredExceptionHandler(1, MainLoopHandler);
}
