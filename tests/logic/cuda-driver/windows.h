#pragma once
using HMODULE = void*;
using FARPROC = void (*)();
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x800
HMODULE LoadLibraryExA(const char*, void*, unsigned);
FARPROC GetProcAddress(HMODULE, const char*);
