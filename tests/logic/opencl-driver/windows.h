#pragma once
typedef void* HMODULE;
#define MAX_PATH 260
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x800
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x100
unsigned GetSystemDirectoryA(char*, unsigned);
unsigned GetEnvironmentVariableA(const char*, char*, unsigned);
HMODULE LoadLibraryExA(const char*, void*, unsigned);
