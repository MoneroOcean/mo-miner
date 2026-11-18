// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

// Release-only entrypoint for in-memory executable packing. UPX owns decompression;
// restore the exact original section protections before entering the unchanged CRT.
// No CRT, allocator, exceptions or security-cookie dependency is safe at this point.
#include <windows.h>
#pragma section(".mompp", read)
// Page map patched by the packer: magic "MOMPP094", version 1, then the range count and original
// image size at [3] and [4], and from [8] one {rva, size, section flags} triple per range.
extern "C" __declspec(allocate(".mompp")) const volatile unsigned int momPageMap[1024] = {
  0x504d4f4d, 0x34393050, 1
};
extern "C" IMAGE_DOS_HEADER __ImageBase;
extern "C" __declspec(noreturn) void wmainCRTStartup();

static DWORD protection(unsigned flags) {
  if (flags & IMAGE_SCN_MEM_EXECUTE)
    return PAGE_EXECUTE_READ;
  if (flags & IMAGE_SCN_MEM_WRITE)
    return PAGE_READWRITE;
  // Some original compiler target-symbol sections are deliberately not loaded/readable.
  return (flags & IMAGE_SCN_MEM_READ) ? PAGE_READONLY : PAGE_NOACCESS;
}

// Section bytes rounded up to whole pages; zero when the size cannot be rounded in 32 bits.
static unsigned sectionPages(const IMAGE_SECTION_HEADER& section) {
  const unsigned bytes = section.Misc.VirtualSize > section.SizeOfRawData
    ? section.Misc.VirtualSize : section.SizeOfRawData;
  if (bytes > 0xfffff000)
    return 0;
  return (bytes + 4095) & ~4095u;
}

// Validate the page map against the loaded PE headers, then apply each recorded range's original
// protection and tighten any section tail beyond the original image. Any inconsistency fails
// before or without widening protections.
static bool restorePages() {
  if (momPageMap[0] != 0x504d4f4d || momPageMap[1] != 0x34393050 || momPageMap[2] != 1)
    return false;
  const unsigned count = momPageMap[3];
  const unsigned originalSize = momPageMap[4];
  if (!count || count > 64 || originalSize < 4096 || originalSize % 4096)
    return false;
  if (__ImageBase.e_magic != IMAGE_DOS_SIGNATURE || __ImageBase.e_lfanew < 64 ||
      __ImageBase.e_lfanew > 4096 - static_cast<LONG>(sizeof(IMAGE_NT_HEADERS64)))
    return false;
  auto* base = reinterpret_cast<unsigned char*>(&__ImageBase);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + __ImageBase.e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE ||
      nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC ||
      nt->OptionalHeader.SectionAlignment != 4096 ||
      nt->OptionalHeader.SizeOfImage < originalSize || nt->OptionalHeader.SizeOfImage % 4096 ||
      !nt->FileHeader.NumberOfSections || nt->FileHeader.NumberOfSections > 64)
    return false;

  // Recorded ranges must be ordered, page-aligned, cover the original image after the headers
  // exactly, and never request writable+executable or inaccessible-but-mapped protections.
  unsigned previous = 4096;
  for (unsigned i = 0; i < count; ++i) {
    const unsigned rva = momPageMap[8 + 3 * i];
    const unsigned size = momPageMap[9 + 3 * i];
    const unsigned flags = momPageMap[10 + 3 * i];
    if (rva < previous || rva % 4096 || !size || size % 4096 || rva > originalSize ||
        size > originalSize - rva ||
        (!(flags & IMAGE_SCN_MEM_READ) && (flags & 0xa0000000)) ||
        (flags & 0xa0000000) == 0xa0000000)
      return false;
    previous = rva + size;
  }
  if (previous != originalSize)
    return false;

  const auto* sections = IMAGE_FIRST_SECTION(nt);
  if (reinterpret_cast<const unsigned char*>(sections + nt->FileHeader.NumberOfSections) >
      base + 4096)
    return false;
  previous = 4096;
  for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
    const unsigned rva = sections[i].VirtualAddress;
    const unsigned size = sectionPages(sections[i]);
    if (rva < previous || rva % 4096 || !size || rva > nt->OptionalHeader.SizeOfImage ||
        size > nt->OptionalHeader.SizeOfImage - rva)
      return false;
    previous = rva + size;
  }

  for (unsigned i = 0; i < count; ++i) {
    const unsigned rva = momPageMap[8 + 3 * i];
    const unsigned size = momPageMap[9 + 3 * i];
    const unsigned flags = momPageMap[10 + 3 * i];
    DWORD old = 0;
    if (!VirtualProtect(base + rva, size, protection(flags), &old))
      return false;
  }
  // Preserve the unpacker's later-thread TLS callback as RX, but do not leave an RWX
  // tail or freeze non-executable imports/data. Original section ranges stay unchanged.
  for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
    const auto& section = sections[i];
    const unsigned end = section.VirtualAddress + sectionPages(section);
    const unsigned begin =
      section.VirtualAddress > originalSize ? section.VirtualAddress : originalSize;
    if (begin >= end)
      continue;
    DWORD old = 0;
    if (!VirtualProtect(base + begin, end - begin, protection(section.Characteristics), &old))
      return false;
  }
  return FlushInstructionCache(GetCurrentProcess(), nullptr, 0) != FALSE;
}

extern "C" __declspec(noreturn) void momPackedEntry() {
  if (!restorePages())
    ExitProcess(69);
  wmainCRTStartup();
}
