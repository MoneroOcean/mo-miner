// Link against the actual original ICD: distinguish a rejected load from a DLL loaded then
// discarded after export checks. The canary is test-only, never part of the native application.
#include <windows.h>
#ifdef MOM_OPENCL_CANARY_DLL
BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
  (void)module; (void)reserved;
  if (reason == DLL_PROCESS_ATTACH) {
    char marker[32768];
    const DWORD size = GetEnvironmentVariableA("MOM_OPENCL_CANARY_MARKER", marker, sizeof(marker));
    if (size && size < sizeof(marker)) {
      HANDLE file = CreateFileA(marker, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
      if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
  }
  return TRUE;
}
#else
#include "icd.h"
#include <stdio.h>
#include <string.h>
#include <sddl.h>

// Lower only this disposable probe child's token. High-integrity SSH sessions make the
// upstream secure_getenv ignore variables already, hiding the ordinary miner path.
static int mediumChild(char** argv) {
  HANDLE current = NULL, restricted = NULL; PSID sid = NULL;
  PROCESS_INFORMATION child = {0}; STARTUPINFOA startup = {0};
  char executable[MAX_PATH], command[32768]; int status = 10;
  const char* stage = "create restricted token";
  // CreateRestrictedToken preserves the handle's access rights; lowering integrity needs
  // TOKEN_ADJUST_DEFAULT on the new handle, not a change to the user's account or policy.
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT, &current) ||
      !CreateRestrictedToken(current, DISABLE_MAX_PRIVILEGE, 0, NULL, 0, NULL, 0, NULL, &restricted) ||
      !ConvertStringSidToSidA("S-1-16-8192", &sid)) goto done;
  TOKEN_MANDATORY_LABEL label = {{sid, SE_GROUP_INTEGRITY}};
  stage = "set child integrity";
  if (!SetTokenInformation(restricted, TokenIntegrityLevel, &label, sizeof(label) + GetLengthSid(sid))) goto done;
  stage = "build child command";
  const DWORD length = GetModuleFileNameA(NULL, executable, sizeof(executable));
  if (!length || length >= sizeof(executable)) goto done;
  for (int i = 2; i < 5; ++i) if (strchr(argv[i], '"') || strchr(argv[i], '\r') || strchr(argv[i], '\n')) goto done;
  const int bytes = snprintf(command, sizeof(command), "\"%s\" --medium-child \"%s\" \"%s\" \"%s\"", executable, argv[2], argv[3], argv[4]);
  if (bytes < 0 || (size_t)bytes >= sizeof(command)) goto done;
  startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE); startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE); startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  stage = "launch restricted child";
  if (!CreateProcessAsUserA(restricted, executable, command, NULL, NULL, TRUE, 0, NULL, NULL, &startup, &child)) goto done;
  stage = "wait for child";
  if (WaitForSingleObject(child.hProcess, 20000) != WAIT_OBJECT_0) {TerminateProcess(child.hProcess, 11); WaitForSingleObject(child.hProcess, 5000); goto done;}
  DWORD code = 0; if (!GetExitCodeProcess(child.hProcess, &code)) goto done;
  status = (int)code; printf("Medium-integrity child exited: %lu\n", (unsigned long)code);
done:
  if (status == 10) fprintf(stderr, "Medium child setup failed (%s): %lu\n", stage, (unsigned long)GetLastError());
  if (child.hThread) CloseHandle(child.hThread);
  if (child.hProcess) CloseHandle(child.hProcess);
  if (sid) LocalFree(sid);
  if (restricted) CloseHandle(restricted);
  if (current) CloseHandle(current);
  return status;
}

// Verify the actual child label; a launcher option alone is not evidence of changed integrity.
static int isMedium(void) {
  HANDLE token = NULL; unsigned char buffer[SECURITY_MAX_SID_SIZE + sizeof(TOKEN_MANDATORY_LABEL)]; DWORD size;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return 0;
  const BOOL ok = GetTokenInformation(token, TokenIntegrityLevel, buffer, sizeof(buffer), &size); CloseHandle(token);
  if (!ok) return 0;
  const PSID sid = ((TOKEN_MANDATORY_LABEL*)buffer)->Label.Sid;
  const UCHAR count = *GetSidSubAuthorityCount(sid);
  return count && *GetSidSubAuthority(sid, count - 1) == SECURITY_MANDATORY_MEDIUM_RID;
}
int main(int argc, char** argv) {
  if (argc == 5 && !strcmp(argv[1], "medium")) return mediumChild(argv);
  if (argc == 5 && !strcmp(argv[1], "--medium-child")) {
    if (!isMedium()) return 9;
    --argc; ++argv;
  }
  if (argc != 4 || (strcmp(argv[1], "load") && strcmp(argv[1], "environment"))) return 2;
  if (GetFileAttributesA(argv[3]) != INVALID_FILE_ATTRIBUTES) return 3;
  if (GetLastError() != ERROR_FILE_NOT_FOUND && GetLastError() != ERROR_PATH_NOT_FOUND) return 4;
  if (!SetEnvironmentVariableA("MOM_OPENCL_CANARY_MARKER", argv[3])) return 5;
  void* library = NULL;
  if (!strcmp(argv[1], "load")) {
    library = khrIcdOsLibraryLoad(argv[2]);
    if (library) khrIcdOsLibraryUnload(library);
  } else {
    if (!SetEnvironmentVariableA("OCL_ICD_FILENAMES", argv[2])) return 6;
    khrIcdVendorsEnumerateEnv();
    SetEnvironmentVariableA("OCL_ICD_FILENAMES", NULL);
  }
  if (library || GetFileAttributesA(argv[3]) != INVALID_FILE_ATTRIBUTES) {
    puts("Untrusted application DLL was loaded"); return 86;
  }
  puts("Native OpenCL rejected the application DLL without loading it");
  return 0;
}
#endif
