/* ito.exe: runs Ito on the Python runtime unzipped beside it, wherever that folder is.

   A GUI-subsystem program, so a double-click opens no console window: the app runs on
   runtime\pythonw.exe and reports problems in its own window. Started from a terminal,
   it attaches to that console and uses runtime\python.exe so output and --help appear.
   The app and everything it starts live in a job that dies with this process. */

#include <windows.h>
#include <wchar.h>

#define LONGEST 32768

static HANDLE inheritable(DWORD which, const wchar_t *device, DWORD access) {
    HANDLE handle = GetStdHandle(which);
    if (handle && handle != INVALID_HANDLE_VALUE) {
        SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        return handle;
    }
    SECURITY_ATTRIBUTES attributes = {sizeof attributes, NULL, TRUE};
    return CreateFileW(device, access, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes,
                       OPEN_EXISTING, 0, NULL);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show) {
    static wchar_t folder[LONGEST], python[LONGEST], command[LONGEST], message[LONGEST];
    (void)instance, (void)previous, (void)show;
    DWORD length = GetModuleFileNameW(NULL, folder, LONGEST);
    wchar_t *name = wcsrchr(folder, L'\\');
    if (!length || length >= LONGEST || !name) return 1;
    *name = 0;

    BOOL console = AttachConsole(ATTACH_PARENT_PROCESS);
    swprintf(python, LONGEST, L"%ls\\runtime\\%ls", folder,
             console ? L"python.exe" : L"pythonw.exe");
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES) {
        swprintf(message, LONGEST,
                 L"Ito is incomplete: %ls is missing.\n\nUnzip the whole ito folder and "
                 L"start ito.exe from there.", python);
        MessageBoxW(NULL, message, L"Ito", MB_ICONERROR);
        return 1;
    }
    /* -I ignores PYTHONHOME, PYTHONPATH and user site-packages from the pilot's machine. */
    swprintf(command, LONGEST, L"\"%ls\" -I -X utf8 -m ito.app %ls", python, arguments);

    STARTUPINFOW startup = {sizeof startup};
    if (console) {
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = inheritable(STD_INPUT_HANDLE, L"CONIN$", GENERIC_READ);
        startup.hStdOutput = inheritable(STD_OUTPUT_HANDLE, L"CONOUT$", GENERIC_WRITE);
        startup.hStdError = inheritable(STD_ERROR_HANDLE, L"CONOUT$", GENERIC_WRITE);
    }
    HANDLE job = CreateJobObjectW(NULL, NULL);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits);

    PROCESS_INFORMATION app;
    if (!CreateProcessW(python, command, NULL, NULL, console, CREATE_SUSPENDED, NULL, NULL,
                        &startup, &app)) {
        swprintf(message, LONGEST, L"Ito could not start %ls (error %lu).", python,
                 GetLastError());
        MessageBoxW(NULL, message, L"Ito", MB_ICONERROR);
        return 1;
    }
    AssignProcessToJobObject(job, app.hProcess);
    ResumeThread(app.hThread);
    WaitForSingleObject(app.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(app.hProcess, &code);
    return (int)code;
}
