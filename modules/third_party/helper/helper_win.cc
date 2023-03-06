#include "helper_win.h"
#include "rtc_base/logging.h"

#include <shlobj.h>

namespace helper {

std::wstring getXuanXingMeetPath() {
	
    PWSTR appDataPath = nullptr;
	
    std::wstring resultPath;
    if (SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &appDataPath) == S_OK) {
        CoTaskMemFree(appDataPath);
        resultPath = std::wstring(appDataPath) + L"\\XuanXingMeet\\";
    } else {
        CoTaskMemFree(appDataPath);
        RTC_LOG(LS_ERROR) << "Failed to get appdata path";
    }

    return resultPath;
}

std::string GetXuanXingMeetDirWin() {
    std::wstring wsPath = getXuanXingMeetPath();
    if (wsPath == L"") {
        RTC_LOG(LS_ERROR) << "path is null";
        return "";
    }
    const wchar_t* pwsPath = wsPath.c_str();
    char cPath[MAX_PATH];
    wcstombs(cPath, pwsPath, MAX_PATH);
    return std::string(cPath);
}

} // namespace helper