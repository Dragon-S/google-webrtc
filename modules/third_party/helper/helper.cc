#include "helper.h"
#ifdef WEBRTC_WIN
#include "helper_win.h"
#elif WEBRTC_MAC
#include "helper_mac.h"
#endif

namespace helper {

std::string GetXuanXingMeetDir() {
#ifdef WEBRTC_WIN
    return GetXuanXingMeetDirWin();
#elif WEBRTC_MAC
    return GetXuanXingMeetDirMac();
#endif
}

} // namespace helper
