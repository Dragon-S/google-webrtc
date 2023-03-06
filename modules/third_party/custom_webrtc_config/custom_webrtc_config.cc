#include "modules/third_party/custom_webrtc_config/custom_webrtc_config.h"

#include <utility>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/platform_thread_types.h"
#include "modules/third_party/helper/helper.h"

namespace webrtc {

namespace {
bool IsValidCustomWebrtcConfigThread() {
  const rtc::PlatformThreadRef main_thread = rtc::CurrentThreadRef();
  return rtc::IsThreadRefEqual(main_thread, rtc::CurrentThreadRef());
}

bool FindFlagWithFile (const std::string file_path, const std::string flag) {
  std::ifstream file(file_path);
  if (!file.good()) {
    RTC_LOG(LS_ERROR) << "[CustomWebrtcConfig]:: Read file fail. File::" 
      << file_path << ", Flag:: " << flag;
    return false;
  }
  
  std::string str;
  while (getline(file, str)) {
    if (str == flag) {
      return true;
    }
  }

  file.close();

  RTC_LOG(LS_ERROR) << "[CustomWebrtcConfig]:: Not find flag. File::" 
    << file_path << ", Flag:: " << flag;

  return false;
}

std::time_t GetLastModificationTime(const std::string file_path) {
  struct stat result;
  if (stat(file_path.c_str(), &result) != 0) {
    RTC_LOG(LS_ERROR) << "[CustomWebrtcConfig] :: Get file state fail. File:: "
      << file_path;
  }

  return result.st_mtime;
}

}  // namespace

CustomWebrtcConfig::CustomWebrtcConfig() {
  std::string xxm_dir = helper::GetXuanXingMeetDir();
  config_file_path_ = xxm_dir + "webrtc_config.txt";

  //初始化文件最后修改时间
  last_modify_time_ = GetLastModificationTime(config_file_path_);

  //初始化各开关状态
  personal_ns_state_ = PersonalNsState(config_file_path_);
}

bool CustomWebrtcConfig::PersonalNsState (const std::string config_file_path) {
  std::string flag = "personal_ns=1";
  return FindFlagWithFile(config_file_path, flag);
}

bool CustomWebrtcConfig::GetPersonalNsState() {
  RTC_DCHECK(IsValidCustomWebrtcConfigThread());

  std::time_t last_time = GetLastModificationTime(config_file_path_);
  if (std::difftime(last_modify_time_, last_time) != 0) {
    last_modify_time_ = last_time;
    personal_ns_state_ = PersonalNsState(config_file_path_);
  }

  return personal_ns_state_;
}

}  // namespace webrtc
