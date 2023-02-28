#ifndef MODULES_THIRTD_PARTY_CUSTOM_WEBRTC_CONFIG_H_
#define MODULES_THIRTD_PARTY_CUSTOM_WEBRTC_CONFIG_H_

#include <memory>
#include <iostream>
#include <fstream>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>

namespace webrtc {

class CustomWebrtcConfig {
 public:
  explicit CustomWebrtcConfig();
  virtual ~CustomWebrtcConfig() = default;

  bool GetPersonalNsState();

 private:
  bool PersonalNsState (const std::string config_file_path);

  std::string config_file_path_;
  std::time_t last_modify_time_;

  bool personal_ns_state_ = false;
};

}  // namespace webrtc

#endif  // MODULES_THIRTD_PARTY_CUSTOM_WEBRTC_CONFIG_H_
