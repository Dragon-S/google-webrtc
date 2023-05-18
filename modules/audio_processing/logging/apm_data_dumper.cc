/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/audio_processing/logging/apm_data_dumper.h"

#include "absl/strings/string_view.h"
#include "rtc_base/strings/string_builder.h"
#include "modules/third_party/helper/helper.h"
#include "rtc_base/logging.h"

#if defined(WEBRTC_MAC)
#include <unistd.h>
#include <sys/stat.h>
#else
#include <windows.h>
#include <shlobj.h>
#endif

// Check to verify that the define is properly set.
#if !defined(WEBRTC_APM_DEBUG_DUMP) || \
    (WEBRTC_APM_DEBUG_DUMP != 0 && WEBRTC_APM_DEBUG_DUMP != 1)
#error "Set WEBRTC_APM_DEBUG_DUMP to either 0 or 1"
#endif

namespace webrtc {
namespace {

#if WEBRTC_APM_DEBUG_DUMP == 1

#if defined(WEBRTC_WIN)
constexpr char kPathDelimiter = '\\';
#else
constexpr char kPathDelimiter = '/';
#endif

#if defined(WEBRTC_MAC)
std::string GetApmDumpDir() {
  std::string apm_dump_dir = helper::GetXuanXingMeetDir() + "apm_dump";
  // 检查文件夹是否存在
  if (access(apm_dump_dir.c_str(), F_OK) != -1) {
      RTC_LOG(LS_INFO) << "Folder exists. apm_dump_dir:: " << apm_dump_dir;
  } else {
      // 创建文件夹
      if (mkdir(apm_dump_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
          RTC_LOG(LS_INFO) << "Folder created successfully. apm_dump_dir:: " << apm_dump_dir;
      } else {
          RTC_LOG(LS_ERROR) << "Failed to create folder. apm_dump_dir:: " << apm_dump_dir;
          return "";
      }
  }
  return apm_dump_dir;
}
#else
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
std::string GetApmDumpDir() {
    std::wstring apm_dump_dir_wstr = getXuanXingMeetPath() + L"apm_dump";

    int size_needed = WideCharToMultiByte(CP_UTF8, 0, apm_dump_dir_wstr.c_str(), -1, NULL, 0, NULL, NULL);
    std::string apm_dump_dir(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, apm_dump_dir_wstr.c_str(), -1, &apm_dump_dir[0], size_needed, NULL, NULL);

    if (CreateDirectoryW(apm_dump_dir_wstr.c_str(), NULL)) {
        RTC_LOG(LS_ERROR) << "Folder created successfully. apm_dump_dir:: " << apm_dump_dir;
    } else {
        if (GetLastError() == ERROR_ALREADY_EXISTS) {
          RTC_LOG(LS_INFO) << "Folder exists. apm_dump_dir:: " << apm_dump_dir;
        } else {
          RTC_LOG(LS_ERROR) << "Failed to create folder. apm_dump_dir:: " << apm_dump_dir;
          return "";
        }
    }

    return apm_dump_dir;
}
#endif

std::string FormFileName(absl::string_view output_dir,
                         absl::string_view name,
                         int instance_index,
                         int reinit_index,
                         absl::string_view suffix) {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  if (!output_dir.empty()) {
    ss << output_dir;
    if (output_dir.back() != kPathDelimiter) {
      ss << kPathDelimiter;
    }
  }
  ss << name << "_" << instance_index << "-" << reinit_index << suffix;
  return ss.str();
}
#endif

}  // namespace

#if WEBRTC_APM_DEBUG_DUMP == 1
ApmDataDumper::ApmDataDumper(int instance_index)
    : instance_index_(instance_index) {}
#else
ApmDataDumper::ApmDataDumper(int instance_index) {}
#endif

ApmDataDumper::~ApmDataDumper() = default;

#if WEBRTC_APM_DEBUG_DUMP == 1
bool ApmDataDumper::recording_activated_ = false;
absl::optional<int> ApmDataDumper::dump_set_to_use_;
char ApmDataDumper::output_dir_[] = "";

FILE* ApmDataDumper::GetRawFile(absl::string_view name) {
  std::string apm_dump_dir = GetApmDumpDir();
  RTC_CHECK(apm_dump_dir.length()) << "apm_dump_dir" << " does not exist. ";
  std::string filename = FormFileName(apm_dump_dir.c_str(), name, instance_index_,
                                      recording_set_index_, ".dat");
  auto& f = raw_files_[filename];
  if (!f) {
#if defined(WEBRTC_WIN)
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, NULL, 0);
    std::wstring filename_wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, &filename_wstr[0], size_needed);
    f.reset(_wfopen(filename_wstr.c_str(), L"wb"));
#else
    f.reset(fopen(filename.c_str(), "wb"));
#endif
    RTC_CHECK(f.get()) << "Cannot write to " << filename << ".";
  }
  return f.get();
}

WavWriter* ApmDataDumper::GetWavFile(absl::string_view name,
                                     int sample_rate_hz,
                                     int num_channels,
                                     WavFile::SampleFormat format) {
  std::string apm_dump_dir = GetApmDumpDir();
  RTC_CHECK(apm_dump_dir.length()) << "apm_dump_dir" << " does not exist. ";
  std::string filename = FormFileName(apm_dump_dir.c_str(), name, instance_index_,
                                      recording_set_index_, ".wav");
  auto& f = wav_files_[filename];
  if (!f) {
    f.reset(
        new WavWriter(filename.c_str(), sample_rate_hz, num_channels, format));
  }
  return f.get();
}
#endif

}  // namespace webrtc
