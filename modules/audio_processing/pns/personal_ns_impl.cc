/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "personal_ns_impl.h"

#include "modules/audio_processing/audio_buffer.h"
#include "personal_ns.h"
#include "rtc_base/logging.h"


namespace webrtc {

PersonalNsImpl::PersonalNsImpl() {}

PersonalNsImpl::~PersonalNsImpl() {}

void PersonalNsImpl::Initialize(size_t channels, int sample_rate_hz) {
  //RTC_DCHECK(sample_rate_hz == 48000);
  channels_ = channels;
  sample_rate_hz_ = sample_rate_hz;
  
  std::vector<std::unique_ptr<pns::PersonalNs> > new_personalNs;
  if (enabled_) {
    new_personalNs.resize(channels);
    for (size_t i = 0; i < channels; i++) {
      new_personalNs[i].reset(pns::PersonalNs::createPns());
      new_personalNs[i]->extractEmbedder();
    }
  }
  personalNs_.swap(new_personalNs);
}

int PersonalNsImpl::ProcessCaptureAudio(AudioBuffer* audio) {
  RTC_DCHECK(audio);
  if (!enabled_) {
    return AudioProcessing::kNotEnabledError;
  }

  for (size_t i = 0; i < personalNs_.size(); i++) {
      personalNs_[i]->processFrame(audio->split_bands_f(i)[0]);
  }
  return AudioProcessing::kNoError;
}

int PersonalNsImpl::Enable(bool enable) {
  if (enabled_ != enable) {
    enabled_ = enable;
    Initialize(channels_, sample_rate_hz_);
  }
  return AudioProcessing::kNoError;
}

bool PersonalNsImpl::is_enabled() const {
  return enabled_;
}


}  // namespace webrtc
