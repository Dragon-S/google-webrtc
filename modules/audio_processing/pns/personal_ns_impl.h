/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_PERSONAL_NS_IMPL_H_
#define MODULES_AUDIO_PROCESSING_PERSONAL_NS_IMPL_H_

#include <memory>
#include <vector>

#include "modules/audio_processing/include/audio_processing.h"
#include "personal_ns.h"

namespace webrtc {

class AudioFrame;

class PersonalNsImpl {
 public:
  explicit PersonalNsImpl();
  ~PersonalNsImpl();

  // TODO(peah): Fold into ctor, once public API is removed.
  void Initialize(size_t channels, int sample_rate_hz);
  int ProcessCaptureAudio(AudioBuffer* audio);

  int Enable(bool enable);
  bool is_enabled() const;

 private:
  bool enabled_ = false;
  size_t channels_ = 0;
  int sample_rate_hz_ = 0;
  std::vector<std::unique_ptr<pns::PersonalNs> > personalNs_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_PERSONAL_NS_IMPL_H_
