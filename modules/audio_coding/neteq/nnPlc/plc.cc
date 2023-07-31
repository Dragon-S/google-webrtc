#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include "fft.h"
#include "nn.h"
#include "plc.h"

namespace nnPlc {

Plc::~Plc() {
    fft_.release();
    nn_.release();
}

int Plc::initialize() {
    phaseBase_.resize(processSize_, 0.0f);
    phaseInc_.resize(processSize_, 0.0f);
    for(size_t i = 0; i < processSize_; i++) {
        phaseInc_[i] = 0.014f * acosf(-1.0f) * (float)i / (float)fftSize_;
    }
    real_.resize(fftSize_, 0.0f);
    imag_.resize(fftSize_, 0.0f);
    logX_.resize(processSize_ + 1, 0.0f);
    ampY_.resize(processSize_, 0.0f);
    phaY_.resize(processSize_, 0.0f);

    fft_.reset(new Fft());
    fft_->initialize();
    nn_.reset(new NN());
    nn_->initialize();

    return 0;
}

void Plc::reset() {
    plcCount_ = 0;
}

void Plc::process(const float* input, float* output) {
    if (plcCount_ == 0) {
        std::fill(imag_.begin(), imag_.end(), 0.0f);
        std::fill(real_.begin(), real_.end(), 0.0f);
        for (size_t i = 0; i < blockSize_; i++) {
            real_[i] = input[i];
        }

        fft_->FftForward(real_, imag_);
        std::transform(real_.begin(), real_.begin() + processSize_, imag_.begin(), phaseBase_.begin(), [this](float a, float b) { 
            float p = atanf(b / (a + 1e-6));
            if ((a * b < 0.0f) && (a < 0.0f)) {
                return p + mPI;
            }
            else if ((a * b > 0.0f) && (a < 0.0f)) {
                return p - mPI;
            }
            return p;
        });
        std::transform(real_.begin(), real_.begin() + processSize_, imag_.begin(), logX_.begin(), [](float a, float b) { return sqrtf(a * a + b * b); });
        std::transform(logX_.begin(), logX_.end() - 1, logX_.begin(), [](float a) { return a < 0.0032f ? 0.0032f : (a > 316.2277f ? 316.2277f : a); });
        std::transform(logX_.begin(), logX_.end() - 1, logX_.begin(), [](float a) { return (20.0f * log10f(a + 1e-6f) + 50.0f) / 100.0f; });
    }

    logX_[processSize_] = (float)(plcCount_ / 36);
    nn_->process(logX_.data(), ampY_.data());
    std::transform(ampY_.begin(), ampY_.end(), ampY_.begin(), [](float a) { return powf(10.0f, a * 5.0f - 2.5f); });

    //
    for (size_t j = 0; j < 3; j++) {
        std::transform(phaseBase_.begin(), phaseBase_.end(), phaseInc_.begin(), phaY_.begin(), [this](float a, float b) { return a + b * (float)plcCount_; });
        std::transform(phaY_.begin(), phaY_.end(), ampY_.begin(), real_.begin(), [](float a, float b) { return b * cosf(a); });
        std::transform(phaY_.begin(), phaY_.end(), ampY_.begin(), imag_.begin(), [](float a, float b) { return b * sinf(a); });
        for (size_t i = processSize_ + 1; i < fftSize_; i++) {
            real_[i] = real_[fftSize_ - i];
            imag_[i] = -imag_[fftSize_ - i];
        }
        fft_->FftBackward(real_, imag_);

        if (j == 0) { 
            for (size_t i = 0; i < blockSize_; i++) {
                output[i] = real_[i];
            }
        } else if (j == 1) {
            float t1 = 0.0f;
            float t2 = (float)overlapSize_;   
            size_t k = 0;
            for (size_t i = blockSize_ - overlapSize_; i < blockSize_; i++) {
                output[i] = (t2 * output[i] + t1 * real_[k++]) / (float)overlapSize_;
                t1 += 1.0f;
                t2 -= 1.0f;
            }
            for (size_t i = blockSize_; i < 2 * blockSize_ - overlapSize_; i++) {
                output[i] = real_[k++];
            }
        } else if (j == 2) {
            float t1 = 0.0f;
            float t2 = (float)overlapSize_;   
            size_t k = 0;
            for (size_t i = 2 * (blockSize_ - overlapSize_); i < 2 * blockSize_ - overlapSize_; i++) {
                output[i] = (t2 * output[i] + t1 * real_[k++]) / (float)overlapSize_;
                t1 += 1.0f;
                t2 -= 1.0f;
            }
            for (size_t i = 2 * blockSize_ - overlapSize_; i < 3 * blockSize_ - 2 * overlapSize_; i++) {
                output[i] = real_[k++];
            }            
        } else {
            // never get here, do nothing
        }
        plcCount_ += 1;
    }
}

} // namespace nnPlc