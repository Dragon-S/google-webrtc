#include "mel_feature.h"
#include "pns_fft.h"

#include <vector>
#include <algorithm>

namespace pns
{
MelFeature::MelFeature(size_t frameSize, size_t fftSize, size_t melNumber, float lowestEnergy)
        : frameSize_(frameSize),
        fftSize_(fftSize),
        melNumber_(melNumber),
        lowestEnergy_(lowestEnergy) {
    fft_.reset(new PnsFft(fftSize));
    
    getMelFilters();
}   

MelFeature::~MelFeature() {
    fft_.reset();
    melFilters_.clear();
}

std::vector<std::vector<float> > MelFeature::getFeatures(float* speech, size_t length) {
    std::vector<std::vector<float> > mels;
    
    // total number of mel-features
    size_t featureNumber = (length - fftSize_) / frameSize_ + 1;
    if (featureNumber < 10) {
        return mels;
    }

    //
    size_t position = 0;
    std::vector<float> segment(fftSize_, 0.0f);
    for (size_t i = 0; i < featureNumber; i++) {
        // std::copy(std::begin(speech) + position, std::begin(speech) + position + fftSize_, segment.begin());
        for (size_t j = 0; j < fftSize_; j++) {
            segment[j] = speech[position + j];
        }
        position += frameSize_;

        float energry = 0.0f;
        // ignore silent segment
        for (size_t j = 0; j < fftSize_; j++) {
            energry += (segment[j] * segment[j]);
        }
        float logEnergy = 20.0f * log10f(energry + 0.0001f);
        if (logEnergy >= lowestEnergy_) {
            std::vector<float> mel = getMel(segment);
            mels.push_back(mel);
        }
    }

    if (mels.size() < 320) {
        mels.clear();
        return mels;
    }

    size_t inx = 0;
    while (mels.size() < 640) {
        std::vector<float> v = mels[inx];
        mels.push_back(v);
        inx += 1;
    }

    
    return mels;
}

std::vector<float> MelFeature::getMel(std::vector<float>& segment) {
    // get power specturm
    std::vector<float> imag(fftSize_, 0.0f);
    fft_->Fft(segment.data(), imag.data());
    std::transform(segment.begin(), segment.begin() + fftSize_ / 2 + 1, imag.begin(), segment.begin(), [](float a, float b) { return a * a + b * b; });

    // get mel-coefficentes
    std::vector<float> mel;
    for (size_t i = 0; i < melNumber_; i++) {
        float x = getMelInnerDot(melFilters_[i], segment);
        mel.push_back(x);
    }

    return mel;
}

void MelFeature::getMelFilters() {
    melFilters_.clear();

#ifdef USEHTK
    // convert frequency to mel
    std::vector<float> melBins(fftSize_ / 2 + 1, 0.0f);
    for (size_t i = 0; i < fftSize_ / 2 + 1; i++) {
        melBins[i] = 2595.0f * log10f(1.0f + (i * 16000.0f / fftSize_) / 700.0f);
    }   

    float maxMel = 2595.0f * log10(1.0f + 8000.0f / 700.0f);
    float halfMelBandWidth = maxMel / (melNumber_ + 1.0f);
    for (size_t i = 0; i < melNumber_; i++) {
        std::vector<float> melFilterI(fftSize_, 0.0f);

        // calculate i-th filter
        float startMel = i * halfMelBandWidth;
        float centerMel = (i + 1) * halfMelBandWidth;
        float endMel = (i + 2) * halfMelBandWidth;
        for (size_t j = 0; j < fftSize_ / 2 + 1; j++) {
            if ((melBins[j] > startMel) && (melBins[j] < endMel)) {
                if (melBins[j] <= centerMel) {
                    melFilterI[j] = (melBins[j] - startMel) / halfMelBandWidth;
                } else {
                    melFilterI[j] = (endMel - melBins[j]) / halfMelBandWidth;
                }
            } 
        }

        melFilters_.push_back(melFilterI);
    }
#else
    for (size_t i = 0; i < melNumber_; i++) {
        std::vector<float> melFilterI(fftSize_ / 2 + 1, 0.0f);

        for (size_t j = 0; j < fftSize_ / 2 + 1; j++) {
            melFilterI[j] = pnsMelFilters[i][j];
        }

        melFilters_.push_back(melFilterI);
    }
#endif
}

float MelFeature::getMelInnerDot(std::vector<float>& filter, std::vector<float> specturm) {
    float x = 0.0f;
    for (size_t i = 0; i < fftSize_ / 2 + 1; i++) {
        x += filter[i] * specturm[i];
    }
    return log10f(x + 0.0001f);
}

} // namespace pns

//#define UNITTEST
#ifdef UNITTEST

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>

int main() {
    std::unique_ptr<pns::MelFeature> melFeature_;
    melFeature_.reset(new pns::MelFeature(160, 512, 40, 2.0f));
    std::cout << "melFeature instance create OK" << std::endl;

    // mel file
    std::ofstream outputStream("output.bin", std::ios_base::out | std::ios_base::binary);
    // speech file
    std::ifstream pcmStream("..\\..\\..\\dataset\\test\\0\\enrollUtterance.wav", std::ios_base::in | std::ios_base::binary);
    
    pcmStream.seekg(0, std::ios_base::end);
    size_t length = pcmStream.tellg();
    assert(length % sizeof(int16_t) == 0);
    size_t sampleNumber = length / sizeof(int16_t);

    int16_t * uttranceSpeechBuf = new int16_t[sampleNumber];
    pcmStream.seekg(44, std::ios_base::beg);
    pcmStream.read((char *)uttranceSpeechBuf, length);
    float * uttranceSpeech = new float[sampleNumber];
    for (size_t i = 0; i < sampleNumber; i++) {
        uttranceSpeech[i] = uttranceSpeechBuf[i] / 32768.0f;
    }

    std::vector<std::vector<float> > mels = melFeature_->getFeatures(uttranceSpeech, sampleNumber);

    
    pcmStream.close();
    delete []uttranceSpeechBuf;
    delete []uttranceSpeech;

    return 0;
}

#endif // UNITTEST