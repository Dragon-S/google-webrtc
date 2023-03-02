#include "embedder.h"

#include <array>
#include <cmath>
#include <algorithm>
#if defined(WEBRTC_WIN)
#include <windows.h>
#include <windowsx.h>
#endif
#include <cassert>

#if defined(WEBRTC_WIN)
#pragma comment(lib, "onnxruntime.lib")
#endif

namespace pns
{
Embedder::Embedder(const ORTCHAR_T* modelPath, size_t melNumber, size_t embedderDim) : embedderDim_(embedderDim) {
    std::array<int64_t, 3> inputShape{8, 1, static_cast<long long>(melNumber)};
    std::array<int64_t, 3> hiddenShape{3, 1, 768};
    std::array<int64_t, 3> cellShape{3, 1, 768};
    std::array<int64_t, 2> outputShape{1, static_cast<long long>(embedderDim)};
    inputData_.resize(8 * melNumber, 0.0f);
    hiddenData_.resize(3 * 768, 0.0f);
    cellData_.resize(3 * 768, 0.0f);
    outputData_.resize(embedderDim, 0.0f);

    session_ = Ort::Session(env_, modelPath, Ort::SessionOptions{nullptr});

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputData_.data(), inputData_.size(), inputShape.data(), inputShape.size()));
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hiddenData_.data(), hiddenData_.size(), hiddenShape.data(), hiddenShape.size()));
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, cellData_.data(), cellData_.size(), cellShape.data(), cellShape.size()));
    outputTensor_ = Ort::Value::CreateTensor<float>(memoryInfo, outputData_.data(), outputData_.size(), outputShape.data(), outputShape.size());
}

Embedder::~Embedder() {}

std::vector<float> Embedder::process(std::vector<std::vector<float> >& features) {
    const char* inputNames[] = {"input", "onnx::Slice_1", "onnx::Slice_2"};
    const char* outputNames[] = {"243"};

    assert(features.size() >= 640);
    size_t melNumber = features[0].size();
    std::vector<float> result(embedderDim_, 0.0f);

    for (size_t i = 0; i < 80; i++) {
        for (size_t j = 0; j < 8; j++) {
            size_t inx = j * 80 + i;
            std::copy(features[inx].begin(), features[inx].end(), inputData_.begin() + j * melNumber);
        }

        session_.Run(Ort::RunOptions{nullptr}, inputNames, inputTensors_.data(), 3, outputNames, &outputTensor_, 1);
        std::transform(outputData_.begin(), outputData_.end(), result.begin(), result.begin(), [](float a, float b) { return a + b; });
    }
    std::transform(result.begin(), result.end(), result.begin(), [](float a) { return a / 80.0f; });
    return result;
}

} // namespace pns

//#define UNITTEST
#ifdef UNITTEST

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    std::unique_ptr<pns::Embedder> embedder_;
    embedder_.reset(new pns::Embedder(8, 40, 256));
    std::cout << "embedder_ instance create OK" << std::endl;

    // load npy datas
    uint8_t buf[10];
    std::ifstream npyStream("melFeature_0.npy", std::ios_base::in | std::ios_base::binary);
    npyStream.read((char *)&buf[0], 10);                                                                                                                                                                                                                                                                                          
    std::cout << (uint16_t)buf[8] << std::endl;
    npyStream.seekg((int)buf[8] + 10);

    std::vector<float> dataBuf(320, 1.0f);
    npyStream.read((char*)dataBuf.data(), 3200 * sizeof(float));
    
    std::vector<std::vector<float> > mels;
    for (size_t i = 0; i < 8; i++) {
        std::vector<float> mel(40, 1.0f);
        std::copy(dataBuf.begin() + i * 40, dataBuf.begin() + (i + 1) * 40, mel.begin());
        mels.push_back(mel);
    }
    std::vector<float> dVector = embedder_->process(mels);

    return 0;
}

#endif // UNITTEST