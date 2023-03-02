#ifndef VOICE_FILTER_H_
#define VOICE_FILTER_H_

#include <vector>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextra-semi"
#include "onnxruntime/include/onnxruntime_cxx_api.h"
#pragma clang diagnostic pop

namespace pns
{
class VoiceFilter {
public:
    VoiceFilter(const ORTCHAR_T* modelPath, size_t featureDim, size_t embedderDim, size_t hiddenNumber);

    ~VoiceFilter();

    void setDVector(std::vector<float>& dVector);

    std::vector<float> process(std::shared_ptr<std::vector<float> > specturm);

private:
    size_t startCount_;

    size_t featureDim_;

    Ort::Env env_;

    Ort::Session session_{nullptr};

    std::vector<Ort::Value> inputTensors_;

    std::vector<Ort::Value> outputTensors_;

    std::vector<float> specturmData_;

    std::vector<float> embedderData_;

    std::vector<float> hiddenData_;

    std::vector<float> cellData_;

    std::vector<float> outputData_;
};

} // namespace pns


#endif // VOICE_FILTER_H_