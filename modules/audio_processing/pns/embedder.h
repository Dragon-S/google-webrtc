#ifndef EMBEDDER_H_
#define EMBEDDER_H_

#include <vector>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextra-semi"
#include "onnxruntime/include/onnxruntime_cxx_api.h"
#pragma clang diagnostic pop

namespace pns
{
class Embedder {
public:
    Embedder(const ORTCHAR_T*, size_t melNumber, size_t embedderDim);

    ~Embedder();

    std::vector<float> process(std::vector<std::vector<float> >& features);

private:
    size_t embedderDim_;

    Ort::Env env_;

    Ort::Session session_{nullptr};

    std::vector<Ort::Value> inputTensors_;

    Ort::Value outputTensor_{nullptr};

    std::vector<float> inputData_;

    std::vector<float> hiddenData_;

    std::vector<float> cellData_;

    std::vector<float> outputData_;
};

} // namespace pns

#endif // EMBEDDER_H_