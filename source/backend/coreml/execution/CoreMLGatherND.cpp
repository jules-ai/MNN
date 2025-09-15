//
//  CoreMLGatherND.cpp
//  MNN
//
//  Created by MNN on 2021/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CoreMLGatherND.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CoreMLGatherND::CoreMLGatherND(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : CoreMLCommonExecution(b, op) {
    // Do nothing
}

ErrorCode CoreMLGatherND::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2);

    auto params = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];

    auto coreMLBackend = static_cast<CoreMLBackend *>(backend());
    std::string indicesName = coreMLBackend->getTensorName(indices);
    std::string paramName = coreMLBackend->getTensorName(params);

    // GatherND Layer
    auto gatherLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
    core_ml__specification__neural_network_layer__init(gatherLayer);
    mCoreMLBackend->setLayerName(gatherLayer, "GatherND");
    gatherLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER_ND;
    gatherLayer->gathernd = mCoreMLBackend->create<CoreML__Specification__GatherNDLayerParams>();
    core_ml__specification__gather_ndlayer_params__init(gatherLayer->gathernd);

    mCoreMLBackend->setLayerInputs(gatherLayer, {paramName, indicesName});
    mCoreMLBackend->setLayerOutputs(gatherLayer, {mCoreMLBackend->getTensorName(output)});
    mCoreMLBackend->addLayer(gatherLayer);

    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLGatherND, OpType_GatherND);
} // namespace MNN
