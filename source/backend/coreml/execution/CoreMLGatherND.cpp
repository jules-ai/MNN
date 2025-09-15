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

    // Handle indices type. CoreML's GatherND requires Int32 indices.
    if (indices->getType().code != halide_type_int || indices->getType().bits != 32) {
        if (TensorUtils::getDescribe(indices)->usage == Tensor::InsideDescribe::CONSTANT) {
            // If indices are constant, we can cast them on the CPU and create a new constant layer.
            indicesName = indicesName + "_casted_to_int32";

            auto castedIndicesLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(castedIndicesLayer);
            mCoreMLBackend->setLayerName(castedIndicesLayer, indicesName);

            castedIndicesLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LOAD_CONSTANT_ND;
            castedIndicesLayer->loadconstantnd = mCoreMLBackend->create<CoreML__Specification__LoadConstantNDLayerParams>();
            core_ml__specification__load_constant_ndlayer_params__init(castedIndicesLayer->loadconstantnd);

            auto shape = indices->shape();
            castedIndicesLayer->loadconstantnd->n_shape = shape.size();
            castedIndicesLayer->loadconstantnd->shape = mCoreMLBackend->create<uint64_t>(shape.size());
            for (int i = 0; i < shape.size(); i++) {
                castedIndicesLayer->loadconstantnd->shape[i] = shape[i];
            }

            castedIndicesLayer->loadconstantnd->data = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
            core_ml__specification__weight_params__init(castedIndicesLayer->loadconstantnd->data);

            auto n_bytes = indices->elementSize() * sizeof(int32_t);
            castedIndicesLayer->loadconstantnd->data->rawvalue.len = n_bytes;
            castedIndicesLayer->loadconstantnd->data->rawvalue.data = mCoreMLBackend->create<uint8_t>(n_bytes);

            // Perform cast
            auto host = indices->host<void>();
            auto dest = castedIndicesLayer->loadconstantnd->data->rawvalue.data;
            if (indices->getType().code == halide_type_int && indices->getType().bits == 64) {
                auto src = (int64_t*)host;
                auto dst = (int32_t*)dest;
                for (int i=0; i<indices->elementSize(); ++i) {
                    dst[i] = (int32_t)src[i];
                }
            } else {
                MNN_ERROR("CoreML GatherND: unhandled indices type for casting!\n");
                return NOT_SUPPORT;
            }

            mCoreMLBackend->setLayerOutputs(castedIndicesLayer, {indicesName});
            mCoreMLBackend->addLayer(castedIndicesLayer);

        } else {
            MNN_ERROR("CoreML GatherND: indices must be Int32 or a constant that can be cast to Int32.\n");
            return NOT_SUPPORT;
        }
    }

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
