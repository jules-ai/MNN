//
//  CoreMLGatherV2.cpp
//  MNN
//
//  Created by MNN on 2021/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CoreMLGatherV2.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CoreMLGatherV2::CoreMLGatherV2(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : CoreMLCommonExecution(b, op) {
    // Do nothing
}

ErrorCode CoreMLGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 || inputs.size() == 3);

    auto params = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];

    int axis = 0;
    if (inputs.size() == 3) {
        auto axisTensor = inputs[2];
        if (TensorUtils::getDescribe(axisTensor)->usage != Tensor::InsideDescribe::CONSTANT) {
            MNN_ERROR("CoreML GatherV2: axis must be constant!\n");
            return NOT_SUPPORT;
        }
        axis = axisTensor->host<int32_t>()[0];
    } else {
        auto p = mOp->main_as_Axis();
        if (p) {
            axis = p->axis();
        }
    }

    auto paramsRank = params->dimensions();
    if (axis < 0) {
        axis += paramsRank;
    }

    auto coreMLBackend = static_cast<CoreMLBackend *>(backend());
    std::string indicesName = coreMLBackend->getTensorName(indices);

    // Handle indices type. CoreML's GatherND requires Int32 indices.
    if (indices->getType().code != halide_type_int || indices->getType().bits != 32) {
        if (TensorUtils::getDescribe(indices)->usage == Tensor::InsideDescribe::CONSTANT) {
            // If indices are constant, we can cast them on the CPU and create a new constant layer.
            indicesName = indicesName + "_casted_to_int32";

            auto castedIndicesLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(castedIndicesLayer);
            mCoreMLBackend->setLayerName(castedIndicesLayer, std::move(indicesName));

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
                MNN_ERROR("CoreML GatherV2: unhandled indices type for casting!\n");
                return NOT_SUPPORT;
            }

            mCoreMLBackend->setLayerOutputs(castedIndicesLayer, {indicesName});
            mCoreMLBackend->addLayer(castedIndicesLayer);

        } else {
            MNN_ERROR("CoreML GatherV2: indices must be Int32 or a constant that can be cast to Int32.\n");
            return NOT_SUPPORT;
        }
    }

    std::string paramName = coreMLBackend->getTensorName(params);

    if (axis != 0) {
        // Transpose params to move the target axis to the front
        std::vector<int> perm(paramsRank);
        perm[0] = axis;
        int current = 1;
        for (int i = 0; i < paramsRank; ++i) {
            if (i == axis) continue;
            perm[current++] = i;
        }

        auto transposeParamLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(transposeParamLayer);
        mCoreMLBackend->setLayerName(transposeParamLayer, "GatherV2_Transpose_Before");
        transposeParamLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TRANSPOSE;
        transposeParamLayer->transpose = mCoreMLBackend->create<CoreML__Specification__TransposeLayerParams>();
        core_ml__specification__transpose_layer_params__init(transposeParamLayer->transpose);
        transposeParamLayer->transpose->n_axes = perm.size();
        transposeParamLayer->transpose->axes = mCoreMLBackend->create<uint64_t>(perm.size());
        for(int i=0; i<perm.size(); ++i) {
            transposeParamLayer->transpose->axes[i] = perm[i];
        }

        auto transposedParamName = paramName + "_transposed";
        mCoreMLBackend->setLayerInputs(transposeParamLayer, {paramName});
        mCoreMLBackend->setLayerOutputs(transposeParamLayer, {transposedParamName});
        mCoreMLBackend->addLayer(transposeParamLayer);
        paramName = transposedParamName;
    }

    auto gatherOutName = mCoreMLBackend->getTensorName(output);
    if (axis != 0) {
        gatherOutName += "_gathered";
    }

    // GatherND Layer
    auto gatherLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
    core_ml__specification__neural_network_layer__init(gatherLayer);
    mCoreMLBackend->setLayerName(gatherLayer, "GatherV2_GatherND");
    gatherLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER_ND;
    gatherLayer->gathernd = mCoreMLBackend->create<CoreML__Specification__GatherNDLayerParams>();
    core_ml__specification__gather_ndlayer_params__init(gatherLayer->gathernd);

    mCoreMLBackend->setLayerInputs(gatherLayer, {paramName, indicesName});
    mCoreMLBackend->setLayerOutputs(gatherLayer, {gatherOutName});
    mCoreMLBackend->addLayer(gatherLayer);

    if (axis != 0) {
        // Transpose output back
        int indicesRank = indices->dimensions();
        int outputRank = output->dimensions();
        std::vector<int> finalPerm(outputRank);
        int p = 0;
        for (int i = 0; i < axis; i++) {
            finalPerm[p++] = i + indicesRank;
        }
        for (int i = 0; i < indicesRank; i++) {
            finalPerm[p++] = i;
        }
        for (int i = axis + 1; i < paramsRank; i++) {
            finalPerm[p++] = i + indicesRank - 1;
        }

        auto transposeOutLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(transposeOutLayer);
        mCoreMLBackend->setLayerName(transposeOutLayer, "GatherV2_Transpose_After");
        transposeOutLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TRANSPOSE;
        transposeOutLayer->transpose = mCoreMLBackend->create<CoreML__Specification__TransposeLayerParams>();
        core_ml__specification__transpose_layer_params__init(transposeOutLayer->transpose);
        transposeOutLayer->transpose->n_axes = finalPerm.size();
        transposeOutLayer->transpose->axes = mCoreMLBackend->create<uint64_t>(finalPerm.size());
        for(int i=0; i<finalPerm.size(); ++i) {
            transposeOutLayer->transpose->axes[i] = finalPerm[i];
        }

        mCoreMLBackend->setLayerInputs(transposeOutLayer, {gatherOutName});
        mCoreMLBackend->setLayerOutputs(transposeOutLayer, {mCoreMLBackend->getTensorName(output)});
        mCoreMLBackend->addLayer(transposeOutLayer);
    }

    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_GatherV2);
REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_Gather);
} // namespace MNN
