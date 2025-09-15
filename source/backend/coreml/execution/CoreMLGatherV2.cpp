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

    // Params and Indices
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
        // For Gather op
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

    if (axis == 0) {
        // If axis is 0, just use a simple GatherND
        auto gatherLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(gatherLayer);
        mCoreMLBackend->setLayerName(gatherLayer, "GatherV2_Axis0");
        gatherLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER_ND;
        gatherLayer->gathernd = mCoreMLBackend->create<CoreML__Specification__GatherNDLayerParams>();
        core_ml__specification__gather_ndlayer_params__init(gatherLayer->gathernd);

        mCoreMLBackend->setLayerInputs(gatherLayer, {mCoreMLBackend->getTensorName(params), mCoreMLBackend->getTensorName(indices)});
        mCoreMLBackend->setLayerOutputs(gatherLayer, {mCoreMLBackend->getTensorName(output)});
        mCoreMLBackend->addLayer(gatherLayer);
        return NO_ERROR;
    }

    // Axis is not 0, need transpose
    // 1. Transpose params
    std::vector<int> perm(paramsRank);
    perm[0] = axis;
    int current = 1;
    for (int i = 0; i < paramsRank; ++i) {
        if (i == axis) {
            continue;
        }
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

    auto transposedParamName = mCoreMLBackend->getTensorName(params) + "_transposed";
    mCoreMLBackend->setLayerInputs(transposeParamLayer, {mCoreMLBackend->getTensorName(params)});
    mCoreMLBackend->setLayerOutputs(transposeParamLayer, {transposedParamName});
    mCoreMLBackend->addLayer(transposeParamLayer);

    // 2. GatherND
    auto gatherLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
    core_ml__specification__neural_network_layer__init(gatherLayer);
    mCoreMLBackend->setLayerName(gatherLayer, "GatherV2_GatherND");
    gatherLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER_ND;
    gatherLayer->gathernd = mCoreMLBackend->create<CoreML__Specification__GatherNDLayerParams>();
    core_ml__specification__gather_ndlayer_params__init(gatherLayer->gathernd);

    auto gatherOutName = mCoreMLBackend->getTensorName(output) + "_gathered";
    mCoreMLBackend->setLayerInputs(gatherLayer, {transposedParamName, mCoreMLBackend->getTensorName(indices)});
    mCoreMLBackend->setLayerOutputs(gatherLayer, {gatherOutName});
    mCoreMLBackend->addLayer(gatherLayer);

    // 3. Transpose output back
    auto outputRank = output->dimensions();
    std::vector<int> backPerm(outputRank);
    current = axis;
    for (int i=0; i<axis; ++i) {
        backPerm[i] = i;
    }
    for (int i=axis; i<outputRank; ++i) {
        if (i < axis + indices->dimensions() -1) {
            backPerm[i] = i;
        } else {
            backPerm[i] = current++;
        }
    }
    // This is hard, let's use a simpler way. The output shape is params.shape[:axis] + indices.shape + params.shape[axis+1:]
    // After transpose, the shape is params.shape[axis] + params.shape[:axis] + params.shape[axis+1:]
    // After gather, the shape is indices.shape + params.shape[:axis] + params.shape[axis+1:]
    // So we need to permute it back
    // The new rank is rank(params) + rank(indices) - 1
    int indicesRank = indices->dimensions();
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

    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_GatherV2);
REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_Gather);
} // namespace MNN
