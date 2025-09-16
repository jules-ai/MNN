//
//  CoreMLGatherV2.cpp
//  MNN
//
//  Created by MNN on 2021/06/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLGatherV2.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {


CoreMLGatherV2::CoreMLGatherV2(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER;
    mLayer_->gather = mCoreMLBackend->create<CoreML__Specification__GatherLayerParams>();
    core_ml__specification__gather_layer_params__init(mLayer_->gather);
    int axis = 0;
    if (inputs.size() == 3) {
        auto axis_tensor = inputs[2];
        axis = axis_tensor->host<int32_t>()[0];
    }
    if (mOp->main_type() == OpParameter_Axis) {
        axis = mOp->main_as_Axis()->axis();
    }
    mLayer_->gather->axis = axis;
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0]), mCoreMLBackend->getTensorName(inputs[1])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_GatherV2)

} // namespace MNN
