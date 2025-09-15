//
//  CoreMLCast.cpp
//  MNN
//
//  Created by MNN on 2021/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CoreMLCast.hpp"

namespace MNN {

// CoreML only supports limited cast type
static CoreML__Specification__CastLayerParams__CastOutputType getCastOutputType(DataType dt) {
    switch (dt) {
        case DataType_DT_FLOAT:
            return CORE_ML__SPECIFICATION__CAST_LAYER_PARAMS__CAST_OUTPUT_TYPE__FLOAT_TYPE;
        case DataType_DT_INT32:
            return CORE_ML__SPECIFICATION__CAST_LAYER_PARAMS__CAST_OUTPUT_TYPE__INT32_TYPE;
        default:
            return CORE_ML__SPECIFICATION__CAST_LAYER_PARAMS__CAST_OUTPUT_TYPE__NOT_SET;
    }
}


CoreMLCast::CoreMLCast(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : CoreMLCommonExecution(b, op) {
    // nothing to do
}

ErrorCode CoreMLCast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cast = mOp->main_as_CastParam();
    auto outputType = getCastOutputType(cast->dstT());
    if (outputType == CORE_ML__SPECIFICATION__CAST_LAYER_PARAMS__CAST_OUTPUT_TYPE__NOT_SET) {
        MNN_ERROR("CoreML Cast: unsupported dst data type %d\n", cast->dstT());
        return NOT_SUPPORT;
    }

    auto castLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
    core_ml__specification__neural_network_layer__init(castLayer);
    mCoreMLBackend->setLayerName(castLayer, "Cast");

    castLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CAST;
    castLayer->cast = mCoreMLBackend->create<CoreML__Specification__CastLayerParams>();
    core_ml__specification__cast_layer_params__init(castLayer->cast);
    castLayer->cast->outputtype = outputType;

    mCoreMLBackend->setLayerInputs(castLayer, {mCoreMLBackend->getTensorName(inputs[0])});
    mCoreMLBackend->setLayerOutputs(castLayer, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(castLayer);

    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLCast, OpType_Cast);

} // namespace MNN
