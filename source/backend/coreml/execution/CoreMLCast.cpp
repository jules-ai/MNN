//
//  CoreMLCast.cpp
//  MNN
//
//  Created by MNN on 2021/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CoreMLCast.hpp"

namespace MNN {

CoreMLCast::CoreMLCast(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : CoreMLCommonExecution(b, op) {
    // Do nothing
}

ErrorCode CoreMLCast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ERROR("CoreML Cast is not supported yet!\n");
    return NOT_SUPPORT;
}

REGISTER_COREML_OP_CREATOR(CoreMLCast, OpType_Cast);
} // namespace MNN
