//
//  CoreMLGatherV2.hpp
//  MNN
//
//  Created by MNN on 2021/06/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLGATHERV2_HPP
#define MNN_COREMLGATHERV2_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLGatherV2 : public CoreMLCommonExecution {
public:
    CoreMLGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLGatherV2() = default;
};
} // namespace MNN

#endif // MNN_COREMLGATHERV2_HPP
