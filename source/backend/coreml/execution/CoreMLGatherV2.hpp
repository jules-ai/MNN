//
//  CoreMLGatherV2.hpp
//  MNN
//
//  Created by MNN on 2021/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CoreMLGatherV2_hpp
#define CoreMLGatherV2_hpp

#include "CoreMLCommonExecution.hpp"

namespace MNN {

class CoreMLGatherV2 : public CoreMLCommonExecution {
public:
    CoreMLGatherV2(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    virtual ~CoreMLGatherV2() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CoreMLGatherV2_hpp */
