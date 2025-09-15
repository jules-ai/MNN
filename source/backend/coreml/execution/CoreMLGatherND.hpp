//
//  CoreMLGatherND.hpp
//  MNN
//
//  Created by MNN on 2021/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CoreMLGatherND_hpp
#define CoreMLGatherND_hpp

#include "CoreMLCommonExecution.hpp"

namespace MNN {

class CoreMLGatherND : public CoreMLCommonExecution {
public:
    CoreMLGatherND(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    virtual ~CoreMLGatherND() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CoreMLGatherND_hpp */
