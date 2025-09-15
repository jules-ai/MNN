//
//  CoreMLCast.hpp
//  MNN
//
//  Created by MNN on 2021/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CoreMLCast_hpp
#define CoreMLCast_hpp

#include "CoreMLCommonExecution.hpp"

namespace MNN {

class CoreMLCast : public CoreMLCommonExecution {
public:
    CoreMLCast(Backend *b, const Op *op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    virtual ~CoreMLCast() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CoreMLCast_hpp */
