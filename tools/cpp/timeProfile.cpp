//
//  timeProfile.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE
#include <stdlib.h>
#include <cstring>
#include <memory>
#include <string>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include "Profiler.hpp"
#include <MNN/Tensor.hpp>
#include "revertMNNModel.hpp"
#include <sstream>

#define MNN_PRINT_TIME_BY_NAME

using namespace MNN;

int main(int argc, const char* argv[]) {
    {
        if (argc < 2) {
            MNN_PRINT("========================================================================\n");
            MNN_PRINT("Arguments: model.MNN runLoops forwardType inputSize numberThread\n");
            MNN_PRINT("========================================================================\n");
            return -1;
        }
    }
    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    // read args
    const char* fileName = argv[1];
    int runTime          = 100;
    if (argc > 2) {
        runTime = ::atoi(argv[2]);
    }
    auto type = MNN_FORWARD_CPU;
    if (argc > 3) {
        type = (MNNForwardType)atoi(argv[3]);
        printf("Use extra forward type: %d\n", type);
    }

    // input dims
    std::vector<int> inputDims;
    if (argc > 4) {
        std::string inputShape(argv[4]);
        const char* delim = "x";
        std::ptrdiff_t p1 = 0, p2;
        while (1) {
            p2 = inputShape.find(delim, p1);
            if (p2 != std::string::npos) {
                inputDims.push_back(atoi(inputShape.substr(p1, p2 - p1).c_str()));
                p1 = p2 + 1;
            } else {
                inputDims.push_back(atoi(inputShape.substr(p1).c_str()));
                break;
            }
        }
    }
    for (auto dim : inputDims) {
        MNN_PRINT("%d ", dim);
    }
    MNN_PRINT("\n");
    int threadNumber = 1;
    if (argc > 5) {
        threadNumber = ::atoi(argv[5]);
        MNN_PRINT("Set ThreadNumber = %d\n", threadNumber);
    }

    auto precision = BackendConfig::PrecisionMode::Precision_Normal;
    if (argc > 6) {
        precision = (BackendConfig::PrecisionMode)atoi(argv[6]);
        printf("Use precision type: %d\n", precision);
    }

    float sparsity = 0.0f;
    if(argc >= 8) {
        sparsity = atof(argv[7]);
    }


    // revert MNN model if necessary
    auto revertor = std::unique_ptr<Revert>(new Revert(fileName));
    revertor->initialize(sparsity);
    auto modelBuffer = revertor->getBuffer();
    auto bufferSize  = revertor->getBufferSize();

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    auto net = std::shared_ptr<Interpreter>(Interpreter::createFromBuffer(modelBuffer, bufferSize));
    if (nullptr == net) {
        return 0;
    }
    revertor.reset();
    net->setSessionMode(Interpreter::Session_Debug);

    // create session
    MNN::ScheduleConfig config;
    config.type           = type;
    config.numThread      = threadNumber;
    BackendConfig backendConfig;
    backendConfig.precision = precision;
    config.backendConfig  = &backendConfig;
    MNN::Session* session = NULL;
    session               = net->createSession(config);
    auto inputTensor      = net->getSessionInput(session, NULL);
    if (!inputDims.empty()) {
        net->resizeTensor(inputTensor, inputDims);
        net->resizeSession(session);
    }
    auto allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto inputTensor = iter.second;
        auto size = inputTensor->size();
        if (size <= 0) {
            continue;
        }
        MNN::Tensor tempTensor(inputTensor, inputTensor->getDimensionType());
        ::memset(tempTensor.host<void>(), 0, tempTensor.size());
        inputTensor->copyFromHostTensor(&tempTensor);
    }
    net->releaseModel();
    std::shared_ptr<MNN::Tensor> inputTensorUser(MNN::Tensor::createHostTensorFromDevice(inputTensor, false));
    auto outputTensor = net->getSessionOutput(session, NULL);
    if (outputTensor->size() <= 0) {
        MNN_ERROR("Output not available\n");
        return 0;
    }
    std::shared_ptr<MNN::Tensor> outputTensorUser(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));

    auto profiler      = MNN::Profiler::getInstance();
    auto beginCallBack = [&](const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
        profiler->start(info);
        return true;
    };
    auto afterCallBack = [&](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
        for (auto o : tensors) {
            o->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        }
        for (size_t i = 0; i < tensors.size(); i++)
        {
            auto tensor = tensors[i];
                std::ostringstream oss;
                for (int j = 0; j < tensor->dimensions(); j++) {
                    oss << (j ? " X " : "") << tensor->length(j);
                }

                MNN_PRINT("Dimensions: %d, %s, OP name %s : %lu\n", tensor->dimensions(), oss.str().c_str(), info->name().c_str(), i);
            //if(inputs[i]->dimensions() == 4) MNN_PRINT("tensor = %s op = %s size = %u shape = %d %d %d %d\n",inputs[i]->name().c_str(),info->name().c_str(),inputs[i]->size(), inputs[i]->batch(), inputs[i]->channel(), inputs[i]->height(), inputs[i]->width());
        }
        profiler->end(info);
        return true;
    };

    AUTOTIME;
    // just run
    for (int i = 0; i < runTime; ++i) {
        inputTensor->copyFromHostTensor(inputTensorUser.get());
        net->runSessionWithCallBackInfo(session, beginCallBack, afterCallBack);
        outputTensor->copyToHostTensor(outputTensorUser.get());
    }

#ifdef MNN_PRINT_TIME_BY_NAME
    profiler->printTimeByName(runTime);
#endif
    profiler->printSlowOp("Convolution", 20, 0.03f);
    profiler->printTimeByType(runTime);

    
    {
        MNN_PRINT("**** Result ****\n");
        MNN_PRINT("output size:%d\n", outputTensorUser->elementSize());
        auto type = outputTensorUser->getType();

        auto size = outputTensorUser->elementSize();
        std::vector<std::pair<int, float>> tempValues(size);
        if (type.code == halide_type_float)
        {
            MNN_PRINT("output type: float\n");
            auto values = outputTensorUser->host<float>();
            for (int i = 0; i < size; ++i)
            {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1)
        {
            MNN_PRINT("output type: uint8\n");
            auto values = outputTensorUser->host<uint8_t>();
            for (int i = 0; i < size; ++i)
            {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }

        int length = size > 10 ? 10 : size;

        for (int i = 0; i < length; ++i)
        {
            MNN_PRINT("%d, %f\n", tempValues[i].first, tempValues[i].second);
        }
    }
    return 0;
}
