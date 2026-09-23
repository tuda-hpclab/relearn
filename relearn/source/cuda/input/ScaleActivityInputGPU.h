#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/input/ScaleActivityInputBase.h"
#include "cuda/CudaConfig.h"
#include "cuda/input/ActivityInput.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <functional>
#include <memory>
#include <vector>

/**
 * GPU implementation of ScaleActivityInput: dispatches a fixed CudaConfig::scaling_function_enum
 * to a kernel (functors can't cross into device code, so the CPU flavor's std::function-based
 * constructor is kept only for interface parity and is unsupported at runtime here).
 */
class ScaleActivityInputGPU : public ScaleActivityInputBase {
public:
    /**
     * @brief Kept for interface parity with ScaleActivityInputCPU; unsupported at runtime since
     *      functors can't cross into device code. Use the scaling_function_enum constructor instead.
     */
    ScaleActivityInputGPU(const int _number_ranks, std::shared_ptr<ActivityInput> activity_input, [[maybe_unused]] std::function<activity_type(activity_type)> scaling_function)
        : ScaleActivityInputBase(_number_ranks, std::move(activity_input)){
            CPU_NOT_SUPPORTED
        }

        /**
         * @brief Constructs a new instance of type ScaleActivityInputGPU with 0 neurons and the passed values for all parameters
         * @param activity_input The activity input which should be scaled, not empty
         * @param _scaling_function_enum The scaling function to dispatch to a kernel
         * @exception Throws a RelearnException if any argument is empty or the scaling function is unknown
         */
        ScaleActivityInputGPU(const auto _number_ranks, std::shared_ptr<ActivityInput> activity_input, const CudaConfig::scaling_function_enum _scaling_function_enum, const activity_type)
        : ScaleActivityInputBase(_number_ranks, std::move(activity_input))
        , scaling_function_enum(_scaling_function_enum) {
        RelearnException::check(scaling_function_enum == CudaConfig::LINEAR || scaling_function_enum == CudaConfig::LOGARITHMIC || scaling_function_enum == CudaConfig::HYPERBOLIC_TANGENT, "SynapticInputCalculator::SynapticInputCalculator: scaling_function does not exist.");
    }

    /**
     * @brief Not supported: the GPU path only calls the stream-taking overload below.
     * @exception Throws a RelearnException, always.
     */
    std::vector<EventWrapper> update_input_range([[maybe_unused]] const step_type, const NeuronID, const NeuronID, const std::shared_ptr<StreamWrapper>&) override {
        CPU_NOT_SUPPORTED
    }

    using ActivityInput::update_input_range;

private:
    CudaConfig::scaling_function_enum scaling_function_enum;
};
