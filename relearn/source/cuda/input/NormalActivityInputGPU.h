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

#include "Config.h"

#include "neurons/input/NormalActivityInputBase.h"
#include "cuda/random/RandomNumberHost.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/**
 * GPU implementation of NormalActivityInput: values are drawn on the device from a curand state
 * registered for this input at init()/create_neurons() time.
 */
class NormalActivityInputGPU : public NormalActivityInputBase {
public:
    NormalActivityInputGPU(const int _number_ranks, const activity_type mean_input, const activity_type stddev_input)
        : NormalActivityInputBase(_number_ranks, mean_input, stddev_input) { }

    NormalActivityInputGPU(const NormalActivityInputGPU&) = delete;
    NormalActivityInputGPU& operator=(const NormalActivityInputGPU&) = delete;

    NormalActivityInputGPU(NormalActivityInputGPU&&) = default;
    NormalActivityInputGPU& operator=(NormalActivityInputGPU&&) = default;

    ~NormalActivityInputGPU() override {
        cudaFree_bridge(d_curand_state);
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    std::vector<EventWrapper> update_input_range([[maybe_unused]] step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) override;

    void init(const number_neurons_type number_neurons) override {
        ActivityInput::init(number_neurons);

        init_gpu();
    }

    void create_neurons(const number_neurons_type creation_count) override {
        ActivityInput::create_neurons(creation_count);

        init_gpu();
    }

    void* d_curand_state = nullptr;

private:
    std::uint32_t random_key{};

    void init_gpu() {
        random_key = RandomNumbers::register_random_numbers(RandomNumberKey::NORMAL_INPUT, RandomNumberType::NORMAL, get_number_neurons(), Config::random_seed);
    }
};
