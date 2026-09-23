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

#include "neurons/input/SynapticActivityInputBase.h"
#include "cuda/input/Handle.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"

#include <memory>
#include <optional>
#include <vector>

/**
 * GPU implementation of SynapticActivityInput: update_local_input/update_distant_input work on
 * device pointers and return the CUDA events they queued.
 *
 * The base class's ActivityInput::_input is never populated on the GPU path -- production code
 * (NeuronModel::get_d_input()) consumes d_input_local/d_input_distant directly as device pointers
 * and never touches _input. The get_input()/get_input_arr() overrides here make the host-visible
 * getters (used by tests and any other CPU-side inspection) reflect the actual GPU-computed values
 * by combining the two device arrays on demand instead of reading the always-zero base class array.
 */
class SynapticActivityInputGPU : public SynapticActivityInputBase {
public:
    explicit SynapticActivityInputGPU(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator)
        : SynapticActivityInputBase(_number_ranks, std::move(communicator)) {
        local_stream = std::make_shared<StreamWrapper>();
        distant_stream = std::make_shared<StreamWrapper>();
    }

    SynapticActivityInputGPU(const SynapticActivityInputGPU&) = delete;
    SynapticActivityInputGPU& operator=(const SynapticActivityInputGPU&) = delete;

    SynapticActivityInputGPU(SynapticActivityInputGPU&&) = default;
    SynapticActivityInputGPU& operator=(SynapticActivityInputGPU&&) = default;

    // d_scales is a non-owning pointer into a subclass-owned LazySyncedArray (see e.g.
    // SynapticScalingActivityInput::scales) -- nothing to free here.
    ~SynapticActivityInputGPU() override = default;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    std::vector<EventWrapper> update_input_range(step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) override;

    void init(const number_neurons_type number_neurons) override {
        ActivityInput::init(number_neurons);
        d_input_local = LazySyncedArray<activity_type>{ 0, local_stream };
        d_input_local.resize(number_neurons, 0);
        d_input_distant = LazySyncedArray<activity_type>{ 0, distant_stream };
        d_input_distant.resize(number_neurons, 0);
    }

    void create_neurons(const number_neurons_type creation_count) override {
        ActivityInput::create_neurons(creation_count);
        d_input_local.resize(get_number_neurons(), 0);
        d_input_distant.resize(get_number_neurons(), 0);
    }

    [[nodiscard]] activity_type get_input(const NeuronID neuron_id) const override {
        return get_input(neuron_id.get_neuron_id());
    }

    [[nodiscard]] activity_type get_input(const NeuronID::value_type neuron_id) const override {
        const auto local = d_input_local.host();
        const auto distant = d_input_distant.host();
        RelearnException::check(neuron_id < local.size(), "SynapticActivityInputGPU::get_input: NeuronID {} is requested, but only {} are stored", neuron_id, local.size());
        return local[neuron_id] + distant[neuron_id];
    }

    [[nodiscard]] std::span<const activity_type> get_input() const noexcept override {
        const auto local = d_input_local.host();
        const auto distant = d_input_distant.host();
        combined_input_cache.resize(local.size());
        for (std::size_t i = 0; i < local.size(); ++i) {
            combined_input_cache[i] = local[i] + distant[i];
        }
        return combined_input_cache;
    }

protected:
    const activity_type* d_scales{};

    virtual std::optional<EventWrapper> update_local_input(number_neurons_type first, number_neurons_type last, activity_type* d_input,
                                                           const FiredStatus* d_fired,
                                                           const activity_type* d_scales, const std::shared_ptr<StreamWrapper>& stream_wrapper) = 0;

    virtual std::optional<EventWrapper> update_distant_input(const number_neurons_type first, const number_neurons_type last, activity_type* d_input, const activity_type* /*d_scales*/,
                                                             std::unique_ptr<FireStatusCommunicatorHandle>& fire_status_handle, const std::shared_ptr<StreamWrapper>& stream_wrapper) = 0;

    [[nodiscard]] std::vector<const LazySyncedArray<activity_type>*> get_input_arr_const() const override;
    [[nodiscard]] std::vector<LazySyncedArray<activity_type>*> get_input_arr() override;

private:
    std::shared_ptr<StreamWrapper> local_stream;
    std::shared_ptr<StreamWrapper> distant_stream;
    LazySyncedArray<activity_type> d_input_local;
    LazySyncedArray<activity_type> d_input_distant;
    mutable std::vector<activity_type> combined_input_cache;
};
