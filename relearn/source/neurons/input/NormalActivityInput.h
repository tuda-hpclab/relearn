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

/**
 * @brief NormalActivityInput resolves, at compile time, to the normal-activity-input
 *      implementation for this build: NormalActivityInputGPU when RELEARN_CUDA_ENABLED,
 *      NormalActivityInputCPU otherwise. A build only ever compiles one of the two, so this is a
 *      plain type alias rather than a runtime choice.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/NormalActivityInputGPU.h"
using NormalActivityInput = NormalActivityInputGPU;
#else
#include "NormalActivityInputCPU.h"
using NormalActivityInput = NormalActivityInputCPU;
#endif

#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <vector>

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/random/RandomNumberHost.h"
#include "cuda/util/Util.h"

#include <cstddef>
#include <span>
#endif

/**
 * @brief This class provides a normally distributed input, but the values are pre-drawn at the initialization
 */
class FastNormalActivityInput : public ActivityInput {
public:
    using activity_type = ActivityInput::activity_type;
    using number_neurons_type = ActivityInput::number_neurons_type;
    using step_type = ActivityInput::step_type;

    static constexpr activity_type default_mean_activity{ 0.0 };
    static constexpr activity_type min_mean_activity{ -10000.0 };
    static constexpr activity_type max_mean_activity{ 10000.0 };

    static constexpr activity_type default_stddev_activity{ 0.0 };
    static constexpr activity_type min_stddev_activity{ 0.0 };
    static constexpr activity_type max_stddev_activity{ 10000.0 };

    /**
     * @brief Constructs a new object with the given mean and standard deviation
     * @param mean_input The mean input
     * @param stddev_input The standard deviation input, > 0.0
     * @param _multiplier The multiplier of how many values are pre-drawn, > 0
     * @exception Throws a RelearnException if stddev_input <= 0.0
     */
    FastNormalActivityInput(const int _number_ranks, const activity_type mean_input, const activity_type stddev_input, const std::size_t _multiplier)
        : ActivityInput(_number_ranks)
        , mean(mean_input)
        , stddev(stddev_input)
        , multiplier(_multiplier) {
        RelearnException::check(stddev > activity_type{ 0 }, "FastNormalActivityInput::FastNormalActivityInput: The standard deviation must be larger than 0.0");
        RelearnException::check(_multiplier > 0, "FastNormalActivityInput::FastNormalActivityInput: The multiplier must be larger than 0");
#ifdef RELEARN_CUDA_ENABLED
        CPU_NOT_SUPPORTED
#endif
    }

    FastNormalActivityInput(const FastNormalActivityInput&) = delete;
    FastNormalActivityInput& operator=(const FastNormalActivityInput&) = delete;

    FastNormalActivityInput(FastNormalActivityInput&&) = delete;
    FastNormalActivityInput& operator=(FastNormalActivityInput&&) = delete;

#ifndef RELEARN_CUDA_ENABLED
    ~FastNormalActivityInput() override = default;
#endif

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(number_neurons_type number_neurons) override;

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(number_neurons_type creation_count) override;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
#ifdef RELEARN_CUDA_ENABLED
    std::vector<EventWrapper> update_input_range(step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) override;
#else
    void update_input_range(step_type step, NeuronID first, NeuronID last) override;
#endif

    using ActivityInput::update_input_range;

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] activity_type get_input(const NeuronID neuron_id) const override {
        const auto number_neurons = get_number_neurons();
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_neurons, "FastNormalActivityInput::get_input: id is too large: {}", neuron_id);
        return pre_drawn_values[offset + local_neuron_id];
    }

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] activity_type get_input(const NeuronID::value_type neuron_id) const override {
        const auto number_neurons = get_number_neurons();

        RelearnException::check(neuron_id < number_neurons, "FastNormalActivityInput::get_input: id is too large: {}", neuron_id);
        return pre_drawn_values[offset + neuron_id];
    }

    /**
     * @brief Returns the calculated background activity for all. Changes after calls to update_input_range(...)
     * @return The background activity for all neurons
     */
    [[nodiscard]] std::span<const activity_type> get_input() const noexcept override {
        const auto number_neurons = get_number_neurons();

        const auto* const pointer = pre_drawn_values.data();

        return std::span<const activity_type>{ pointer + offset, number_neurons };
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput) + (sizeof(activity_type) * pre_drawn_values.size());
        footprint->emplace("FastNormalActivityInput", total_size);
    }

    /**
     * Returns the mean of the normal distribution
     * @return Mean of the normal distribution
     */
    [[nodiscard]] activity_type get_mean() const noexcept {
        return mean;
    }

    /**
     * Standard deviation of the normal distribution
     * @return Deviation of the normal distribution
     */
    [[nodiscard]] activity_type get_stddev() const noexcept {
        return stddev;
    }

private:
    activity_type mean{ default_mean_activity };
    activity_type stddev{ default_stddev_activity };
    std::size_t multiplier{ 1 };
    std::size_t offset{ 0 };

    std::vector<activity_type, RelearnAllocator<activity_type>> pre_drawn_values;

#ifdef RELEARN_CUDA_ENABLED
public:
    ~FastNormalActivityInput() override = default;

    std::size_t get_offset() override {
        return offset;
    }

private:
#endif
};
