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

#include "ActivityInput.h"

#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

class NeuronMonitor;

/**
 * @brief This class provides a normally distributed input
 *      Holds everything that is common to the CPU and GPU normal activity inputs;
 *      NormalActivityInputCPU and NormalActivityInputGPU add the parts that differ
 *      (update_input_range, and the curand state the GPU side needs to draw from).
 */
class NormalActivityInputBase : public ActivityInput {
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
     * @exception Throws a RelearnException if stddev_input <= 0.0
     */
    NormalActivityInputBase(const int _number_ranks, const activity_type mean_input, const activity_type stddev_input)
        : ActivityInput(_number_ranks)
        , mean(mean_input)
        , stddev(stddev_input) {
        RelearnException::check(stddev > activity_type{ 0 }, "NormalActivityInputBase::NormalActivityInputBase: The standard deviation must be larger than 0.0");
    }

    NormalActivityInputBase(const NormalActivityInputBase&) = delete;
    NormalActivityInputBase& operator=(const NormalActivityInputBase&) = delete;

    NormalActivityInputBase(NormalActivityInputBase&&) = default;
    NormalActivityInputBase& operator=(NormalActivityInputBase&&) = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("NormalActivityInput", total_size);
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

protected:
    activity_type mean{ default_mean_activity };
    activity_type stddev{ default_stddev_activity };
};
