#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticActivityInput.h"

#include <memory>
#include <span>
#include <vector>

/**
 * This class calculates the synaptic input with each synapse having a relative weight of the number
 * of incoming synapses to a neuron.
 * The synaptic input is the sum of all synapses whose source neuron fired in the last step.
 * Inhibitory synapses have a negative weight.
 */
class SynapticScalingActivityInput : public SynapticActivityInput {
public:
    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @param total_scale The total scale for all synapses, must be > 0
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticScalingActivityInput(std::shared_ptr<FiredStatusCommunicator> communicator, const double total_scale = 1.0)
        : SynapticActivityInput(std::move(communicator))
        , t_scale{ total_scale } {
        RelearnException::check(total_scale > 0.0, "Total scale must be > 0");
    }

    SynapticScalingActivityInput(const SynapticScalingActivityInput&) = default;
    SynapticScalingActivityInput& operator=(const SynapticScalingActivityInput&) = default;

    SynapticScalingActivityInput(SynapticScalingActivityInput&&) = default;
    SynapticScalingActivityInput& operator=(SynapticScalingActivityInput&&) = default;

    ~SynapticScalingActivityInput() override = default;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        SynapticActivityInput::init(number_neurons);

        scales.resize(number_neurons, 0.0);
        needs_scale_update = true;
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        SynapticActivityInput::create_neurons(creation_count);

        const auto old_size = scales.size();
        scales.resize(old_size + creation_count, 0.0);
        needs_scale_update = true;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        SynapticActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(SynapticActivityInput) + scales.size() * sizeof(double);
        footprint->emplace("SynapticScalingActivityInput", total_size);
    }

    /**
     * @brief Returns a view of the scales
     * @return The scales
     */
    [[nodiscard]] std::span<const double> get_scales() const noexcept {
        return scales;
    }

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) override;

private:
    void ensure_scales();

    std::vector<double> scales{};
    double t_scale{ 1.0 };
    bool needs_scale_update{ false };
};
