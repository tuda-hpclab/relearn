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

/**
 * This class calculates the synaptic input with each synapse having the same absolute weight of 1.
 * The synaptic input is the sum of all synapses whose source neuron fired in the last step.
 * Inhibitory synapses have a negative weight.
 */
class SynapticEquallyWeightedActivityInput : public SynapticActivityInput {
public:
    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticEquallyWeightedActivityInput(std::shared_ptr<FiredStatusCommunicator> communicator)
        : SynapticActivityInput(std::move(communicator)) {
    }

    SynapticEquallyWeightedActivityInput(const SynapticEquallyWeightedActivityInput&) = default;
    SynapticEquallyWeightedActivityInput& operator=(const SynapticEquallyWeightedActivityInput&) = default;

    SynapticEquallyWeightedActivityInput(SynapticEquallyWeightedActivityInput&&) = default;
    SynapticEquallyWeightedActivityInput& operator=(SynapticEquallyWeightedActivityInput&&) = default;

    ~SynapticEquallyWeightedActivityInput() override = default;

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<double> input, const NeuronID first, const NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<double> input, const NeuronID first, const NeuronID last) override;
};
