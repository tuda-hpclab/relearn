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

#include "SynapticActivityInput.h"

#include <memory>

/**
 * This class calculates the synaptic input with each synapse having the same absolute weight of 1.
 * The synaptic input is the sum of all synapses whose source neuron fired in the last step.
 * Inhibitory synapses have a negative weight.
 *      Holds everything that is common to the CPU and GPU flavors; SynapticEquallyWeightedActivityInputCPU
 *      and SynapticEquallyWeightedActivityInputGPU add the parts that differ (update_local_input/
 *      update_distant_input's signature and body).
 */
class SynapticEquallyWeightedActivityInputBase : public SynapticActivityInput {
public:
    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticEquallyWeightedActivityInputBase(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator, activity_type synapse_conductance_)
        : SynapticActivityInput(_number_ranks, std::move(communicator))
        , synapse_conductance(synapse_conductance_) {
    }

    SynapticEquallyWeightedActivityInputBase(const SynapticEquallyWeightedActivityInputBase&) = delete;
    SynapticEquallyWeightedActivityInputBase& operator=(const SynapticEquallyWeightedActivityInputBase&) = delete;

    SynapticEquallyWeightedActivityInputBase(SynapticEquallyWeightedActivityInputBase&&) = default;
    SynapticEquallyWeightedActivityInputBase& operator=(SynapticEquallyWeightedActivityInputBase&&) = default;

    ~SynapticEquallyWeightedActivityInputBase() override = default;

    activity_type synapse_conductance{};
};
