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

#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"

#include <boost/functional/hash.hpp>

#include <unordered_map>
#include <utility>
#include <vector>

/**
 * Holds everything that is common to the CPU and GPU flavors; SynapticIndividuallyWeightedActivityInputCPU
 * and SynapticIndividuallyWeightedActivityInputGPU add the parts that differ (update_local_input/
 * update_distant_input's signature and body; the GPU flavor doesn't use the weight map at all).
 */
class SynapticIndividuallyWeightedActivityInputBase : public SynapticActivityInput {
public:
    using weight_type = activity_type;
    using weight_vector_type = std::vector<weight_type>;
    using weight_map_type = std::unordered_map<std::pair<NeuronID::value_type, RankNeuronId>, weight_vector_type, boost::hash<std::pair<NeuronID::value_type, RankNeuronId>>>; // first neuron in pair resembles target neuron, second resembles source neuron

    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticIndividuallyWeightedActivityInputBase(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator)
        : SynapticActivityInput(_number_ranks, std::move(communicator)) {
    }

    SynapticIndividuallyWeightedActivityInputBase(const SynapticIndividuallyWeightedActivityInputBase&) = delete;
    SynapticIndividuallyWeightedActivityInputBase& operator=(const SynapticIndividuallyWeightedActivityInputBase&) = delete;

    SynapticIndividuallyWeightedActivityInputBase(SynapticIndividuallyWeightedActivityInputBase&&) = default;
    SynapticIndividuallyWeightedActivityInputBase& operator=(SynapticIndividuallyWeightedActivityInputBase&&) = default;

    ~SynapticIndividuallyWeightedActivityInputBase() override = default;

    void set_weight_map(weight_map_type&& new_weight_map) noexcept {
        weight_map = std::move(new_weight_map);
    }

protected:
    weight_map_type get_weight_map() const noexcept {
        return weight_map;
    }

    weight_map_type weight_map;
};
