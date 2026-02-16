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

#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"

#include "mpi-wrapper/MPIInfo.h"

#include <boost/functional/hash.hpp>

#include <unordered_map>
#include <utility>

class SynapticIndividuallyWeightedActivityInput : public SynapticActivityInput {
public:
    using weight_type = double;
    using weight_vector_type = std::vector<weight_type>;
    using weight_map_type = std::unordered_map<std::pair<NeuronID::value_type, RankNeuronId>, weight_vector_type, boost::hash<std::pair<NeuronID::value_type, RankNeuronId>>>; // first neuron in pair resembles target neuron, second resembles source neuron

    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticIndividuallyWeightedActivityInput(std::shared_ptr<FiredStatusCommunicator> communicator)
        : SynapticActivityInput(std::move(communicator)) {
    }

    SynapticIndividuallyWeightedActivityInput(const SynapticIndividuallyWeightedActivityInput&) = default;
    SynapticIndividuallyWeightedActivityInput& operator=(const SynapticIndividuallyWeightedActivityInput&) = default;

    SynapticIndividuallyWeightedActivityInput(SynapticIndividuallyWeightedActivityInput&&) = default;
    SynapticIndividuallyWeightedActivityInput& operator=(SynapticIndividuallyWeightedActivityInput&&) = default;

    ~SynapticIndividuallyWeightedActivityInput() override = default;

    void set_weight_map(weight_map_type&& new_weight_map) noexcept {
        weight_map = std::move(new_weight_map);
    }

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) override;

    [[nodiscard]] const weight_vector_type& get_weights(const weight_map_type& weight_map_to_use, const NeuronID target_neuron, const RankNeuronId source_neuron) const;

    [[nodiscard]] const weight_vector_type& get_weights(const weight_map_type& weight_map_to_use, const NeuronID::value_type target_neuron, const RankNeuronId source_neuron) const;

    weight_map_type get_weight_map() const noexcept {
        return weight_map;
    }

    weight_map_type weight_map{};
};