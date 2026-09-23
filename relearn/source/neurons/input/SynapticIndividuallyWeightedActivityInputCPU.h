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

#include "SynapticIndividuallyWeightedActivityInputBase.h"

#include <span>

/**
 * CPU implementation of SynapticIndividuallyWeightedActivityInput: update_local_input/
 * update_distant_input compute on host spans, reading per-synapse weights from the weight map.
 */
class SynapticIndividuallyWeightedActivityInputCPU : public SynapticIndividuallyWeightedActivityInputBase {
public:
    using SynapticIndividuallyWeightedActivityInputBase::SynapticIndividuallyWeightedActivityInputBase;

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;

    [[nodiscard]] const weight_vector_type& get_weights(const weight_map_type& weight_map_to_use, NeuronID target_neuron, RankNeuronId source_neuron) const;

    [[nodiscard]] const weight_vector_type& get_weights(const weight_map_type& weight_map_to_use, NeuronID::value_type target_neuron, RankNeuronId source_neuron) const;
};
