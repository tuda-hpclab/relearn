/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RankNeuronIdAdapter.h"

#include "neurons/helper/RankNeuronId.h"

#include <sstream>
#include <string>

std::string RankNeuronIdAdapter::codify_rank_neuron_id(const RankNeuronId& rni) {
    auto ss = std::stringstream{};
    ss << rni.get_rank().get_rank() << ':' << (rni.get_neuron_id().get_neuron_id() + 1);
    return ss.str();
}

RankNeuronId RankNeuronIdAdapter::add_one_to_neuron_id(const RankNeuronId& rni) {
    const auto& [rank, neuron_id] = rni;
    const auto id = neuron_id.get_neuron_id();
    return { rank, NeuronID(id + 1) };
}

RankNeuronId RankNeuronIdAdapter::substract_one_from_neuron_id(const RankNeuronId& rni) {
    const auto& [rank, neuron_id] = rni;
    const auto id = neuron_id.get_neuron_id();
    return { rank, NeuronID(id - 1) };
}
