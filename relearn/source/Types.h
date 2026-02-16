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

#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/Synapse.h"
#include "util/BoundingBox.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace RelearnTypes {
// These types are 'easy'

using box_size_type = Vec3d;
using position_type = Vec3d;
using bounding_box_type = BoundingBox<box_size_type::value_type>;

using plastic_synapse_weight = int;
using static_synapse_weight = double;

using neuron_id = std::size_t;

using counter_type = unsigned int;

using calcium_type = double;

using step_type = std::uint32_t;
using number_neurons_type = std::uint64_t;

using group_name = std::string;
using group_id = std::size_t;

using group_names = std::vector<group_name>;
using group_names_unordered = std::unordered_set<group_name>;

using group_ids = std::vector<group_id>;
using group_ids_unordered = std::unordered_set<group_id>;

} // namespace RelearnTypes

using PlasticLocalSynapse = Synapse<NeuronID, NeuronID, RelearnTypes::plastic_synapse_weight>;
using PlasticDistantInSynapse = Synapse<NeuronID, RankNeuronId, RelearnTypes::plastic_synapse_weight>;
using PlasticDistantOutSynapse = Synapse<RankNeuronId, NeuronID, RelearnTypes::plastic_synapse_weight>;
using PlasticDistantSynapse = Synapse<RankNeuronId, RankNeuronId, RelearnTypes::plastic_synapse_weight>;

using PlasticLocalSynapses = std::vector<PlasticLocalSynapse>;
using PlasticDistantInSynapses = std::vector<PlasticDistantInSynapse>;
using PlasticDistantOutSynapses = std::vector<PlasticDistantOutSynapse>;
using PlasticDistantSynapses = std::vector<PlasticDistantSynapse>;

using StaticLocalSynapse = Synapse<NeuronID, NeuronID, RelearnTypes::static_synapse_weight>;
using StaticDistantInSynapse = Synapse<NeuronID, RankNeuronId, RelearnTypes::static_synapse_weight>;
using StaticDistantOutSynapse = Synapse<RankNeuronId, NeuronID, RelearnTypes::static_synapse_weight>;
using StaticDistantSynapse = Synapse<RankNeuronId, RankNeuronId, RelearnTypes::static_synapse_weight>;

using StaticLocalSynapses = std::vector<StaticLocalSynapse>;
using StaticDistantInSynapses = std::vector<StaticDistantInSynapse>;
using StaticDistantOutSynapses = std::vector<StaticDistantOutSynapse>;
using StaticDistantSynapses = std::vector<StaticDistantSynapse>;
