#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BasicTypes.h"

#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/Synapse.h"
#include "util/NeuronID.h"

#include <vector>

// The synapses of the simulation, in every combination of the two weights and the four ways a synapse can
// cross a rank boundary: a local synapse connects two neurons of this rank, an in-synapse has its source on
// another rank, an out-synapse has its target there, and a distant synapse has both on other ranks.
//
// These aliases are not in RelearnTypes because they name the synapses of the whole project, which is also
// how they are spelled everywhere.

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
