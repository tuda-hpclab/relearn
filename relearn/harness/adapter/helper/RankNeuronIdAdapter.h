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

#include "neurons/helper/RankNeuronId.h"

#include <string>

class RankNeuronIdAdapter {
public:
    static std::string codify_rank_neuron_id(const RankNeuronId& rni);

    static RankNeuronId add_one_to_neuron_id(const RankNeuronId& rni);

    static RankNeuronId substract_one_from_neuron_id(const RankNeuronId& rni);
};
