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

#include "sim/SynapseLoader.h"

#include <memory>
#include <utility>
#include <vector>

class Partition;

class RandomSynapseLoader : public SynapseLoader {
public:
    /**
     * @brief Constructs a RandomSynapseLoader with the given Partition.
     *      Does not provide any synapses
     * @param _partition The partition to use
     */
    explicit RandomSynapseLoader(std::shared_ptr<Partition> _partition)
        : SynapseLoader(std::move(_partition)) { }

protected:
    synapses_pair_type internal_load_synapses() override {
        return synapses_pair_type{};
    }
};