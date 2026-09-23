#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "types/SynapseTypes.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

class Essentials;
class Partition;

/**
 * SynapseLoader is a type that abstracts away the mechanics of how synapses are loaded.
 */
class SynapseLoader {
protected:
    using plastic_synapses = std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses>;
    using static_synapses = std::tuple<StaticLocalSynapses, StaticDistantInSynapses, StaticDistantOutSynapses>;

    using synapses_pair_type = std::pair<static_synapses, plastic_synapses>;

public:
    /**
     * @brief Constructs a SynapseLoader with the given Partition
     * @param _partition The partition to use
     */
    explicit SynapseLoader(std::shared_ptr<Partition> _partition)
        : partition(std::move(_partition)) { }

    virtual ~SynapseLoader() = default;

    SynapseLoader(const SynapseLoader& other) = default;
    SynapseLoader(SynapseLoader&& other) = default;

    SynapseLoader& operator=(const SynapseLoader& other) = default;
    SynapseLoader& operator=(SynapseLoader&& other) = default;

    /**
     * @brief Loads all synapses that affect the local neurons, which are
     *      (1) local synapses (local neuron to local neuron)
     *      (2) in synapses (non-local neuron to local neuron)
     *      (3) out synapses (local neuron to non-local neuron)
     *      for both plastic and static synapses
     * @param essentials The essentials to print the metrics
     * @return Two tuples of (local, in, out) synapses, the first one being static synapses, the second one being plastic synapses
     */
    synapses_pair_type load_synapses(const std::unique_ptr<Essentials>& essentials);

protected:
    std::shared_ptr<Partition> partition;

    virtual synapses_pair_type internal_load_synapses() = 0;
};
