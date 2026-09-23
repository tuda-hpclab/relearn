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

#include "sim/random/BoxBasedRandomSubdomainAssignment.h"
#include "types/BasicTypes.h"

#include <memory>

class Partition;

/**
 * This class fills every subdomain with neurons at random positions. It assigns a fixed number
 * of neurons to each MPI rank.
 * It inherits from BoxBasedRandomSubdomainAssignment.
 */
class SubdomainFromNeuronPerRank : public BoxBasedRandomSubdomainAssignment {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object with the specified parameters
     * @param _number_neurons_per_rank The number of neurons per MPI rank, must be > 0
     * @param fraction_excitatory_neurons The fraction of excitatory neurons, must be in [0.0, 1.0]
     * @param um_per_neuron The box length in which a single neuron is placed, must be > 0.0
     * @param _partition The partition that stores all information for the subdomain calculations
     * @exception Throws a RelearnException if number_neurons_per_rank == 0, fraction_excitatory_neurons is not from [0.0, 1.0], or um_per_neuron <= 0.0
     */
    SubdomainFromNeuronPerRank(number_neurons_type _number_neurons_per_rank, percentage_type fraction_excitatory_neurons, space_type um_per_neuron, std::shared_ptr<Partition> _partition);

    SubdomainFromNeuronPerRank(const SubdomainFromNeuronPerRank& other) = delete;
    SubdomainFromNeuronPerRank(SubdomainFromNeuronPerRank&& other) = delete;

    SubdomainFromNeuronPerRank& operator=(const SubdomainFromNeuronPerRank& other) = delete;
    SubdomainFromNeuronPerRank& operator=(SubdomainFromNeuronPerRank&& other) = delete;

    ~SubdomainFromNeuronPerRank() override = default;

    /**
     * @brief Prints relevant metrics to the essentials
     * @param essentials The essentials
     */
    void print_essentials(const std::unique_ptr<Essentials>& essentials) override;

protected:
    void fill_all_subdomains() override;

private:
    number_neurons_type number_neurons_per_rank{ Constants::uninitialized };
};
