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

#include "Types.h"

#include "neurons/enums/SynapticElementType.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include "mpi-wrapper/MPIRank.h"

#include <filesystem>
#include <memory>
#include <random>
#include <utility>
#include <vector>

class SynapsesFactory {
public:
    constexpr static RelearnTypes::static_synapse_weight bound_static_synapse_weight = 10.0;
    constexpr static RelearnTypes::plastic_synapse_weight bound_plastic_synapse_weight = 10;
    constexpr static int upper_bound_num_synapses = 100;

    static NeuronID::value_type get_random_number_synapses(std::mt19937& mt);

    static RelearnTypes::plastic_synapse_weight get_random_plastic_synapse_weight(std::mt19937& mt);

    static RelearnTypes::static_synapse_weight get_random_static_synapse_weight(std::mt19937& mt);

    static std::vector<std::tuple<NeuronID, NeuronID, RelearnTypes::plastic_synapse_weight>> get_random_plastic_synapses(size_t number_neurons, size_t number_synapses, std::mt19937& mt);

    static std::vector<PlasticLocalSynapse> generate_all_to_all(NeuronID::value_type number_neurons, std::mt19937& mt);

    static std::vector<PlasticLocalSynapse> generate_derangements(NeuronID::value_type number_neurons, NeuronID::value_type number_derangements, std::mt19937& mt);

    static std::vector<PlasticLocalSynapse> generate_plastic_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::mt19937& mt);

    static std::vector<StaticLocalSynapse> generate_static_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::mt19937& mt);

    static std::vector<PlasticDistantInSynapse> generate_plastic_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt);

    static std::vector<StaticDistantInSynapse> generate_static_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt);

    static std::vector<PlasticDistantOutSynapse> generate_plastic_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt);

    static std::vector<StaticDistantOutSynapse> generate_static_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt);

    static std::vector<PlasticDistantInSynapse> generate_plastic_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses,
                                                                                     std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt);

    static std::vector<StaticDistantInSynapse> generate_static_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses,
                                                                                   std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt);

    static std::vector<PlasticDistantOutSynapse> generate_plastic_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses,
                                                                                       std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt);

    static std::vector<StaticDistantOutSynapse> generate_static_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses,
                                                                                     std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt);

    static std::vector<PlasticLocalSynapse> generate_local_synapses(NeuronID::value_type number_neurons, std::mt19937& mt);

    static PlasticLocalSynapses generate_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron);

    static PlasticDistantInSynapses generate_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron);

    static PlasticDistantOutSynapses generate_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron);

    static void generate_random_network(const std::filesystem::path& directory, const std::vector<NeuronID::value_type>& number_neurons_per_rank,
                                        const std::vector<NeuronID::value_type>& number_local_synapses, const std::vector<NeuronID::value_type>& number_distant_in_synapses, std::mt19937& mt);
};
