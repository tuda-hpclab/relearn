/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "synapses_factory.h"

#include "Types.h"

#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIRank.h"

#include "adapter/network_graph/NetworkGraphAdapter.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

NeuronID::value_type SynapsesFactory::get_random_number_synapses(std::mt19937& mt) {
    return RandomFactory::get_random_integer<NeuronID::value_type>(1, upper_bound_num_synapses, mt);
}

RelearnTypes::plastic_synapse_weight SynapsesFactory::get_random_plastic_synapse_weight(std::mt19937& mt) {
    auto weight = RandomFactory::get_random_integer<RelearnTypes::plastic_synapse_weight>(-bound_plastic_synapse_weight, bound_plastic_synapse_weight, mt);

    while (weight == 0) {
        weight = RandomFactory::get_random_integer<RelearnTypes::plastic_synapse_weight>(-bound_plastic_synapse_weight, bound_plastic_synapse_weight, mt);
    }

    return weight;
}

RelearnTypes::static_synapse_weight SynapsesFactory::get_random_static_synapse_weight(std::mt19937& mt) {
    auto weight = RandomFactory::get_random_double<RelearnTypes::static_synapse_weight>(-bound_static_synapse_weight, bound_static_synapse_weight, mt);

    while (weight == 0) {
        weight = RandomFactory::get_random_double<RelearnTypes::static_synapse_weight>(-bound_static_synapse_weight, bound_static_synapse_weight, mt);
    }

    return weight;
}

std::vector<std::tuple<NeuronID, NeuronID, RelearnTypes::plastic_synapse_weight>> SynapsesFactory::get_random_plastic_synapses(size_t number_neurons, size_t number_synapses, std::mt19937& mt) {
    auto synapses = std::vector<std::tuple<NeuronID, NeuronID, RelearnTypes::plastic_synapse_weight>>(number_synapses);

    for (auto i = 0ULL; i < number_synapses; i++) {
        const auto source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto target_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = get_random_plastic_synapse_weight(mt);

        synapses[i] = { source_id, target_id, weight };
    }

    return synapses;
}

std::vector<PlasticLocalSynapse> SynapsesFactory::generate_all_to_all(NeuronID::value_type number_neurons, std::mt19937& mt) {
    auto synapses = std::vector<PlasticLocalSynapse>{};
    synapses.reserve(number_neurons * number_neurons);

    for (const auto& source_id : NeuronID::range(number_neurons)) {
        for (const auto& target_id : NeuronID::range(number_neurons)) {
            if (source_id.get_neuron_id() == target_id.get_neuron_id()) {
                continue;
            }

            const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
            synapses.emplace_back(target_id, source_id, weight);
        }
    }

    return synapses;
}

std::vector<PlasticLocalSynapse> SynapsesFactory::generate_derangements(NeuronID::value_type number_neurons, NeuronID::value_type number_derangements, std::mt19937& mt) {
    auto synapses = std::vector<PlasticLocalSynapse>{};
    synapses.reserve(number_neurons * number_derangements);

    for (auto i = 0ULL; i < number_derangements; i++) {
        const auto& source_ids = NeuronID::range(number_neurons);
        const auto& target_ids = RandomFactory::get_random_derangement(number_neurons, mt);

        for (auto j = 0ULL; j < number_neurons; j++) {
            const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
            synapses.emplace_back(NeuronID(false, target_ids[j]), source_ids[j], weight);
        }
    }

    return synapses;
}

std::vector<PlasticLocalSynapse> SynapsesFactory::generate_plastic_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::plastic_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<PlasticLocalSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<StaticLocalSynapse> SynapsesFactory::generate_static_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::static_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<StaticLocalSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<PlasticDistantInSynapse> SynapsesFactory::generate_plastic_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<NeuronID, RankNeuronId>, RelearnTypes::plastic_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons, mt);
        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mpiPP::MPIRank::root_rank(), mt);
        const auto source = RankNeuronId{ source_rank, source_id };

        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<PlasticDistantInSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<StaticDistantInSynapse> SynapsesFactory::generate_static_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<NeuronID, RankNeuronId>, RelearnTypes::static_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons, mt);
        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mpiPP::MPIRank::root_rank(), mt);
        const auto source = RankNeuronId{ source_rank, source_id };

        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<StaticDistantInSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<PlasticDistantOutSynapse> SynapsesFactory::generate_plastic_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<RankNeuronId, NeuronID>, RelearnTypes::plastic_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons, mt);
        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mpiPP::MPIRank::root_rank(), mt);
        const auto target = RankNeuronId{ target_rank, target_id };

        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<PlasticDistantOutSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<StaticDistantOutSynapse> SynapsesFactory::generate_static_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, int number_ranks, NeuronID::value_type number_foreign_neurons, std::mt19937& mt) {
    auto synapse_map = std::map<std::pair<RankNeuronId, NeuronID>, RelearnTypes::static_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons, mt);
        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mpiPP::MPIRank::root_rank(), mt);
        const auto target = RankNeuronId{ target_rank, target_id };

        const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<StaticDistantOutSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<PlasticDistantInSynapse> SynapsesFactory::generate_plastic_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt) {
    const auto number_ranks = number_foreign_neurons.size();

    auto synapse_map = std::map<std::pair<NeuronID, RankNeuronId>, RelearnTypes::plastic_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, my_rank, mt);
        const auto source_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons[source_rank.get_rank_cast()], mt);
        const auto source = RankNeuronId{ source_rank, source_id };

        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<PlasticDistantInSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<StaticDistantInSynapse> SynapsesFactory::generate_static_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt) {
    const auto number_ranks = number_foreign_neurons.size();

    auto synapse_map = std::map<std::pair<NeuronID, RankNeuronId>, RelearnTypes::static_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, my_rank, mt);
        const auto source_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons[source_rank.get_rank_cast()], mt);
        const auto source = RankNeuronId{ source_rank, source_id };

        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<StaticDistantInSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<PlasticDistantOutSynapse> SynapsesFactory::generate_plastic_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt) {
    const auto number_ranks = number_foreign_neurons.size();

    auto synapse_map = std::map<std::pair<RankNeuronId, NeuronID>, RelearnTypes::plastic_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, my_rank, mt);
        const auto target_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons[target_rank.get_rank_cast()], mt);
        const auto target = RankNeuronId{ target_rank, target_id };

        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<PlasticDistantOutSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<StaticDistantOutSynapse> SynapsesFactory::generate_static_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses, std::vector<NeuronID::value_type> number_foreign_neurons, mpiPP::MPIRank my_rank, std::mt19937& mt) {
    const auto number_ranks = number_foreign_neurons.size();

    auto synapse_map = std::map<std::pair<RankNeuronId, NeuronID>, RelearnTypes::static_synapse_weight>{};
    auto added_synapses = NeuronID::value_type{ 0 };
    while (added_synapses < number_synapses) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, my_rank, mt);
        const auto target_id = NeuronIdFactory::get_random_neuron_id(number_foreign_neurons[target_rank.get_rank_cast()], mt);
        const auto target = RankNeuronId{ target_rank, target_id };

        const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
        added_synapses++;
    }

    auto synapses = std::vector<StaticDistantOutSynapse>{};
    synapses.reserve(synapse_map.size());

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [target, source] = pair;
        if (weight != 0) {
            synapses.emplace_back(target, source, weight);
        } else {
            synapses.emplace_back(target, source, 1);
        }
    }

    return synapses;
}

std::vector<PlasticLocalSynapse> SynapsesFactory::generate_local_synapses(NeuronID::value_type number_neurons, std::mt19937& mt) {
    const auto number_synapses = get_random_number_synapses(mt);

    auto synapse_map = std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::plastic_synapse_weight>{};
    for (auto i = 0ULL; i < number_synapses; i++) {
        const auto source = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto target = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto weight = get_random_plastic_synapse_weight(mt);

        synapse_map[{ target, source }] += weight;
    }

    return synapse_map | ranges::views::filter(utility::not_equal_to(0), detail::to_edge_weight) | ranges::views::transform([](const auto& val) -> PlasticLocalSynapse {
               return { val.first.first, val.first.second, detail::to_edge_weight(val) };
           })
           | ranges::to_vector;
}

PlasticLocalSynapses SynapsesFactory::generate_local_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron) {
    auto synapses = std::vector<PlasticLocalSynapse>{};
    synapses.reserve(number_neurons * number_synapses_per_neuron);

    auto mt = std::mt19937{};
    auto uid = std::uniform_int_distribution<std::uint64_t>(0, number_neurons - 1);

    for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
            auto random_id = uid(mt);

            const auto source_id = NeuronID{ neuron_id };
            const auto target_id = NeuronID{ random_id };

            const auto weight = 1;

            synapses.emplace_back(target_id, source_id, weight);
        }
    }

    return synapses;
}

PlasticDistantInSynapses SynapsesFactory::generate_distant_in_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron) {
    auto synapses = std::vector<PlasticDistantInSynapse>{};
    synapses.reserve(number_neurons * number_synapses_per_neuron);

    auto mt = std::mt19937{};
    auto uid = std::uniform_int_distribution<std::uint64_t>(0, number_neurons - 1);
    auto uid_rank = std::uniform_int_distribution<int>(1, 31);

    for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
            auto random_id = uid(mt);
            auto random_rank = uid_rank(mt);

            const auto source_id = NeuronID{ random_id };
            const auto target_id = NeuronID{ neuron_id };

            const auto rni = RankNeuronId{ mpiPP::MPIRank(random_rank), source_id };

            const auto weight = 1;

            synapses.emplace_back(target_id, rni, weight);
        }
    }

    return synapses;
}

PlasticDistantOutSynapses SynapsesFactory::generate_distant_out_synapses(NeuronID::value_type number_neurons, NeuronID::value_type number_synapses_per_neuron) {
    auto synapses = std::vector<PlasticDistantOutSynapse>{};
    synapses.reserve(number_neurons * number_synapses_per_neuron);

    auto mt = std::mt19937{};
    auto uid = std::uniform_int_distribution<std::uint64_t>(0, number_neurons - 1);
    auto uid_rank = std::uniform_int_distribution<int>(1, 31);

    for (auto neuron_id = 0ULL; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0ULL; synapse_id < number_synapses_per_neuron; synapse_id++) {
            auto random_id = uid(mt);
            auto random_rank = uid_rank(mt);

            const auto source_id = NeuronID{ neuron_id };
            const auto target_id = NeuronID{ random_id };

            const auto rni = RankNeuronId{ mpiPP::MPIRank(random_rank), target_id };

            const auto weight = 1;

            synapses.emplace_back(rni, source_id, weight);
        }
    }

    return synapses;
}

void SynapsesFactory::generate_random_network(const std::filesystem::path& directory, const std::vector<NeuronID::value_type>& number_neurons_per_rank, const std::vector<NeuronID::value_type>& number_local_synapses, const std::vector<NeuronID::value_type>& number_distant_in_synapses, std::mt19937& mt) {

    const auto number_ranks = number_neurons_per_rank.size();

    auto local_synapses = std::vector<PlasticLocalSynapses>{};
    local_synapses.reserve(number_ranks);

    auto distant_in_synapses = std::vector<PlasticDistantInSynapses>{};
    distant_in_synapses.reserve(number_ranks);

    auto distant_out_synapses = std::vector<PlasticDistantOutSynapses>{ number_ranks, PlasticDistantOutSynapses{} };

    for (auto rank = std::size_t{ 0 }; rank < number_ranks; rank++) {
        auto locals = SynapsesFactory::generate_plastic_local_synapses(number_neurons_per_rank[rank], number_local_synapses[rank], mt);
        local_synapses.emplace_back(std::move(locals));

        const auto target_rank = mpiPP::MPIRank{ static_cast<int>(rank) };
        auto distances = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons_per_rank[rank], number_distant_in_synapses[rank], number_neurons_per_rank, target_rank, mt);
        for (const auto& [target_id, source_rni, weight] : distances) {
            const auto& [source_rank, source_id] = source_rni;

            distant_out_synapses[source_rank.get_rank_cast()].emplace_back(RankNeuronId{ target_rank, target_id }, source_id, weight);
        }

        distant_in_synapses.emplace_back(std::move(distances));
    }

    for (auto rank = std::size_t{ 0 }; rank < number_ranks; rank++) {
        auto in_network_path = directory / ("rank_" + std::to_string(rank) + "_in_network.txt");
        auto in_network_file = std::ofstream{ in_network_path };

        auto out_network_path = directory / ("rank_" + std::to_string(rank) + "_out_network.txt");
        auto out_network_file = std::ofstream{ out_network_path };

        for (const auto& [target_id, source_id, weight] : local_synapses[rank]) {
            in_network_file << rank << ' ' << (target_id.get_neuron_id() + 1) << '\t' << rank << ' ' << (source_id.get_neuron_id() + 1) << '\t' << weight << "\t1\n";
            out_network_file << rank << ' ' << (target_id.get_neuron_id() + 1) << '\t' << rank << ' ' << (source_id.get_neuron_id() + 1) << '\t' << weight << "\t1\n";
        }

        for (const auto& [target_id, source_rni, weight] : distant_in_synapses[rank]) {
            const auto& [source_rank, source_id] = source_rni;

            in_network_file << rank << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                            << source_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << '\t' << weight << "\t1\n";
        }

        for (const auto& [target_rni, source_id, weight] : distant_out_synapses[rank]) {
            const auto& [target_rank, target_id] = target_rni;

            out_network_file << target_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                             << rank << ' ' << (source_id.get_neuron_id() + 1) << '\t' << weight << "\t1\n";
        }

        in_network_file.flush();
        in_network_file.close();

        out_network_file.flush();
        out_network_file.close();
    }
}
