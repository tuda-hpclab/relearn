#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/LocalGroupTranslator.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * Class finds the group ids of neurons on other ranks through mpi communication
 */
class GlobalGroupMapper {
public:
    /**
     * Constructor
     * @param _local_group_translator The local group translator
     * @param number_ranks Number of mpi ranks
     * @param my_rank Current mpi rank
     */
    GlobalGroupMapper(std::shared_ptr<LocalGroupTranslator> _local_group_translator, const int number_ranks, const mpiPP::MPIRank my_rank)
        : local_group_translator(std::move(_local_group_translator))
        , num_ranks(number_ranks)
        , rank(my_rank) {
    }

    /**
     * Indicates that someone wants to know the group ids for a neuron. Group ids will be requested with the next communication
     * @param rni The neuron whose group ids we want to know
     */
    void request_group_ids(const RankNeuronId& rni) {
        if (rank == rni.get_rank() || known_mappings.contains(rni)) {
            return;
        }

        const auto& [rni_rank, rni_id] = rni;

        next_request.emplace_back(rni_rank, rni_id);
    }

    /**
     * @brief Clears all known mappings
     */
    void clear_cache() noexcept {
        known_mappings.clear();
    }

    /**
     * Returns the group ids for a neuron. Must have been requested before the last communication
     * @param rni The neuron whose group id we want to know
     * @return Group ids
     */
    RelearnTypes::group_ids get_group_ids(const RankNeuronId& rni) {
        if (rni.get_rank() == rank) {
            return local_group_translator->get_group_ids_for_neuron_id(rni.get_neuron_id().get_neuron_id());
        }
        return known_mappings[rni];
    }

    /**
     * Start communication with other mpi ranks and exchange requested group ids
     */
    void exchange_requests() {
        send_requests();
        next_request.clear();
    }

private:
    std::shared_ptr<LocalGroupTranslator> local_group_translator;
    std::unordered_map<RankNeuronId, RelearnTypes::group_ids> known_mappings;
    RelearnTypes::comm_map_group<NeuronID> next_request{ mpiPP::MPIInfo::get_number_ranks() };

    int num_ranks;
    mpiPP::MPIRank rank;

    static constexpr auto max_size_map = 10000;

    void send_requests() {
        const auto& received_requested_data = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests<NeuronID>(next_request);
        answer_requests(received_requested_data);
    }

    void answer_requests(const RelearnTypes::comm_map_group<NeuronID>& received_data) {
        auto answer_data = RelearnTypes::comm_map_group<RelearnTypes::group_id>{ num_ranks };
        auto answer_sizes = RelearnTypes::comm_map_group<std::size_t>{ num_ranks };

        const auto sizes = received_data.get_request_sizes_vector();

        for (auto requesting_rank = 0; requesting_rank < num_ranks; requesting_rank++) {
            for (auto i = 0U; i < sizes[static_cast<std::size_t>(requesting_rank)]; i++) {
                const auto& neuron_id = received_data.get_requests(mpiPP::MPIRank{ requesting_rank })[i];
                const auto& group_ids = local_group_translator->get_group_ids_for_neuron_id(neuron_id.get_neuron_id());
                const auto length = group_ids.size();
                for (const auto& group_id : group_ids) {
                    answer_data.emplace_back(mpiPP::MPIRank{ requesting_rank }, group_id);
                }
                answer_sizes.emplace_back(mpiPP::MPIRank{ requesting_rank }, length);
            }
        }

        const auto& received_answer_data = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests<RelearnTypes::group_id>(answer_data);
        const auto& received_answer_sizes = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests<std::size_t>(answer_sizes);
        parse_answer(received_answer_data, received_answer_sizes);
    }

    void parse_answer(const RelearnTypes::comm_map_group<RelearnTypes::group_id>& answer_data, const RelearnTypes::comm_map_group<std::size_t>& answer_sizes) {
        const auto& number_requests = answer_sizes.get_request_sizes_vector();

        for (auto mpi_rank = 0; mpi_rank < num_ranks; mpi_rank++) {
            if (number_requests[static_cast<std::size_t>(mpi_rank)] == 0) {
                continue;
            }

            const auto& sizes = answer_sizes.get_requests(mpiPP::MPIRank{ mpi_rank });
            const auto& flat_group_ids_data = answer_data.get_requests(mpiPP::MPIRank{ mpi_rank });
            const auto& next_requests = next_request.get_requests(mpiPP::MPIRank{ mpi_rank });

            auto offset = std::ptrdiff_t{ 0 };
            for (auto i = 0U; i < number_requests[static_cast<std::size_t>(mpi_rank)]; i++) {
                const auto number_group_ids = static_cast<std::ptrdiff_t>(sizes[i]);
                const auto neuron_id = next_requests[i];
                const auto rank_neuron_id = RankNeuronId{ mpiPP::MPIRank{ mpi_rank }, NeuronID{ neuron_id } };
                const auto group_ids = RelearnTypes::group_ids(flat_group_ids_data.begin() + offset, flat_group_ids_data.begin() + offset + number_group_ids);
                known_mappings.emplace(rank_neuron_id, group_ids);
                offset += number_group_ids;
            }
        }
    }
};
