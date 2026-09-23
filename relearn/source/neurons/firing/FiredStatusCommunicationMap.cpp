/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusCommunicationMap.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/CommunicationTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/map.hpp>

#include <algorithm>
#include <set>

void FiredStatusCommunicationMap::commit_local_fired_status([[maybe_unused]] const step_type step) {
    outgoing_ids.clear();

    if (const auto num_ranks = get_number_ranks(); num_ranks == 1) {
        return;
    }

    /**
     * Check which of my neurons fired and determine which ranks need to know about it.
     * That is, they contain the neurons connecting the axons of my firing neurons.
     */

    const auto disable_flags = extra_infos->get_disable_flags();
    const auto fired_status = get_fired_status_recorder()->get_fired();
    const auto num_local_neurons = get_number_local_neurons();

    Timers::start(TimerRegion::PREPARE_SENDING_SPIKES);

    const auto not_disabled_not_inactive = [&disable_flags, &fired_status](const NeuronID& neuron_id) {
        const auto id = neuron_id.get_neuron_id();
        auto res = disable_flags[id] != UpdateStatus::Disabled && fired_status[id] != FiredStatus::Inactive;
        return res;
    };

    // For my neurons
    for (const auto neuron_id : NeuronIDRange::range(num_local_neurons)
                                    | ranges::views::filter(not_disabled_not_inactive)) {
        // Don't send firing neuron id to myself as I already have this info
        const auto& [distant_out_edges_plastic, distant_out_edges_static] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());

        // Find all target neurons which should receive the signal fired.
        // That is, neurons which connect axons from neuron "neuron_id"
        auto send_fired_neurons = [this, neuron_id](const auto& distant_out_edges) {
            ranges::for_each(distant_out_edges | ranges::views::keys, [this, &neuron_id](const RankNeuronId& target_rank) {
                outgoing_ids.append(target_rank.get_rank(), neuron_id);
            });
        };

        send_fired_neurons(distant_out_edges_static);
        send_fired_neurons(distant_out_edges_plastic);

    } // For my neurons

    Timers::stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);
}

void FiredStatusCommunicationMap::exchange_fired_status([[maybe_unused]] const step_type step) {
    if (get_number_ranks() == 1) {
        return;
    }
    Timers::start(TimerRegion::EXCHANGE_NEURON_IDS);

    const auto& [in_ranks_plastic, _1] = network_graph->get_ranks_in_connected();
    const auto& [out_ranks_plastic, _2] = network_graph->get_ranks_out_connected();
    incoming_ids = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_ids, in_ranks_plastic, out_ranks_plastic);

    Timers::stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);
}

bool FiredStatusCommunicationMap::contains(const mpiPP::MPIRank rank, const NeuronID neuron_id) const {
    const auto num_ranks = get_number_ranks();
    RelearnException::check(rank.is_initialized(), "FiredStatusCommunicationMap::contains: rank is not initialized");
    RelearnException::check(rank.get_rank() < num_ranks,
                            "FiredStatusCommunicationMap::contains: rank {} is larger than the number of ranks {}",
                            rank, num_ranks);
    RelearnException::check(neuron_id.is_initialized(),
                            "FiredStatusCommunicationMap::contains: The neuron id is not initialized: {}", neuron_id);

    const auto& firings_ids_opt = incoming_ids.get_optional_requests(rank);
    if (!firings_ids_opt.has_value()) {
        return false;
    }

    const auto& firing_ids = firings_ids_opt.value().get();
    const auto contains_id = std::ranges::binary_search(firing_ids, neuron_id);

    return contains_id;
}

void FiredStatusCommunicationMap::set_incoming_ids(const std::set<RankNeuronId>& rnis) {
    incoming_ids = RelearnTypes::comm_map_firing<NeuronID>{ get_number_ranks() };

    for (const auto& [rank, neuron_id] : rnis) {
        incoming_ids.emplace_back(rank, neuron_id);
    }
}
