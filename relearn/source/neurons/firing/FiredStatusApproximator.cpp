/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusApproximator.h"

#include "Types3.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIAdvancedCommunicationPatterns.h"
#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unordered_map>

void FiredStatusApproximator::commit_local_fired_status(const step_type /*step*/) {
    const auto fired_status = get_fired_status_recorder()->get_fired();

    for (auto i = std::size_t{ 0 }; i < fired_status.size(); i++) {
        if (fired_status[i] == FiredStatus::Fired) {
            accumulated_fired[i]++;
        }
    }
}

void FiredStatusApproximator::exchange_fired_status([[maybe_unused]] const step_type step) {
    // This is a no-op
}

bool FiredStatusApproximator::contains(const mpiPP::MPIRank rank, const NeuronID neuron_id) const {
    const auto rank_it = rank.get_rank_cast();
    RelearnException::check(rank_it < firing_rate_cache.size(), "FiredStatusApproximator::contains: Rank {} is too large for the sizes {}.", rank, firing_rate_cache.size());

    const auto& rank_cache = firing_rate_cache[rank_it];

    const auto pos = rank_cache.find(neuron_id);
    if (pos == rank_cache.end()) {
        return false;
    }

    const auto firing_rate = pos->second;

    const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::FiringStatusApproximator, 0.0, 1.0);
    return firing_rate >= random_number;
}

void FiredStatusApproximator::notify_of_plasticity_change(const step_type step) {
    RelearnException::check(last_synced <= step, "FiredStatusApproximator::notify_of_plasticity_change: step is smaller than last_synced: {} > {}", last_synced, step);
    const auto steps_since_last_sync = step - last_synced;

    if (steps_since_last_sync > 0) {
        const auto steps_since_last_sync_inv = 1.0 / steps_since_last_sync;
        for (auto i = std::size_t{ 0 }; i < accumulated_fired.size(); i++) {
            latest_firing_rate[i] = steps_since_last_sync_inv * static_cast<double>(accumulated_fired[i]);
        }

        std::ranges::fill(accumulated_fired, 0);
    }

    struct communication_type {
        NeuronID neuron_id;
        double firing_rate{};
    };

    const auto num_local_neurons = get_number_local_neurons();

    const auto size_hint = std::min<std::size_t>(static_cast<std::size_t>(get_number_ranks()), num_local_neurons);
    auto outgoing_firing_rates = RelearnTypes::comm_map_firing<communication_type>{ get_number_ranks(), size_hint };

    auto add_to_communication_map = [&outgoing_firing_rates, num_local_neurons, this](const auto& synapses) {
        for (const auto neuron_id : NeuronID::range(num_local_neurons)) {
            const auto it = neuron_id.get_neuron_id();

            for (const auto& [target_id, weight] : synapses[it]) {
                const auto& [target_rank, target_neuron_id] = target_id;

                outgoing_firing_rates.append(target_rank, { neuron_id, latest_firing_rate[it] });
            }
        }
    };

    const auto& [outgoing_plastic_synapses, outgoing_static_synapses] = network_graph->get_all_distant_out_edges();
    add_to_communication_map(outgoing_plastic_synapses);
    add_to_communication_map(outgoing_static_synapses);

    const auto& [in_ranks_plastic, _1] = network_graph->get_ranks_in_connected();
    const auto& [out_ranks_plastic, _2] = network_graph->get_ranks_out_connected();

    const auto& incoming_firing_rates = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_firing_rates, in_ranks_plastic, out_ranks_plastic);

    for (auto& rank_cache : firing_rate_cache) {
        rank_cache.clear();
    }

    for (const auto& [rank, values] : incoming_firing_rates) {
        auto& cache = firing_rate_cache[static_cast<std::size_t>(rank.get_rank())];
        for (const auto& [neuron_id, firing_rate] : values) {
            cache[neuron_id] = firing_rate;
        }
    }

    last_synced = step;
}

void FiredStatusApproximator::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_easy_footprint = sizeof(*this) - sizeof(FiredStatusCommunicator)
                                   + (accumulated_fired.capacity() * sizeof(std::size_t))
                                   + (latest_firing_rate.capacity() * sizeof(double));

    auto my_hard_footprint = firing_rate_cache.capacity() * sizeof(std::unordered_map<NeuronID, double>);
    for (const auto& cache : firing_rate_cache) {
        // Some internet approximation of an unordered_map's size
        const auto hard_value = cache.size() * (sizeof(double) + sizeof(void*)) + cache.bucket_count() * (sizeof(void*) + sizeof(std::size_t));
        my_hard_footprint += static_cast<std::size_t>(static_cast<double>(hard_value) * 1.5);
    }

    footprint->emplace("FiredStatusApproximator", my_hard_footprint + my_easy_footprint);

    FiredStatusCommunicator::record_memory_footprint(footprint);
}
