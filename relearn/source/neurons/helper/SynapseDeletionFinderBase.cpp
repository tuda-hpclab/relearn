/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinderBase.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cmath>

void SynapseDeletionFinderBase::set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) { // NOLINT(performance-unnecessary-value-param) - moved into extra_info below
    const auto full = new_extra_info != nullptr;
    RelearnException::check(full, "SynapseDeletionFinderBase::set_extra_infos: new_extra_info is empty");

    extra_info = std::move(new_extra_info);
}

void SynapseDeletionFinderBase::set_fired_status_recorder(std::shared_ptr<FiredStatusRecorder> new_fired_status_recorder) { // NOLINT(performance-unnecessary-value-param) - moved into fired_status_recorder below
    const auto full = new_fired_status_recorder != nullptr;
    RelearnException::check(full,
                            "SynapseDeletionFinderBase::set_fired_status_recorder: new_fired_status_recorder is empty");

    fired_status_recorder = std::move(new_fired_status_recorder);
}

std::vector<mpiPP::MPIRank> SynapseDeletionFinderBase::build_rank_whitelist() const {
    auto whitelist = std::vector<mpiPP::MPIRank>{};

    const auto& distant_count = network_graph->plastic_network_graph.get_distant_count();
    whitelist.reserve(distant_count.size());
    for (auto i = 0U; i < distant_count.size(); i++) {
        if (distant_count[i] > 0) {
            whitelist.emplace_back(i);
        }
    }

    return whitelist;
}

std::vector<RankNeuronId>
SynapseDeletionFinderBase::register_synapses(const NeuronID neuron_id, const ElementType element_type,
                                             const SignalType signal_type) {
    auto register_out_edges = [](const auto& distant_out_edges, const auto& local_out_edges) {
        auto neuron_ids = std::vector<RankNeuronId>{};
        neuron_ids.reserve((distant_out_edges.size() + local_out_edges.size()) * 2);

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        for (const auto& [rni, weight] : distant_out_edges) {
            /**
             * Create "edge weight" number of synapses and add them to the synapse list
             * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
             */

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [local_neuron_id, weight] : local_out_edges) {
            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, local_neuron_id);
            }
        }

        return neuron_ids;
    };

    auto register_in_edges = [signal_type](const auto& distant_in_edges, const auto& local_in_edges) {
        auto neuron_ids = std::vector<RankNeuronId>{};
        neuron_ids.reserve((distant_in_edges.size() + local_in_edges.size()) * 2);

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        for (const auto& [rni, weight] : distant_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [local_neuron_id, weight] : local_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, local_neuron_id);
            }
        }

        return neuron_ids;
    };

    auto current_synapses = std::vector<RankNeuronId>{};
    if (element_type == ElementType::Axon) {
        const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id.get_neuron_id());

        current_synapses = register_out_edges(distant_out_edges, local_out_edges);
    } else {
        const auto& [distant_in_edges, _1] = network_graph->get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [local_in_edges, _2] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());

        current_synapses = register_in_edges(distant_in_edges, local_in_edges);
    }

    return current_synapses;
}
