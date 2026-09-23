/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Neurons.h"

#include "Config.h"

#include "cuda/random/RandomNumberHost.h"
#include "io/Event.h"
#include "io/LogFiles.h"
#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "neurons/models/NeuronModel.h"
#include "sim/Essentials.h"
#include "structure/Partition.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/Accumulate.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/StatisticalMeasures.h"
#include "util/Timers.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPISynchronization.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>
#include <mpi-wrapper/rma/RMAWindow.h>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

void Neurons::init(const number_neurons_type number_neurons_init, std::vector<NeuronsExtraInfo::position_type> pos) {
    RelearnException::check(this->number_neurons == 0, "Neurons::init: Was already initialized");
    RelearnException::check(number_neurons_init > 0, "Neurons::init: number_neurons_init was 0");

    number_neurons = number_neurons_init;

    RelearnException::check(network_graph->get_number_neurons() != 0, "Neurons::init: The network graph must already be initialized");

    neuron_model->set_network_graph(network_graph);
    neuron_model->set_extra_infos(extra_info);

    neuron_model->init(number_neurons);
    extra_info->init(number_neurons);
    extra_info->set_positions(std::move(pos));

    synaptic_elements->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);
    calcium_calculator->init(number_neurons);

    synaptic_elements->set_extra_infos(extra_info);

    algorithm->set_neuron_extra_infos(extra_info);
    algorithm->set_network_graph(network_graph);
    algorithm->set_synaptic_elements(synaptic_elements);
    algorithm->set_probability_kernel(std::move(probability_kernel));
    algorithm->init(number_neurons);
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Octree created");

    synapse_deletion_finder->set_extra_infos(extra_info);
    synapse_deletion_finder->set_network_graph(network_graph);
    synapse_deletion_finder->set_fired_status_recorder(neuron_model->get_fired_status_recorder());
}

void Neurons::init_synaptic_elements(const PlasticLocalSynapses& local_synapses_plastic, const PlasticDistantInSynapses& in_synapses_plastic, const PlasticDistantOutSynapses& out_synapses_plastic) {
    last_created_local_synapses = local_synapses_plastic;
    last_created_in_synapses = in_synapses_plastic;
    last_created_out_synapses = out_synapses_plastic;

    for (const auto id : NeuronIDRange::range_id(number_neurons)) {
        const auto [axon_connections, _1] = network_graph->get_number_out_edges(id);
        const auto [dendrites_ex_connections, _2] = network_graph->get_number_excitatory_in_edges(id);
        const auto [dendrites_in_connections, _3] = network_graph->get_number_inhibitory_in_edges(id);

        const auto axons_cast = utility::safe_cast<unsigned int>(axon_connections);
        const auto dendrites_ex_cast = utility::safe_cast<unsigned int>(dendrites_ex_connections);
        const auto dendrites_in_cast = utility::safe_cast<unsigned int>(dendrites_in_connections);

        synaptic_elements->add_connected_elements(axons_cast, id, SynapticElementType::Axon);
        synaptic_elements->add_connected_elements(dendrites_ex_cast, id, SynapticElementType::DendriteExcitatory);
        synaptic_elements->add_connected_elements(dendrites_in_cast, id, SynapticElementType::DendriteInhibitory);
    }

    check_signal_types(network_graph, synaptic_elements->get_signal_types(), mpiPP::MPIInfo::get_my_rank());

    network_graph->rebuild();
}

void Neurons::check_signal_types(const std::shared_ptr<NetworkGraph>& network_graph,
                                 const std::span<const SignalType> signal_types, const mpiPP::MPIRank my_rank) {

    for (const auto neuron_id : NeuronIDRange::range_id(signal_types.size())) {
        const auto& signal_type = signal_types[neuron_id];

        const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id);
        for (const auto& [tgt_rni, weight] : distant_out_edges) {
            RelearnException::check((SignalType::Excitatory == signal_type && weight > 0) || (SignalType::Inhibitory == signal_type && weight < 0),
                                    "Neuron has outgoing connections not matching its signal type. {} {} -> {} {} {}",
                                    my_rank, neuron_id, tgt_rni, signal_type, weight);
        }

        const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id);
        for (const auto& [tgt_rni, weight] : local_out_edges) {
            RelearnException::check((SignalType::Excitatory == signal_type && weight > 0) || (SignalType::Inhibitory == signal_type && weight < 0),
                                    "Neuron has outgoing connections not matching its signal type. {} {} -> {} {} {}",
                                    my_rank, neuron_id, tgt_rni, signal_type, weight);
        }
    }
}

std::pair<RelearnTypes::number_synapse_type, RelearnTypes::comm_map_deletion<SynapseDeletionRequest>> Neurons::disable_neurons(const step_type step, const std::span<const NeuronID> local_neuron_ids, const int num_ranks) {
    const auto transformed_ids = local_neuron_ids | ranges::views::transform([](auto val) { return val.get_neuron_id(); }) | ranges::to_vector;

    extra_info->set_disabled_neurons(local_neuron_ids);

    neuron_model->disable_neurons(local_neuron_ids);

    auto deleted_axon_ex_connections = std::vector<RelearnTypes::counter_type>(number_neurons, 0);
    auto deleted_axon_in_connections = std::vector<RelearnTypes::counter_type>(number_neurons, 0);
    auto deleted_dend_ex_connections = std::vector<RelearnTypes::counter_type>(number_neurons, 0);
    auto deleted_dend_in_connections = std::vector<RelearnTypes::counter_type>(number_neurons, 0);

    auto number_deleted_out_inh_edges_within = RelearnTypes::number_synapse_type{ 0 };
    auto number_deleted_out_exc_edges_within = RelearnTypes::number_synapse_type{ 0 };

    auto number_deleted_out_inh_edges_to_outside = RelearnTypes::number_synapse_type{ 0 };
    auto number_deleted_out_exc_edges_to_outside = RelearnTypes::number_synapse_type{ 0 };

    auto number_deleted_distant_out_exc = RelearnTypes::number_synapse_type{ 0 };
    auto number_deleted_distant_out_inh = RelearnTypes::number_synapse_type{ 0 };
    auto number_deleted_distant_in_exc = RelearnTypes::number_synapse_type{ 0 };
    auto number_deleted_distant_in_inh = RelearnTypes::number_synapse_type{ 0 };

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(num_ranks));
    auto synapse_deletion_requests_outgoing = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(num_ranks, size_hint);

    for (const auto neuron_id : local_neuron_ids) {
        const auto id = neuron_id.get_neuron_id();
        RelearnException::check(id < number_neurons,
                                "Neurons::disable_neurons: There was a too large id: {} vs {}", neuron_id,
                                number_neurons);

        const auto [local_out_edges_ref, _1] = network_graph->get_local_out_edges(id);
        const auto [distant_out_edges_ref, _2] = network_graph->get_distant_out_edges(id);

        auto local_out_edges = local_out_edges_ref;
        auto distant_out_edges = distant_out_edges_ref;

        for (const auto& [target_neuron_id, weight] : local_out_edges) {
            network_graph->add_synapse(PlasticLocalSynapse(target_neuron_id, neuron_id, -weight));

            // Shall target_neuron_id also be disabled? Important: Do not remove synapse twice in this case
            const bool is_within = std::ranges::binary_search(local_neuron_ids, target_neuron_id);
            const auto local_target_neuron_id = target_neuron_id.get_neuron_id();

            if (weight > 0) {
                deleted_dend_ex_connections[local_target_neuron_id]++;

                if (is_within) {
                    number_deleted_out_exc_edges_within++;
                    deleted_axon_ex_connections[id]++;
                } else {
                    number_deleted_out_exc_edges_to_outside++;
                }
            } else {
                deleted_dend_in_connections[local_target_neuron_id]++;

                if (is_within) {
                    number_deleted_out_inh_edges_within++;
                    deleted_axon_in_connections[id]++;
                } else {
                    number_deleted_out_inh_edges_to_outside++;
                }
            }
        }

        for (const auto& [target_neuron_id, weight] : distant_out_edges) {
            network_graph->add_synapse(PlasticDistantOutSynapse(target_neuron_id, neuron_id, -weight));

            if (weight > 0) {
                deleted_axon_ex_connections[id]++;
                number_deleted_distant_out_exc++;

                synapse_deletion_requests_outgoing.append(target_neuron_id.get_rank(), { neuron_id, target_neuron_id.get_neuron_id(), ElementType::Axon, SignalType::Excitatory });
            } else {
                deleted_axon_in_connections[id]++;
                number_deleted_distant_out_inh++;

                synapse_deletion_requests_outgoing.append(target_neuron_id.get_rank(), { neuron_id, target_neuron_id.get_neuron_id(), ElementType::Axon, SignalType::Inhibitory });
            }
        }
    }

    auto number_deleted_in_edges_from_outside = RelearnTypes::number_synapse_type{ 0 };

    for (const auto neuron_id : local_neuron_ids) {
        const auto id = neuron_id.get_neuron_id();
        const auto [local_in_edges_ref, _1] = network_graph->get_local_in_edges(id);
        const auto [distant_in_edges_ref, _2] = network_graph->get_distant_in_edges(id);

        auto local_in_edges = local_in_edges_ref;
        auto distant_in_edges = distant_in_edges_ref;

        for (const auto& [source_neuron_id, weight] : local_in_edges) {
            network_graph->add_synapse(PlasticLocalSynapse(neuron_id, source_neuron_id, -weight));

            if (weight > 0) {
                deleted_axon_ex_connections[source_neuron_id.get_neuron_id()]++;
            } else {
                deleted_axon_in_connections[source_neuron_id.get_neuron_id()]++;
            }

            const bool is_within = std::ranges::binary_search(local_neuron_ids, source_neuron_id);

            if (is_within) {
                RelearnException::fail(
                    "Neurons::disable_neurons: While disabling neurons, found a within-in-edge that has not been deleted");
            } else {
                number_deleted_in_edges_from_outside++;
            }
        }

        for (const auto& [source_neuron_id, weight] : distant_in_edges) {
            network_graph->add_synapse(PlasticDistantInSynapse(neuron_id, source_neuron_id, -weight));

            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            synapse_deletion_requests_outgoing.append(source_neuron_id.get_rank(), { neuron_id, source_neuron_id.get_neuron_id(), ElementType::Dendrite, signal_type });

            if (weight > 0) {
                deleted_dend_ex_connections[id]++;
                number_deleted_distant_in_exc++;
            } else {
                deleted_dend_in_connections[id]++;
                number_deleted_distant_in_inh++;
            }
        }
    }

    const auto number_deleted_edges_within = number_deleted_out_inh_edges_within + number_deleted_out_exc_edges_within;

    synaptic_elements->disconnect_elements(deleted_axon_ex_connections, SynapticElementType::Axon);
    synaptic_elements->disconnect_elements(deleted_axon_in_connections, SynapticElementType::Axon);
    synaptic_elements->disconnect_elements(deleted_dend_ex_connections, SynapticElementType::DendriteExcitatory);
    synaptic_elements->disconnect_elements(deleted_dend_in_connections, SynapticElementType::DendriteInhibitory);

    synaptic_elements->disable_neurons(transformed_ids);

    neuron_model->notify_of_plasticity_change(step);

#ifdef RELEARN_CUDA_ENABLED
    network_graph->sync_with_gpu();
#endif

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Deleted {} in-edges with and ({}, {}) out-edges (exc., inh.) within the deleted portion",
                                 number_deleted_edges_within,
                                 number_deleted_out_exc_edges_within,
                                 number_deleted_out_inh_edges_within);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Deleted {} in-edges and ({}, {}) out-edges  (exc., inh.) connecting to the outside",
                                 number_deleted_in_edges_from_outside,
                                 number_deleted_out_exc_edges_to_outside,
                                 number_deleted_out_inh_edges_to_outside);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Deleted ({},{}) in-edges (exc., inh.) and ({},{}) out-edges connecting to the other ranks",
                                 number_deleted_distant_in_exc, number_deleted_distant_in_inh,
                                 number_deleted_distant_out_exc, number_deleted_distant_out_inh);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Deleted {} in-edges and ({}, {}) out-edges (exc., inh.) altogether",
                                 number_deleted_edges_within + number_deleted_in_edges_from_outside,
                                 number_deleted_out_exc_edges_within + number_deleted_out_exc_edges_to_outside,
                                 number_deleted_out_inh_edges_within + number_deleted_out_inh_edges_to_outside);

    const auto deleted_connections = number_deleted_distant_out_exc + number_deleted_distant_out_inh + number_deleted_distant_in_inh + number_deleted_distant_in_exc
                                     + number_deleted_in_edges_from_outside + number_deleted_out_inh_edges_to_outside + number_deleted_out_exc_edges_to_outside
                                     + number_deleted_out_exc_edges_within + number_deleted_out_inh_edges_within;

    return std::make_pair(deleted_connections, synapse_deletion_requests_outgoing);
}

void Neurons::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(number_neurons > 0, "Neurons::create_neurons: Was not initialized");
    RelearnException::check(creation_count > 0, "Neurons::create_neurons: Cannot create 0 neurons");

    const auto current_size = number_neurons;
    const auto new_size = current_size + creation_count;

    local_group_translator->create_neurons(creation_count);
    neuron_model->create_neurons(creation_count);
    calcium_calculator->create_neurons(creation_count);
    extra_info->create_neurons(creation_count);

    network_graph->create_neurons(creation_count);

    synaptic_elements->create_neurons(creation_count);

    algorithm->create_neurons(creation_count);

    number_neurons = new_size;
}

void Neurons::register_neuron_monitor(NeuronMonitor& monitor) {
    neuron_model->register_neuron_monitor(monitor);
    synaptic_elements->register_neuron_monitor(monitor);
    calcium_calculator->register_neuron_monitor(monitor);
}

void Neurons::update_electrical_activity(const step_type step) {
    neuron_model->update_electrical_activity(step);

#ifndef RELEARN_CUDA_ENABLED
    const auto& fired = neuron_model->get_fired();
    calcium_calculator->update_calcium(step, fired);

    Timers::start(TimerRegion::CALC_CALCIUM_EXTREME_VALUES);
    const auto calcium_values = calcium_calculator->get_calcium();
    const auto current_min_id = calcium_calculator->get_current_minimum().get_neuron_id();
    const auto current_max_id = calcium_calculator->get_current_maximum().get_neuron_id();

    LogFiles::write_to_file(LogFiles::EventType::ExtremeCalciumValues, false, "{};{:.6f};{};{:.6f}",
                            current_min_id, calcium_values[current_min_id], current_max_id, calcium_values[current_max_id]);
    Timers::stop_and_add(TimerRegion::CALC_CALCIUM_EXTREME_VALUES);
#else
    const auto* d_fired = neuron_model->get_fired_status_recorder()->get_d_fired_const();
    calcium_calculator->update_calcium(step, d_fired);
#endif
}

void Neurons::update_number_synaptic_elements_delta([[maybe_unused]] const step_type step) {

#ifndef RELEARN_CUDA_ENABLED
    const auto& calcium = calcium_calculator->get_calcium();
    const auto& target_calcium = calcium_calculator->get_target_calcium();
    synaptic_elements->update_number_elements(step, calcium, target_calcium);
#else
    const auto* d_calcium = calcium_calculator->get_d_calcium_const();
    const auto* d_target_calcium = calcium_calculator->get_d_target_calcium_const();

    synaptic_elements->update_number_elements(d_calcium, d_target_calcium);
#endif
}

void Neurons::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this);
    footprint->emplace("Neurons", my_footprint);

    partition->record_memory_footprint(footprint);
    local_group_translator->record_memory_footprint(footprint);
    algorithm->record_memory_footprint(footprint);
    network_graph->record_memory_footprint(footprint);
    neuron_model->record_memory_footprint(footprint);
    calcium_calculator->record_memory_footprint(footprint);
    synaptic_elements->record_memory_footprint(footprint);
    synapse_deletion_finder->record_memory_footprint(footprint);
    extra_info->record_memory_footprint(footprint);
    footprint->emplace("Random GPU", RandomNumbers::get_memory_usage());
}

void Neurons::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    network_graph->record_usage_footprint(footprint);
}

StatisticalMeasures Neurons::global_statistics(const std::span<const double> local_values, [[maybe_unused]] const mpiPP::MPIRank root) const {
    const auto disable_flags = extra_info->get_disable_flags();
    const auto [d_my_min, d_my_max, d_my_acc, d_num_values] = Util::min_max_acc(local_values, extra_info->get_disable_flags());
    const auto my_avg = d_my_acc / static_cast<double>(d_num_values);

    const auto d_min = mpiPP::MPIReductions::reduce_min(d_my_min);
    const auto d_max = mpiPP::MPIReductions::reduce_max(d_my_max);

    const auto num_values = static_cast<double>(mpiPP::MPIReductions::all_reduce_sum(d_num_values));

    // Get global avg at all ranks (needed for variance)
    const auto avg = mpiPP::MPIReductions::all_reduce_sum(my_avg) / mpiPP::MPIInfo::get_number_ranks();

    /**
     * Calc variance
     */
    const auto my_var = ranges::accumulate(NeuronIDRange::range_id(number_neurons)
                                               | ranges::views::filter(utility::not_equal_to(UpdateStatus::Disabled), utility::lookup(disable_flags))
                                               | ranges::views::transform([&local_values, avg](const auto neuron_id) {
                                                     const auto val = local_values[neuron_id] - avg;
                                                     return val * val;
                                                 }),
                                           0.0)
                        / num_values;

    // Get global variance at rank "root"
    const auto var = mpiPP::MPIReductions::reduce_sum(my_var);

    // Calc standard deviation
    const auto std = std::sqrt(var);

    return { .min = d_min, .max = d_max, .avg = avg, .var = var, .std = std };
}

RelearnTypes::number_synapse_type Neurons::create_synapses() {
    // const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    Event::create_and_print_duration_begin_event("Neurons::prepare_update_connectivity", { EventCategory::mpi, EventCategory::calculation }, {}, true);

    const auto signal_types = synaptic_elements->get_signal_types();

    const auto& vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto& vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto& vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    algorithm->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
    Event::create_and_print_duration_end_event(true);

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    mpiPP::MPISynchronization::barrier();

    // Delegate the creation of new synapses to the algorithm
    Event::create_and_print_duration_begin_event("Neurons::update_connectivity", { EventCategory::mpi, EventCategory::calculation }, {}, true);
    auto [num_synapses_created, local_synapses, distant_in_synapses, distant_out_synapses]
        = algorithm->update_connectivity(number_neurons);
    Event::create_and_print_duration_end_event(true);

    // Update the network graph all at once
    // TODO That is too slow
    Timers::start(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);
    Event::create_and_print_duration_begin_event("Neurons::add_edges", { EventCategory::mpi, EventCategory::calculation }, {}, true);
    network_graph->add_edges(local_synapses, distant_in_synapses, distant_out_synapses);
    Event::create_and_print_duration_end_event(true);
    Timers::stop_and_add(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);

    last_created_local_synapses = std::move(local_synapses);
    last_created_in_synapses = std::move(distant_in_synapses);
    last_created_out_synapses = std::move(distant_out_synapses);

    return num_synapses_created;
}

void Neurons::debug_check_counts() {
    if (!Config::do_debug_checks) {
        return;
    }

    RelearnException::check(network_graph != nullptr, "Neurons::debug_check_counts: network_graph is nullptr");

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    network_graph->debug_check();

    const auto ga = synaptic_elements->get_grown_elements(SynapticElementType::Axon);
    const auto ged = synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory);
    const auto gid = synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory);

    const auto& va = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto& ved = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto& vid = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto& ca = synaptic_elements->get_connected_elements(SynapticElementType::Axon);
    const auto& ced = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto& cid = synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory);

    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto integral_axons = va[neuron_id] + ca[neuron_id];
        const auto integral_excitatory_dendrites = ved[neuron_id] + ced[neuron_id];
        const auto integral_inhibitory_dendrites = vid[neuron_id] + cid[neuron_id];

        RelearnException::check(integral_axons == static_cast<unsigned int>(ga[neuron_id]),
                                "Neurons::debug_check_counts: Neuron {} has {} (connected & vacant) axons but {} grown axons (rank {})",
                                neuron_id, integral_axons, ga[neuron_id], my_rank);

        RelearnException::check(integral_excitatory_dendrites == static_cast<unsigned int>(ged[neuron_id]),
                                "Neurons::debug_check_counts: Neuron {} has {} (connected & vacant) excitatory dendrites but {} grown dendrites (rank {})",
                                neuron_id, integral_excitatory_dendrites, ged[neuron_id], my_rank);

        RelearnException::check(integral_inhibitory_dendrites == static_cast<unsigned int>(gid[neuron_id]),
                                "Neurons::debug_check_counts: Neuron {} has {} (connected & vacant) inhibitory dendrites but {} grown dendrites (rank {})",
                                neuron_id, integral_inhibitory_dendrites, gid[neuron_id], my_rank);

        const auto [number_out_edges, _1] = network_graph->get_number_out_edges(neuron_id);
        const auto [number_excitatory_in_edges, _2] = network_graph->get_number_excitatory_in_edges(neuron_id);
        const auto [number_inhibitory_in_edges, _3] = network_graph->get_number_inhibitory_in_edges(neuron_id);

        RelearnException::check(ca[neuron_id] == static_cast<unsigned int>(number_out_edges),
                                "Neurons::debug_check_counts: Neuron {} has {} axons but {} out edges (rank {})",
                                neuron_id, ca[neuron_id], number_out_edges, my_rank);

        RelearnException::check(ced[neuron_id] == static_cast<unsigned int>(number_excitatory_in_edges),
                                "Neurons::debug_check_counts: Neuron {} has {} excitatory dendrites but {} excitatory in edges (rank {})",
                                neuron_id, ced[neuron_id], number_excitatory_in_edges, my_rank);

        RelearnException::check(cid[neuron_id] == static_cast<unsigned int>(number_inhibitory_in_edges),
                                "Neurons::debug_check_counts: Neuron {} has {} inhibitory dendrites but {} inhibitory in edges (rank {})",
                                neuron_id, cid[neuron_id], number_inhibitory_in_edges, my_rank);
    }
}

StatisticalMeasures Neurons::get_statistics(const NeuronAttribute attribute) const {
    auto buffer = std::vector<RelearnTypes::calcium_type>{};
    auto buffer_unsigned = std::vector<RelearnTypes::counter_type>{};

    switch (attribute) {
    case NeuronAttribute::Calcium:
        return global_statistics(calcium_calculator->get_calcium(), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::TargetCalcium:
        return global_statistics(calcium_calculator->get_target_calcium(), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::CalciumDifference:
        buffer = std::vector<RelearnTypes::calcium_type>{ calcium_calculator->get_calcium().begin(), calcium_calculator->get_calcium().end() };
        for (auto i = 0U; i < buffer.size(); i++) {
            buffer[i] -= calcium_calculator->get_target_calcium()[i];
        }
        return global_statistics(std::span<const RelearnTypes::calcium_type>{ buffer }, mpiPP::MPIRank::root_rank());

    case NeuronAttribute::X:
        return global_statistics(neuron_model->get_x(), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::Fired:
        return global_statistics(neuron_model->get_fired(), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::InputActivity:
        return global_statistics(neuron_model->get_input(), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::AxonsGrown:
        return global_statistics(synaptic_elements->get_grown_elements(SynapticElementType::Axon), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::AxonsConnected:
        return global_statistics(synaptic_elements->get_connected_elements(SynapticElementType::Axon), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatory:
        return global_statistics(synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatoryConnected:
        return global_statistics(synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitory:
        return global_statistics(synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory), mpiPP::MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitoryConnected:
        return global_statistics(synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory), mpiPP::MPIRank::root_rank());
    }

    RelearnException::fail("Neurons::get_statistics: Got an unsupported attribute: {}", static_cast<int>(attribute));

    return {};
}

std::tuple<RelearnTypes::number_synapse_type, RelearnTypes::number_synapse_type, RelearnTypes::number_synapse_type> Neurons::update_connectivity(const step_type step) {
    RelearnException::check(network_graph != nullptr, "Network graph is nullptr");
    RelearnException::check(algorithm != nullptr, "Algorithm is nullptr");

    // Drain the background spike exchange before ANY MPI on the main thread.
    // delete_synapses() calls MPIAdvancedCommunicationPatterns::exchange_requests,
    // and create_synapses() issues collectives — both corrupt UCX shared transport
    // state if the background point-to-point exchange is still in flight.
    neuron_model->wait_for_spike_exchange();
    debug_check_counts();
    const auto& [num_axons_deleted, num_dendrites_deleted] = synapse_deletion_finder->delete_synapses();
    debug_check_counts();
    const auto num_synapses_created = create_synapses();
    debug_check_counts();

    neuron_model->notify_of_plasticity_change(step);

    network_graph->rebuild();

    return { num_axons_deleted, num_dendrites_deleted, num_synapses_created };
}

RelearnTypes::number_synapse_type Neurons::delete_disabled_distant_synapses(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& list, const mpiPP::MPIRank my_rank) {
    auto num_synapses_deleted = RelearnTypes::number_synapse_type{ 0 };

    const auto& disable_flags = extra_info->get_disable_flags();

    for (const auto& [other_rank, requests] : list) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            if (disable_flags[my_neuron_id.get_neuron_id()] != UpdateStatus::Enabled) {
                continue;
            }

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                RelearnException::fail("Local synapse deletion is not allowed via mpi");
            }

            if (ElementType::Dendrite == element_type) {
                const auto& [out_edges, _1] = network_graph->get_distant_out_edges(my_neuron_id.get_neuron_id());
                RelearnTypes::plastic_synapse_weight weight = 0;
                for (const auto& [target, edge_weight] : out_edges) {
                    if (target.get_rank() == other_rank && target.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                RelearnException::check(weight != 0, "Couldnot find the weight of the connection");
                network_graph->add_synapse(
                    PlasticDistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, -weight));
            } else {
                const auto& [in_edges, _2] = network_graph->get_distant_in_edges(my_neuron_id.get_neuron_id());
                RelearnTypes::plastic_synapse_weight weight = 0;
                for (const auto& [source, edge_weight] : in_edges) {
                    if (source.get_rank() == other_rank && source.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                network_graph->add_synapse(
                    PlasticDistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight));
            }

            const auto synaptic_element_type = get_synaptic_element_type(get_other_element_type(element_type), signal_type);
            synaptic_elements->disconnect_elements(1, my_neuron_id.get_neuron_id(), synaptic_element_type);
        }
    }

    return num_synapses_deleted;
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(const step_type step,
                                                                        const RelearnTypes::number_synapse_type sum_axon_deleted,
                                                                        const RelearnTypes::number_synapse_type sum_dendrites_deleted,
                                                                        const RelearnTypes::number_synapse_type sum_synapses_created) {

    const auto sum_dends_exc_vacant = ranges::accumulate(synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory), 0U);
    const auto sum_dends_inh_vacant = ranges::accumulate(synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory), 0U);

    const auto sum_axons_vacant = ranges::accumulate(synaptic_elements->get_vacant_elements(SynapticElementType::Axon), 0U);

    // Get global sums at rank 0
    const auto sums_local = std::array<std::int64_t, 6>{ sum_axons_vacant,
                                                         sum_dends_exc_vacant,
                                                         sum_dends_inh_vacant,
                                                         static_cast<std::int64_t>(sum_axon_deleted),
                                                         static_cast<std::int64_t>(sum_dendrites_deleted),
                                                         static_cast<std::int64_t>(sum_synapses_created) };

    const auto sums_global = mpiPP::MPIReductions::reduce_componentwise_sum(sums_local);

    // Output data
    if (mpiPP::MPIRank::root_rank() == mpiPP::MPIInfo::get_my_rank()) {
        constexpr auto cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            LogFiles::write_to_file(LogFiles::EventType::Sums, false,
                                    "# SUMS OVER ALL NEURONS\n{1:{0}}{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}{7:{0}}",
                                    cwidth,
                                    "# step",
                                    "Axons exc. (vacant)",
                                    "Dends exc. (vacant)",
                                    "Dends inh. (vacant)",
                                    "Synapses (axons) deleted",
                                    "Synapses (dendrites) deleted",
                                    "Synapses created");
        }

        LogFiles::write_to_file(LogFiles::EventType::Sums, false,
                                "{2:<{0}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}",
                                cwidth,
                                Constants::print_precision,
                                step,
                                sums_global[0],
                                sums_global[1],
                                sums_global[2],
                                sums_global[3] / 2,
                                sums_global[4] / 2,
                                sums_global[5] / 2);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(const step_type step) const {
    return;
    const auto& calcium_statistics = get_statistics(NeuronAttribute::Calcium);
    const auto& calcium_difference_statistics = get_statistics(NeuronAttribute::CalciumDifference);
    const auto& axons_statistics = get_statistics(NeuronAttribute::AxonsGrown);
    const auto& axons_connected_statistics = get_statistics(NeuronAttribute::AxonsConnected);
    const auto& dendrites_excitatory_statistics = get_statistics(NeuronAttribute::DendritesExcitatory);
    const auto& dendrites_excitatory_connected_statistics = get_statistics(NeuronAttribute::DendritesExcitatoryConnected);

    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    constexpr auto cwidth = 20; // Column width

    // Write headers to file if not already done so
    if (0 == step) {
        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
                                "# ALL NEURONS\n{1:{0}}"
                                "{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}"
                                "{7:{0}}{8:{0}}{9:{0}}{10:{0}}{11:{0}}"
                                "{12:{0}}{13:{0}}{14:{0}}{15:{0}}{16:{0}}"
                                "{17:{0}}{18:{0}}{19:{0}}{20:{0}}{21:{0}}"
                                "{22:{0}}{23:{0}}{24:{0}}{25:{0}}{26:{0}}"
                                "{27:{0}}{28:{0}}{29:{0}}{30:{0}}{31:{0}}",
                                cwidth,
                                "# step",
                                "C (avg)",
                                "C (min)",
                                "C (max)",
                                "C (var)",
                                "C (std_dev)",
                                "C diff (avg)",
                                "C diff (min)",
                                "C diff (max)",
                                "C diff (var)",
                                "C diff (std_dev)",
                                "axons (avg)",
                                "axons (min)",
                                "axons (max)",
                                "axons (var)",
                                "axons (std_dev)",
                                "axons.c (avg)",
                                "axons.c (min)",
                                "axons.c (max)",
                                "axons.c (var)",
                                "axons.c (std_dev)",
                                "den.ex (avg)",
                                "den.ex (min)",
                                "den.ex (max)",
                                "den.ex (var)",
                                "den.ex (std_dev)",
                                "den.ex.c (avg)",
                                "den.ex.c (min)",
                                "den.ex.c (max)",
                                "den.ex.c (var)",
                                "den.ex.c (std_dev)");

        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
                                "# step",
                                "C (avg)",
                                "C (min)",
                                "C (max)",
                                "C (var)",
                                "C (std_dev)",
                                "C diff (avg)",
                                "C diff (min)",
                                "C diff (max)",
                                "C diff (var)",
                                "C diff (std_dev)",
                                "C (std_dev)",
                                "axons (avg)",
                                "axons (min)",
                                "axons (max)",
                                "axons (var)",
                                "axons (std_dev)",
                                "axons.c (avg)",
                                "axons.c (min)",
                                "axons.c (max)",
                                "axons.c (var)",
                                "axons.c (std_dev)",
                                "den.ex (avg)",
                                "den.ex (min)",
                                "den.ex (max)",
                                "den.ex (var)",
                                "den.ex (std_dev)",
                                "den.ex.c (avg)",
                                "den.ex.c (min)",
                                "den.ex.c (max)",
                                "den.ex.c (var)",
                                "den.ex.c (std_dev)");
    }

    // Write data at step "step"
    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
                            "{2:<{0}}"
                            "{3:<{0}.{1}f}{4:<{0}.{1}f}{5:<{0}.{1}f}{6:<{0}.{1}f}{7:<{0}.{1}f}"
                            "{8:<{0}.{1}f}{9:<{0}.{1}f}{10:<{0}.{1}f}{11:<{0}.{1}f}{12:<{0}.{1}f}"
                            "{13:<{0}.{1}f}{14:<{0}.{1}f}{15:<{0}.{1}f}{16:<{0}.{1}f}{17:<{0}.{1}f}"
                            "{18:<{0}.{1}f}{19:<{0}.{1}f}{20:<{0}.{1}f}{21:<{0}.{1}f}{22:<{0}.{1}f}"
                            "{23:<{0}.{1}f}{24:<{0}.{1}f}{25:<{0}.{1}f}{26:<{0}.{1}f}{27:<{0}.{1}f}"
                            "{28:<{0}.{1}f}{29:<{0}.{1}f}{30:<{0}.{1}f}{31:<{0}.{1}f}{32:<{0}.{1}f}",
                            cwidth,
                            Constants::print_precision,
                            step,
                            calcium_statistics.avg,
                            calcium_statistics.min,
                            calcium_statistics.max,
                            calcium_statistics.var,
                            calcium_statistics.std,
                            calcium_difference_statistics.avg,
                            calcium_difference_statistics.min,
                            calcium_difference_statistics.max,
                            calcium_difference_statistics.var,
                            calcium_difference_statistics.std,
                            axons_statistics.avg,
                            axons_statistics.min,
                            axons_statistics.max,
                            axons_statistics.var,
                            axons_statistics.std,
                            axons_connected_statistics.avg,
                            axons_connected_statistics.min,
                            axons_connected_statistics.max,
                            axons_connected_statistics.var,
                            axons_connected_statistics.std,
                            dendrites_excitatory_statistics.avg,
                            dendrites_excitatory_statistics.min,
                            dendrites_excitatory_statistics.max,
                            dendrites_excitatory_statistics.var,
                            dendrites_excitatory_statistics.std,
                            dendrites_excitatory_connected_statistics.avg,
                            dendrites_excitatory_connected_statistics.min,
                            dendrites_excitatory_connected_statistics.max,
                            dendrites_excitatory_connected_statistics.var,
                            dendrites_excitatory_connected_statistics.std);

    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
                            "{};"
                            "{};{};{};{};{};"
                            "{};{};{};{};{};"
                            "{};{};{};{};{};"
                            "{};{};{};{};{};"
                            "{};{};{};{};{};"
                            "{};{};{};{};{}",
                            step,
                            calcium_statistics.avg,
                            calcium_statistics.min,
                            calcium_statistics.max,
                            calcium_statistics.var,
                            calcium_statistics.std,
                            calcium_difference_statistics.avg,
                            calcium_difference_statistics.min,
                            calcium_difference_statistics.max,
                            calcium_difference_statistics.var,
                            calcium_difference_statistics.std,
                            axons_statistics.avg,
                            axons_statistics.min,
                            axons_statistics.max,
                            axons_statistics.var,
                            axons_statistics.std,
                            axons_connected_statistics.avg,
                            axons_connected_statistics.min,
                            axons_connected_statistics.max,
                            axons_connected_statistics.var,
                            axons_connected_statistics.std,
                            dendrites_excitatory_statistics.avg,
                            dendrites_excitatory_statistics.min,
                            dendrites_excitatory_statistics.max,
                            dendrites_excitatory_statistics.var,
                            dendrites_excitatory_statistics.std,
                            dendrites_excitatory_connected_statistics.avg,
                            dendrites_excitatory_connected_statistics.min,
                            dendrites_excitatory_connected_statistics.max,
                            dendrites_excitatory_connected_statistics.var,
                            dendrites_excitatory_connected_statistics.std);
}

void Neurons::print_calcium_statistics_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    const auto& calcium = calcium_calculator->get_calcium();
    const auto& calcium_statistics = global_statistics(calcium, mpiPP::MPIRank::root_rank());
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    essentials->insert("Calcium-Minimum", calcium_statistics.min);
    essentials->insert("Calcium-Average", calcium_statistics.avg);
    essentials->insert("Calcium-Maximum", calcium_statistics.max);
}

void Neurons::print_synaptic_changes_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    auto helper = [&essentials, this](const std::string& message, const auto synaptic_element_type) {
        const auto local_adds = synaptic_elements->get_total_additions(synaptic_element_type);
        const auto local_dels = synaptic_elements->get_total_deletions(synaptic_element_type);

        const auto global_adds = mpiPP::MPIReductions::reduce_sum(local_adds);
        const auto global_dels = mpiPP::MPIReductions::reduce_sum(local_dels);

        if (mpiPP::MPIRank::root_rank() == mpiPP::MPIInfo::get_my_rank()) {
            essentials->insert(message + "Additions", global_adds);
            essentials->insert(message + "Deletions", global_dels);
        }
    };

    helper("Axons-", SynapticElementType::Axon);
    helper("Dendrites-Excitatory-", SynapticElementType::DendriteExcitatory);
    helper("Dendrites-Inhibitory-", SynapticElementType::DendriteInhibitory);
}

void Neurons::print_network_graph_to_log_file(const step_type step, const bool with_prefix) const {
    if (with_prefix) {
        const auto prefix = "step_" + std::to_string(step) + "_";
        LogFiles::save_and_open_new(LogFiles::EventType::InNetwork, prefix + "in_network", "network/");
        LogFiles::save_and_open_new(LogFiles::EventType::OutNetwork, prefix + "out_network", "network/");
    } else {
        LogFiles::save_and_open_new(LogFiles::EventType::InNetwork, "in_network", "network/");
        LogFiles::save_and_open_new(LogFiles::EventType::OutNetwork, "out_network", "network/");
    }

    network_graph->update_host_if_necessary();

    auto ss_in_network = std::stringstream{};
    auto ss_out_network = std::stringstream{};

    const auto& [plastic_distant_out, static_distant_out] = network_graph->get_all_distant_out_edges();
    const auto& [plastic_local_out, static_local_out] = network_graph->get_all_local_out_edges();

    const auto& [plastic_distant_in, static_distant_in] = network_graph->get_all_distant_in_edges();
    const auto& [plastic_local_in, static_local_in] = network_graph->get_all_local_in_edges();

    NeuronIO::write_out_synapses(static_local_out, static_distant_out, plastic_local_out, plastic_distant_out, mpiPP::MPIInfo::get_my_rank(),
                                 partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
                                 partition->get_total_number_neurons(), ss_out_network, step);

    NeuronIO::write_in_synapses(static_local_in, static_distant_in, plastic_local_in, plastic_distant_in, mpiPP::MPIInfo::get_my_rank(),
                                partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
                                partition->get_total_number_neurons(), ss_in_network, step);

    LogFiles::write_to_file(LogFiles::EventType::InNetwork, false, ss_in_network.str());
    LogFiles::write_to_file(LogFiles::EventType::OutNetwork, false, ss_out_network.str());
}

void Neurons::print_positions_to_log_file() const {
    auto sstream = std::stringstream{};
    NeuronIO::write_neuron_positions_and_signals_componentwise(NeuronIDRange::range(number_neurons) | ranges::to_vector, extra_info->get_positions(),
                                                               synaptic_elements->get_signal_types(), sstream, partition->get_total_number_neurons(), partition->get_simulation_box_size(), partition->get_all_local_subdomain_boundaries());

    LogFiles::write_to_file(LogFiles::EventType::Positions, false, sstream.str());
    LogFiles::flush_file(LogFiles::EventType::Positions);
}

void Neurons::print_groups_to_log_file() const {
    auto sstream = std::stringstream{};
    NeuronIO::write_neuron_groups(sstream, local_group_translator);

    LogFiles::write_to_file(LogFiles::EventType::Groups, false, sstream.str());
    LogFiles::flush_file(LogFiles::EventType::Groups);
}

void Neurons::print_group_mapping_to_log_file() const {
    auto sstream = std::stringstream{};
    NeuronIO::write_group_names(sstream, local_group_translator);

    LogFiles::write_to_file(LogFiles::EventType::GroupMapping, false, sstream.str());
    LogFiles::flush_file(LogFiles::EventType::GroupMapping);
}

void Neurons::print_groups_to_file_name_to_log_file() const {
    auto sstream = std::stringstream{};
    NeuronIO::write_group_name_to_file_name(sstream, local_group_translator);

    LogFiles::write_to_file(LogFiles::EventType::GroupToFileMapping, false, sstream.str());
    LogFiles::flush_file(LogFiles::EventType::GroupToFileMapping);
}

void Neurons::print() {
    const auto& calcium = calcium_calculator->get_calcium();

    // Column widths
    constexpr auto cwidth_left = 6;
    constexpr auto cwidth = 20;

    // Heading
    LogFiles::write_to_file(LogFiles::EventType::Cout, true,
                            "{2:<{1}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}{10:<{0}}", cwidth, cwidth_left,
                            "gid", "x", "AP", "refractory_time", "C", "A_ex", "A_in", "D_ex", "D_in");

    // Values
    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        LogFiles::write_to_file(LogFiles::EventType::Cout, true,
                                "{3:<{1}}{4:<{0}.{2}f}{5:<{0}}{6:<{0}.{2}f}{7:<{0}.{2}f}{8:<{0}.{2}f}{9:<{0}.{2}f}",
                                cwidth, cwidth_left, Constants::print_precision,
                                local_neuron_id,
                                neuron_model->get_x(neuron_id),
                                neuron_model->has_fired(neuron_id),
                                calcium[local_neuron_id],
                                synaptic_elements->get_grown_elements(SynapticElementType::Axon),
                                synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory),
                                synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory));
    }
}

void Neurons::print_info_for_algorithm() {
    const auto axons_counts = synaptic_elements->get_grown_elements(SynapticElementType::Axon);
    const auto dendrites_exc_counts = synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory);
    const auto dendrites_inh_counts = synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory);

    const auto& axons_connected_counts = synaptic_elements->get_connected_elements(SynapticElementType::Axon);
    const auto& dendrites_exc_connected_counts = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto& dendrites_inh_connected_counts = synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory);

    // Column widths
    const int cwidth_small = 8;
    const int cwidth_medium = 16;
    const int cwidth_big = 27;

    auto sstream = std::stringstream{};
    auto my_string = std::string{};

    // Heading
    sstream << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region"
            << std::setw(cwidth_medium) << "position";
    sstream << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
    sstream << std::setw(cwidth_big) << "inh_den (exist|connected)\n";

    // Values
    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        sstream << std::left << std::setw(cwidth_small) << neuron_id;

        const auto [x, y, z] = extra_info->get_position(neuron_id);

        my_string = "(" + std::to_string(x) + ',' + std::to_string(y) + ',' + std::to_string(z) + ")";
        sstream << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_counts[local_neuron_id]) + "|" + std::to_string(axons_connected_counts[local_neuron_id]);
        sstream << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_counts[local_neuron_id]) + "|" + std::to_string(dendrites_exc_connected_counts[local_neuron_id]);
        sstream << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_counts[local_neuron_id]) + "|" + std::to_string(dendrites_inh_connected_counts[local_neuron_id]);
        sstream << std::setw(cwidth_big) << my_string;

        sstream << '\n';
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, sstream.str());
}

void Neurons::print_calcium_values_to_file(const step_type current_step) {
    if (LogFiles::get_log_status(LogFiles::EventType::CalciumValues)) {
        return;
    }

    const auto& calcium = calcium_calculator->get_calcium();

    auto sstream = std::stringstream{};

    sstream << '#' << current_step;
    for (const auto val : calcium) {
        sstream << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::CalciumValues, false, sstream.str());
}

void Neurons::print_fire_rate_to_file(const step_type current_step) {
    if (LogFiles::get_log_status(LogFiles::EventType::FireRates)) {
        return;
    }

    const auto& fired_recorder = neuron_model->get_fired_status_recorder();
    const auto& fire_recorder = fired_recorder->get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);

    auto sstream = std::stringstream{};

    sstream << '#' << current_step;
    for (const auto val : fire_recorder) {
        // This gives the frequency in [Hz] as every step is 1 ms
        const auto frequency = static_cast<RelearnTypes::fire_rate_type>(val) / static_cast<RelearnTypes::fire_rate_type>(Config::fire_rate_log_step) * RelearnTypes::fire_rate_type{ 1000 };
        sstream << ';' << frequency;
    }

    LogFiles::write_to_file(LogFiles::EventType::FireRates, false, sstream.str());
}
//
// void Neurons::print_fire_steps_to_file(const step_type current_step, const step_type steps_since_last_print) const {
//     if (LogFiles::get_log_status(LogFiles::EventType::FireSteps)) {
//         return;
//     }
//
//     const auto& fired_recorder = neuron_model->get_fired_status_recorder();
//     const auto fire_history_size = fired_recorder->get_fire_history_size();
//     RelearnException::check(steps_since_last_print <= fire_history_size, "Neurons::Print_fire_steps_to_file: Fire history is too small {} {}", steps_since_last_print, fire_history_size);
//     auto ss = std::stringstream{};
//
//     const auto offset_bitset = std::min(static_cast<std::int64_t>(fire_history_size - 1), static_cast<std::int64_t>(steps_since_last_print));
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
//         const auto& fire_history = fired_recorder->get_fire_history(neuron_id);
//
//         ss << neuron_id.get_neuron_id() + 1 << ',';
//         for (auto i = offset_bitset; i >= 0; i--) {
//             if (fire_history.test(static_cast<unsigned int>(i))) {
//                 const auto step = current_step - i;
//                 ss << step << ',';
//             }
//         }
//         ss << '\n';
//     }
//
//     LogFiles::write_to_file(LogFiles::EventType::FireSteps, false, ss.rdbuf()->view());
// }
