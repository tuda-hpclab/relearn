/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GroupMonitor.h"

#include "Config.h"
#include "Types.h"

#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/GlobalGroupMapper.h"
#include "sim/Simulation.h"
#include "util/Timers.h"

#include "cpp-utility/ranges/Functional.hpp"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/map.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>

GroupMonitor::GroupMonitor(std::shared_ptr<Neurons> _neurons,
                           std::shared_ptr<GlobalGroupMapper> _global_group_mapper,
                           const RelearnTypes::group_id _group_id,
                           RelearnTypes::group_name _group_name,
                           const int rank,
                           std::filesystem::path _path,
                           const bool monitor_connectivity)
    : neurons(std::move(_neurons))
    , my_rank(rank)
    , flag_monitor_connectivity(monitor_connectivity)
    , path(std::move(_path))
    , group_name(std::move(_group_name))
    , group_id(_group_id)
    , global_group_mapper(std::move(_global_group_mapper)) {
    write_header();
}

void GroupMonitor::monitor_connectivity() {
    if (!flag_monitor_connectivity) {
        return;
    }

    const auto& local_group_translator = neurons->get_local_group_translator();

    Timers::start(TimerRegion::GROUP_MONITORS_LOCAL_EDGES);
    for (const auto& [target, source, weight] : neurons->last_created_local_synapses) {
        const auto target_group_ids = local_group_translator->get_group_ids_for_neuron_id_unordered(target.get_neuron_id());
        if (!target_group_ids.contains(group_id)) {
            continue;
        }

        const auto other_group_ids = local_group_translator->get_group_ids_for_neuron_id(source.get_neuron_id());
        const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
        for (const auto& source_group_id : other_group_ids) {
            add_ingoing_connection({ my_rank, source_group_id, target, signal_type }, weight);
        }
    }
    Timers::stop_and_add(TimerRegion::GROUP_MONITORS_LOCAL_EDGES);

    Timers::start(TimerRegion::GROUP_MONITORS_DISTANT_EDGES);
    for (const auto& [target, source, weight] : neurons->last_created_in_synapses) {
        const auto target_group_ids = local_group_translator->get_group_ids_for_neuron_id_unordered(target.get_neuron_id());
        if (!target_group_ids.contains(group_id)) {
            continue;
        }

        // Other group is on different mpi rank. Save connection for communication over mpi
        const auto& other_rank = source.get_rank().get_rank();
        const auto other_group_ids = global_group_mapper->get_group_ids(source);
        const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
        for (const auto& source_group_id : other_group_ids) {
            add_ingoing_connection({ other_rank, source_group_id, target, signal_type }, weight);
        }
    }
    Timers::stop_and_add(TimerRegion::GROUP_MONITORS_DISTANT_EDGES);
}

void GroupMonitor::record_data(const NeuronID neuron_id) {
    Timers::start(TimerRegion::GROUP_MONITORS_DELETIONS);
    if (flag_monitor_connectivity) {
        // Deletions
        const auto& deletions_in_step = neurons->get_extra_info()->get_deletions_log(neuron_id);
        for (const auto& [other_neuron_id, weight] : deletions_in_step) {
            const auto other_group_ids = global_group_mapper->get_group_ids(other_neuron_id);
            const auto& other_rank = other_neuron_id.get_rank().get_rank();
            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            for (const auto& other_group_id : other_group_ids) {
                deletions[{ other_rank, other_group_id }]++;
                remove_ingoing_connection(GroupConnection(other_rank, other_group_id, neuron_id, signal_type), weight);
            }
        }
    }
    Timers::stop_and_add(TimerRegion::GROUP_MONITORS_DELETIONS);

    Timers::start(TimerRegion::GROUP_MONITORS_STATISTICS);

    const auto& synaptic_elements = neurons->get_synaptic_elements();
    const auto id = neuron_id.get_neuron_id();

    // Store statistics
    internal_statistics.axons_grown += synaptic_elements->get_grown_elements(SynapticElementType::Axon)[id];
    internal_statistics.den_ex_grown += synaptic_elements->get_grown_elements(SynapticElementType::DendriteExcitatory)[id];
    internal_statistics.den_inh_grown += synaptic_elements->get_grown_elements(SynapticElementType::DendriteInhibitory)[id];

    internal_statistics.axons_conn += synaptic_elements->get_connected_elements(SynapticElementType::Axon)[id];
    internal_statistics.den_ex_conn += synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory)[id];
    internal_statistics.den_inh_conn += synaptic_elements->get_connected_elements(SynapticElementType::DendriteInhibitory)[id];

    internal_statistics.activity_input += neurons->neuron_model->get_activity_input()->get_input(neuron_id);

    internal_statistics.calcium += neurons->get_calcium(neuron_id);

    const auto& fired_recorder = neurons->get_neuron_model()->get_fired_status_recorder();
    const auto fired = fired_recorder->get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);

    internal_statistics.fired_fraction += static_cast<double>(fired[neuron_id.get_neuron_id()]) / static_cast<double>(Config::plasticity_update_step);
    internal_statistics.num_enabled_neurons++;

    Timers::stop_and_add(TimerRegion::GROUP_MONITORS_STATISTICS);
}

void GroupMonitor::prepare_recording() {
    deletions = EnsembleDeletions{};
    internal_statistics = InternalStatistics{};
}

void GroupMonitor::finish_recording() {
    data.emplace_back(connections, deletions, internal_statistics);
}

void GroupMonitor::write_header() const {
    auto out = std::ofstream(path, std::ios_base::app);

    // Header
    out << "# Connections from ensemble " << group_name << " (" << my_rank << ':' << group_id << ") to ..."
        << "\n";
    out << "# Rank: " << my_rank << "\n";
    out << "# Group id: " << group_id << "\n";
    out << "# Group name: " << group_name << "\n";
}

void GroupMonitor::write_data_to_file() {
    auto out = std::ofstream(path, std::ios_base::app);

    const auto unique_group_ids1 = data
                                   | ranges::views::for_each(utility::element<0>)
                                   | ranges::views::keys;

    const auto unique_group_ids2 = data
                                   | ranges::views::for_each(utility::element<1>)
                                   | ranges::views::keys;

    const auto unique_group_ids = ranges::views::concat(unique_group_ids1, unique_group_ids2) | ranges::to<std::set>;

    // Header
    out << "# Step;";
    for (const auto& [rank, _group_id] : unique_group_ids) {
        out << rank << ':' << _group_id << "ex;"
            << rank << ':' << _group_id << "in;"
            << rank << ':' << _group_id << "del;";
    }

    out << "Axons grown;Axons conn;Den ex grown;Den ex conn;Den inh grown;Den inh conn;Activity input;Calcium;Fire rate;Enabled neurons;\n";

    // Data
    for (auto& [connection_data, deletion_data, internal_statistics_data] : data) {
        out << step << ';';

        for (const auto& rank_group_id : unique_group_ids) {
            const auto& _connections = connection_data[rank_group_id];
            out << std::to_string(_connections.den_ex) << ';';
            out << std::to_string(_connections.den_inh) << ';';

            const auto& deletions_in_step = deletion_data[rank_group_id];
            out << std::to_string(deletions_in_step) << ';';
        }

        out << internal_statistics_data.axons_grown << ';';
        out << internal_statistics_data.axons_conn << ';';
        out << internal_statistics_data.den_ex_grown << ';';
        out << internal_statistics_data.den_ex_conn << ';';
        out << internal_statistics_data.den_inh_grown << ';';
        out << internal_statistics_data.den_inh_conn << ';';
        out << internal_statistics_data.activity_input << ';';
        out << internal_statistics_data.calcium << ';';
        out << internal_statistics_data.fired_fraction << ';';
        out << internal_statistics_data.num_enabled_neurons << ';';

        out << '\n';
        step += Config::neuron_monitor_log_step;
    }

    data.clear();
}

void GroupMonitor::request_data() const {
    if (!flag_monitor_connectivity) {
        return;
    }

    const auto& local_group_translator = neurons->get_local_group_translator();

    Timers::start(TimerRegion::GROUP_MONITORS_DISTANT_EDGES);
    for (const auto& [target, source, _] : neurons->last_created_in_synapses) {
        const auto target_group_ids = local_group_translator->get_group_ids_for_neuron_id_unordered(target.get_neuron_id());
        if (!target_group_ids.contains(group_id)) {
            continue;
        }

        // Other group is on different mpi rank. Save connection for communication over mpi
        global_group_mapper->request_group_ids(source);
    }
    Timers::stop_and_add(TimerRegion::GROUP_MONITORS_DISTANT_EDGES);
}

void GroupMonitor::add_ingoing_connection(const GroupMonitor::GroupConnection& connection, const RelearnTypes::plastic_synapse_weight weight) {
    auto& conn = connections[{ connection.from_rank, connection.from_group }];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex += std::abs(weight);
    } else {
        conn.den_inh += std::abs(weight);
    }
}

void GroupMonitor::remove_ingoing_connection(const GroupMonitor::GroupConnection& connection, const RelearnTypes::plastic_synapse_weight weight) {
    auto& conn = connections[{ connection.from_rank, connection.from_group }];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex -= std::abs(weight);
    } else {
        conn.den_inh -= std::abs(weight);
    }
}