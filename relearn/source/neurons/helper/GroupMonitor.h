#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/SynapticElementType.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <boost/functional/hash.hpp>
#pragma GCC diagnostic pop

#include <cstdint>
#include <filesystem>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

class GlobalGroupMapper;
class Neurons;
class Simulation;

/**
 * Monitors the number of connections between groups and more group statistics
 */
class GroupMonitor {
public:
    struct GroupConnection {
    public:
        GroupConnection() = default;

        GroupConnection(const int _from_rank, const RelearnTypes::group_id _from_group,
                        const NeuronID _to_local_neuron_id, const SignalType _signal_type)
            : from_rank(_from_rank)
            , from_group(_from_group)
            , to_local_neuron_id(_to_local_neuron_id)
            , signal_type(_signal_type) { }

        int from_rank{ -1 };
        RelearnTypes::group_id from_group{};
        NeuronID to_local_neuron_id;
        SignalType signal_type{};
    };

    /**
     * Construct an object for monitoring a specific group on this mpi rank
     * @param _neurons The neurons object
     * @param _global_group_mapper Global group id mapper
     * @param _group_id Id of the group that will be monitored
     * @param _group_name Name of the group that will be monitored
     * @param rank The mpi rank of this process
     * @param _path The output file path
     * @param monitor_connectivity If the connectivity should be monitored
     */
    GroupMonitor(std::shared_ptr<Neurons> _neurons, std::shared_ptr<GlobalGroupMapper> _global_group_mapper,
                 RelearnTypes::group_id _group_id, RelearnTypes::group_name _group_name, int rank, std::filesystem::path _path, bool monitor_connectivity);

    /**
     * @brief Requests the data from all other ranks which is relevant for the monitoring of this group
     */
    void request_data() const;

    /**
     * Add an ingoing connection to the group. This method shall be called by other group monitors with ingoing connections to this group
     * @param connection Connection whose source is this group
     * @param weight The weight of the connection
     */
    void add_ingoing_connection(const GroupConnection& connection, RelearnTypes::plastic_synapse_weight weight);

    /**
     * @brief Removes an incoming connection from the group.
     *      This method shall be called by other group monitors with ingoing connections to this group
     * @param connection Connection whose source is this group
     * @param weight The weight of the connection
     */
    void remove_ingoing_connection(const GroupMonitor::GroupConnection& connection, RelearnTypes::plastic_synapse_weight weight);

    /**
     * Prepares the monitor for a new logging step. Call this method before each logging step.
     */
    void prepare_recording();

    /**
     * Add the data of a single neuron to the recording. The neuron must be part of the ensemble.
     * Call this method with each neuron of the ensemble in each logging step
     * @param neuron_id Neuron which is part of the ensemble
     */
    void record_data(NeuronID neuron_id);

    /**
     * Indicates end of a single logging step. Call this method after the data off each neuron was recorded.
     */
    void finish_recording();

    /**
     * Write all recorded data to a csv file
     */
    void write_data_to_file();

    /**
     * Returns the name of the group that is monitored
     * @return Group name
     */
    [[nodiscard]] const RelearnTypes::group_name& get_group_name() const noexcept {
        return group_name;
    }

    void monitor_connectivity();

    /**
     * Returns the id of the group that is monitored
     * @return Group id
     */
    [[nodiscard]] const RelearnTypes::group_id& get_group_id() const noexcept {
        return group_id;
    }

private:
    /**
     * Number of connections to another ensemble in a single step
     */
    struct ConnectionCount {
        RelearnTypes::plastic_synapse_weight den_ex = 0;
        RelearnTypes::plastic_synapse_weight den_inh = 0;
    };

    std::shared_ptr<Neurons> neurons;

    int my_rank;

    bool flag_monitor_connectivity{ true };

    RelearnTypes::step_type step = 0;

    std::filesystem::path path;

    RelearnTypes::group_name group_name;

    RelearnTypes::group_id group_id;

    struct InternalStatistics {
        RelearnTypes::grown_type axons_grown = 0;
        RelearnTypes::grown_type den_ex_grown = 0;
        RelearnTypes::grown_type den_inh_grown = 0;
        std::uint64_t axons_conn = 0;
        std::uint64_t den_ex_conn = 0;
        std::uint64_t den_inh_conn = 0;
        RelearnTypes::activity_type activity_input = 0;
        RelearnTypes::calcium_type calcium = 0;
        RelearnTypes::fire_rate_type fired_fraction = 0.0;
        RelearnTypes::number_neurons_type num_enabled_neurons = 0;
    };

    using EnsembleConnections = std::unordered_map<std::pair<int, RelearnTypes::group_id>, ConnectionCount,
                                                   boost::hash<std::pair<int, RelearnTypes::group_id>>>;
    using EnsembleDeletions = std::unordered_map<std::pair<int, RelearnTypes::group_id>, std::int64_t,
                                                 boost::hash<std::pair<int, RelearnTypes::group_id>>>;

    /**
     * For current logging step: Maps for each ensemble the number of connections
     */
    EnsembleConnections connections;
    EnsembleDeletions deletions;
    InternalStatistics internal_statistics{};

    /**
     * Complete data of all earlier logging steps
     */
    std::vector<std::tuple<EnsembleConnections, EnsembleDeletions, InternalStatistics>> data;

    std::shared_ptr<GlobalGroupMapper> global_group_mapper;
    void write_header() const;
};
