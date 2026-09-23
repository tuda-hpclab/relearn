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

#include "Config.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "sim/Essentials.h"
#include "types/AlgorithmTypes.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/StatisticalMeasures.h"

#include <cpp-utility/Interval.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

enum class NetworkGPUType;

namespace utility {
class MemoryFootprint;
}

class Algorithm;
class GlobalGroupMapper;
class GroupMonitor;
class KernelBase;
class NeuronMonitor;
class NeuronToSubdomainAssignment;
class Neurons;
class Partition;

/**
 * This class encapsulates all necessary attributes of a simulation.
 * The neuron model, the synaptic elements, and the subdomain assignment must be set before calling initialize,
 * which in turn must happen before calling simulate.
 */
class Simulation {
public:
    using step_type = RelearnTypes::step_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using calcium_type = RelearnTypes::calcium_type;
    using acceptance_criterion_type = RelearnTypes::acceptance_criterion_type;
    using percentage_type = RelearnTypes::percentage_type;

    /**
     * @brief Constructs a new object with the given partition and essentials.
     * @param _essentials The essentials container for this simulation
     * @param _partition The partition for this simulation
     */
    Simulation(std::unique_ptr<Essentials> _essentials, std::shared_ptr<Partition> _partition);

    /**
     * @brief Registers a monitor for the given neuron id.
     *      Does not check for duplicates, etc.
     * @param neuron_id The local neuron id that should be monitored
     */
    void register_neuron_monitor(NeuronID neuron_id);

    /**
     * @brief Enables the group monitor for all groups. Must be called before initialize()
     * @param enable If the group monitor shall be enabled
     */
    void enable_group_monitor(const bool enable, const bool monitor_connectivity) {
        RelearnException::check(enable || !monitor_connectivity, "Simulation::enable_group_monitor: You cant monitor the connectivity without the group monitor enabled");
        group_monitor_enabled = enable;
        group_monitor_connectivity = monitor_connectivity;
    }

    /**
     * @brief Sets the acceptance criterion (theta) for the barnes hut algorithm
     * @param value The acceptance criterion (theta) in [0.0, BarnesHut::max_theta]
     * @exception Throws a RelearnException if value is not from [0.0, BarnesHut::max_theta]
     */
    void set_acceptance_criterion_for_barnes_hut(acceptance_criterion_type value);

    void set_algorithm_vector_for_combined_algorithms(RelearnTypes::AlgorithmConfigs&& algorithm_configs_to_use);

    void set_indices_and_neurons_for_combined_algorithms(const RelearnTypes::AlgorithmIndexWithNeuronsType& inds_and_neurons);

    void set_network_type(const NetworkGPUType _network_gpu_type) {
        network_gpu_type = _network_gpu_type;
    }

    /**
     * @brief Sets the neuron model used for the simulation
     * @param _neuron_model The neuron model
     */
    void set_neuron_model(std::unique_ptr<NeuronModel>&& _neuron_model) noexcept;

    /**
     * @brief Sets the calcium calculator used for the simulation
     * @param calculator The calcium calculator
     */
    void set_calcium_calculator(std::unique_ptr<CalciumCalculator>&& calculator) noexcept;

    /**
     * @brief Sets the synaptic elements used for the simulation
     * @param _synaptic_elements The synaptic elements
     */
    void set_synaptic_elements(std::shared_ptr<SynapticElements>&& _synaptic_elements) noexcept;

    /**
     * @brief Sets the synapse deletion finder
     * @param sdf The synapse deletion finder
     */
    void set_synapse_deletion_finder(std::unique_ptr<SynapseDeletionFinder>&& sdf) noexcept;

    /**
     * @brief Sets the enable interrupts during the simulation.
     *      An enable interrupt is a pair of (1) the simulation set (2) all local ids that should be enabled
     * @param interrupts The enable interrupts
     */
    void set_enable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts);

    /**
     * @brief Sets the disable interrupts during the simulation.
     *      An disable interrupt is a pair of (1) the simulation set (2) all local ids that should be disabled
     * @param interrupts The disable interrupts
     */
    void set_disable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts);

    /**
     * @brief Sets the creation interrupts during the simulation.
     *      An creation interrupt is a pair of (1) the simulation set (2) the number of neurons to create
     * @param interrupts The creation interrupts
     */
    void set_creation_interrupts(std::vector<std::pair<step_type, number_neurons_type>> interrupts) noexcept;

    /**
     * @brief Sets the algorithm that is used for finding target neurons.
     * @param new_algorithm_enum The desired algorithm
     */
    void set_algorithm(AlgorithmEnum new_algorithm_enum) noexcept;

    /**
     * @brief Sets the probability kernel that is used for the simulation
     * @param kernel The kernel
     */
    void set_probability_kernel(std::unique_ptr<KernelBase>&& kernel) noexcept;

    /**
     * @brief Sets the percentage of neurons that fired in the 0th simulation step.
     * @param percentage The percentage, must be 0.0 <= percentage <= 1.0
     * @exception Throws a RelearnException if the percentage is out of bounds
     */
    void set_percentage_initial_fired_neurons(RelearnTypes::percentage_type percentage);

    /**
     * @brief Sets the subdomain assignment that determines how the neurons are loaded.
     * @param subdomain_assignment The desired subdomain assignment
     */
    void set_subdomain_assignment(std::unique_ptr<NeuronToSubdomainAssignment>&& subdomain_assignment) noexcept;

    /**
     * @brief Sets the list of neurons into a static sate. Only static connections are allowed from and to a static neuron
     * @param _static_neurons Vector with neuron ids for the local rank
     */
    void set_static_neurons(std::vector<NeuronID> _static_neurons);

    /**
     * @brief Sets the new interval determining when the electrical activity is updated
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_update_electrical_activity_interval(const auto& interval) {
        interval_update_electrical_activity = interval;
    }

    /**
     * @brief Sets the new interval determining when the synaptic elements is updated
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_update_synaptic_elements_interval(const auto& interval) {
        interval_update_synaptic_elements = interval;
    }

    /**
     * @brief Sets the new interval determining when the plasticity is updated
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_update_plasticity_interval(const auto& interval) {
        interval_update_plasticity = interval;
    }

    /**
     * @brief Sets the new interval determining when the neuron monitors are updated
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_update_neuron_monitor_interval(const auto& interval) {
        interval_neuron_monitor = interval;
    }

    /**
     * @brief Sets the new interval determining when the group monitors are updated
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_update_group_monitor_interval(const auto& interval) {
        interval_group_monitor = interval;
    }

    /**
     * @brief Sets the new interval determining when the calcium is logged
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_log_calcium_interval(const auto& interval) {
        interval_calcium_log = interval;
    }

    /**
     * @brief Sets the new interval determining when the fire rate is logged
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_log_fire_rate_interval(const auto& interval) {
        interval_fire_rate_log = interval;
    }

    /**
     * @brief Sets the new interval determining when the synaptic input is logged
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_log_synaptic_input_interval(const auto& interval) {
        interval_synaptic_input_log = interval;
    }

    /**
     * @brief Sets the new interval determining when the network is logged
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_log_network_interval(const auto& interval) {
        interval_network_log = interval;
    }

    /**
     * @brief Sets the new interval determining when the statistics are logged
     * @param interval The new interval with first and last step, as well as the frequency
     */
    void set_log_statistics_interval(const auto& interval) {
        interval_statistics_log = interval;
    }

    /**
     * @brief Initializes the simulation and all other objects.
     * @exception Throws a RelearnException if one object is missing or something went wrong otherwise
     */
    void initialize();

    /**
     * @brief Simulates the neurons for the requested number of steps. Every step_monitor-th step, records all neuron monitors
     * @param number_steps The number of simulation steps, must be > 0
     * @exception Throws a RelearnException if number_steps == 0
     */
    void simulate(step_type number_steps);

    /**
     * @brief Finalizes the simulation in the sense that it prints the final statistics.
     *      Does not perform any "irreversible" steps and does not finalize MPI.
     *      All MPI processes must call finalize
     */
    void finalize() const;

    void final_timer_print() const;

    /**
     * @brief Returns an std::shared_ptr to the partition object
     * @return The partition object
     */
    [[nodiscard]] const std::shared_ptr<Partition>& get_partition() const noexcept {
        return partition;
    }

    /**
     * @brief Returns an std::shared_ptr to the neurons object
     * @return The neurons object
     */
    [[nodiscard]] std::shared_ptr<Neurons> get_neurons() const noexcept {
        return neurons;
    }

    void print_memory_footprint() const;

    /**
     * @brief Returns the neuron monitor
     * @return A constant reference to the neuron monitor
     */
    [[nodiscard]] const std::unique_ptr<NeuronMonitor>& get_monitor() const noexcept {
        return neuron_monitor;
    }

    /**
     * @brief Adds the statistics for the global statistics overview
     *      Does nothing if the statistics has been added before
     * @param neuron_attribute_to_observe The statistics that should be observed
     */
    void add_statistical_overview(NeuronAttribute neuron_attribute_to_observe) noexcept {
        if (statistics.find(neuron_attribute_to_observe) == statistics.end()) {
            statistics.emplace(neuron_attribute_to_observe, std::vector<StatisticalMeasures>{});
        }
    }

    /**
     * @brief Returns the statistics observed for the requested attribute
     * @param neuron_attribute_to_observe The statistics
     * @exception Throws a RelearnException if the statistics have not been observed
     * @return A constants reference to the statistics
     */
    [[nodiscard]] const std::vector<StatisticalMeasures>& get_statistics(NeuronAttribute neuron_attribute_to_observe) const {
        if (statistics.find(neuron_attribute_to_observe) == statistics.end()) {
            RelearnException::fail("Simulation::get_statistics: The attribute was not observed: {}", static_cast<int>(neuron_attribute_to_observe));
        }

        const auto& return_value = statistics.at(neuron_attribute_to_observe);

        return return_value;
    }

    /**
     * @brief Records one snapshot of each neuron monitor
     */
    void snapshot_monitors();

    /**
     * @brief Returns the group monitors
     * @return The group monitors
     */
    [[nodiscard]] const std::shared_ptr<std::unordered_map<RelearnTypes::group_id, GroupMonitor>>& get_group_monitors() const noexcept {
        return group_monitors;
    }

private:
    std::unique_ptr<Essentials> essentials{};
    std::unique_ptr<utility::MemoryFootprint> footprint{};
    std::unique_ptr<utility::MemoryFootprint> usage_footprint{};

    std::unique_ptr<KernelBase> probability_kernel{};

    std::shared_ptr<Partition> partition;

    std::unique_ptr<NeuronToSubdomainAssignment> neuron_to_subdomain_assignment{};

    std::shared_ptr<SynapticElements> synaptic_elements;

    std::vector<NeuronID> static_neurons{};
    std::vector<std::string> static_groups{};

    std::unique_ptr<NeuronModel> neuron_models{};
    std::unique_ptr<CalciumCalculator> calcium_calculator{};
    std::shared_ptr<Neurons> neurons;
    std::unique_ptr<SynapseDeletionFinder> synapse_deletion_finder{};

    std::unique_ptr<NeuronMonitor> neuron_monitor{};
    std::shared_ptr<std::unordered_map<RelearnTypes::group_id, GroupMonitor>> group_monitors{};
    std::shared_ptr<GlobalGroupMapper> global_group_mapper;

    std::vector<std::pair<step_type, std::vector<NeuronID>>> enable_interrupts{};
    std::vector<std::pair<step_type, std::vector<NeuronID>>> disable_interrupts{};
    std::vector<std::pair<step_type, number_neurons_type>> creation_interrupts{};

    std::map<NeuronAttribute, std::vector<StatisticalMeasures>> statistics{};

    std::function<calcium_type(int, NeuronID::value_type)> target_calcium_calculator{};
    std::function<calcium_type(int, NeuronID::value_type)> initial_calcium_initiator{};

    utility::Interval<step_type> interval_update_electrical_activity{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = RelearnTypes::step_type{ 1 } };
    utility::Interval<step_type> interval_update_synaptic_elements{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = RelearnTypes::step_type{ 1 } };
    utility::Interval<step_type> interval_update_plasticity{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::plasticity_update_step };

    utility::Interval<step_type> interval_neuron_monitor{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::neuron_monitor_log_step };
    utility::Interval<step_type> interval_group_monitor{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::group_monitor_log_step };

    utility::Interval<step_type> interval_calcium_log{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::calcium_log_step };
    utility::Interval<step_type> interval_fire_rate_log{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::fire_rate_log_step };
    utility::Interval<step_type> interval_synaptic_input_log{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::synaptic_input_log_step };
    utility::Interval<step_type> interval_network_log{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::network_log_step };
    utility::Interval<step_type> interval_flush_all_logs_step{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::flush_monitor_step };

    utility::Interval<step_type> interval_statistics_log{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::statistics_log_step };

    percentage_type percentage_initially_fired{ 0.0 };

    bool group_monitor_enabled{ false };
    bool group_monitor_connectivity{ true };

    acceptance_criterion_type accept_criterion{ 0.0 };

    RelearnTypes::AlgorithmConfigs algorithms;
    RelearnTypes::AlgorithmIndexWithNeuronsType indices_and_neurons;

    AlgorithmEnum algorithm_enum{};
    NetworkGPUType network_gpu_type{};

    std::int64_t total_synapse_creations{ 0 };
    std::int64_t total_synapse_deletions{ 0 };

    std::int64_t delta_synapse_creations{ 0 };
    std::int64_t delta_synapse_deletions{ 0 };

    step_type step{ 1 };

    std::chrono::system_clock::time_point start_time;
};
