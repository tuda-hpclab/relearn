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

#include "Config.h"
#include "Types.h"
#include "Types3.h"

#include "algorithm/Algorithm.h"
#include "algorithm/Kernel/KernelBase.h"
#include "models/NeuronModel.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "synaptic_elements/SynapticElements.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/StatisticalMeasures.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <range/v3/functional/arithmetic.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class GroupMonitor;
class Essentials;
class LocalGroupTranslator;
class NetworkGraph;
class NeuronMonitor;
class Partition;

/**
 * This class gathers all information for the neurons and provides the primary interface for the simulation
 */
class Neurons {
    friend class GroupMonitor;

public:
    using step_type = RelearnTypes::step_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Creates a new object with the passed Partition, NeuronModel, Axons, DendritesExc, and DendritesInh
     * @param part The partition, is only used for printing, must not be empty
     * @param model_ptr The electrical model for the neurons, must not be empty
     * @param calculator_ptr The calcium calculator, must not be empty
     * @param network_graph_ptr The network graph for the connections, must not be empty
     * @exception Throws a RelearnException if any of the pointers is empty
     */
    Neurons(std::shared_ptr<Partition> part,
            std::unique_ptr<NeuronModel> model_ptr,
            std::unique_ptr<CalciumCalculator> calculator_ptr,
            std::shared_ptr<NetworkGraph> network_graph_ptr,
            std::shared_ptr<SynapticElements> synaptic_elements_ptr,
            std::unique_ptr<SynapseDeletionFinder> synapse_del_ptr)
        : partition(std::move(part))
        , neuron_model(std::move(model_ptr))
        , calcium_calculator(std::move(calculator_ptr))
        , network_graph(std::move(network_graph_ptr))
        , synaptic_elements(std::move(synaptic_elements_ptr))
        , synapse_deletion_finder(std::move(synapse_del_ptr))
        , extra_info(std::make_shared<NeuronsExtraInfo>()) {

        const bool all_filled = this->partition && this->network_graph && neuron_model && calcium_calculator && synaptic_elements && synapse_deletion_finder;
        RelearnException::check(all_filled, "Neurons::Neurons: Neurons was constructed with some null arguments");
    }

    ~Neurons() = default;

    Neurons(const Neurons& other) = delete;
    Neurons(Neurons&& other) = default;

    Neurons& operator=(const Neurons& other) = delete;
    Neurons& operator=(Neurons&& other) = default;

    /**
     * @brief Initializes this class and all models with number_neurons_init, i.e.,
     *      (a) Initializes the electrical model
     *      (b) Initializes the extra infos
     *      (c) Initializes the synaptic models
     *      (d) Enables all neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     * @param number_neurons_init The number of local neurons
     * @param pos The positions of the neurons
     * @exception Throws a RelearnException if something unexpected happened
     */
    void init(number_neurons_type number_neurons_init, std::vector<NeuronsExtraInfo::position_type> pos);

    /**
     * @brief Creates creation_count many new neurons with default values
     *      (a) Creates neurons in the electrical model
     *      (b) Creates neurons in the extra infos
     *      (c) Creates neurons in the synaptic models
     *      (d) Enables all created neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     *      (f) Inserts the newly created neurons into the octree
     * @param creation_count The number of newly created neurons
     * @exception Throws a RelearnException if something unexpected happens
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * Returns the algorithm that calculates to which neuron a neuron connects during the plasticity update
     * @return The algorithm
     */
    [[nodiscard]] const std::shared_ptr<Algorithm>& get_algorithm() const {
        return algorithm;
    }

    /**
     * @brief Sets the algorithm that calculates to which neuron a neuron connects during the plasticity update
     * @param algorithm_ptr The pointer to the algorithm
     */
    void set_algorithm(std::shared_ptr<Algorithm> algorithm_ptr) noexcept {
        algorithm = std::move(algorithm_ptr);
    }

    /**
     * @brief Sets the probability kernel that is used for the simulation
     * @param kernel The kernel
     */
    void set_probability_kernel(std::unique_ptr<KernelBase>&& kernel) noexcept {
        probability_kernel = std::move(kernel);
    }

    /**
     * @brief Sets the group translator that translates between the local group id on the current mpi rank and its group name
     * @param new_local_group_translator The local group translator for this mpi rank
     */
    void set_local_group_translator(std::shared_ptr<LocalGroupTranslator> new_local_group_translator) {
        this->local_group_translator = std::move(new_local_group_translator);
    }

    /**
     * @brief Neurons that are static are only allowed to have static connections. Plastic connections cannot be added during the simulation. This method marks the given neurons as static
     * @param static_neurons List of neuron ids that will be marked as static
     * @throws RelearnException When a static neuron is loaded with a plastic connection
     */
    void set_static_neurons(const std::span<const NeuronID> static_neurons) {
        extra_info->set_static_neurons(static_neurons);

        for (const auto neuron_id : static_neurons) {
            const auto id = neuron_id.get_neuron_id();

            const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(id);
            RelearnException::check(distant_out_edges.empty(), "Plastic connection from a static neuron is forbidden. {} (static)  -> ?", neuron_id);

            const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(id);
            RelearnException::check(local_out_edges.empty(), "Plastic connection from a static neuron is forbidden. {} (static)  -> ?", neuron_id);

            const auto& [distant_in_edges, _3] = network_graph->get_distant_in_edges(id);
            RelearnException::check(distant_in_edges.empty(), "Plastic connection from a static neuron is forbidden. ? -> {} (static)", neuron_id);

            const auto& [local_in_edges, _4] = network_graph->get_local_in_edges(id);
            RelearnException::check(local_in_edges.empty(), "Plastic connection from a static neuron is forbidden. ? -> {} (static)", neuron_id);
        }
    }

    /**
     * @brief Returns the number of neurons in this object
     * @return The number of neurons in this object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_neurons;
    }

    /**
     * @brief Returns the group translate that translates between the local group id on the current mpi rank and its group name
     * @return the local group translator
     */
    [[nodiscard]] const std::shared_ptr<LocalGroupTranslator>& get_local_group_translator() const {
        return local_group_translator;
    }

    /**
     * @brief Returns a constant reference to the extra information
     * @return The extra information for the neurons
     */
    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_info() const noexcept {
        return extra_info;
    }

    /**
     * @brief Returns a constant reference to the neuron model
     * @return The neuron model for the neurons
     */
    [[nodiscard]] const std::shared_ptr<NeuronModel>& get_neuron_model() const noexcept {
        return neuron_model;
    }

    [[nodiscard]] const std::shared_ptr<SynapticElements>& get_synaptic_elements() const noexcept {
        return synaptic_elements;
    }

    [[nodiscard]] const std::shared_ptr<NetworkGraph>& get_network_graph() const noexcept {
        return network_graph;
    }

    /**
     * @brief Returns the current calcium value of the neuron
     * @param neuron_id Local neuron id
     * @return Calcium of the neuron
     */
    [[nodiscard]] double get_calcium(const NeuronID neuron_id) const {
        return calcium_calculator->get_calcium()[neuron_id.get_neuron_id()];
    }

    /**
     * @brief Sets the signal types in the extra infos
     * @param signal_types The signal types
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_signal_types
     */
    void set_signal_types(std::vector<SignalType> signal_types) {
        synaptic_elements->set_signal_types(std::move(signal_types));
    }

    /**
     * @brief Manually sets the fired status of the neurons
     * @param fired The fired status of the neurons
     * @exception Throws a RelearnException if fired.size() is not equal to the number of local neurons
     */
    void set_fired(const std::span<const FiredStatus> fired) {
        RelearnException::check(fired.size() == number_neurons, "Neurons::set_fired: The sizes didn't match: {} vs {}", fired.size(), number_neurons);

        for (const auto neuron_id : NeuronID::range(number_neurons)) {
            neuron_model->set_fired(neuron_id, fired[neuron_id.get_neuron_id()]);
        }
    }

    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return extra_info->get_disable_flags();
    }

    /**
     * @brief Initializes the synaptic elements with respect to the network graph, i.e.,
     *      adds the synapses from the network graph as connected counts to the synaptic elements models
     */
    void init_synaptic_elements(const PlasticLocalSynapses& local_synapses_plastic, const PlasticDistantInSynapses& in_synapses_plastic, const PlasticDistantOutSynapses& out_synapses_plastic);

    /**
     * @brief Disables all neurons with specified ids
     *      If a neuron is already disabled, nothing happens for that one
     *      Otherwise, also deletes all synapses from the disabled neurons
     *      Returns a RelearnTypes::comm_map_deletion containing the mpi requests for deleting distant connections on other ranks to the disabled neurons on this rank.
     * @param step The current simulation step
     * @exception Throws RelearnExceptions if something unexpected happens
     * @return Pair of number of local synapse deletion and requests for deletions on other ranks
     */
    std::pair<std::size_t, RelearnTypes::comm_map_deletion<SynapseDeletionRequest>> disable_neurons(step_type step, std::span<const NeuronID> local_neuron_ids, int num_ranks);

    /**
     * @brief Enables all neurons with specified ids
     *      If a neuron is already enabled, nothing happens for that one
     * @exception Throws RelearnExceptions if something unexpected happens
     */
    void enable_neurons(const std::span<const NeuronID> neuron_ids) {
        extra_info->set_enabled_neurons(neuron_ids);
        neuron_model->enable_neurons(neuron_ids);
    }

    /**
     * @brief Calls update_electrical_activity from the electrical model with the stored network graph,
     *      and updates the calcium values afterwards
     * @param step The current update step
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_electrical_activity(step_type step);

    /**
     * @brief Updates the delta of the synaptic elements for (1) axons, (2) excitatory dendrites, (3) inhibitory dendrites
     * @param step The current update step
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_number_synaptic_elements_delta(step_type step);

    /**
     * @brief Updates the plasticity by
     *      (1) Deleting superfluous synapses
     *      (2) Creating new synapses with the stored algorithm
     * @param step The current simulation step
     * @exception Throws a RelearnException if the network graph, the octree, or the algorithm is empty,
     *      or something unexpected happens
     * @return Returns a tuple with (1) the number of deleted synapses, and (2) the number of created synapses
     */
    [[nodiscard]] std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> update_connectivity(step_type step);

    /**
     * @brief Calculates the number vacant axons and dendrites (excitatory, inhibitory) and prints them to LogFiles::EventType::Sums
     *      Performs communication with MPI
     * @param step The current simulation step
     * @param sum_axon_deleted The number of delected axons
     * @param sum_dendrites_deleted The number of deleted synapses (locally)
     * @param sum_synapses_created The number of created synapses (locally)
     */
    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step_type step, std::uint64_t sum_axon_deleted, std::uint64_t sum_dendrites_deleted, std::uint64_t sum_synapses_created);

    /**
     * @brief Prints the overview of the neurons to LogFiles::EventType::NeuronsOverview
     *      Performs communication with MPI
     * @param step The current simulation step
     */
    void print_neurons_overview_to_log_file_on_rank_0(step_type step) const;

    /**
     * @brief Prints the steps when each neuron on the current rank fired since the last write to a file
     * @param current_step The current step
     * @param steps_since_last_print Steps since last print
     */
    void print_fire_steps_to_file(step_type current_step, step_type steps_since_last_print) const;

    /**
     * @brief Inserts the calcium statistics in the essentials
     *      Performs communication with MPI
     * @param essentials The essentials
     */
    void print_calcium_statistics_to_essentials(const std::unique_ptr<Essentials>& essentials);

    /**
     * @brief Inserts the calcium statistics in the essentials
     *      Performs communication with MPI
     * @param essentials The essentials
     */
    void print_synaptic_changes_to_essentials(const std::unique_ptr<Essentials>& essentials);

    /**
     * @brief Prints the network graph to LogFiles::EventType::Network. Stores current step in file name and log
     * @param step The current simulation step
     * @param with_prefix If the file name should contain the current step as prefix
     */
    void print_network_graph_to_log_file(step_type step, bool with_prefix) const;

    /**
     * @brief Prints the neuron positions to LogFiles::EventType::Positions
     */
    void print_positions_to_log_file() const;

    /**
     * @brief Prints the neuron groups to LogFiles::EventType::Groups (each group with associated neurons)
     */
    void print_groups_to_log_file() const;

    /**
     * @brief Writes the group mapping (names and number of neurons) to the file
     */
    void print_group_mapping_to_log_file() const;

    /**
     * @brief Writes the group name to file name mapping to the file
     */
    void print_groups_to_file_name_to_log_file() const;

    /**
     * @brief Prints some overview to LogFiles::EventType::Cout
     */
    void print();

    /**
     * @brief Prints some algorithm overview to LogFiles::EventType::Cout
     */
    void print_info_for_algorithm();

    /**
     * @brief Prints the calcium values for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_calcium_values_to_file(step_type current_step);

    /**
     * @brief Writes the fire rates (since the last reset) to the file
     */
    void print_fire_rate_to_file(step_type current_step);

    /**
     * @brief Performs debug checks on the synaptic element models if Config::do_debug_checks
     * @exception Throws a RelearnException if a check fails
     */
    void debug_check_counts();

    /**
     * @brief Returns a statistical measure for the selected attribute, considering all MPI ranks.
     *      Performs communication across MPI processes
     * @param attribute The selected attribute of the neurons
     * @return The statistical measure across all MPI processes. All MPI processes have the same return value
     */
    [[nodiscard]] StatisticalMeasures get_statistics(NeuronAttribute attribute) const;

    /**
     * @brief Checks if the weights of the out-going connections match their signal type
     * @param network_graph Network graph with all connections of the current mpi rank
     * @param signal_types Vector of SignalTypes. Neuron i has signal_type[i]
     * @throws RelearnException If signal_type does not match weight
     */
    static void check_signal_types(const std::shared_ptr<NetworkGraph>& network_graph, std::span<const SignalType> signal_types, mpiPP::MPIRank my_rank);

    /**
     * Processes the requests of other mpi ranks to delete distant synapses on this rank to disabled remote neurons
     * @param list The communication map
     * @param my_rank Current mpi rank
     * @return Number of deletions
     */
    [[nodiscard]] std::size_t delete_disabled_distant_synapses(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& list, mpiPP::MPIRank my_rank);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);

private:
    [[nodiscard]] StatisticalMeasures global_statistics(std::span<const double> local_values, mpiPP::MPIRank root) const;

    template <typename T>
    [[nodiscard]] StatisticalMeasures global_statistics_integral(const std::span<const T> local_values, const mpiPP::MPIRank root) const {
        auto values = local_values
                      | ranges::views::transform(ranges::convert_to<double>{})
                      | ranges::to_vector;
        return global_statistics(std::move(values), root);
    }

    [[nodiscard]] std::uint64_t create_synapses();

    number_neurons_type number_neurons = 0;

    PlasticLocalSynapses last_created_local_synapses{};
    PlasticDistantInSynapses last_created_in_synapses{};
    PlasticDistantOutSynapses last_created_out_synapses{};

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<NeuronModel> neuron_model{};
    std::unique_ptr<CalciumCalculator> calcium_calculator{};

    std::shared_ptr<NetworkGraph> network_graph{};

    std::shared_ptr<SynapticElements> synaptic_elements{};
    std::unique_ptr<SynapseDeletionFinder> synapse_deletion_finder{};

    std::shared_ptr<LocalGroupTranslator> local_group_translator;

    std::shared_ptr<Algorithm> algorithm{};
    std::unique_ptr<KernelBase> probability_kernel{};

    std::shared_ptr<NeuronsExtraInfo> extra_info{};
};
