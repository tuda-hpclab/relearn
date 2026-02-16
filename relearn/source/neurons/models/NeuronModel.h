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

#include "Types.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace utility {
class MemoryFootprint;
}

class ActivityInput;
class GroupMonitor;
class FiredStatusCommunicator;
class NetworkGraph;
class NeuronMonitor;
class NeuronsExtraInfo;

/**
 * This class provides the basic interface for every neuron model, that is, the rules by which a neuron spikes.
 * The calculations should focus solely on the spiking behavior, and should not account for any plasticity changes.
 * The object itself stores only the local portion of the neuron population.
 * This class performs communication with MPI.
 */
class NeuronModel {
    friend class GroupMonitor;
    friend class NeuronModelsTest;

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new instance of type NeuronModel with 0 neurons.
     * @param h The step size for the numerical integration, >0
     * @param activity_input The object that is responsible for calculating the input the neurons receive, not empty
     * @param fired_status_communicator The object that is responsible for communicating the fired status, not empty
     * @exception Throws a RelearnException if h == 0 or one of the pointers is empty
     */
    NeuronModel(unsigned int h,
                std::shared_ptr<ActivityInput>&& activity_input,
                std::shared_ptr<FiredStatusCommunicator>&& fired_status_communicator);

    virtual ~NeuronModel() = default;

    NeuronModel(const NeuronModel& other) = delete;
    NeuronModel& operator=(const NeuronModel& other) = delete;

    NeuronModel(NeuronModel&& other) = default;
    NeuronModel& operator=(NeuronModel&& other) = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info);

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph);

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial membrane potential and initial synaptic inputs to 0.0 and fired to false
     * @param number_neurons The number of local neurons to store in this class
     */
    virtual void init(number_neurons_type number_neurons);

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    virtual void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    virtual void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    virtual void disable_neurons(std::span<const NeuronID> neuron_ids);

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    virtual void enable_neurons(std::span<const NeuronID> neuron_ids);

    /**
     * @brief Creates an object of type T wrapped inside an std::unique_ptr
     * @tparam T The type of NeuronModel that shall be constructed, must inherit from NeuronModel
     * @tparam Ts The types of parameters for the constructor of T
     * @param args The arguments that shall be passed to the constructor of T
     * @return A new instance of type T wrapped inside an std::unique_ptr
     */
    template <typename T, typename... Ts>
        requires(std::is_base_of_v<NeuronModel, T>)
    [[nodiscard]] static std::unique_ptr<T> create(Ts... args) {
        return std::make_unique<T>(args...);
    }

    /**
     * @brief Sets if a neuron fired for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value True iff the neuron fired in the current simulation step
     */
    void set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
        fired_status_recorder->set_fired(neuron_id, new_value);
    }

    /**
     * @brief Sets if a neuron fired for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value True iff the neuron fired in the current simulation step
     */
    void set_fired(const NeuronID::value_type neuron_id, const FiredStatus new_value) {
        fired_status_recorder->set_fired(neuron_id, new_value);
    }

    /**
     * @brief Returns a bool that indicates if the neuron with the passed local id spiked in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return True iff the neuron spiked
     */
    [[nodiscard]] bool has_fired(NeuronID neuron_id) const;

    /**
     * @brief Returns a vector of flags that indicate if the neuron with the local id spiked in the current simulation step
     * @return A constant reference to the vector of flags. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const FiredStatus> get_fired() const noexcept;

    /**
     * @brief Returns a double that indicates the neuron's membrane potential in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The neuron's membrane potential
     */
    [[nodiscard]] double get_x(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return x[local_neuron_id];
    }

    /**
     * @brief Returns a span of doubles that indicate the neurons' respective membrane potential in the current simulation step
     * @return A span of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const double> get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns a span of doubles that indicate the neurons' respective input in the current simulation step
     * @return A span of doubles. It is not invalidated by calls to other methods but for init(...) and create_neurons(...)
     */
    [[nodiscard]] std::span<const double> get_input() const noexcept;

    /**
     * @brief Returns the numerical integration's step size
     * @return The step size
     */
    [[nodiscard]] unsigned int get_h() const noexcept {
        return precision_h;
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Performs one step of simulating the electrical activity for all neurons.
     *      This method performs communication via MPI.
     * @param step The current update step
     */
    void update_electrical_activity(step_type step);

    /**
     * @brief This function is used for benchmarking purposes
     * @param step The current update step
     */
    void update_electrical_activity_benchmark(step_type step);

    /**
     * @brief Notifies this class and the input calculators that the plasticity has changed.
     *      Some might cache values, which than can be recalculated
     * @param step The current simulation step
     */
    void notify_of_plasticity_change(step_type step);

    /**
     * @brief Returns the recorder for the fired status
     * @return The recorder
     */
    [[nodiscard]] const std::shared_ptr<FiredStatusRecorder>& get_fired_status_recorder() const noexcept {
        return fired_status_recorder;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);

    static constexpr unsigned int default_h{ 10 };
    static constexpr unsigned int min_h{ 1 };
    static constexpr unsigned int max_h{ 1000 };

protected:
    virtual void update_activity() = 0;

    virtual void update_activity_benchmark() {
        update_activity();
    }

    /**
     * @brief Provides a hook to initialize all neurons with local id in [start_id, end_id)
     *      This method exists because of the order of operations when creating neurons
     * @param start_id The first local neuron id to initialize
     * @param end_id The next to last local neuron id to initialize
     */
    virtual void init_neurons(number_neurons_type start_id, number_neurons_type end_id) = 0;

    /**
     * @brief Sets the membrane potential for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value The new membrane potential
     */
    void set_x(const NeuronID::value_type neuron_id, const double new_value) {
        x[neuron_id] = new_value;
    }

    /**
     * @brief Sets the membrane potential for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value The new membrane potential
     */
    void set_x(const NeuronID neuron_id, const double new_value) {
        x[neuron_id.get_neuron_id()] = new_value;
    }

    [[nodiscard]] double get_x(const NeuronID::value_type neuron_id) const {
        RelearnException::check(neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return x[neuron_id];
    }

    [[nodiscard]] std::span<double> get_x_internal() {
        return x;
    }

    [[nodiscard]] double get_input(const NeuronID neuron_id) const {
        return act_input->get_input(neuron_id);
    }

    [[nodiscard]] double get_input(const NeuronID::value_type neuron_id) const {
        return act_input->get_input(neuron_id);
    }

    [[nodiscard]] const std::shared_ptr<ActivityInput>& get_activity_input() const noexcept {
        return act_input;
    }

    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_infos() const noexcept {
        return extra_infos;
    }

    [[nodiscard]] const std::shared_ptr<FiredStatusCommunicator>& get_fired_status_communicator() const noexcept {
        return fired_status_comm;
    }

private:
    // My local number of neurons
    number_neurons_type number_local_neurons{ 0 };

    // Model parameters for all neurons
    unsigned int precision_h{ default_h }; // Precision for Euler integration

    // Variables for each neuron where the array index denotes the local neuron ID
    std::vector<double, RelearnAllocator<double>> x{}; // The membrane potential (in equations usually v(t))

    std::shared_ptr<ActivityInput> act_input{};

    std::shared_ptr<FiredStatusRecorder> fired_status_recorder{};
    std::shared_ptr<FiredStatusCommunicator> fired_status_comm{};

    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    std::shared_ptr<NetworkGraph> network_graph{};
};
