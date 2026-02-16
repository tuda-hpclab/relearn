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

#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>

class NetworkGraph;
class NeuronMonitor;
class NeuronsExtraInfo;

/**
 * @brief This class provides an interface for the input neurons receive during the simulation.
 */
class ActivityInput {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    ActivityInput() = default;

    ActivityInput(const ActivityInput&) = default;
    ActivityInput& operator=(const ActivityInput&) = default;

    ActivityInput(ActivityInput&&) = default;
    ActivityInput& operator=(ActivityInput&&) = default;

    virtual ~ActivityInput() = default;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    virtual void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_local_neurons == 0, "ActivityInput::init: Method was called twice");
        RelearnException::check(number_neurons != 0, "ActivityInput::init: Must initialize with at least 1 neuron");

        number_local_neurons = number_neurons;
        _input.resize(number_local_neurons, 0.0);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(number_local_neurons != 0, "ActivityInput::create_neurons: init(...) was not called before");
        RelearnException::check(creation_count != 0, "ActivityInput::create_neurons: creation_count is 0");

        number_local_neurons += creation_count;
        _input.resize(number_local_neurons, 0.0);
    }

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    virtual void register_neuron_monitor(NeuronMonitor& monitor) = 0;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its input
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    virtual void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "ActivityInput::set_extra_infos: new_extra_info is empty");
        _extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @exception Might throw a RelearnException
     */
    void update_input(step_type step) {
        update_input_range(step, NeuronID{ 0 }, NeuronID{ get_number_neurons() });
    }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Might throw a RelearnException
     */
    virtual void update_input_range(step_type step, NeuronID first, NeuronID last) = 0;

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] virtual double get_input(const NeuronID neuron_id) const {
        const auto index = neuron_id.get_neuron_id();
        RelearnException::check(index < _input.size(), "ActivityInput::get_input: NeuronID {} is request, but only {} are stored", neuron_id, _input.size());

        return _input[index];
    }

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] virtual double get_input(const NeuronID::value_type neuron_id) const {
        RelearnException::check(neuron_id < _input.size(), "ActivityInput::get_input: NeuronID {} is request, but only {} are stored", neuron_id, _input.size());

        return _input[neuron_id];
    }

    /**
     * @brief Returns the input for all neurons. Changes after calls to update_input_range(...)
     * @return The input for all neurons
     */
    [[nodiscard]] virtual std::span<const double> get_input() const noexcept {
        return _input;
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        RelearnException::check(footprint != nullptr, "ActivityInput::record_memory_footprint: footprint is empty");
        footprint->emplace("ActivityInput", sizeof(*this) + (sizeof(double) * _input.size()));
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    virtual void set_network_graph(const std::shared_ptr<NetworkGraph>& new_network_graph) {
        const auto is_filled = new_network_graph != nullptr;
        RelearnException::check(is_filled, "SynapticActivityInput::set_network_graph: new_network_graph is empty");

        _network_graph = new_network_graph;
    }

protected:
    /**
     * @brief Returns the held NeuronsExtraInfo
     * @return The held NeuronsExtraInfo, can be empty
     */
    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_infos() const noexcept {
        return _extra_infos;
    }

    [[nodiscard]] const std::shared_ptr<NetworkGraph>& get_network_graph() const noexcept {
        return _network_graph;
    }

    /**
     * @brief Returns a span to the modifiable input for internal update
     * @return The input
     */
    [[nodiscard]] virtual std::span<double> get_input_internal() noexcept {
        return _input;
    }

private:
    number_neurons_type number_local_neurons{ 0 };
    std::vector<double, RelearnAllocator<double>> _input{};

    std::shared_ptr<NeuronsExtraInfo> _extra_infos{};
    std::shared_ptr<NetworkGraph> _network_graph{};
};
