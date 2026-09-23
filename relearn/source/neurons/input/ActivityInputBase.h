#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfo.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <span>
#include <vector>

class NetworkGraph;
class NeuronMonitor;

/**
 * @brief This class provides an interface for the input neurons receive during the simulation.
 *      Holds everything that is common to the CPU and GPU activity inputs; ActivityInputCPU and
 *      ActivityInputGPU add the parts that differ (update_input_range's signature and return type
 *      differ by build: CPU updates in place, GPU returns the CUDA events it queued). ActivityInput
 *      itself (ActivityInput.h) resolves to whichever of the two this build compiles.
 *      Templated on the per-neuron array storage (Storage<T>): ActivityInputCPU instantiates it
 *      with std::vector, ActivityInputGPU with LazySyncedArray.
 */
template <template <typename...> class Storage>
class ActivityInputBase {
public:
    using activity_type = RelearnTypes::activity_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    explicit ActivityInputBase(const int _number_ranks)
        : number_ranks(_number_ranks) { }

    // _input is a LazySyncedArray, which is move-only, so copying is already impossible;
    // this makes that explicit instead of an unenforced (and clang-flagged) "= default".
    ActivityInputBase(const ActivityInputBase&) = delete;
    ActivityInputBase& operator=(const ActivityInputBase&) = delete;

    ActivityInputBase(ActivityInputBase&&) = default;
    ActivityInputBase& operator=(ActivityInputBase&&) = default;

    virtual ~ActivityInputBase() = default;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    virtual void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_local_neurons == 0, "ActivityInputBase::init: Method was called twice");
        RelearnException::check(number_neurons != 0, "ActivityInputBase::init: Must initialize with at least 1 neuron");

        number_local_neurons = number_neurons;
        _input.resize(number_local_neurons);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(number_local_neurons != 0, "ActivityInputBase::create_neurons: init(...) was not called before");
        RelearnException::check(creation_count != 0, "ActivityInputBase::create_neurons: creation_count is 0");

        number_local_neurons += creation_count;
        _input.resize(number_local_neurons);
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
    virtual void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) { // NOLINT(performance-unnecessary-value-param) - virtual base of a 4-way override family (Combined/Synaptic/Flexible/ScaleActivityInput); also moved into _extra_infos below
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "ActivityInputBase::set_extra_infos: new_extra_info is empty");
        _extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] virtual activity_type get_input(const NeuronID neuron_id) const {
        const auto index = neuron_id.get_neuron_id();
        RelearnException::check(index < _input.size(), "ActivityInputBase::get_input: NeuronID {} is request, but only {} are stored", neuron_id, _input.size());

        return _input[index];
    }

    /**
     * @brief Returns input for the given neuron. Changes after calls to update_input_range(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The input for the given neuron
     */
    [[nodiscard]] virtual activity_type get_input(const NeuronID::value_type neuron_id) const {
        RelearnException::check(neuron_id < _input.size(), "ActivityInputBase::get_input: NeuronID {} is request, but only {} are stored", neuron_id, _input.size());

        return _input[neuron_id];
    }

    /**
     * @brief Returns the input for all neurons. Changes after calls to update_input_range(...)
     * @return The input for all neurons
     */
    [[nodiscard]] virtual std::span<const activity_type> get_input() const noexcept {
        return _input;
    }

    [[nodiscard]] virtual std::vector<Storage<activity_type>*> get_input_arr() {
        std::vector<Storage<activity_type>*> v{};
        v.emplace_back(&_input);
        return v;
    }

    [[nodiscard]] virtual std::vector<const Storage<activity_type>*> get_input_arr_const() const {
        std::vector<const Storage<activity_type>*> v{};
        v.emplace_back(&_input);
        return v;
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
        RelearnException::check(footprint != nullptr, "ActivityInputBase::record_memory_footprint: footprint is empty");
        footprint->emplace("ActivityInput", sizeof(*this) + (sizeof(activity_type) * _input.size()));
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

    [[nodiscard]] int get_number_ranks() const noexcept {
        return number_ranks;
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
    [[nodiscard]] virtual std::span<activity_type> get_input_internal() noexcept {
        return _input;
    }

    Storage<activity_type> _input{};

private:
    number_neurons_type number_local_neurons{ 0 };
    int number_ranks{};
    std::shared_ptr<NeuronsExtraInfo> _extra_infos;
    std::shared_ptr<NetworkGraph> _network_graph;
};
