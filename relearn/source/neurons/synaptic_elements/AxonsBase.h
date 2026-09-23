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
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/synaptic_elements/ElementBase.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

class GrowthrateCalculator;

class SynapticElementsAdapter;

/**
 * @brief Holds everything that is common to the CPU and GPU axons; AxonsCPU and AxonsGPU add the
 *      parts that differ (get_cuda_handle(_const)/get_d_signal_types only exist on GPU, and
 *      commit_updates takes an extra stream parameter on GPU).
 *      Templated on the per-neuron array storage (Storage<T>): AxonsCPU instantiates it with
 *      std::vector, AxonsGPU with LazySyncedArray -- this is the only thing that differs about
 *      signal_types between the two, and keeping it a template parameter means this header has no
 *      dependency on anything under cuda/.
 */
template <template <typename...> class Storage>
class AxonsBase {
public:
    friend class SynapticElementsAdapter;

    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;
    using calcium_type = RelearnTypes::calcium_type;
    using counter_type = RelearnTypes::counter_type;
    using grown_type = RelearnTypes::grown_type;

    AxonsBase() = default;

    AxonsBase(const AxonsBase& other) = delete;
    AxonsBase(AxonsBase&& other) = default;

    AxonsBase& operator=(const AxonsBase& other) = delete;
    AxonsBase& operator=(AxonsBase&& other) = default;

    virtual ~AxonsBase() = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its elements
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "Axons::set_extra_infos: new_extra_info is empty");

        extra_infos = std::move(new_extra_info);
        axons_base.set_extra_infos(extra_infos);
    }

    /**
     * @brief Sets the function that calculates the number of grown elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_grown_elements_calculator(std::function<grown_type(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "Axons::set_grown_elements_calculator: calculator is empty");

        axons_base.set_grown_elements_calculator(std::move(calculator));
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_delta_since_last_update_calculator(std::function<grown_type(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "Axons::set_delta_since_last_update_calculator: calculator is empty");

        axons_base.set_delta_since_last_update_calculator(std::move(calculator));
    }

    /**
     * @brief Sets the function that calculates the number of connected elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_connected_elements_calculator(std::function<counter_type(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "Axons::set_connected_elements_calculator: calculator is empty");

        axons_base.set_connected_elements_calculator(std::move(calculator));
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must return values from [0.0, 1.0]
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_vacant_retract_ratio_calculator(std::function<grown_type(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "Axons::set_vacant_retract_ratio_calculator: calculator is empty");

        axons_base.set_vacant_retract_ratio_calculator(std::move(calculator));
    }

    /**
     * @brief Sets the function that calculates the minimum calcium that is required for the elements to grow.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_minimum_calcium_calculator(std::function<calcium_type(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "Axons::set_minimum_calcium_calculator: calculator is empty");

        axons_base.set_minimum_calcium_calculator(std::move(calculator));
    }

    /**
     * @brief Initializes the object to contain number_neurons elements.
     * @param number_neurons The number of that should be stored, >0
     * @exception Throws a RelearnException if number_neurons == 0 or if init() has been called before
     */
    virtual void init(number_neurons_type number_neurons);

    /**
     * @brief Creates additional creation_count elements.
     * @param creation_count The number of that should be created, >0
     * @exception Throws a RelearnException if number_neurons == 0 or if init() was not called before
     */
    virtual void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Disables the specified neurons. Sets the grown elements, deltas, vacant and connected elements to 0.
     * @param disabled_neuron_ids The neurons to disable
     * @exception Throws a RelearnException if a neuron_id is too large
     */
    virtual void disable_neurons(std::span<const number_neurons_type> disabled_neuron_ids);

    /**
     * @brief Sets the signal types for all neurons at once
     * @param types The new signal types
     */
    void set_signal_types(std::vector<SignalType> types) {
        RelearnException::check(types.size() == size, "Axons::set_signal_type: Mismatching size of type vectors");
        signal_types = Storage<SignalType>(std::move(types));

        excitatory_axon_ids.clear();
        inhibitory_axon_ids.clear();

        excitatory_axon_ids.reserve(size);
        inhibitory_axon_ids.reserve(size);

        for (auto i = number_neurons_type{ 0 }; i < size; i++) {
            if (signal_types[i] == SignalType::Excitatory) { // NOLINT(bugprone-branch-clone) - excitatory vs inhibitory container, not identical branches
                excitatory_axon_ids.emplace_back(i);
            } else {
                inhibitory_axon_ids.emplace_back(i);
            }
        }
    }

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Adds a delta for the specified neuron. This delta has to be applied via commit_updates(...).
     *      Affects the axons specified.
     * @param delta The delta to add
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_to_delta(const grown_type delta, const number_neurons_type neuron_id) {
        axons_base.add_to_delta(delta, neuron_id);
    }

    /**
     * @brief Adds deltas for all neurons. These deltas has to be applied via commit_updates(...).
     *      Affects the axons specified.
     * @param deltas The deltas to add
     * @exception Throws a RelearnException if deltas.size() != size
     */
    void add_to_delta(const std::span<const grown_type> deltas) {
        axons_base.add_to_delta(deltas);
    }

    /**
     * @brief Adds the specified amount to the connected elements of the specified neuron.
     *      Also reduces the number of vacant elements by the same amount (or sets it to zero).
     *      Affects the axons specified.
     * @param newly_added The amount of newly connected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_connected_elements(const counter_type newly_added, const number_neurons_type neuron_id) {
        axons_base.add_connected_elements(newly_added, neuron_id);
    }

    /**
     * @brief Adds the specified amount to the connected elements of each neuron.
     *      Also reduces the number of vacant elements by the same amount (or sets it to zero).
     *      Affects the axons specified.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes does not have the correct size
     */
    void add_connected_elements(const std::span<const counter_type> changes) {
        axons_base.add_connected_elements(changes);
    }

    /**
     * @brief Removes the specified amount of connected elements from the specified neuron.
     *      Also increases the number of vacant elements by the same amount.
     *      Affects the axons specified.
     * @param newly_free The amount of newly free elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if the number of connected elements is smaller than newly_free
     */
    void remove_connected_elements(const counter_type newly_free, const number_neurons_type neuron_id) {
        axons_base.remove_connected_elements(newly_free, neuron_id);
    }

    /**
     * @brief Removes the amount of connected elements from the neurons.
     *      Also increases the number of vacant elements by the same amount.
     *      Affects the axons specified.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes does not have the correct size or if the number of connected elements is smaller than the specified amount for one neuron
     */
    void remove_connected_elements(const std::span<const counter_type> changes) {
        axons_base.remove_connected_elements(changes);
    }

    /**
     * @brief Connects the specified number of elements to the specified neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param newly_connected The number of newly connected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough vacant elements
     */
    void connect_elements(const counter_type newly_connected, const number_neurons_type neuron_id) {
        axons_base.connect_elements(newly_connected, neuron_id);
    }

    /**
     * @brief Connects the specified number of elements to each neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough vacant elements
     */
    void connect_elements(const std::span<const counter_type> changes) {
        axons_base.connect_elements(changes);
    }

    /**
     * @brief Disconnects the specified number of elements to the specified neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param newly_disconnected The number of newly disconnected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough connected elements
     */
    void disconnect_elements(const counter_type newly_disconnected, const number_neurons_type neuron_id) {
        axons_base.disconnect_elements(newly_disconnected, neuron_id);
    }

    /**
     * @brief Disconnects the specified number of elements to each neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough connected elements
     */
    void disconnect_elements(const std::span<const counter_type> changes) {
        axons_base.disconnect_elements(changes);
    }

    /**
     * @brief Returns the size
     * @return The size
     */
    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

    /**
     * @brief Returns the neuron ids of the excitatory neurons in increasing order
     * @return The ids
     */
    [[nodiscard]] std::span<const number_neurons_type> get_excitatory_axon_ids() const noexcept {
        return excitatory_axon_ids;
    }

    /**
     * @brief Returns the neuron ids of the inhibitory neurons in increasing order
     * @return The ids
     */
    [[nodiscard]] std::span<const number_neurons_type> get_inhibitory_axon_ids() const noexcept {
        return inhibitory_axon_ids;
    }

    /**
     * @brief Returns the total number of additions over the lifetime of this object
     * @return The total number of additions for the specified type
     */
    [[nodiscard]] grown_type get_total_additions() const noexcept {
        return axons_base.get_total_additions();
    }

    /**
     * @brief Returns the total number of deletions over the lifetime of this object
     * @return The total number of deletions for the specified type
     */
    [[nodiscard]] grown_type get_total_deletions() const noexcept {
        return axons_base.get_total_deletions();
    }

    /**
     * @brief Returns the signal types of the elements, indexed by the local neuron id
     * @return The signal types
     */
    [[nodiscard]] std::span<const SignalType> get_signal_types() const noexcept {
        return signal_types;
    }

    /**
     * @brief Returns the number of grown elements, indexed by the local neuron id
     * @return The number of grown elements
     */
    [[nodiscard]] std::span<const grown_type> get_grown_elements() const noexcept {
        return axons_base.get_grown_elements();
    }

    /**
     * @brief Returns the accumulated changes to the grown elements, indexed by the local neuron id (the built-up difference from the electrical updates)
     * @return The accumulated changes
     */
    [[nodiscard]] std::span<const grown_type> get_deltas() const noexcept {
        return axons_base.get_deltas();
    }

    /**
     * @brief Returns the number of vacant elements, indexed by the local neuron id (how many elements are grown and not connected)
     * @return The vacant elements
     */
    [[nodiscard]] std::span<const counter_type> get_vacant_elements() const noexcept {
        return axons_base.get_vacant_elements();
    }

    /**
     * @brief Returns the number of connected elements, indexed by the local neuron id (how many elements from the neuron are connected via synapses)
     * @return The connected elements
     */
    [[nodiscard]] std::span<const counter_type> get_connected_elements() const noexcept {
        return axons_base.get_connected_elements();
    }

    /**
     * @brief Returns the vacant retract ratio, indexed by the local neuron id
     * @return The vacant retract ratio
     */
    [[nodiscard]] std::span<const grown_type> get_vacant_retract_ratio() const noexcept {
        return axons_base.get_vacant_retract_ratio();
    }

    /**
     * @brief Returns the minimum calcium, indexed by the local neuron id
     * @return The vacant minimum calcium
     */
    [[nodiscard]] std::span<const calcium_type> get_minimum_calcium() const noexcept {
        return axons_base.get_minimum_calcium();
    }

    /**
     * @brief Returns the number of axonal boutons for the specified neuron.
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The number of boutons
     */
    [[nodiscard]] virtual counter_type get_number_boutons([[maybe_unused]] const number_neurons_type neuron_id) const {
        RelearnException::check(neuron_id < size, "Axons::get_number_boutons: neuron_id is too large: {}", neuron_id);
        return counter_type{ 1 };
    }

    /**
     * @brief Returns for a specified neuron the position of the specified axonal bouton.
     * @param neuron_id The neuron's id
     * @param bouton_id The bouton's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The position of the axonal bouton
     */
    [[nodiscard]] virtual RelearnTypes::position_type get_bouton_position(number_neurons_type neuron_id, std::size_t bouton_id) const;

    /**
     * @brief Returns for a specified neuron the position of an axonal bouton.
     *      Can use a random generator to determine which bouton's position is returned.
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The position of the axonal bouton
     */
    [[nodiscard]] virtual RelearnTypes::position_type get_bouton_position(number_neurons_type neuron_id) const;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        axons_base.record_memory_footprint(footprint, "Axon Base");

        constexpr auto my_size = sizeof(*this) - sizeof(ElementBase);

        const auto signal_type_size = sizeof(SignalType) * signal_types.capacity();
        const auto excitatory_axon_ids_size = sizeof(number_neurons_type) * excitatory_axon_ids.capacity();
        const auto inhibitory_axon_ids_size = sizeof(number_neurons_type) * inhibitory_axon_ids.capacity();

        footprint->emplace("Axons", my_size + signal_type_size + excitatory_axon_ids_size + inhibitory_axon_ids_size);
    }

protected:
    number_neurons_type size{};
    std::shared_ptr<NeuronsExtraInfo> extra_infos;

    Storage<SignalType> signal_types{};

    std::vector<number_neurons_type> excitatory_axon_ids{};
    std::vector<number_neurons_type> inhibitory_axon_ids{};

    ElementBase axons_base;
};

template <template <typename...> class Storage>
void AxonsBase<Storage>::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "Axons::init: init() was called previously");
    RelearnException::check(number_neurons > 0, "Axons::init: number_neurons must be > 0");

    size = number_neurons;

    signal_types.resize(size);

    axons_base.init(number_neurons);
}

template <template <typename...> class Storage>
void AxonsBase<Storage>::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "Axons::create_neurons: init() must be called before create_neurons()");
    RelearnException::check(creation_count > 0, "Axons::create_neurons: creation_count must be > 0");

    const auto old_size = size;
    const auto new_size = old_size + creation_count;
    size = new_size;

    signal_types.resize(new_size);

    axons_base.create_neurons(creation_count);
}

template <template <typename...> class Storage>
void AxonsBase<Storage>::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    axons_base.disable_neurons(disabled_neuron_ids);
}

template <template <typename...> class Storage>
void AxonsBase<Storage>::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Axons (exc.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Excitatory) {
            return utility::cast<float>(axons_base.get_grown_elements()[neuron_id]);
        }
        return 0.0F; }, []() { }, []() { });

    monitor.register_paramter("Axons (inh.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Inhibitory) {
            return utility::cast<float>(axons_base.get_grown_elements()[neuron_id]);
        }
        return 0.0F; }, []() { }, []() { });

    monitor.register_paramter("Axons (exc.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Excitatory) {
            return static_cast<float>(axons_base.get_connected_elements()[neuron_id]);
        }
        return 0.0F; }, []() { }, []() { });

    monitor.register_paramter("Axons (inh.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Inhibitory) {
            return static_cast<float>(axons_base.get_connected_elements()[neuron_id]);
        }
        return 0.0F; }, []() { }, []() { });
}

template <template <typename...> class Storage>
RelearnTypes::position_type AxonsBase<Storage>::get_bouton_position(const number_neurons_type neuron_id, const std::size_t bouton_id) const {
    RelearnException::check(neuron_id < size, "Axons::get_bouton_position: neuron_id is too large");
    RelearnException::check(bouton_id == 0, "Axons::get_bouton_position: bouton_id must be 0");
    return extra_infos->get_position(NeuronID{ neuron_id });
}

template <template <typename...> class Storage>
RelearnTypes::position_type AxonsBase<Storage>::get_bouton_position(const number_neurons_type neuron_id) const {
    RelearnException::check(neuron_id < size, "Axons::get_bouton_position: neuron_id is too large");
    return extra_infos->get_position(NeuronID{ neuron_id });
}
