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
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "neurons/synaptic_elements/GrowthCurves.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

class GrowthrateCalculator;
class NeuronMonitor;

class SynapticElementsAdapter;

/**
 * @brief Holds everything that is common to the CPU and GPU synaptic elements; SynapticElementsCPU
 *      and SynapticElementsGPU add the parts that differ: update_number_elements takes different
 *      argument types (a step and spans of host calcium data on CPU, device calcium pointers on
 *      GPU), commit_updates dispatches to per-element-type CUDA streams on GPU, and the GPU
 *      constructor additionally creates those streams.
 */
class SynapticElementsBase {
public:
    friend class SynapticElementsAdapter;

    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;
    using calcium_type = RelearnTypes::calcium_type;
    using counter_type = RelearnTypes::counter_type;
    using grown_type = RelearnTypes::grown_type;

    /**
     * @brief Constructs the new synaptic elements with the given axons and dendrites
     * @param _axons The axons, must not be empty
     * @param _dendrites The dendrites, must not be empty
     * @exception Throws a RelearnException if axons or dendrites is empty
     */
    SynapticElementsBase(std::shared_ptr<Axons> _axons, std::shared_ptr<Dendrites> _dendrites)
        : axons{ std::move(_axons) }
        , dendrites{ std::move(_dendrites) } {
        RelearnException::check(axons != nullptr, "SynapticElements::SynapticElements: axons is empty");
        RelearnException::check(dendrites != nullptr, "SynapticElements::SynapticElements: dendrites is empty");
    }

    SynapticElementsBase(const SynapticElementsBase& other) = delete;
    SynapticElementsBase(SynapticElementsBase&& other) = default;

    SynapticElementsBase& operator=(const SynapticElementsBase& other) = delete;
    SynapticElementsBase& operator=(SynapticElementsBase&& other) = default;

    virtual ~SynapticElementsBase() = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its elements
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "SynapticElements::set_extra_infos: new_extra_info is empty");

        axons->set_extra_infos(new_extra_info);
        dendrites->set_extra_infos(new_extra_info);
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Sets the calculator for the growth rate.
     * @param new_growthrate_calculator The new growth rate calculator, must not be empty
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if new_growthrate_calculator is empty
     */
    void set_growthrate_calculator(std::shared_ptr<GrowthrateCalculator> new_growthrate_calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto is_filled = new_growthrate_calculator != nullptr;
        RelearnException::check(is_filled, "SynapticElements::set_growthrate_calculator: new_growthrate_calculator is empty");

        if (element_type == ElementType::Axon) {
            growthrate_calculator_axon = std::move(new_growthrate_calculator);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            if (signal_type == SignalType::Excitatory) { // NOLINT(bugprone-branch-clone) - dispatches to a different member field, not identical branches
                growthrate_calculator_excitatory_dendrites = std::move(new_growthrate_calculator);
            } else {
                growthrate_calculator_inhibitory_dendrites = std::move(new_growthrate_calculator);
            }
        }
    }

    /**
     * @brief Sets the function that calculates the number of grown elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_grown_elements_calculator(std::function<grown_type(number_neurons_type)> calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "SynapticElements::set_grown_elements_calculator: calculator is empty");

        if (element_type == ElementType::Axon) {
            axons->set_grown_elements_calculator(std::move(calculator));
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->set_grown_elements_calculator(std::move(calculator), signal_type);
        }
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_delta_since_last_update_calculator(std::function<grown_type(number_neurons_type)> calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "SynapticElements::set_delta_since_last_update_calculator: calculator is empty");

        if (element_type == ElementType::Axon) {
            axons->set_delta_since_last_update_calculator(std::move(calculator));
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->set_delta_since_last_update_calculator(std::move(calculator), signal_type);
        }
    }

    /**
     * @brief Sets the function that calculates the number of connected elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_connected_elements_calculator(std::function<counter_type(number_neurons_type)> calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "SynapticElements::set_connected_elements_calculator: calculator is empty");

        if (element_type == ElementType::Axon) {
            axons->set_connected_elements_calculator(std::move(calculator));
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->set_connected_elements_calculator(std::move(calculator), signal_type);
        }
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must return values from [0.0, 1.0]
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_vacant_retract_ratio_calculator(std::function<grown_type(number_neurons_type)> calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "SynapticElements::set_vacant_retract_ratio_calculator: calculator is empty");

        if (element_type == ElementType::Axon) {
            axons->set_vacant_retract_ratio_calculator(std::move(calculator));
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->set_vacant_retract_ratio_calculator(std::move(calculator), signal_type);
        }
    }

    /**
     * @brief Sets the function that calculates the minimum calcium that is required for the elements to grow.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @param synaptic_element_type The type of the elements for which to set the calculator
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_minimum_calcium_calculator(std::function<calcium_type(number_neurons_type)> calculator, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "SynapticElements::set_minimum_calcium_calculator: calculator is empty");

        if (element_type == ElementType::Axon) {
            axons->set_minimum_calcium_calculator(std::move(calculator));
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->set_minimum_calcium_calculator(std::move(calculator), signal_type);
        }
    }

    /**
     * @brief Initializes the object to contain number_neurons elements.
     * @param number_neurons The number of that should be stored, >0
     * @exception Throws a RelearnException if number_neurons == 0 or if init() has been called before
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Creates additional creation_count elements.
     * @param creation_count The number of that should be created, >0
     * @exception Throws a RelearnException if number_neurons == 0 or if init() was not called before
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Disables the specified neurons. Sets the grown elements, deltas, vacant and connected elements to 0 for both the axons and dendrites->
     * @param disabled_neuron_ids The neurons to disable
     * @exception Throws a RelearnException if a neuron_id is too large
     */
    void disable_neurons(std::span<const number_neurons_type> disabled_neuron_ids);

    /**
     * @brief Sets the signal types for all neurons at once
     * @param types The new signal types
     */
    void set_signal_types(std::vector<SignalType> types) {
        axons->set_signal_types(std::move(types));
    }

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Adds a delta for the specified neuron. This delta has to be applied via commit_updates(...).
     *      Affects the dendrites specified.
     * @param delta The delta to add
     * @param neuron_id The neuron's id
     * @param synaptic_element_type The type of the elements for which to add
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_to_delta(const grown_type delta, const number_neurons_type neuron_id, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->add_to_delta(delta, neuron_id);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->add_to_delta(delta, neuron_id, signal_type);
        }
    }

    /**
     * @brief Adds deltas for all neurons. These deltas has to be applied via commit_updates(...).
     *      Affects the dendrites specified.
     * @param deltas The deltas to add
     * @param synaptic_element_type The type of the elements for which to add
     * @exception Throws a RelearnException if deltas.size() != size
     */
    void add_to_delta(const std::span<const grown_type> deltas, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->add_to_delta(deltas);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->add_to_delta(deltas, signal_type);
        }
    }

    /**
     * @brief Adds the specified amount to the connected elements of the specified neuron.
     *      Also reduces the number of vacant elements by the same amount (or sets it to zero).
     *      Affects the elements specified.
     * @param newly_added The amount of newly connected elements
     * @param neuron_id The neuron's id
     * @param synaptic_element_type The type of the elements for which to add
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_connected_elements(const counter_type newly_added, const number_neurons_type neuron_id, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->add_connected_elements(newly_added, neuron_id);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->add_connected_elements(newly_added, neuron_id, signal_type);
        }
    }

    /**
     * @brief Adds the specified amount to the connected elements of each neuron.
     *      Also reduces the number of vacant elements by the same amount (or sets it to zero).
     *      Affects the elements specified.
     * @param changes The changes for each neuron
     * @param synaptic_element_type The type of the elements for which to add
     * @exception Throws a RelearnException if changes does not have the correct size
     */
    void add_connected_elements(const std::span<const counter_type> changes, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->add_connected_elements(changes);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->add_connected_elements(changes, signal_type);
        }
    }

    /**
     * @brief Removes the specified amount of connected elements from the specified neuron.
     *      Also increases the number of vacant elements by the same amount.
     *      Affects the elements specified.
     * @param newly_free The amount of newly free elements
     * @param neuron_id The neuron's id
     * @param synaptic_element_type The type of the elements for which to remove
     * @exception Throws a RelearnException if neuron_id is too large or if the number of connected elements is smaller than newly_free
     */
    void remove_connected_elements(const counter_type newly_free, const number_neurons_type neuron_id, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->remove_connected_elements(newly_free, neuron_id);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->remove_connected_elements(newly_free, neuron_id, signal_type);
        }
    }

    /**
     * @brief Removes the amount of connected elements from the neurons.
     *      Also increases the number of vacant elements by the same amount.
     *      Affects the elements specified.
     * @param changes The changes for each neuron
     * @param synaptic_element_type The type of the elements for which to remove
     * @exception Throws a RelearnException if changes does not have the correct size or if the number of connected elements is smaller than the specified amount for one neuron
     */
    void remove_connected_elements(const std::span<const counter_type> changes, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->remove_connected_elements(changes);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->remove_connected_elements(changes, signal_type);
        }
    }

    /**
     * @brief Connects the specified number of elements to the specified neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param newly_connected The number of newly connected elements
     * @param neuron_id The neuron's id
     * @param synaptic_element_type The type of the elements for which to connect
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough vacant elements
     */
    void connect_elements(const counter_type newly_connected, const number_neurons_type neuron_id, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->connect_elements(newly_connected, neuron_id);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->connect_elements(newly_connected, neuron_id, signal_type);
        }
    }

    /**
     * @brief Connects the specified number of elements to each neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @param synaptic_element_type The type of the elements for which to connect
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough vacant elements
     */
    void connect_elements(const std::span<const counter_type> changes, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->connect_elements(changes);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->connect_elements(changes, signal_type);
        }
    }

    /**
     * @brief Disconnects the specified number of elements to the specified neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param newly_disconnected The number of newly disconnected elements
     * @param neuron_id The neuron's id
     * @param synaptic_element_type The type of the elements for which to disconnect
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough connected elements
     */
    void disconnect_elements(const counter_type newly_disconnected, const number_neurons_type neuron_id, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->disconnect_elements(newly_disconnected, neuron_id);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->disconnect_elements(newly_disconnected, neuron_id, signal_type);
        }
    }

    /**
     * @brief Disconnects the specified number of elements to each neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @param synaptic_element_type The type of the elements for which to disconnect
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough connected elements
     */
    void disconnect_elements(const std::span<const counter_type> changes, const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            axons->disconnect_elements(changes);
        } else {
            const auto signal_type = get_signal_type(synaptic_element_type);
            dendrites->disconnect_elements(changes, signal_type);
        }
    }

    /**
     * @brief Returns the size
     * @return The size
     */
    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

    /**
     * @brief Returns the axons
     * @return A view on the axons
     */
    [[nodiscard]] const std::shared_ptr<Axons>& get_axons() const noexcept {
        return axons;
    }

    /**
     * @brief Returns the dendrites
     * @return A view on the dendrites
     */
    [[nodiscard]] const std::shared_ptr<Dendrites>& get_dendrites() const noexcept {
        return dendrites;
    }

    /**
     * @brief Returns the total number of additions over the lifetime of this object
     * @param synaptic_element_type The type of the elements for which to retrieve the value
     * @return The total number of additions for the specified type
     */
    [[nodiscard]] grown_type get_total_additions(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_total_additions();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_total_additions(signal_type);
    }

    /**
     * @brief Returns the total number of deletions over the lifetime of this object
     * @param synaptic_element_type The type of the elements for which to retrieve the value
     * @return The total number of deletions for the specified type
     */
    [[nodiscard]] grown_type get_total_deletions(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_total_deletions();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_total_deletions(signal_type);
    }

    /**
     * @brief Returns the signal types of the elements, indexed by the local neuron id
     * @return The signal types
     */
    [[nodiscard]] std::span<const SignalType> get_signal_types() const noexcept {
        return axons->get_signal_types();
    }

    /**
     * @brief Returns the number of grown elements, indexed by the local neuron id
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The number of grown elements
     */
    [[nodiscard]] std::span<const grown_type> get_grown_elements(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_grown_elements();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_grown_elements(signal_type);
    }

    /**
     * @brief Returns the accumulated changes to the grown elements, indexed by the local neuron id (the built-up difference from the electrical updates)
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The accumulated changes
     */
    [[nodiscard]] std::span<const grown_type> get_deltas(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_deltas();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_deltas(signal_type);
    }

    /**
     * @brief Returns the number of vacant elements, indexed by the local neuron id (how many elements are grown and not connected)
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The vacant elements
     */
    [[nodiscard]] std::span<const counter_type> get_vacant_elements(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_vacant_elements();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_vacant_elements(signal_type);
    }

    /**
     * @brief Returns the number of connected elements, indexed by the local neuron id (how many elements from the neuron are connected via synapses)
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The connected elements
     */
    [[nodiscard]] std::span<const counter_type> get_connected_elements(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_connected_elements();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_connected_elements(signal_type);
    }

    /**
     * @brief Returns the vacant retract ratio, indexed by the local neuron id
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The vacant retract ratio
     */
    [[nodiscard]] std::span<const grown_type> get_vacant_retract_ratio(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_vacant_retract_ratio();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_vacant_retract_ratio(signal_type);
    }

    /**
     * @brief Returns the minimum calcium, indexed by the local neuron id
     * @param synaptic_element_type The type of the elements for which to retrieve the values
     * @return The vacant minimum calcium
     */
    [[nodiscard]] std::span<const calcium_type> get_minimum_calcium(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->get_minimum_calcium();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->get_minimum_calcium(signal_type);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        axons->record_memory_footprint(footprint);
        dendrites->record_memory_footprint(footprint);

        constexpr auto my_size = sizeof(*this) - sizeof(Axons) - sizeof(Dendrites);

        footprint->emplace("SynapticElements", my_size);
    }

    static constexpr calcium_type default_eta_Axons{ utility::as<calcium_type>(0.4) };         // In Sebastian's work: 0.0
    static constexpr calcium_type default_eta_Dendrites_exc{ utility::as<calcium_type>(0.1) }; // In Sebastian's work: 0.0
    static constexpr calcium_type default_eta_Dendrites_inh{ 0.0 };                            // In Sebastian's work: 0.0
    static constexpr grown_type default_nu{ utility::as<grown_type>(1e-4) };                   // In Sebastian's work: 1e-5
    static constexpr grown_type default_vacant_retract_ratio{ 0.0 };
    static constexpr grown_type default_vacant_elements_initially_lower_bound{ 0.0 };
    static constexpr grown_type default_vacant_elements_initially_upper_bound{ 0.0 };
    static constexpr grown_type default_min_elements{ 0.0 };
    static constexpr grown_type default_max_elements{ std::numeric_limits<grown_type>::max() };

    static constexpr calcium_type min_min_C_level_to_grow{ 0.0 };
    static constexpr calcium_type min_C_target{ 0.0 };
    static constexpr grown_type min_nu{ 0.0 };
    static constexpr grown_type min_vacant_retract_ratio{ 0.0 };
    static constexpr grown_type min_vacant_elements_initially{ 0.0 };

    static constexpr calcium_type max_min_C_level_to_grow{ 10.0 };
    static constexpr calcium_type max_C_target{ 100.0 };
    static constexpr grown_type max_nu{ 1.0 };
    static constexpr grown_type max_vacant_retract_ratio{ 1.0 };
    static constexpr grown_type max_vacant_elements_initially{ 1000.0 };

protected:
    number_neurons_type size{};
    std::shared_ptr<NeuronsExtraInfo> extra_infos;

    std::shared_ptr<GrowthrateCalculator> growthrate_calculator_axon;
    std::shared_ptr<GrowthrateCalculator> growthrate_calculator_excitatory_dendrites;
    std::shared_ptr<GrowthrateCalculator> growthrate_calculator_inhibitory_dendrites;

    std::shared_ptr<Axons> axons;
    std::shared_ptr<Dendrites> dendrites;
};
