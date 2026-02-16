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

#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <functional>
#include <span>
#include <vector>

namespace details {
template <typename T>
T get_zero([[maybe_unused]] const RelearnTypes::number_neurons_type nnt) {
    return T(0);
}
} // namespace details

class SynapticElementsAdapter;

/**
 * This class encapsulates the data used for excitatory/inhibitory axons/dendrites.
 * It allows for an initial value for the grown and connected elements, the delta since the last update, and the retract ratio of vacant elements.
 * It provides the number of vacant elements based on the other values.
 */
class ElementBase {
public:
    friend class SynapticElementsAdapter;

    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;
    using calcium_type = RelearnTypes::calcium_type;

    ElementBase() = default;

    ElementBase(const ElementBase& other) = default;
    ElementBase(ElementBase&& other) = default;

    ElementBase& operator=(const ElementBase& other) = default;
    ElementBase& operator=(ElementBase&& other) = default;

    ~ElementBase() = default;

    /**
     * @brief Sets the function that calculates the number of grown elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_grown_elements_calculator(std::function<double(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "ElementBase::set_grown_elements_calculator: calculator is empty");

        grown_elements_calculator = std::move(calculator);
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_delta_since_last_update_calculator(std::function<double(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "ElementBase::set_delta_since_last_update_calculator: calculator is empty");

        delta_since_last_update_calculator = std::move(calculator);
    }

    /**
     * @brief Sets the function that calculates the number of connected elements for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_connected_elements_calculator(std::function<unsigned int(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "ElementBase::set_connected_elements_calculator: calculator is empty");

        connected_elements_calculator = std::move(calculator);
    }

    /**
     * @brief Sets the function that calculates the delta for each neuron.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must return values from [0.0, 1.0]
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_vacant_retract_ratio_calculator(std::function<double(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "ElementBase::set_vacant_retract_ratio_calculator: calculator is empty");

        vacant_retract_ratio_calculator = std::move(calculator);
    }

    /**
     * @brief Sets the function that calculates the minimum calcium that is required for the elements to grow.
     *      Is used in init(...) and create_neurons(...).
     * @param calculator The function, must not be empty, and must not return a value < 0.0
     * @exception Throws a RelearnException if calculator is empty
     */
    void set_minimum_calcium_calculator(std::function<double(number_neurons_type)> calculator) {
        const auto function_full = calculator.operator bool();
        RelearnException::check(function_full, "ElementBase::set_minimum_calcium_calculator: calculator is empty");

        minimum_calcium_calculator = std::move(calculator);
    }

    /**
     * @brief Initializes the object to contain number_neurons elements.
     *      Uses the calculators for grown elements, delta since last update, and connected elements.
     *      Calculates the number of vacant elements based on the other values.
     * @param number_neurons The number of neurons to initialize, >0
     * @exception Throws a RelearnException if the object was already initialized, if number_neurons == 0,
     *      or if the calculator for the grown elements returns a value < 0.0
     */
    void init(const number_neurons_type number_neurons) {
        RelearnException::check(size == 0, "ElementBase::init: Already initialized");
        RelearnException::check(number_neurons > 0, "ElementBase::init: number_neurons must be > 0");

        size = number_neurons;

        grown_elements.resize(size);
        delta_since_last_update.resize(size);
        vacant_elements.resize(size);
        connected_elements.resize(size);

        vacant_retract_ratio.resize(size);
        minimum_calcium.resize(size);

        init_values(number_neurons_type{ 0 }, size);
    }

    /**
     * @brief Creates the specified number of additional neurons.
     *      Uses the calculators for grown elements, delta since last update, and connected elements.
     *      Calculates the number of vacant elements based on the other values.
     * @param creation_count The number of that should be created, >0
     * @exception Throws a RelearnException if the object was not initialized already, if creation_count == 0,
     *      or if the calculator for the grown elements returns a value < 0.0
     */
    void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(size > 0, "ElementBase::create_neurons: Not initialized");
        RelearnException::check(creation_count > 0, "ElementBase::create_neurons: creation_count must be > 0");

        const auto old_size = size;
        const auto new_size = old_size + creation_count;
        size = new_size;

        grown_elements.resize(size);
        delta_since_last_update.resize(size);
        vacant_elements.resize(size);
        connected_elements.resize(size);

        vacant_retract_ratio.resize(size);
        minimum_calcium.resize(size);

        init_values(old_size, new_size);
    }

    /**
     * @brief Disables the specified neurons. Sets the grown elements, deltas, vacant and connected elements to 0.
     * @param disabled_neuron_ids The neurons to disable
     * @exception Throws a RelearnException if a neuron_id is too large
     */
    void disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
        for (const auto neuron_id : disabled_neuron_ids) {
            RelearnException::check(neuron_id < size, "ElementBase::disable_neurons: Cannot disable a neuron with a too large id");
            grown_elements[neuron_id] = 0.0;
            delta_since_last_update[neuron_id] = 0.0;
            vacant_elements[neuron_id] = 0;
            connected_elements[neuron_id] = 0;
        }
    }

    /**
     * @brief Adds a delta for the specified neuron. This delta has to be applied via commit_updates(...).
     * @param delta The delta to add
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_to_delta(const double delta, const number_neurons_type neuron_id) {
        RelearnException::check(neuron_id < size, "ElementBase::add_to_delta: neuron_id is too large: {}", neuron_id);
        delta_since_last_update[neuron_id] += delta;
    }

    /**
     * @brief Adds deltas for all neurons. These deltas has to be applied via commit_updates(...).
     * @param deltas The deltas to add
     * @exception Throws a RelearnException if deltas.size() != size
     */
    void add_to_delta(const std::span<const double> deltas) {
        RelearnException::check(deltas.size() == size, "ElementBase::add_to_delta: deltas.size() != size");

#pragma omp parallel for shared(deltas) default(none)
        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            delta_since_last_update[neuron_id] += deltas[neuron_id];
        }
    }

    /**
     * @brief Adds the specified amount to the connected elements of the specified neuron.
     *      Also adds the amount of grown elements.
     * @param newly_added The amount of newly connected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     */
    void add_connected_elements(const unsigned int newly_added, const number_neurons_type neuron_id) {
        RelearnException::check(neuron_id < size, "ElementBase::add_connected_elements: neuron_id is too large: {}", neuron_id);
        connected_elements[neuron_id] += newly_added;
        grown_elements[neuron_id] += static_cast<double>(newly_added);
    }

    /**
     * @brief Adds the specified amount to the connected elements of each neuron.
     *      Also adds the amount of grown elements.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes does not have the correct size
     */
    void add_connected_elements(const std::span<const unsigned int> changes) {
        RelearnException::check(changes.size() == size, "ElementBase::add_connected_elements: changes.size() != size");

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            const auto change = changes[neuron_id];
            connected_elements[neuron_id] += change;
            grown_elements[neuron_id] += static_cast<double>(change);
        }
    }

    /**
     * @brief Removes the specified amount of connected elements from the specified neuron.
     *      Also increases the number of vacant elements by the same amount.
     * @param newly_free The amount of newly free elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if the number of connected elements is smaller than newly_free
     */
    void remove_connected_elements(const unsigned int newly_free, const number_neurons_type neuron_id) {
        RelearnException::check(neuron_id < size, "ElementBase::remove_connected_elements: neuron_id is too large: {}", neuron_id);
        RelearnException::check(connected_elements[neuron_id] >= newly_free,
                                "ElementBase::remove_connected_elements: Cannot delete more connections than present for neuron {}: {} vs {}", neuron_id, newly_free, connected_elements[neuron_id]);
        RelearnException::check(static_cast<unsigned int>(grown_elements[neuron_id]) >= newly_free,
                                                          "ElementBase::remove_connected_elements: Cannot delete more connections than present for neuron {}: {} vs {}", neuron_id, newly_free, connected_elements[neuron_id]);

        connected_elements[neuron_id] -= newly_free;
        vacant_elements[neuron_id] += newly_free;
    }

    /**
     * @brief Removes the amount of connected elements from the neurons.
     *      Also increases the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes does not have the correct size or if the number of connected elements is smaller than the specified amount for one neuron
     */
    void remove_connected_elements(const std::span<const unsigned int> changes) {
        RelearnException::check(changes.size() == size, "ElementBase::add_connected_elements: changes.size() != size");

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            const auto change = changes[neuron_id];
            RelearnException::check(connected_elements[neuron_id] >= change,
                                    "ElementBase::remove_connected_elements: Cannot delete more connections than present for neuron {}: {} vs {}", neuron_id, change, connected_elements[neuron_id]);

            connected_elements[neuron_id] -= change;
            vacant_elements[neuron_id] += change;
        }
    }

    /**
     * @brief Connects the specified number of elements to the specified neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param newly_connected The number of newly connected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough vacant elements
     */
    void connect_elements(const unsigned int newly_connected, const number_neurons_type neuron_id) {
        RelearnException::check(neuron_id < size, "ElementBase::connect_elements: neuron_id is too large: {}", neuron_id);

        const auto vacant = vacant_elements[neuron_id];
        RelearnException::check(vacant >= newly_connected, "ElementBase::connect_elements: Not enough vacant elements for neuron {}: {} vs {}", neuron_id, newly_connected, vacant);

        connected_elements[neuron_id] += newly_connected;
        vacant_elements[neuron_id] -= newly_connected;
    }

    /**
     * @brief Connects the specified number of elements to each neuron.
     *      Reduces the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough vacant elements
     */
    void connect_elements(const std::span<const unsigned int> changes) {
        RelearnException::check(changes.size() == size, "ElementBase::connect_elements: changes.size() != size");

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            const auto change = changes[neuron_id];
            const auto vacant = vacant_elements[neuron_id];

            RelearnException::check(vacant >= change, "ElementBase::connect_elements: Not enough vacant elements for neuron {}: {} vs {}", neuron_id, change, vacant);

            connected_elements[neuron_id] += change;
            vacant_elements[neuron_id] -= change;
        }
    }

    /**
     * @brief Disconnects the specified number of elements to the specified neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param newly_disconnected The number of newly disconnected elements
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large or if there are not enough connected elements
     */
    void disconnect_elements(const unsigned int newly_disconnected, const number_neurons_type neuron_id) {
        RelearnException::check(neuron_id < size, "ElementBase::disconnect_elements: neuron_id is too large: {}", neuron_id);

        const auto connected = connected_elements[neuron_id];
        RelearnException::check(connected >= newly_disconnected, "ElementBase::disconnect_elements: Not enough connected elements for neuron {}: {} vs {}", neuron_id, newly_disconnected, connected);

        connected_elements[neuron_id] -= newly_disconnected;
        vacant_elements[neuron_id] += newly_disconnected;
    }

    /**
     * @brief Disconnects the specified number of elements to each neuron.
     *      Increases the number of vacant elements by the same amount.
     * @param changes The changes for each neuron
     * @exception Throws a RelearnException if changes is not the same size as this, or if there are not enough connected elements
     */
    void disconnect_elements(const std::span<const unsigned int> changes) {
        RelearnException::check(changes.size() == size, "ElementBase::disconnect_elements: changes.size() != size");

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            const auto change = changes[neuron_id];
            const auto connected = connected_elements[neuron_id];

            RelearnException::check(connected >= change, "ElementBase::disconnect_elements: Not enough connected elements for neuron {}: {} vs {}", neuron_id, change, connected);

            connected_elements[neuron_id] -= change;
            vacant_elements[neuron_id] += change;
        }
    }

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<unsigned int> commit_updates() {
        auto number_deletions = std::vector<unsigned int>{};
        number_deletions.resize(size, 0U);

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            number_deletions[neuron_id] = update(neuron_id);
        }

        return number_deletions;
    }

    /**
     * @brief Returns the total number of additions over the lifetime of this object
     * @return The total number of additions
     */
    [[nodiscard]] double get_total_additions() const noexcept {
        return total_additions;
    }

    /**
     * @brief Returns the total number of deletions over the lifetime of this object
     * @return The total number of deletions
     */
    [[nodiscard]] double get_total_deletions() const noexcept {
        return total_deletions;
    }

    /**
     * @brief Returns the number of grown elements, indexed by the local neuron id
     * @return The number of grown elements
     */
    [[nodiscard]] std::span<const double> get_grown_elements() const noexcept {
        return grown_elements;
    }

    /**
     * @brief Returns the accumulated changes to the grown elements, indexed by the local neuron id (the built-up difference from the electrical updates)
     * @return The accumulated changes
     */
    [[nodiscard]] std::span<const double> get_deltas() const noexcept {
        return delta_since_last_update;
    }

    /**
     * @brief Returns the number of vacant elements, indexed by the local neuron id (how many elements are grown and not connected)
     * @return The vacant elements
     */
    [[nodiscard]] std::span<const unsigned int> get_vacant_elements() const noexcept {
        return vacant_elements;
    }

    /**
     * @brief Returns the number of connected elements, indexed by the local neuron id (how many elements from the neuron are connected via synapses)
     * @return The connected elements
     */
    [[nodiscard]] std::span<const unsigned int> get_connected_elements() const noexcept {
        return connected_elements;
    }

    /**
     * @brief Returns the vacant retract ratio, indexed by the local neuron id
     * @return The vacant retract ratio
     */
    [[nodiscard]] std::span<const double> get_vacant_retract_ratio() const noexcept {
        return vacant_retract_ratio;
    }

    /**
     * @brief Returns the minimum calcium, indexed by the local neuron id
     * @return The vacant minimum calcium
     */
    [[nodiscard]] std::span<const calcium_type> get_minimum_calcium() const noexcept {
        return minimum_calcium;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     * @param key The key to use for the memory footprint
     */
    template <typename T>
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, T&& key) {
        const auto size_grown = sizeof(double) * grown_elements.capacity();
        const auto size_delta = sizeof(double) * delta_since_last_update.capacity();
        const auto size_vacant = sizeof(unsigned int) * vacant_elements.capacity();
        const auto size_connected = sizeof(unsigned int) * connected_elements.capacity();
        const auto size_vacant_retract = sizeof(double) * vacant_retract_ratio.capacity();
        const auto size_minimum_calcium = sizeof(calcium_type) * minimum_calcium.capacity();

        const auto my_size = sizeof(*this);

        const auto total_size = size_grown + size_delta + size_vacant + size_connected + size_vacant_retract + size_minimum_calcium + my_size;

        footprint->emplace(std::forward<T>(key), total_size);
    }

private:
    void init_values(const number_neurons_type begin, const number_neurons_type end) {
        for (auto i = begin; i < end; i++) {
            const auto forced_grown_elements = grown_elements_calculator(i);
            const auto forced_delta_since_last_update = delta_since_last_update_calculator(i);
            const auto forced_connected_elements = connected_elements_calculator(i);
            const auto forced_vacant_retract_ratio = vacant_retract_ratio_calculator(i);
            const auto forced_minimum_calcium = minimum_calcium_calculator(i);

            RelearnException::check(forced_grown_elements >= 0.0, "ElementBase::init_values: grown_elements_calculator returned {} for id {}", forced_grown_elements, i);
            RelearnException::check(forced_vacant_retract_ratio >= 0.0, "ElementBase::init_values: vacant_retract_ratio_calculator returned {} for id {}", forced_vacant_retract_ratio, i);
            RelearnException::check(forced_vacant_retract_ratio <= 1.0, "ElementBase::init_values: vacant_retract_ratio_calculator returned {} for id {}", forced_vacant_retract_ratio, i);
            RelearnException::check(forced_minimum_calcium >= 0.0, "ElementBase::init_values: minimum_calcium_calculator returned {} for id {}", forced_minimum_calcium, i);

            const auto cast_grown_elements = static_cast<unsigned int>(forced_grown_elements);

            grown_elements[i] = forced_grown_elements;
            delta_since_last_update[i] = forced_delta_since_last_update;
            connected_elements[i] = forced_connected_elements;

            vacant_retract_ratio[i] = forced_vacant_retract_ratio;
            minimum_calcium[i] = forced_minimum_calcium;

            if (forced_connected_elements >= cast_grown_elements) {
                vacant_elements[i] = 0;
            } else {
                vacant_elements[i] = cast_grown_elements - forced_connected_elements;
            }
        }
    }

    [[nodiscard]] unsigned int update(const number_neurons_type neuron_id) {
        const auto current_count = grown_elements[neuron_id];
        const auto current_delta = delta_since_last_update[neuron_id];
        const auto current_connected_count_integral = connected_elements[neuron_id];
        const auto retract_ratio = vacant_retract_ratio[neuron_id];

        const auto current_connected_count = static_cast<double>(current_connected_count_integral);
        const auto current_vacant = current_count - current_connected_count;

        RelearnException::check(current_count >= 0.0, "ElementBase::update: {}", current_count);
        RelearnException::check(current_connected_count >= 0.0, "ElementBase::update: {}", current_connected_count);
        RelearnException::check(current_vacant >= 0.0, "ElementBase::update: {}", current_count - current_connected_count);

        if (current_delta >= 0.0) {
            total_additions += current_delta;
        } else {
            total_deletions += std::min(-current_delta, current_count);
        }

        if (const auto new_vacant = current_vacant + current_delta; new_vacant >= 0.0) {
            // The vacant portion after caring for the delta
            // No deletion of bound synaptic elements required, connected_elements stays the same

            const auto new_vacant_after_retract = (1 - retract_ratio) * new_vacant;
            const auto new_count = new_vacant_after_retract + current_connected_count;
            RelearnException::check(new_count >= current_connected_count, "ElementBase::update: new count is smaller than connected count");

            grown_elements[neuron_id] = new_count;
            delta_since_last_update[neuron_id] = 0.0;
            vacant_elements[neuron_id] = static_cast<unsigned int>(new_vacant_after_retract);
            // connected_elements does not need to change

            return 0U;
        }

        if (current_count + current_delta <= 0.0) {
            // More bound elements should be deleted than are available.
            // Now, neither vacant (see if branch above) nor bound elements are left.

            grown_elements[neuron_id] = 0.0;
            delta_since_last_update[neuron_id] = 0.0;
            vacant_elements[neuron_id] = 0U;
            connected_elements[neuron_id] = 0;

            return current_connected_count_integral;
        }

        const auto new_count = current_count + current_delta;
        const auto new_connected_count = std::floor(new_count);
        const auto num_vacant = new_count - new_connected_count;

        const auto retracted_new_count = ((1 - retract_ratio) * num_vacant) + new_connected_count;

        RelearnException::check(num_vacant >= 0.0, "ElementBase::update: num_vacant is negative");
        RelearnException::check(num_vacant < 1.0, "ElementBase::update: num_vacant is larger than 1.0");

        grown_elements[neuron_id] = retracted_new_count;
        delta_since_last_update[neuron_id] = 0.0;
        vacant_elements[neuron_id] = 0U;
        connected_elements[neuron_id] = static_cast<unsigned int>(new_connected_count);

        const auto deleted_counts = current_connected_count - new_connected_count;

        RelearnException::check(deleted_counts >= 0.0, "ElementBase::update: deleted was negative");
        const auto num_delete_connected = static_cast<unsigned int>(deleted_counts);

        return num_delete_connected;
    }

    number_neurons_type size{};
    double total_additions{ 0.0 };
    double total_deletions{ 0.0 };

    std::vector<double> grown_elements{};
    std::vector<double> delta_since_last_update{};
    std::vector<unsigned int> vacant_elements{};
    std::vector<unsigned int> connected_elements{};

    std::vector<double> vacant_retract_ratio{};
    std::vector<calcium_type> minimum_calcium{};

    std::function<double(number_neurons_type)> grown_elements_calculator{ details::get_zero<double> };
    std::function<double(number_neurons_type)> delta_since_last_update_calculator{ details::get_zero<double> };
    std::function<unsigned int(number_neurons_type)> connected_elements_calculator{ details::get_zero<unsigned int> };

    std::function<double(number_neurons_type)> vacant_retract_ratio_calculator{ details::get_zero<double> };
    std::function<double(number_neurons_type)> minimum_calcium_calculator{ details::get_zero<double> };
};
