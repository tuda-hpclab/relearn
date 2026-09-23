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

#include "ElementBaseBase.h"

#include "types/BasicTypes.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <vector>

/**
 * CPU implementation of ElementBase: updates are committed via a host loop, and there is no
 * device-side memory footprint to report.
 */
class ElementBaseCPU : public ElementBaseBase<std::vector> {
public:
    /**
     * @brief Initializes the object to contain number_neurons elements.
     *      Uses the calculators for grown elements, delta since last update, and connected elements.
     *      Calculates the number of vacant elements based on the other values.
     * @param number_neurons The number of neurons to initialize, >0
     * @exception Throws a RelearnException if the object was already initialized, if number_neurons == 0,
     *      or if the calculator for the grown elements returns a value < 0.0
     */
    void init(const RelearnTypes::number_neurons_type number_neurons) {
        RelearnException::check(size == 0, "ElementBaseCPU::init: Already initialized");
        RelearnException::check(number_neurons > 0, "ElementBaseCPU::init: number_neurons must be > 0");

        size = number_neurons;

        grown_elements.resize(size);
        delta_since_last_update.resize(size);
        vacant_elements.resize(size);
        vacant_retract_ratio.resize(size);
        minimum_calcium.resize(size);
        connected_elements.resize(size);

        init_values(RelearnTypes::number_neurons_type{ 0 }, size);
    }

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates() {
        auto number_deletions = std::vector<counter_type>{};
        number_deletions.resize(size, 0U);

        for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < size; neuron_id++) {
            number_deletions[neuron_id] = update(neuron_id);
        }

        return number_deletions;
    }

    /**
     * @brief Returns the total number of additions over the lifetime of this object
     * @return The total number of additions
     */
    [[nodiscard]] grown_type get_total_additions() const {
        return total_additions;
    }

    /**
     * @brief Returns the total number of deletions over the lifetime of this object
     * @return The total number of deletions
     */
    [[nodiscard]] grown_type get_total_deletions() const {
        return total_deletions;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     * @param key The key to use for the memory footprint
     */
    template <typename T>
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, T&& key) {
        const auto size_grown = sizeof(grown_type) * grown_elements.capacity();
        const auto size_delta = sizeof(grown_type) * delta_since_last_update.capacity();
        const auto size_vacant = sizeof(counter_type) * vacant_elements.capacity();
        const auto size_connected = sizeof(counter_type) * connected_elements.capacity();
        const auto size_vacant_retract = sizeof(grown_type) * vacant_retract_ratio.capacity();
        const auto size_minimum_calcium = sizeof(calcium_type) * minimum_calcium.capacity();

        const auto my_size = sizeof(*this);
        const auto total_size = size_grown + size_delta + size_vacant + size_connected + size_vacant_retract + size_minimum_calcium + my_size;
        footprint->emplace(key, total_size);
    }
};
