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

#include <memory>

namespace utility {
class MemoryFootprint;
}

/**
 * Calculates the growth rate for the synaptic elements.
 * Offers an interface for, e.g., dampening
 */
class GrowthrateCalculator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    GrowthrateCalculator() = default;
    virtual ~GrowthrateCalculator() = default;

    GrowthrateCalculator(const GrowthrateCalculator&) = default;
    GrowthrateCalculator& operator=(const GrowthrateCalculator&) = default;

    GrowthrateCalculator(GrowthrateCalculator&&) = default;
    GrowthrateCalculator& operator=(GrowthrateCalculator&&) = default;

    /**
     * @brief Initializes the object with the given number of neurons
     * @param number_neurons The number of neurons, not 0
     * @exception Throws a RelearnException if number_neurons is 0
     *      or if called multiple times
     */
    virtual void init(number_neurons_type number_neurons) = 0;

    /**
     * @brief Creates the given number of neurons additionally
     * @param creation_count The number of neurons, not 0
     * @exception Throws a RelearnException if number_neurons is 0
     *      or if init() was not called previously
     */
    virtual void create_neurons(number_neurons_type creation_count) = 0;

    /**
     * @brief Returns the current growth rate for the neurons
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if the id is too large
     * @return The current growth rate, always >= 0
     */
    [[nodiscard]] virtual double get_growth_rate(NeuronID neuron_id) const = 0;

    /**
     * @brief Returns the current growth rate for the neurons
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if the id is too large
     * @return The current growth rate, always >= 0
     */
    [[nodiscard]] virtual double get_growth_rate(NeuronID::value_type neuron_id) const = 0;

    /**
     * @brief Updates the growth rate for each neuron
     * @param current_step The current step
     */
    virtual void update_growth_rate([[maybe_unused]] const RelearnTypes::step_type current_step) { }

    /**
     * @brief Sets the last change in grown elements, might be used for the growth rate
     * @param neuron_id The neuron's id
     * @param last_change The last change in grown elements
     */
    virtual void set_last_change([[maybe_unused]] const NeuronID neuron_id, [[maybe_unused]] const double last_change) { }

    /**
     * @brief Sets the last change in grown elements, might be used for the growth rate
     * @param neuron_id The neuron's id
     * @param last_change The last change in grown elements
     */
    virtual void set_last_change([[maybe_unused]] const NeuronID::value_type neuron_id, [[maybe_unused]] const double last_change) { }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) = 0;

    static constexpr double default_growth_rate{ 1e-5 }; // In Sebastian's work: 1e-5
    static constexpr double min_growth_rate{ 0.0 };
    static constexpr double max_growth_rate{ 1.0 };
};
