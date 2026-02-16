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

#include "GrowthrateCalculator.h"

#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

/**
 * Provides a constant growth rate
 */
class ConstantGrowthrateCalculator : public GrowthrateCalculator {
public:
    /**
     * @brief Creates the object with the given growth rate
     * @param growth_rate The intended growth rate, must be >= min_growth_rate and <= max_growth_rate
     * @exception Throws a RelearnException if growth_rate is not from [min_growth_rate, max_growth_rate]
     */
    ConstantGrowthrateCalculator(const double growth_rate)
        : intended_growth_rate(growth_rate) {
        RelearnException::check(min_growth_rate <= growth_rate,
                                "ConstantGrowthrateCalculator::ConstantGrowthrateCalculator: growth_rate is too small: {}", growth_rate);
        RelearnException::check(growth_rate <= max_growth_rate,
                                "ConstantGrowthrateCalculator::ConstantGrowthrateCalculator: growth_rate is too large: {}", growth_rate);
    }

    /**
     * @brief Initializes the object with the given number of neurons
     * @param number_neurons The number of neurons, not 0
     * @exception Throws a RelearnException if number_neurons is 0
     *      or if called multiple times
     */
    void init([[maybe_unused]] const number_neurons_type number_neurons) override { }

    /**
     * @brief Creates the given number of neurons additionally
     * @param creat_count The number of neurons, not 0
     * @exception Throws a RelearnException if number_neurons is 0
     *      or if init() was not called previously
     */
    void create_neurons([[maybe_unused]] const number_neurons_type creat_count) override { }

    /**
     * @brief Returns the intended growth rate for the neurons
     * @param neuron_id unused
     * @return The intended growth rate
     */
    [[nodiscard]] double get_growth_rate([[maybe_unused]] const NeuronID neuron_id) const noexcept override {
        return intended_growth_rate;
    }

    /**
     * @brief Returns the intended growth rate for the neurons
     * @param neuron_id unused
     * @return The intended growth rate
     */
    [[nodiscard]] double get_growth_rate([[maybe_unused]] const NeuronID::value_type neuron_id) const noexcept override {
        return intended_growth_rate;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("ConstantGrowthrateCalculator", my_footprint);
    }

private:
    double intended_growth_rate{};
};
