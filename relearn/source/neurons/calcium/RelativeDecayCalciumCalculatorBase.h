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

#include "CalciumCalculator.h"

#include "util/RelearnException.h"

#include <cpp-utility/Interval.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

/**
 * @brief This class focuses on calculating the inter-cellular calcium concentration of the neurons.
 *      It offers the functionality for neuron-dependent target values.
 *      This class offers the functionality to let the target calcium decay by a relative amount.
 *      Holds everything that is common to the CPU and GPU flavors; RelativeDecayCalciumCalculatorCPU
 *      and RelativeDecayCalciumCalculatorGPU add the part that differs (update_target_calcium
 *      operates on a host span on the CPU and a device pointer on the GPU).
 */
class RelativeDecayCalciumCalculatorBase : public CalciumCalculator {
public:
    using step_type = CalciumCalculator::step_type;
    using calcium_type = CalciumCalculator::calcium_type;

    /**
     * @brief Constructs an object that can calculate a relative decay
     * @param _decay_amount The amount of relative decay, must be >0.0
     * @param _decay_interval The frequency of the decay, must have a frequency >0
     * @exception Throws a RelearnException if decay_amount <= 0.0 or the frequency == 0
     */
    RelativeDecayCalciumCalculatorBase(const calcium_type _decay_amount, const utility::Interval<step_type>& _decay_interval)
        : decay_amount(_decay_amount)
        , decay_interval(_decay_interval) {

        RelearnException::check(decay_amount >= calcium_type{ 0 } && decay_amount < calcium_type{ 1 }, "RelativeDecayCalciumCalculator::RelativeDecayCalciumCalculator: The decay amount was not from [0, 1)! {}", decay_amount);
        RelearnException::check(decay_interval.frequency > 0, "RelativeDecayCalciumCalculator::RelativeDecayCalciumCalculator: Frequency of decay is 0");
    }

    /**
     * @brief Returns the relative amount of decay for each decaying step
     * @return The absolut amount of decay
     */
    [[nodiscard]] calcium_type get_decay_amount() const noexcept {
        return decay_amount;
    }

    /**
     * @brief Returns the decay interval, i.e., the first and last step of decay, and the frequency inbetween
     * @return The interval of decay
     */
    [[nodiscard]] const utility::Interval<step_type>& get_decay_interval() const noexcept {
        return decay_interval;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(CalciumCalculator);
        footprint->emplace("RelativeDecayCalciumCalculator", my_footprint);

        CalciumCalculator::record_memory_footprint(footprint);
    }

protected:
    calcium_type decay_amount{ 0.0 };
    utility::Interval<step_type> decay_interval{};
};
