#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */
#include "CalciumCalculatorBase.h"

#include "neurons/enums/FiredStatus.h"

#include <span>
#include <vector>

/**
 * @brief CPU implementation of the calcium calculator: the current calcium is updated from a host
 *      span of the fired status.
 */
class CalciumCalculatorCPU : public CalciumCalculatorBase<std::vector> {
public:
    /**
     * @brief Constructs a new calculator without decaying target calcium
     */
    CalciumCalculatorCPU() = default;

    CalciumCalculatorCPU(const CalciumCalculatorCPU&) = delete;
    CalciumCalculatorCPU& operator=(const CalciumCalculatorCPU&) = delete;

    CalciumCalculatorCPU(CalciumCalculatorCPU&&) = default;
    CalciumCalculatorCPU& operator=(CalciumCalculatorCPU&&) = default;

    virtual ~CalciumCalculatorCPU() = default;

    /**
     * @brief Updates the calcium values for each neuron
     * @param step The current update step
     * @param fired_status Indicates if a neuron fired
     * @exception Throws a RelearnException if the size of the vectors doesn't match the size of the stored vectors
     */
    void update_calcium(step_type step, std::span<const FiredStatus> fired_status);

private:
    void update_current_calcium(std::span<const FiredStatus> fired_status) noexcept;
};
