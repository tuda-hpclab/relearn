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

#include "neurons/calcium/CalciumCalculatorBase.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/LazySyncedArray.h"
#include "neurons/enums/FiredStatus.h"

#include <memory>

namespace utility {
class MemoryFootprint;
}

/**
 * @brief GPU implementation of the calcium calculator: the current calcium is updated from a device
 *      pointer to the fired status.
 */
class CalciumCalculatorGPU : public CalciumCalculatorBase<LazySyncedArray> {
public:
    using Base = CalciumCalculatorBase<LazySyncedArray>;

    /**
     * @brief Constructs a new calculator without decaying target calcium
     */
    CalciumCalculatorGPU() = default;

    CalciumCalculatorGPU(const CalciumCalculatorGPU&) = delete;
    CalciumCalculatorGPU& operator=(const CalciumCalculatorGPU&) = delete;

    CalciumCalculatorGPU(CalciumCalculatorGPU&&) = default;
    CalciumCalculatorGPU& operator=(CalciumCalculatorGPU&&) = default;

    virtual ~CalciumCalculatorGPU();

    /**
     * @brief Updates the calcium values for each neuron
     * @param step The current update step
     * @param d_fired Device pointer indicating if a neuron fired
     * @exception Throws a RelearnException if the size of the vectors doesn't match the size of the stored vectors
     */
    void update_calcium(step_type step, const FiredStatus* d_fired);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

    [[nodiscard]] CudaConfig::calcium_type* get_d_calcium() {
        return calcium.get_device_ptr();
    }

    [[nodiscard]] const CudaConfig::calcium_type* get_d_calcium_const() const {
        return calcium.get_device_ptr_const();
    }

    [[nodiscard]] CudaConfig::calcium_type* get_d_target_calcium() {
        return target_calcium.get_device_ptr();
    }

    [[nodiscard]] const CudaConfig::calcium_type* get_d_target_calcium_const() const {
        return target_calcium.get_device_ptr_const();
    }

private:
    void update_current_calcium(const FiredStatus* d_fired);
};
