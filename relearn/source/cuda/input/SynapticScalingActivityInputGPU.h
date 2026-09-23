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

#include "neurons/input/SynapticScalingActivityInputBase.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <memory>
#include <optional>
#include <span>
#include <vector>

/**
 * GPU implementation of SynapticScalingActivityInput. This model is currently unsupported on the
 * GPU (construction fails via CPU_NOT_SUPPORTED); this class exists only so factories compile
 * without needing their own #ifdef around it.
 */
class SynapticScalingActivityInputGPU : public SynapticScalingActivityInputBase<LazySyncedArray> {
public:
    using Base = SynapticScalingActivityInputBase<LazySyncedArray>;

    SynapticScalingActivityInputGPU(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator, const activity_type total_scale = 1.0)
        : Base(_number_ranks, std::move(communicator), total_scale) {
        CPU_NOT_SUPPORTED
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        Base::init(number_neurons);
        d_scales = scales.get_device_ptr_const();
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        Base::create_neurons(creation_count);
        d_scales = scales.get_device_ptr_const();
    }

    /**
     * @brief Returns a view of the scales
     * @return The scales
     */
    [[nodiscard]] std::span<const activity_type> get_scales() const noexcept {
        return scales.host();
    }

protected:
    std::optional<EventWrapper> update_local_input([[maybe_unused]] number_neurons_type first, [[maybe_unused]] number_neurons_type last, [[maybe_unused]] activity_type* d_input,
                                                   [[maybe_unused]] const FiredStatus* d_fired, [[maybe_unused]] const activity_type* _d_scales,
                                                   [[maybe_unused]] const std::shared_ptr<StreamWrapper>& stream_wrapper) override{
        CPU_NOT_SUPPORTED
    }

    std::optional<EventWrapper> update_distant_input([[maybe_unused]] const number_neurons_type first, [[maybe_unused]] const number_neurons_type last, [[maybe_unused]] activity_type* d_input, [[maybe_unused]] const activity_type* _d_scales,
                                                     [[maybe_unused]] std::unique_ptr<FireStatusCommunicatorHandle>& fire_status_handle, [[maybe_unused]] const std::shared_ptr<StreamWrapper>& stream_wrapper) override {
        if (mpiPP::MPIInfo::get_number_ranks() == 1) {
            return { std::nullopt };
        }
        CPU_NOT_SUPPORTED
    }
};
