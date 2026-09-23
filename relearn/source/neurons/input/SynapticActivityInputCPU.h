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

#include "SynapticActivityInputBase.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"

#include <memory>
#include <span>

/**
 * CPU implementation of SynapticActivityInput: update_local_input/update_distant_input work on
 * host spans and update the input in place.
 */
class SynapticActivityInputCPU : public SynapticActivityInputBase {
public:
    explicit SynapticActivityInputCPU(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator)
        : SynapticActivityInputBase(_number_ranks, std::move(communicator)) { }

    SynapticActivityInputCPU(const SynapticActivityInputCPU&) = delete;
    SynapticActivityInputCPU& operator=(const SynapticActivityInputCPU&) = delete;

    SynapticActivityInputCPU(SynapticActivityInputCPU&&) = default;
    SynapticActivityInputCPU& operator=(SynapticActivityInputCPU&&) = default;

    ~SynapticActivityInputCPU() override = default;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range(step_type step, NeuronID first, NeuronID last) override;

protected:
    virtual void update_local_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) = 0;

    virtual void update_distant_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) = 0;
};
