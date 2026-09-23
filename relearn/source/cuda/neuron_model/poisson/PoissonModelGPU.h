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

#include "neurons/models/poisson/PoissonModelBase.h"

#include <cstdint>

namespace models {
/**
 * GPU implementation of PoissonModel: update_activity(_benchmark) launches the Poisson kernel,
 *      drawing from a registered device RNG stream (random_key). init/create_neurons additionally
 *      register that stream and re-initialize the neurons, since the device RNG state needs it.
 */
class PoissonModelGPU : public PoissonModelBase {
public:
    using PoissonModelBase::PoissonModelBase;

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial refractory_time counter to 0 and registers the device RNG stream.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(number_neurons_type number_neurons) override;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(number_neurons_type creation_count) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

protected:
    void update_activity() final;

    void update_activity_benchmark() final;

private:
    std::uint32_t random_key{};
};

} // namespace models
