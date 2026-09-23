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

#include "neurons/models/aeif/AEIFModelBase.h"

namespace models {
/**
 * GPU implementation of AEIFModel: update_activity(_benchmark) launches the AEIF kernel.
 */
class AEIFModelGPU : public AEIFModelBase {
public:
    using AEIFModelBase::AEIFModelBase;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

protected:
    void update_activity() final;

    void update_activity_benchmark() final;
};

} // namespace models
