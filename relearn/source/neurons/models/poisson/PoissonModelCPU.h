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

#include "PoissonModelBase.h"

namespace models {
/**
 * CPU implementation of PoissonModel: update_activity(_benchmark) integrates the model on the host,
 *      drawing from the host RNG.
 */
class PoissonModelCPU : public PoissonModelBase {
public:
    using PoissonModelBase::PoissonModelBase;

protected:
    void update_activity() final;

    void update_activity_benchmark() final;
};

} // namespace models
