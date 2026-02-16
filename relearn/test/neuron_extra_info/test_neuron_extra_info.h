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

#include "RelearnTest.hpp"

#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnAllocator.h"

class NeuronsExtraInfoTest : public RelearnTest {
protected:
    void assert_empty(const NeuronsExtraInfo& extra_info, size_t number_neurons);

    void assert_contains(const NeuronsExtraInfo& extra_info, size_t number_neurons, size_t num_neurons_check,
        const std::vector<Vec3d>& expected_positions);
};
