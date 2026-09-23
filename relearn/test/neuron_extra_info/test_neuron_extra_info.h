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

#include "RelearnTest.hpp"

#include "neurons/NeuronsExtraInfo.h"
#include "types/SpaceTypes.h"
#include "util/RelearnAllocator.h"

class NeuronsExtraInfoTest : public RelearnTest {
protected:
    void assert_empty(const NeuronsExtraInfo& extra_info, RelearnTypes::number_neurons_type number_neurons);

    void assert_contains(const NeuronsExtraInfo& extra_info, RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_neurons_type num_neurons_check,
                         const std::vector<RelearnTypes::position_type>& expected_positions);
};
