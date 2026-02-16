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

#include "Types.h"

#include <filesystem>
#include <vector>

class AxonPositionIO {
public:
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Reads the positions of the axons. The format must be:
     *      <neuron_id + 1> <x> <y> <z>
     *      where an <neuron_id + 1> can occur multiple times, but overall, the neuron ids must be increasing
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if opening the file failed
     * @return The positions indexed by the neuron ids
     */
    [[nodiscard]] static std::vector<std::vector<position_type>> read_axon_positions(const std::filesystem::path& file_path);
};
