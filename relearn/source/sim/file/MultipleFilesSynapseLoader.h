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

#include "sim/SynapseLoader.h"

#include <filesystem>
#include <optional>

class Partition;

class MultipleFilesSynapseLoader : public SynapseLoader {
public:
    /**
     * @brief Constructs a FileSynapseLoader with the given Partition.
     *      Can load synapses from a file
     * @param _partition The partition to use
     * @param path_to_synapses The path to the synapses, can be empty
     */
    MultipleFilesSynapseLoader(std::shared_ptr<Partition> _partition, std::optional<std::filesystem::path> path_to_synapses);

protected:
    synapses_pair_type internal_load_synapses() override;

private:
    std::optional<std::filesystem::path> optional_path_to_file{};
};