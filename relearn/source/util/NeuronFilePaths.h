#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <filesystem>
#include <optional>

/**
 * This struct encapsulates paths used to describe neurons as optionals.
 */
struct NeuronOptFilePaths {
    std::optional<std::filesystem::path> path_to_positions{ std::nullopt };
    std::optional<std::filesystem::path> path_to_groups{ std::nullopt };
};

/**
 * This struct encapsulates paths used to describe neurons.
 */
struct NeuronFilePaths {
    std::filesystem::path path_to_positions;
    std::filesystem::path path_to_groups;
};