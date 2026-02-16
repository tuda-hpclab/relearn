#pragma once

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