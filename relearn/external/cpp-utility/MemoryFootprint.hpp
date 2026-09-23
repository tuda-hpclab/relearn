#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace utility {

/**
 * Stores byte counts keyed by a textual description. Inserting an existing
 * description preserves the first recorded value.
 */
class MemoryFootprint {
public:
    /**
     * @brief Reserves space for the expected number of distinct descriptions.
     * @param reserved_space The expected number of entries.
     */
    explicit MemoryFootprint(const std::size_t reserved_space) {
        memory_description.reserve(reserved_space);
    }

    /**
     * @brief Inserts a description and its byte count if the description is not present.
     * @tparam T A key type from which std::string can be constructed.
     * @param key The description
     * @param memory_size The size in bytes
     */
    template <typename T>
        requires std::constructible_from<std::string, T&&>
    void emplace(T&& key, const std::uint64_t memory_size) {
        memory_description.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::forward<T>(key)),
            std::forward_as_tuple(memory_size));
    }

    /**
     * @brief Returns the stored memory descriptions, i.e., a mapping from description to size
     * @return A constant reference to the description
     */
    [[nodiscard]] const std::unordered_map<std::string, std::uint64_t>& get_descriptions() const noexcept {
        return memory_description;
    }

private:
    std::unordered_map<std::string, std::uint64_t> memory_description{};
};

} // namespace utility
