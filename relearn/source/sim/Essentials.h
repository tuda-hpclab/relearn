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

#include <map>
#include <ostream>
#include <string>
#include <type_traits>

/**
 * Provides the functionality to gather descriptions of the simulation
 * and print them out sorted in one flush.
 */
class Essentials {
public:
    using description_type = std::string;
    using value_type = std::string;

    /**
     * @brief Adds a description and a value to the store.
     *      If the descriptions is already present, the previous value is overwritten.
     * @param description The (new) description
     * @param value The value
     * @tparam T The type of the value, must support std::to_string(T)
     */
    template <typename T>
    void insert(description_type description, T value) {
        if constexpr (std::is_constructible_v<std::string, T>) {
            dictionary[std::move(description)] = std::string(std::move(value));
        } else {
            dictionary[std::move(description)] = std::to_string(value);
        }
    }

    /**
     * @brief Prints all stored entries in the form
     *      <key>: <value>\n
     *      sorted by <key>
     * @param out The outstream to print
     */
    void print(std::ostream& out) {
        for (const auto& [key, value] : dictionary) {
            out << key << ": " << value << '\n';
        }
    }

    /**
     * Return the underlying key-value map of the essentials
     * @return The key-value map
     */
    [[nodiscard]] const std::map<description_type , value_type >& as_map() const {
        return dictionary;
    }

private:
    std::map<description_type, value_type> dictionary{};
};
