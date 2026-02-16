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

#include <fmt/ostream.h>

#include <ostream>
#include <string>

/**
 * This enum is used to differentiate between the algorithms which can be used for creating synapses
 */
enum class AlgorithmEnum : std::uint8_t {
    Naive,
    BarnesHut,
    BarnesHutInverted,
    BarnesHutLocationAware,
    FastMultipoleMethod,
    CombinedAlgorithms,
};

/**
 * @brief Checks if the specified algorithm actually implements the Barnes-Hut algorithm
 * @param algorithm_enum The specified algorithm
 * @return True iff the specified algorithm implements the Barnes-Hut algorithm
 */
[[nodiscard]] constexpr bool is_barnes_hut(const AlgorithmEnum algorithm_enum) {
    return algorithm_enum == AlgorithmEnum::BarnesHut
           || algorithm_enum == AlgorithmEnum::BarnesHutInverted
           || algorithm_enum == AlgorithmEnum::BarnesHutLocationAware
        ;
}

/**
 * @brief Checks if the specified algorithm actually implements the Fast Multipole Method
 * @param algorithm_enum The specified algorithm
 * @return True iff the specified algorithm implements the Fast Multipole Method
 */
[[nodiscard]] constexpr bool is_fast_multipole_method(const AlgorithmEnum algorithm_enum) {
    return algorithm_enum == AlgorithmEnum::FastMultipoleMethod;
}

/**
 * @brief Returns a string representation of the algorithm enum
 * @param algorithm_enum The value to print
 * @return The string representation
 */
[[nodiscard]] inline std::string stringify(const AlgorithmEnum& algorithm_enum) {
    switch (algorithm_enum) {
    case AlgorithmEnum::Naive:
        return "Naive";
    case AlgorithmEnum::BarnesHut:
        return "BarnesHut";
    case AlgorithmEnum::BarnesHutInverted:
        return "BarnesHutInverted";
    case AlgorithmEnum::BarnesHutLocationAware:
        return "BarnesHutLocationAware";
    case AlgorithmEnum::FastMultipoleMethod:
        return "FastMultipoleMethod";
    case AlgorithmEnum::CombinedAlgorithms:
        return "CombinedAlgorithms";

    default:
        return "";
    }
}

/**
 * @brief Pretty-prints the algorithm to the chosen stream
 * @param out The stream to which to print the algorithm
 * @param algorithm_enum The algorithm to print
 * @return The argument out, now altered with the algorithm
 */
inline std::ostream& operator<<(std::ostream& out, const AlgorithmEnum& algorithm_enum) {
    return out << stringify(algorithm_enum);
}

template <>
struct fmt::formatter<AlgorithmEnum> : ostream_formatter { };
