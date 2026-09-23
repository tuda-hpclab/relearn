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

#include <fmt/ostream.h>

#include <ostream>

/**
 * This class indicates the different types of background.
 * There might be none, a constant one, a normally distributed one,
 * or a normally distributed one with predrawn values.
 */
enum class BackgroundActivityCalculatorType : char {
    Null,
    Constant,
    Normal,
    FastNormal,
    Flexible,
};

/**
 * @brief Pretty-prints the background activity calculator type to the chosen stream
 * @param out The stream to which to print the background activity
 * @param calculator_type The background activity to print
 * @return The argument out, now altered with the background activity
 */
inline std::ostream& operator<<(std::ostream& out, const BackgroundActivityCalculatorType& calculator_type) {
    if (calculator_type == BackgroundActivityCalculatorType::Null) {
        return out << "Null";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Constant) {
        return out << "Constant";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Normal) {
        return out << "Normal";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Normal) {
        return out << "FastNormal";
    }

    if (calculator_type == BackgroundActivityCalculatorType::Normal) {
        return out << "Flexible";
    }

    return out;
}

template <>
struct fmt::formatter<BackgroundActivityCalculatorType> : ostream_formatter { };