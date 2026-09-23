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
 * An instance of this enum symbolizes if the calcium calculator type
 */
enum class CalciumCalculatorType : char {
    Normal = 0,
    AbsoluteDecay = 1,
    RelativeDecay = 2,
};

/**
 * @brief Pretty-prints the target calcium calculator to the chosen stream
 * @param out The stream to which to print the target calcium calculator
 * @param calculator_type The target calcium calculator to print
 * @return The argument out, now altered with the target calcium calculator
 */
inline std::ostream& operator<<(std::ostream& out, const CalciumCalculatorType calculator_type) {
    if (calculator_type == CalciumCalculatorType::Normal) {
        return out << "Normal";
    }

    if (calculator_type == CalciumCalculatorType::AbsoluteDecay) {
        return out << "AbsoluteDecay";
    }

    if (calculator_type == CalciumCalculatorType::RelativeDecay) {
        return out << "RelativeDecay";
    }

    return out;
}

template <>
struct fmt::formatter<CalciumCalculatorType> : ostream_formatter { };
