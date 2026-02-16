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

/**
 * This enums lists all types of growth rate calculators
 */
enum class GrowthrateCalculatorType : char {
    Constant,
};

/**
 * @brief Pretty-prints the growth rate calculator type to the chosen stream
 * @param out The stream to which to print the growth rate calculator
 * @param calculator_type The growth rate calculator to print
 * @return The argument out, now altered with the growth rate calculator
 */
inline std::ostream& operator<<(std::ostream& out, const GrowthrateCalculatorType& calculator_type) {
    if (calculator_type == GrowthrateCalculatorType::Constant) {
        return out << "Constant";
    }

    return out;
}

template <>
struct fmt::formatter<GrowthrateCalculatorType> : ostream_formatter { };
