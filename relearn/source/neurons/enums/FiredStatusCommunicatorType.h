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
 * This enums lists all types of fired status communicator
 */
enum class FiredStatusCommunicatorType : char {
    Map,
    Approximator,
};

/**
 * @brief Pretty-prints the fired status communicator type to the chosen stream
 * @param out The stream to which to print the fired status communicator
 * @param calculator_type The fired status communicator to print
 * @return The argument out, now altered with the fired status communicator
 */
inline std::ostream& operator<<(std::ostream& out, const FiredStatusCommunicatorType& calculator_type) {
    if (calculator_type == FiredStatusCommunicatorType::Map) {
        return out << "Map";
    }

    if (calculator_type == FiredStatusCommunicatorType::Approximator) {
        return out << "Approximator";
    }

    return out;
}

template <>
struct fmt::formatter<FiredStatusCommunicatorType> : ostream_formatter { };
