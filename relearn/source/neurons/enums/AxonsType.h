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
 * An instance of this enum symbolizes if the axons type
 */
enum class AxonsType : char {
    Normal = 0,
    MultiPosition = 1,
};

/**
 * @brief Pretty-prints the axon type to the chosen stream
 * @param out The stream to which to print the axon type
 * @param axons_type The axon type  to print
 * @return The argument out, now altered with the axon type
 */
inline std::ostream& operator<<(std::ostream& out, const AxonsType axons_type) {
    if (axons_type == AxonsType::Normal) {
        return out << "Normal";
    }

    if (axons_type == AxonsType::MultiPosition) {
        return out << "MultiPosition";
    }

    return out;
}

template <>
struct fmt::formatter<AxonsType> : ostream_formatter { };
