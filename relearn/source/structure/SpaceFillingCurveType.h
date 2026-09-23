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

#include <fmt/ostream.h>

#include <cstdint>
#include <ostream>

/**
 * This enum reflects the different space-filling curves, it must
 * be kept in sync with the classes to allow a seamless integration
 */
enum class SpaceFillingCurveType : std::uint8_t {
    HilbertCurve,
    MortonCurve,
};

/**
 * @brief Pretty-prints the space-filling curve type to the chosen stream
 * @param out The stream to which to print the space-filling curve type
 * @param curve_type The space-filling curve type to print
 * @return The argument out, now altered with the space-filling curve type
 */
inline std::ostream& operator<<(std::ostream& out, const SpaceFillingCurveType& curve_type) {
    switch (curve_type) {
    case SpaceFillingCurveType::HilbertCurve:
        return out << "Hilbert";
    case SpaceFillingCurveType::MortonCurve:
        return out << "Morton";
    }

    return out << "UNKNOWN";
}

template <>
struct fmt::formatter<SpaceFillingCurveType> : ostream_formatter { };
