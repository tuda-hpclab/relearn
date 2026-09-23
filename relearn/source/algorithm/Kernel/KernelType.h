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
 * This enum reflects the different probability kernels, it must
 * be kept in sync with the classes to allow a seamless integration
 */
enum class KernelType : std::uint8_t {
    Gaussian,
    Linear,
    Gamma,
    Weibull
};

/**
 * @brief Pretty-prints the kernel type to the chosen stream
 * @param out The stream to which to print the kernel type
 * @param kernel_type The kernel type to print
 * @return The argument out, now altered with the kernel type
 */
inline std::ostream& operator<<(std::ostream& out, const KernelType& kernel_type) {
    switch (kernel_type) {
    case KernelType::Gamma:
        return out << "Gamma";
    case KernelType::Gaussian:
        return out << "Gaussian";
    case KernelType::Linear:
        return out << "Linear";
    case KernelType::Weibull:
        return out << "Weibull";
    }

    return out << "UNKNOWN";
}

template <>
struct fmt::formatter<KernelType> : ostream_formatter { };
