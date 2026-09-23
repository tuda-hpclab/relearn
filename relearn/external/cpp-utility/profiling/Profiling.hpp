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

/**
 * @file
 * @brief Convenience umbrella for the profiling module: pulls in the four instrumentation types
 *      (CallCounter, ExitCounter, RegionTimer, LifetimeTracker) and adds combined init/report helpers.
 *
 * The four types are fully independent and can be used on their own; this header only offers a single
 * entry point for programs that use several of them. Call profiling::init() once at the start and
 * profiling::print_report() at the end of main().
 *
 * @code
 *     #include "cpp-utility/profiling/Profiling.hpp"
 *
 *     using namespace utility;
 *
 *     int compute(int x) {
 *         auto exits = ExitCounter{ "compute" };
 *         const auto timer = RegionTimer{ "compute" };
 *         if (x < 0) {
 *             exits.set_exit("negative");
 *             return 0;
 *         }
 *         return x * x;
 *     }
 *
 *     int main() {
 *         profiling::init();
 *         for (auto i = -2; i < 3; i++) {
 *             const auto counter = CallCounter{ "main-loop" };
 *             compute(i);
 *         }
 *         profiling::print_report();
 *     }
 * @endcode
 */

#include "cpp-utility/profiling/CallCounter.hpp"
#include "cpp-utility/profiling/ExitCounter.hpp"
#include "cpp-utility/profiling/LifetimeTracker.hpp"
#include "cpp-utility/profiling/ProfilingOptions.hpp"
#include "cpp-utility/profiling/RegionTimer.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace utility::profiling {

/**
 * @brief Resets all four profiling types, i.e., discards everything recorded so far.
 *      Call this once before the instrumented run
 */
inline void init() {
    CallCounter::init();
    ExitCounter::init();
    RegionTimer::init();
    LifetimeTracker::init();
}

/**
 * @brief Renders a combined report of the four profiling types (using their default clocks).
 *      Sections appear in a fixed order (CallCounter, ExitCounter, RegionTimer, LifetimeTracker),
 *      separated by a blank line; a type without any entries is omitted entirely
 * @exception Throws an Exception if any type's parent links are inconsistent (see the per-type format_report)
 * @return The combined report, or an empty string if nothing was recorded
 */
[[nodiscard]] inline std::string format_report() {
    auto sections = std::vector<std::string>{};
    if (CallCounter::get_number_of_entries() > 0) {
        sections.push_back(CallCounter::format_report());
    }
    if (ExitCounter::get_number_of_entries() > 0) {
        sections.push_back(ExitCounter::format_report());
    }
    if (RegionTimer::get_number_of_entries() > 0) {
        sections.push_back(RegionTimer::format_report());
    }
    if (LifetimeTracker::get_number_of_entries() > 0) {
        sections.push_back(LifetimeTracker::format_report());
    }

    auto result = std::string{};
    for (auto index = std::size_t{ 0 }; index < sections.size(); index++) {
        if (index > 0) {
            result += '\n';
        }
        result += sections[index];
    }
    return result;
}

/**
 * @brief Prints the combined report to the given stream
 * @param output_stream The stream to print to
 */
inline void print_report(std::ostream& output_stream) {
    output_stream << format_report();
}

/**
 * @brief Prints the combined report to the standard output
 */
inline void print_report() {
    fmt::print("{}", format_report());
}

} // namespace utility::profiling
