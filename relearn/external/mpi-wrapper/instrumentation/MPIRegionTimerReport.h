#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/reductions/MPIComponentwiseReductions.h"

#include <cpp-utility/profiling/RegionTimer.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace mpiPP {

/**
 * @brief Reduces a utility::RegionTimer report across the ranks of a communicator and prints it on the root rank.
 *
 * utility::RegionTimer itself stays MPI-free so it can also be used outside distributed programs; this class adds
 * the rank-spanning reduction on top, reusing MPIReductions::reduce_componentwise_sum/max (which is why every
 * per-key reduction happens in one collective call instead of one call per key).
 *
 * Every participating rank must have registered the same timer entries in the same order (true for an SPMD
 * program that instruments the same code path on every rank), since entries are matched across ranks by their
 * position in utility::RegionTimer::get_all_entries(), not by key. A differing number of entries across ranks
 * throws.
 */
class MPIRegionTimerReport {
public:
    /**
     * @brief Reduces the total time of every utility::RegionTimer entry across the communicator (sum, turned into
     *      the average, and separately the maximum) and prints a table with one row per key on the root rank.
     *
     * The reduction is collective and always runs, even while printing is disabled via set_disable_status(); only
     * the root's output is suppressed in that case, so every rank must still call this in lockstep regardless.
     * @param title The title printed above the table
     * @param communicator The communicator to reduce within, defaults to the world communicator
     * @param output_stream The stream the table is printed to on the root rank; must outlive the call
     * @exception Throws an Exception if the number of entries differs across ranks or if mpi reports an error
     */
    static void reduce_and_print(const std::string_view title, const MPICommunicator& communicator = MPICommunicator::World, std::ostream& output_stream = std::cout) {
        const auto entries = utility::RegionTimer::get_all_entries();

        auto local_times_ms = std::vector<double>(entries.size());
        for (auto index = std::size_t{ 0 }; index < entries.size(); index++) {
            local_times_ms[index] = std::chrono::duration<double, std::milli>(entries[index].total_time).count();
        }

        const auto sum_times_ms = MPIReductions::reduce_componentwise_sum(local_times_ms, communicator);
        const auto max_times_ms = MPIReductions::reduce_componentwise_max(local_times_ms, communicator);

        if (disabled_ || !communicator.is_root_rank()) {
            return;
        }

        const auto number_ranks = static_cast<double>(communicator.get_number_ranks());

        output_stream << fmt::format("{}\n{}\n", title, std::string(title.size(), '-'));
        output_stream << fmt::format("{:<32} {:>14} {:>14}\n", "key", "avg [ms]", "max [ms]");
        for (auto index = std::size_t{ 0 }; index < entries.size(); index++) {
            output_stream << fmt::format("{:<32} {:>14.3f} {:>14.3f}\n", entries[index].key, sum_times_ms[index] / number_ranks, max_times_ms[index]);
        }
    }

    /**
     * @brief Enables or disables printing for every reduce_and_print() call at once. The collective reduction
     *      itself always runs regardless; only the root rank's output is suppressed while disabled
     * @param disabled True to disable printing
     */
    static void set_disable_status(const bool disabled) noexcept {
        disabled_ = disabled;
    }

private:
    static inline bool disabled_{ false };
};

} // namespace mpiPP
