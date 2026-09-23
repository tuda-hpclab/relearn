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

#include <fmt/format.h>

#include <concepts>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

namespace utility {

namespace detail {

/**
 * @brief Global switch that silences every BasicStatus instantiation at once. Not thread-safe.
 */
inline bool status_reporting_disabled = false;

/**
 * @brief The requirements a reporter must satisfy so that BasicStatus can determine the global amount of work
 *      and decide which participant emits the output.
 *
 * A participant is one contributor to the same computation, e.g. one MPI rank or one worker. Both operations
 * should be usable on a const reporter. T is the reporter type.
 */
template <typename T>
concept status_reporter = requires(const T reporter, std::uint64_t value) {
    { reporter.reduce_sum(value) } -> std::convertible_to<std::uint64_t>;
    { reporter.is_root() } -> std::convertible_to<bool>;
};

} // namespace detail

/**
 * @brief The default reporter for a single-process computation: there is exactly one participant, so the
 *      reduction returns its argument unchanged and that participant is always the root.
 *
 * A distributed program (e.g. one based on MPI) provides its own reporter instead. Such an adapter forwards
 * the operations to its communication layer, for example:
 * @code
 *     struct MPIStatusReporter {
 *         [[nodiscard]] std::uint64_t reduce_sum(std::uint64_t v) const { return mpiPP::MPIReductions::reduce_sum(v); }
 *         [[nodiscard]] bool is_root() const { return mpiPP::MPIInfo::is_root_rank(); }
 *     };
 *     using MPIStatus = utility::BasicStatus<MPIStatusReporter>;
 * @endcode
 */
struct LocalStatusReporter {
    /**
     * @brief Returns the sum of the value across all participants; here just the value itself
     * @param value The local value
     * @return The value unchanged
     */
    [[nodiscard]] std::uint64_t reduce_sum(const std::uint64_t value) const noexcept {
        return value;
    }

    /**
     * @brief Returns whether this participant emits the output; the single process always does
     * @return Always true
     */
    [[nodiscard]] bool is_root() const noexcept {
        return true;
    }
};

/**
 * @brief A common interface to report the progress of a computation. The output is throttled to roughly one
 *      line per percent of progress (and at most about 100 lines) and is emitted only by the root participant.
 *
 * The class is agnostic about how many participants cooperate on the computation: a Reporter abstracts both the
 * reduction of the total amount of work and the decision which participant prints. The default LocalStatusReporter
 * covers the single-process case; a distributed program plugs in its own reporter (see LocalStatusReporter).
 *
 * @par Collective safety
 * The only cross-participant reduction happens in the constructor, which every participant calls exactly once,
 * so it is always reached in lockstep. report() and finish() never communicate. This is deliberate: the
 * participants are not required to run the same number of iterations, so a reduction inside report() could not
 * be reached consistently and would deadlock a distributed reporter as soon as one participant finished early.
 * Because report() needs no collective, the exact global running total is not available while the computation
 * is in progress; the root participant therefore estimates it from its own progress. The estimate is monotonic
 * and reaches the exact global total (which itself is exact, from the constructor) once the root has finished.
 *
 * @tparam Reporter The reporter that performs the reduction and identifies the root, see detail::status_reporter
 *
 * Example:
 * @code
 *     auto status = utility::Status{ number_of_iterations, "bfs" };
 *     for (auto i = std::uint64_t{ 0 }; i < number_of_iterations; ++i) {
 *         // ... work ...
 *         status.report(i);
 *     }
 *     status.finish();
 * @endcode
 */
template <typename Reporter = LocalStatusReporter>
    requires detail::status_reporter<Reporter>
class BasicStatus {
public:
    /**
     * @brief Constructs the status reporter. Must be called by every participant, as it reduces the iteration
     *      counts across all of them to obtain the global amount of work.
     * @param number_iterations The number of local iterations of this participant
     * @param algorithm_name The name of the algorithm, used as a prefix in the output
     * @param reporter The reporter performing the reduction and identifying the root participant
     * @param output_stream The stream the output is written to; it must outlive this object
     */
    BasicStatus(const std::uint64_t number_iterations, std::string algorithm_name, Reporter reporter = Reporter{}, std::ostream& output_stream = std::cout)
        : reporter_(std::move(reporter))
        , output_stream_(&output_stream)
        , algo_name_(std::move(algorithm_name))
        , num_local_iterations_(number_iterations)
        , num_global_iterations_(reporter_.reduce_sum(number_iterations)) {
    }

    /**
     * @brief Reports that one more local iteration has been completed and prints the throttled progress line.
     *      Performs no communication, so the participants may call it a different number of times.
     * @param current_iteration The number of completed local iterations. Currently informational; the progress
     *      is tracked internally so that the parameter can carry the caller's iteration index unchanged.
     */
    void report([[maybe_unused]] const std::uint64_t current_iteration) {
        if (detail::status_reporting_disabled) {
            return;
        }

        // Only the root participant prints, and nothing below communicates, so report() is safe to call a
        // different number of times on each participant, which happens when their iteration counts differ.
        if (!reporter_.is_root()) {
            return;
        }

        if (num_local_iterations_ == 0) {
            return;
        }

        num_reported_iterations_++;

        if (const auto print_every = num_local_iterations_ / 100; print_every != 0 && num_reported_iterations_ % print_every != 0) {
            return;
        }

        ++num_printed_iterations_;
        if (num_printed_iterations_ > 101) {
            return;
        }

        // The exact global running total would require a reduction in every reporting step, which cannot be
        // reached consistently once the participants run a different number of iterations. The total is
        // therefore estimated from the root's own progress and rounded to the nearest iteration.
        const auto progress_fraction = static_cast<double>(num_reported_iterations_) / static_cast<double>(num_local_iterations_);
        const auto processed_estimate = static_cast<std::uint64_t>(progress_fraction * static_cast<double>(num_global_iterations_) + 0.5);

        *output_stream_ << fmt::format("{}: Processed a total of {} of {} iterations.\n", algo_name_, processed_estimate, num_global_iterations_);
    }

    /**
     * @brief Finishes the status reporting and prints a final line. Only the root participant emits output.
     *      Performs no communication.
     */
    void finish() const {
        if (detail::status_reporting_disabled) {
            return;
        }

        if (!reporter_.is_root()) {
            return;
        }

        *output_stream_ << fmt::format("Processed all iterations of {}\n", algo_name_);
    }

    /**
     * @brief Enables or disables status reporting for every BasicStatus instantiation at once
     * @param status True iff the status reporting should be disabled
     */
    static void set_disable_status(const bool status) noexcept {
        detail::status_reporting_disabled = status;
    }

private:
    Reporter reporter_;
    std::ostream* output_stream_;
    std::string algo_name_{};
    std::uint64_t num_local_iterations_{};
    std::uint64_t num_global_iterations_{};
    std::uint64_t num_reported_iterations_{};
    std::uint64_t num_printed_iterations_{};
};

/**
 * @brief A status reporter for a single-process computation, the recommended default
 */
using Status = BasicStatus<LocalStatusReporter>;

} // namespace utility
