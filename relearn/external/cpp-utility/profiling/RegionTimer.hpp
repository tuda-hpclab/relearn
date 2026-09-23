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

#include "cpp-utility/profiling/ProfilingOptions.hpp"

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <ostream>
#include <source_location>
#include <string>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A flat, per-entry snapshot of a BasicRegionTimer registry, as returned by get_all_entries()
 */
struct RegionTimerEntry {
    /** @brief The key of the entry */
    std::string key;
    /** @brief The accumulated time recorded for this entry */
    std::chrono::nanoseconds total_time;
    /** @brief How often the timer of this entry was activated (started and stopped) */
    std::uint64_t activation_count;
    /** @brief The key of the parent entry, or empty if this is a root entry */
    std::string parent_key;
};

/**
 * @brief Measures how long a region (the lifetime of the timer object) took, accumulating the time per key.
 *
 * The timer starts on construction and stops on destruction, so a timer placed in a scope measures that scope.
 * Call stop() to end the measurement early; the destructor then records nothing more. Timers with the same key
 * accumulate their durations and count as separate activations, so a timer in a loop sums the per-iteration times.
 *
 * A timer can be declared a subregion of another timer via the parent key; the report then shows the child
 * indented under its parent together with its share of the parent's total time.
 *
 * @tparam Clock The clock to use, must satisfy detail::clock_like. Use the RegionTimer alias for the default clock
 *
 * Example:
 * @code
 *     void step() {
 *         const auto timer = RegionTimer{ "step" };
 *         // ... work ...
 *     }  // the elapsed time is recorded here
 * @endcode
 */
template <typename Clock>
    requires detail::clock_like<Clock>
class BasicRegionTimer {
public:
    /**
     * @brief Constructs an anonymous timer (source-location key) and starts measuring
     * @param location The construction source location (defaulted, do not pass explicitly)
     */
    explicit BasicRegionTimer(const std::source_location location = std::source_location::current())
        : BasicRegionTimer(ProfilingOptions{}, location) {
    }

    /**
     * @brief Constructs a timer with the given key and starts measuring
     * @param key The key, must be in kebab-case
     * @exception Throws an Exception if the key is not in kebab-case
     */
    explicit BasicRegionTimer(std::string key)
        : BasicRegionTimer(ProfilingOptions{ .key = std::move(key) }, std::source_location::current()) {
    }

    /**
     * @brief Constructs a timer with the given options and starts measuring.
     *      If timers are currently disabled (see set_enabled), the entry is not registered and nothing is
     *      recorded on destruction or stop(), regardless of the enabled status at that later point in time
     * @param options The options (key, message, parent)
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @exception Throws an Exception if a provided key/parent is not in kebab-case or conflicts with an existing entry
     * @exception Propagates exceptions from Clock::now()
     */
    explicit BasicRegionTimer(ProfilingOptions options, const std::source_location location = std::source_location::current())
        : key_{ enabled_ ? registry_.register_entry(std::move(options), location) : std::string{} }
        , start_time_{ enabled_ ? Clock::now() : typename Clock::time_point{} } {
    }

    BasicRegionTimer(const BasicRegionTimer&) = delete;
    BasicRegionTimer& operator=(const BasicRegionTimer&) = delete;
    BasicRegionTimer(BasicRegionTimer&&) = delete;
    BasicRegionTimer& operator=(BasicRegionTimer&&) = delete;

    /**
     * @brief Records the elapsed time if the timer is still running. Clock/allocation failures are suppressed so
     *      instrumentation cannot terminate stack unwinding
     */
    ~BasicRegionTimer() noexcept {
        try {
            if (running_) {
                record();
            }
        } catch (...) {
            // Profiling is best-effort and must not make the instrumented program fail during destruction.
        }
    }

    /**
     * @brief Stops the timer early and records the elapsed time. The destructor then records nothing more
     * @exception Throws an Exception if the timer was already stopped
     * @exception Propagates exceptions from Clock::now() or from recreating an entry after init()
     */
    void stop() {
        Exception::check(running_, "BasicRegionTimer::stop: the timer was already stopped");
        record();
        running_ = false;
    }

    /**
     * @brief Returns whether the timer is still running (has not been stopped yet)
     * @return True iff the timer is running
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_;
    }

    /**
     * @brief Returns the resolved key of this timer, or an empty string if timers were disabled on construction
     * @return A constant reference to the key
     */
    [[nodiscard]] const std::string& get_key() const noexcept {
        return key_;
    }

    /**
     * @brief Removes all recorded timers
     */
    static void init() {
        registry_.reset();
    }

    /**
     * @brief Enables or disables every BasicRegionTimer<Clock> instantiation of this Clock at once.
     *      While disabled, newly constructed timers skip both the registry and Clock::now(), so measurements
     *      can be skipped cheaply (e.g., on non-reporting participants of a distributed run). Timers already
     *      under way keep the enabled status they had on construction
     * @param enabled True to enable (the default), false to disable
     */
    static void set_enabled(const bool enabled) noexcept {
        enabled_ = enabled;
    }

    /**
     * @brief Returns whether newly constructed timers currently record measurements
     * @return True iff timers are enabled
     */
    [[nodiscard]] static bool is_enabled() noexcept {
        return enabled_;
    }

    /**
     * @brief Returns the accumulated time recorded for the given key
     * @param key The key
     * @exception Throws an Exception if there is no entry with the given key
     * @return The accumulated time
     */
    [[nodiscard]] static std::chrono::nanoseconds get_total_time(const std::string& key) {
        return registry_.get_checked_entry(key).data.total_time;
    }

    /**
     * @brief Returns how often the timer of the given key was activated (started and stopped)
     * @param key The key
     * @exception Throws an Exception if there is no entry with the given key
     * @return The number of activations
     */
    [[nodiscard]] static std::uint64_t get_activations(const std::string& key) {
        return registry_.get_checked_entry(key).data.activation_count;
    }

    /**
     * @brief Returns the number of recorded entries (distinct keys)
     * @return The number of entries
     */
    [[nodiscard]] static std::size_t get_number_of_entries() {
        return registry_.size();
    }

    /**
     * @brief Returns a flat snapshot of every currently registered entry, without keys needing to be known upfront.
     *      Entries appear in registration order (the same order format_report() sorts roots and siblings by),
     *      as a flat list rather than a resolved parent/child tree, which suits building a custom report
     *      (e.g., reducing entries of the same key across the participants of a distributed run)
     * @return The entries
     */
    [[nodiscard]] static std::vector<RegionTimerEntry> get_all_entries() {
        return registry_.map_entries_in_registration_order(
            [](const std::string& key, const TimerData& data, const std::string& parent_key) {
                return RegionTimerEntry{
                    .key = key,
                    .total_time = data.total_time,
                    .activation_count = data.activation_count,
                    .parent_key = parent_key,
                };
            });
    }

    /**
     * @brief Renders the region-timer report as an indented tree. A subregion additionally shows its share
     *      of the parent's total time
     * @exception Throws an Exception if a parent key was never registered or if the parent links contain a cycle
     * @return The report, ending with a newline
     */
    [[nodiscard]] static std::string format_report() {
        return registry_.format_report("RegionTimer", [](const std::string&, const TimerData& data, const TimerData* parent_data) {
            auto line = fmt::format("{} ({} activations", detail::format_duration(data.total_time), data.activation_count);
            if (parent_data != nullptr && parent_data->total_time.count() > 0) {
                const auto percent = static_cast<double>(data.total_time.count()) / static_cast<double>(parent_data->total_time.count()) * 100.0;
                line += fmt::format(", {:.1f}% of parent", percent);
            }
            line += ')';
            return line;
        });
    }

    /**
     * @brief Prints the region-timer report to the given stream
     * @param output_stream The stream to print to
     */
    static void print_report(std::ostream& output_stream) {
        output_stream << format_report();
    }

    /**
     * @brief Prints the region-timer report to the standard output
     */
    static void print_report() {
        fmt::print("{}", format_report());
    }

private:
    struct TimerData {
        std::chrono::nanoseconds total_time{ 0 };
        std::uint64_t activation_count{ 0 };
    };

    void record() {
        // A key is only empty when the timer was constructed while disabled; a resolved key is never empty
        // (auto-generated source-location keys and user-provided kebab-case keys are both non-empty).
        if (key_.empty()) {
            return;
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time_);
        auto& data = registry_.obtain_data(key_);
        data.total_time += elapsed;
        data.activation_count++;
    }

    static inline detail::ProfilingRegistry<TimerData> registry_{};
    static inline bool enabled_{ true };

    std::string key_;
    typename Clock::time_point start_time_{};
    bool running_{ true };
};

/**
 * @brief A region timer using the steady clock, the recommended default for wall-clock measurements
 */
using RegionTimer = BasicRegionTimer<std::chrono::steady_clock>;

} // namespace utility
