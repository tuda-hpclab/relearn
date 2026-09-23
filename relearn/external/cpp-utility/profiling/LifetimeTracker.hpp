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

#include "cpp-utility/data-structure/StatisticalMeasures.hpp"
#include "cpp-utility/profiling/ProfilingOptions.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <ostream>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief Records how long each individual object lived, i.e., stores the lifetime of every tracked object separately.
 *
 * The tracker starts on construction and, on destruction, appends the elapsed lifetime to the entry of its key.
 * Unlike RegionTimer, which accumulates one total per key, this keeps every single lifetime so the report can show
 * statistics (count, min, max, average, variance, standard deviation) over all objects of a key.
 *
 * @tparam Clock The clock to use, must satisfy detail::clock_like. Use the LifetimeTracker alias for the default clock
 *
 * Example:
 * @code
 *     struct Particle {
 *         BasicLifetimeTracker<std::chrono::steady_clock> tracker_{ std::string{ "particle" } };
 *         // ...
 *     };
 * @endcode
 */
template <typename Clock>
    requires detail::clock_like<Clock>
class BasicLifetimeTracker {
public:
    /**
     * @brief Constructs an anonymous tracker (source-location key) and starts measuring its lifetime
     * @param location The construction source location (defaulted, do not pass explicitly)
     */
    explicit BasicLifetimeTracker(const std::source_location location = std::source_location::current())
        : BasicLifetimeTracker(ProfilingOptions{}, location) {
    }

    /**
     * @brief Constructs a tracker with the given key and starts measuring its lifetime
     * @param key The key, must be in kebab-case
     * @exception Throws an Exception if the key is not in kebab-case
     */
    explicit BasicLifetimeTracker(std::string key)
        : BasicLifetimeTracker(ProfilingOptions{ .key = std::move(key) }, std::source_location::current()) {
    }

    /**
     * @brief Constructs a tracker with the given options and starts measuring its lifetime
     * @param options The options (key, message, parent)
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @exception Throws an Exception if a provided key/parent is not in kebab-case or conflicts with an existing entry
     * @exception Propagates exceptions from Clock::now()
     */
    explicit BasicLifetimeTracker(ProfilingOptions options, const std::source_location location = std::source_location::current())
        : key_{ registry_.register_entry(std::move(options), location) }
        , birth_time_{ Clock::now() } {
    }

    BasicLifetimeTracker(const BasicLifetimeTracker&) = delete;
    BasicLifetimeTracker& operator=(const BasicLifetimeTracker&) = delete;
    BasicLifetimeTracker(BasicLifetimeTracker&&) = delete;
    BasicLifetimeTracker& operator=(BasicLifetimeTracker&&) = delete;

    /**
     * @brief Records the elapsed lifetime of this object. Clock/allocation failures are suppressed so instrumentation
     *      cannot terminate stack unwinding
     */
    ~BasicLifetimeTracker() noexcept {
        try {
            const auto lifetime = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - birth_time_);
            registry_.obtain_data(key_).lifetimes.push_back(lifetime);
        } catch (...) {
            // Profiling is best-effort and must not make the instrumented program fail during destruction.
        }
    }

    /**
     * @brief Returns the resolved key of this tracker
     * @return A constant reference to the key
     */
    [[nodiscard]] const std::string& get_key() const noexcept {
        return key_;
    }

    /**
     * @brief Removes all recorded lifetimes
     */
    static void init() {
        registry_.reset();
    }

    /**
     * @brief Returns the individual lifetimes recorded for the given key, in the order they were recorded
     * @param key The key
     * @exception Throws an Exception if there is no entry with the given key
     * @return A view of the recorded lifetimes; it is invalidated when another lifetime is appended or init() is called
     */
    [[nodiscard]] static std::span<const std::chrono::nanoseconds> get_lifetimes(const std::string& key) {
        return std::span<const std::chrono::nanoseconds>{ registry_.get_checked_entry(key).data.lifetimes };
    }

    /**
     * @brief Returns the number of recorded entries (distinct keys)
     * @return The number of entries
     */
    [[nodiscard]] static std::size_t get_number_of_entries() {
        return registry_.size();
    }

    /**
     * @brief Renders the lifetime report as an indented tree. Each entry shows the number of lifetimes and,
     *      if there is at least one, their statistics in a unit chosen from the longest lifetime
     * @exception Throws an Exception if a parent key was never registered or if the parent links contain a cycle
     * @return The report, ending with a newline
     */
    [[nodiscard]] static std::string format_report() {
        return registry_.format_report("LifetimeTracker", [](const std::string&, const LifetimeData& data, const LifetimeData*) {
            if (data.lifetimes.empty()) {
                return std::string{ "0 lifetimes" };
            }

            const auto maximum = *std::ranges::max_element(data.lifetimes);
            const auto [divisor, unit] = detail::pick_time_unit(maximum);

            auto values_in_unit = std::vector<double>{};
            values_in_unit.reserve(data.lifetimes.size());
            for (const auto lifetime : data.lifetimes) {
                values_in_unit.push_back(static_cast<double>(lifetime.count()) / divisor);
            }

            const auto measures = StatisticalMeasures::calculate(values_in_unit);
            return fmt::format("{} lifetimes in {}: {:.2f}", data.lifetimes.size(), unit, measures);
        });
    }

    /**
     * @brief Prints the lifetime report to the given stream
     * @param output_stream The stream to print to
     */
    static void print_report(std::ostream& output_stream) {
        output_stream << format_report();
    }

    /**
     * @brief Prints the lifetime report to the standard output
     */
    static void print_report() {
        fmt::print("{}", format_report());
    }

private:
    struct LifetimeData {
        std::vector<std::chrono::nanoseconds> lifetimes{};
    };

    static inline detail::ProfilingRegistry<LifetimeData> registry_{};

    std::string key_;
    typename Clock::time_point birth_time_{};
};

/**
 * @brief A lifetime tracker using the steady clock, the recommended default for wall-clock measurements
 */
using LifetimeTracker = BasicLifetimeTracker<std::chrono::steady_clock>;

} // namespace utility
