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

#include <cstdint>
#include <ostream>
#include <source_location>
#include <string>
#include <utility>

namespace utility {

/**
 * @brief Counts how often a function (or any region) was reached.
 *
 * The default behavior is automatic: constructing a CallCounter counts one call, so placing one at the
 * top of a function counts every call by RAII. For manual counting use the CallCounter::manual factories,
 * which register the entry but do not count on construction; call count() for each event yourself.
 *
 * All counters with the same key aggregate into one entry. Without a key, a key is derived from the
 * construction source location so repeated calls at the same site still share one entry.
 *
 * Example:
 * @code
 *     void process() {
 *         const auto counter = CallCounter{ "process-calls" };  // counts on construction
 *         // ...
 *     }
 * @endcode
 */
class CallCounter {
public:
    /**
     * @brief Constructs an anonymous counter (source-location key) and counts one call
     * @param location The construction source location (defaulted, do not pass explicitly)
     */
    explicit CallCounter(const std::source_location location = std::source_location::current())
        : CallCounter(ProfilingOptions{}, location, true) {
    }

    /**
     * @brief Constructs a counter with the given key and counts one call
     * @param key The key, must be in kebab-case
     * @exception Throws an Exception if the key is not in kebab-case
     */
    explicit CallCounter(std::string key)
        : CallCounter(ProfilingOptions{ .key = std::move(key) }, std::source_location::current(), true) {
    }

    /**
     * @brief Constructs a counter with the given options and counts one call
     * @param options The options (key, message, parent)
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @exception Throws an Exception if a provided key/parent is not in kebab-case or conflicts with an existing entry
     */
    explicit CallCounter(ProfilingOptions options, const std::source_location location = std::source_location::current())
        : CallCounter(std::move(options), location, true) {
    }

    CallCounter(const CallCounter&) = delete;
    CallCounter& operator=(const CallCounter&) = delete;
    CallCounter(CallCounter&&) = delete;
    CallCounter& operator=(CallCounter&&) = delete;
    ~CallCounter() = default;

    /**
     * @brief Creates an anonymous counter (source-location key) that does not count on construction
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @return The manual counter
     */
    [[nodiscard]] static CallCounter manual(const std::source_location location = std::source_location::current()) {
        return CallCounter(ProfilingOptions{}, location, false);
    }

    /**
     * @brief Creates a counter with the given key that does not count on construction
     * @param key The key, must be in kebab-case
     * @exception Throws an Exception if the key is not in kebab-case
     * @return The manual counter
     */
    [[nodiscard]] static CallCounter manual(std::string key) {
        return CallCounter(ProfilingOptions{ .key = std::move(key) }, std::source_location::current(), false);
    }

    /**
     * @brief Creates a counter with the given options that does not count on construction
     * @param options The options (key, message, parent)
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @exception Throws an Exception if a provided key/parent is not in kebab-case or conflicts with an existing entry
     * @return The manual counter
     */
    [[nodiscard]] static CallCounter manual(ProfilingOptions options, const std::source_location location = std::source_location::current()) {
        return CallCounter(std::move(options), location, false);
    }

    /**
     * @brief Counts one call. Valid on any counter, also on one that already counted on construction
     */
    void count() {
        registry_.obtain_data(key_).count++;
    }

    /**
     * @brief Returns the resolved key of this counter
     * @return A constant reference to the key
     */
    [[nodiscard]] const std::string& get_key() const noexcept {
        return key_;
    }

    /**
     * @brief Removes all recorded counters
     */
    static void init() {
        registry_.reset();
    }

    /**
     * @brief Returns the number of calls recorded for the given key
     * @param key The key
     * @exception Throws an Exception if there is no entry with the given key
     * @return The number of calls
     */
    [[nodiscard]] static std::uint64_t get_count(const std::string& key) {
        return registry_.get_checked_entry(key).data.count;
    }

    /**
     * @brief Returns the number of recorded entries (distinct keys)
     * @return The number of entries
     */
    [[nodiscard]] static std::size_t get_number_of_entries() {
        return registry_.size();
    }

    /**
     * @brief Renders the call-counter report as an indented tree
     * @exception Throws an Exception if a parent key was never registered or if the parent links contain a cycle
     * @return The report, ending with a newline
     */
    [[nodiscard]] static std::string format_report() {
        return registry_.format_report("CallCounter", [](const std::string&, const CallData& data, const CallData*) {
            return fmt::format("{} calls", data.count);
        });
    }

    /**
     * @brief Prints the call-counter report to the given stream
     * @param output_stream The stream to print to
     */
    static void print_report(std::ostream& output_stream) {
        output_stream << format_report();
    }

    /**
     * @brief Prints the call-counter report to the standard output
     */
    static void print_report() {
        fmt::print("{}", format_report());
    }

private:
    struct CallData {
        std::uint64_t count{ 0 };
    };

    CallCounter(ProfilingOptions options, const std::source_location& location, const bool count_on_construction)
        : key_{ registry_.register_entry(std::move(options), location) } {
        if (count_on_construction) {
            registry_.obtain_data(key_).count++;
        }
    }

    static inline detail::ProfilingRegistry<CallData> registry_{};

    std::string key_;
};

} // namespace utility
