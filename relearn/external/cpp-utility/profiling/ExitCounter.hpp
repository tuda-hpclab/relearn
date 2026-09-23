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
#include <string_view>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief Counts how often a function was called and, in addition, how often it was left through each of its exits.
 *
 * Constructing an ExitCounter counts one call. Before returning, label the taken exit with set_exit; if no exit was
 * set, the destructor records the default exit. This distinguishes, e.g., the early-return path from the normal path.
 * All counters with the same key aggregate into one entry (see CallCounter for the key rules).
 *
 * Example:
 * @code
 *     int lookup(int x) {
 *         auto counter = ExitCounter{ "lookup" };
 *         if (x < 0) {
 *             counter.set_exit("invalid-argument");
 *             return -1;
 *         }
 *         return x;  // destructor records the "default" exit
 *     }
 * @endcode
 */
class ExitCounter {
public:
    /** @brief The exit label recorded when no exit was set explicitly */
    static constexpr std::string_view default_exit_label = "default";

    /**
     * @brief Constructs an anonymous counter (source-location key) and counts one call
     * @param location The construction source location (defaulted, do not pass explicitly)
     */
    explicit ExitCounter(const std::source_location location = std::source_location::current())
        : ExitCounter(ProfilingOptions{}, location) {
    }

    /**
     * @brief Constructs a counter with the given key and counts one call
     * @param key The key, must be in kebab-case
     * @exception Throws an Exception if the key is not in kebab-case
     */
    explicit ExitCounter(std::string key)
        : ExitCounter(ProfilingOptions{ .key = std::move(key) }, std::source_location::current()) {
    }

    /**
     * @brief Constructs a counter with the given options and counts one call
     * @param options The options (key, message, parent)
     * @param location The construction source location (defaulted, do not pass explicitly)
     * @exception Throws an Exception if a provided key/parent is not in kebab-case or conflicts with an existing entry
     */
    explicit ExitCounter(ProfilingOptions options, const std::source_location location = std::source_location::current())
        : key_{ registry_.register_entry(std::move(options), location) } {
        registry_.obtain_data(key_).call_count++;
    }

    ExitCounter(const ExitCounter&) = delete;
    ExitCounter& operator=(const ExitCounter&) = delete;
    ExitCounter(ExitCounter&&) = delete;
    ExitCounter& operator=(ExitCounter&&) = delete;

    /**
     * @brief Records the taken exit for this call, i.e., the stored label or the default exit if none was set.
     *      Allocation failures are suppressed so instrumentation cannot terminate stack unwinding
     */
    ~ExitCounter() noexcept {
        try {
            const auto label = exit_label_.empty() ? std::string{ default_exit_label } : exit_label_;

            auto& data = registry_.obtain_data(key_);
            for (auto& [existing_label, existing_count] : data.exit_counts) {
                if (existing_label == label) {
                    existing_count++;
                    return;
                }
            }
            data.exit_counts.emplace_back(label, std::uint64_t{ 1 });
        } catch (...) {
            // Profiling is best-effort and must not make the instrumented program fail during destruction.
        }
    }

    /**
     * @brief Sets the exit label that the destructor will record. Calling it repeatedly keeps the last label
     * @param label The exit label, must be in kebab-case
     * @exception Throws an Exception if the label is not in kebab-case
     */
    void set_exit(std::string label) {
        Exception::check(detail::is_kebab_case(label), "ExitCounter::set_exit: the exit label '{}' is not in kebab-case", label);
        exit_label_ = std::move(label);
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
    [[nodiscard]] static std::uint64_t get_call_count(const std::string& key) {
        return registry_.get_checked_entry(key).data.call_count;
    }

    /**
     * @brief Returns how often the given exit was taken for the given key
     * @param key The key
     * @param exit_label The exit label
     * @exception Throws an Exception if there is no entry with the given key
     * @return The number of times the exit was taken, or 0 if the exit was never recorded
     */
    [[nodiscard]] static std::uint64_t get_exit_count(const std::string& key, const std::string& exit_label) {
        const auto& data = registry_.get_checked_entry(key).data;
        for (const auto& [label, count] : data.exit_counts) {
            if (label == exit_label) {
                return count;
            }
        }
        return 0;
    }

    /**
     * @brief Returns the number of recorded entries (distinct keys)
     * @return The number of entries
     */
    [[nodiscard]] static std::size_t get_number_of_entries() {
        return registry_.size();
    }

    /**
     * @brief Renders the exit-counter report as an indented tree
     * @exception Throws an Exception if a parent key was never registered or if the parent links contain a cycle
     * @return The report, ending with a newline
     */
    [[nodiscard]] static std::string format_report() {
        return registry_.format_report("ExitCounter", [](const std::string&, const ExitData& data, const ExitData*) {
            auto exits = std::string{};
            for (const auto& [label, count] : data.exit_counts) {
                if (!exits.empty()) {
                    exits += ", ";
                }
                exits += fmt::format("{}: {}", label, count);
            }
            return fmt::format("{} calls, exits: [{}]", data.call_count, exits);
        });
    }

    /**
     * @brief Prints the exit-counter report to the given stream
     * @param output_stream The stream to print to
     */
    static void print_report(std::ostream& output_stream) {
        output_stream << format_report();
    }

    /**
     * @brief Prints the exit-counter report to the standard output
     */
    static void print_report() {
        fmt::print("{}", format_report());
    }

private:
    struct ExitData {
        std::uint64_t call_count{ 0 };
        std::vector<std::pair<std::string, std::uint64_t>> exit_counts{};
    };

    static inline detail::ProfilingRegistry<ExitData> registry_{};

    std::string key_;
    std::string exit_label_{};
};

} // namespace utility
