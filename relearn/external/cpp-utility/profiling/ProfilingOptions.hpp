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

#include "cpp-utility/Exception.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief Common options that describe an instrumentation object of the profiling module.
 *      All fields are optional. Designated initializers make the call sites terse and self-documenting,
 *      e.g., ProfilingOptions{ .key = "inner-step", .parent = "main-loop" }.
 *
 * An empty key requests an automatically generated key based on the construction source location,
 * so repeated constructions at the same call site (e.g., in a loop) aggregate into a single entry.
 * A non-empty key must be in kebab-case (see detail::is_kebab_case), same for the parent key.
 */
struct ProfilingOptions {
    /** @brief The key that identifies the entry. Empty requests a source-location based key */
    std::string key{};
    /** @brief An optional message that is appended to the entry in the report. Free-form */
    std::string message{};
    /** @brief The key of the parent entry (of the same type) if this entry is a subregion. Empty means a root entry */
    std::string parent{};
};

namespace detail {

/**
 * @brief Constrains a type to a minimal clock interface, i.e., a nested time_point type and a static now().
 *      This is used instead of std::chrono::is_clock_v to keep the requirement portable across standard libraries
 *      and to allow lightweight fake clocks in tests.
 *      Clock is the candidate clock type
 */
template <typename Clock>
concept clock_like = requires {
    typename Clock::time_point;
    { Clock::now() } -> std::same_as<typename Clock::time_point>;
};

/**
 * @brief Checks whether a key is in kebab-case, i.e., matches [a-z0-9]+(-[a-z0-9]+)*.
 *      In words: non-empty, only lowercase letters, digits and single hyphens,
 *      neither starting nor ending with a hyphen and without consecutive hyphens
 * @param key The key to check
 * @return True iff the key is in kebab-case
 */
[[nodiscard]] constexpr bool is_kebab_case(const std::string_view key) noexcept {
    if (key.empty()) {
        return false;
    }

    auto previous_was_hyphen = false;
    for (auto index = std::size_t{ 0 }; index < key.size(); index++) {
        const auto character = key[index];
        const auto is_lower = character >= 'a' && character <= 'z';
        const auto is_digit = character >= '0' && character <= '9';
        const auto is_hyphen = character == '-';

        if (!is_lower && !is_digit && !is_hyphen) {
            return false;
        }
        if (is_hyphen && (index == 0 || index + 1 == key.size() || previous_was_hyphen)) {
            return false;
        }
        previous_was_hyphen = is_hyphen;
    }

    return true;
}

/**
 * @brief Builds a key from a source location, in the format "{file-basename}:{line}"
 * @param location The source location
 * @return The generated key
 */
[[nodiscard]] inline std::string make_source_location_key(const std::source_location& location) {
    const auto full_path = std::string_view{ location.file_name() };
    const auto last_separator = full_path.find_last_of("/\\");
    const auto file_name = (last_separator == std::string_view::npos) ? full_path : full_path.substr(last_separator + 1);
    return fmt::format("{}:{}", file_name, location.line());
}

/**
 * @brief Picks a human-readable time unit for a duration, based on its magnitude
 * @param duration The duration
 * @return A pair of the divisor (from nanoseconds to the chosen unit) and the unit name ("ns", "us", "ms" or "s")
 */
[[nodiscard]] inline std::pair<double, std::string_view> pick_time_unit(const std::chrono::nanoseconds duration) noexcept {
    const auto nanoseconds_count = static_cast<double>(duration.count());
    const auto absolute = nanoseconds_count < 0.0 ? -nanoseconds_count : nanoseconds_count;

    if (absolute < 1'000.0) {
        return { 1.0, "ns" };
    }
    if (absolute < 1'000'000.0) {
        return { 1'000.0, "us" };
    }
    if (absolute < 1'000'000'000.0) {
        return { 1'000'000.0, "ms" };
    }
    return { 1'000'000'000.0, "s" };
}

/**
 * @brief Formats a duration human-readably with an auto-scaled unit, e.g., "500 ns", "1.50 us", "2.50 ms" or "3.00 s".
 *      Nanoseconds are printed as an integer, the coarser units with two decimal places
 * @param duration The duration
 * @return The formatted duration
 */
[[nodiscard]] inline std::string format_duration(const std::chrono::nanoseconds duration) {
    const auto [divisor, unit] = pick_time_unit(duration);
    if (unit == "ns") {
        return fmt::format("{} ns", duration.count());
    }
    return fmt::format("{:.2f} {}", static_cast<double>(duration.count()) / divisor, unit);
}

/**
 * @brief A string-keyed registry of profiling entries. Each profiling type holds its own instance,
 *      so the four types are completely independent and never share state.
 *
 * Entries are aggregated by key: constructing several instrumentation objects with the same key
 * accumulates into one entry. A parent key links an entry to another entry of the same registry;
 * the link is only resolved when the report is rendered, which tolerates forward references.
 * Registry access is not synchronized; instrumentation, reset, and report generation must be externally serialized.
 *
 * @tparam EntryData The per-entry payload of the owning profiling type (e.g., a call count)
 */
template <typename EntryData>
class ProfilingRegistry {
public:
    /**
     * @brief An entry of the registry: the type-agnostic bookkeeping plus the type-specific payload
     */
    struct Entry {
        std::string message{};
        std::string parent_key{};
        std::size_t registration_index{ 0 };
        EntryData data{};
    };

    /**
     * @brief Registers an entry, aggregating with an existing entry of the same key.
     *      Validates a user-provided key and parent (kebab-case) before mutating anything.
     *      A conflicting non-empty message or parent for an existing key throws;
     *      an empty field is filled in by a later non-empty one
     * @param options The options (key, message, parent) of the instrumentation object
     * @param location The construction source location, used to generate a key if none was given
     * @exception Throws an Exception if a provided key/parent is not kebab-case,
     *      or if the message/parent conflicts with an already registered value for the same key
     * @return The resolved key of the entry
     */
    std::string register_entry(ProfilingOptions options, const std::source_location& location) {
        auto key = std::string{};
        if (options.key.empty()) {
            key = make_source_location_key(location);
        } else {
            Exception::check(is_kebab_case(options.key), "ProfilingRegistry::register_entry: the key '{}' is not in kebab-case", options.key);
            key = std::move(options.key);
        }

        if (!options.parent.empty()) {
            Exception::check(is_kebab_case(options.parent), "ProfilingRegistry::register_entry: the parent key '{}' is not in kebab-case", options.parent);
        }

        const auto entry_iterator = entries_.find(key);
        if (entry_iterator == entries_.end()) {
            auto entry = Entry{};
            entry.message = std::move(options.message);
            entry.parent_key = std::move(options.parent);
            entry.registration_index = next_registration_index_;
            entries_.emplace(key, std::move(entry));
            next_registration_index_++;
            return key;
        }

        auto& entry = entry_iterator->second;
        if (!options.message.empty()) {
            Exception::check(entry.message.empty() || entry.message == options.message,
                             "ProfilingRegistry::register_entry: conflicting message for the key '{}'", key);
        }
        if (!options.parent.empty()) {
            Exception::check(entry.parent_key.empty() || entry.parent_key == options.parent,
                             "ProfilingRegistry::register_entry: conflicting parent for the key '{}'", key);
        }

        // Commit only after both conflict checks so a failed re-registration leaves the existing entry unchanged.
        if (!options.message.empty()) {
            entry.message = std::move(options.message);
        }
        if (!options.parent.empty()) {
            entry.parent_key = std::move(options.parent);
        }
        return key;
    }

    /**
     * @brief Returns a reference to the payload of the entry with the given key, creating the entry if needed.
     *      A stale key after reset therefore self-heals, although creating that replacement entry can allocate
     * @param key The key of the entry
     * @exception std::bad_alloc If a missing entry cannot be allocated
     * @return A reference to the payload
     */
    [[nodiscard]] EntryData& obtain_data(const std::string& key) {
        auto entry_iterator = entries_.find(key);
        if (entry_iterator == entries_.end()) {
            auto entry = Entry{};
            entry.registration_index = next_registration_index_;
            entry_iterator = entries_.emplace(key, std::move(entry)).first;
            next_registration_index_++;
        }
        return entry_iterator->second.data;
    }

    /**
     * @brief Returns the entry with the given key
     * @param key The key of the entry
     * @exception Throws an Exception if there is no entry with the given key
     * @return A constant reference to the entry
     */
    [[nodiscard]] const Entry& get_checked_entry(const std::string& key) const {
        const auto entry_iterator = entries_.find(key);
        Exception::check(entry_iterator != entries_.end(), "ProfilingRegistry::get_checked_entry: there is no entry with the key '{}'", key);
        return entry_iterator->second;
    }

    /**
     * @brief Checks whether an entry with the given key exists
     * @param key The key
     * @return True iff the entry exists
     */
    [[nodiscard]] bool has_entry(const std::string& key) const noexcept {
        return entries_.contains(key);
    }

    /**
     * @brief Returns the number of registered entries
     * @return The number of entries
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return entries_.size();
    }

    /**
     * @brief Removes all entries and resets the registration counter
     */
    void reset() {
        entries_.clear();
        next_registration_index_ = 0;
    }

    /**
     * @brief Returns all entries mapped by the given function, in registration order.
     *      Unlike format_report, this returns a flat list and does not resolve the parent/child tree structure,
     *      which suits callers that reduce or re-render entries themselves (e.g., a distributed report that
     *      merges entries of the same key across participants before printing)
     * @tparam Mapper A callable (const std::string& key, const EntryData& data, const std::string& parent_key) -> T
     *      that produces the mapped value for one entry. parent_key is empty for root entries
     * @param mapper The per-entry mapping function
     * @return A vector of the mapped results, in registration order
     */
    template <typename Mapper>
    [[nodiscard]] auto map_entries_in_registration_order(Mapper&& mapper) const {
        using Result = std::invoke_result_t<Mapper&, const std::string&, const EntryData&, const std::string&>;

        auto result = std::vector<Result>{};
        const auto sorted_entries = get_entries_in_registration_order();
        result.reserve(sorted_entries.size());
        for (const auto* map_entry : sorted_entries) {
            result.push_back(mapper(map_entry->first, map_entry->second.data, map_entry->second.parent_key));
        }
        return result;
    }

    /**
     * @brief Renders a report of all entries as an indented tree.
     *      Roots and siblings appear in registration order, subregions are indented by four spaces per level.
     *      A non-empty message is appended as " - {message}"
     * @tparam LineFormatter A callable (const std::string& key, const EntryData& data, const EntryData* parent_data) -> std::string
     *      that produces the value part of a line (everything after "{key}: "). parent_data is null for root entries
     * @param title The section title, underlined with as many dashes as it has characters
     * @param line_formatter The per-entry value formatter
     * @exception Throws an Exception if a parent key was never registered or if the parent links contain a cycle
     * @return The rendered report, ending with a newline
     */
    template <typename LineFormatter>
    [[nodiscard]] std::string format_report(const std::string_view title, LineFormatter&& line_formatter) const {
        auto result = fmt::format("{}\n{}\n", title, std::string(title.size(), '-'));
        if (entries_.empty()) {
            return result;
        }

        const auto sorted_entries = get_entries_in_registration_order();

        auto children = std::unordered_map<std::string, std::vector<const MapEntry*>>{};
        auto roots = std::vector<const MapEntry*>{};
        for (const auto* map_entry : sorted_entries) {
            const auto& parent_key = map_entry->second.parent_key;
            if (parent_key.empty()) {
                roots.push_back(map_entry);
            } else {
                Exception::check(entries_.contains(parent_key),
                                 "ProfilingRegistry::format_report: the parent '{}' of the entry '{}' was never registered", parent_key, map_entry->first);
                children[parent_key].push_back(map_entry);
            }
        }

        auto rendered_count = std::size_t{ 0 };
        const auto render = [&](const auto& self, const MapEntry* map_entry, const std::size_t depth) -> void {
            const auto& entry = map_entry->second;
            const auto* parent_data = entry.parent_key.empty() ? nullptr : &entries_.at(entry.parent_key).data;

            result += std::string(depth * 4, ' ');
            result += fmt::format("{}: {}", map_entry->first, line_formatter(map_entry->first, entry.data, parent_data));
            if (!entry.message.empty()) {
                result += fmt::format(" - {}", entry.message);
            }
            result += '\n';
            rendered_count++;

            const auto children_iterator = children.find(map_entry->first);
            if (children_iterator != children.end()) {
                for (const auto* child : children_iterator->second) {
                    self(self, child, depth + 1);
                }
            }
        };
        for (const auto* root : roots) {
            render(render, root, 0);
        }

        Exception::check(rendered_count == entries_.size(), "ProfilingRegistry::format_report: the parent links contain a cycle");
        return result;
    }

private:
    using MapEntry = std::pair<const std::string, Entry>;

    /**
     * @brief Returns pointers to all entries, sorted by registration index
     * @return The sorted entries
     */
    [[nodiscard]] std::vector<const MapEntry*> get_entries_in_registration_order() const {
        auto sorted_entries = std::vector<const MapEntry*>{};
        sorted_entries.reserve(entries_.size());
        for (const auto& map_entry : entries_) {
            sorted_entries.push_back(&map_entry);
        }
        std::ranges::sort(sorted_entries, std::less{}, [](const MapEntry* map_entry) { return map_entry->second.registration_index; });
        return sorted_entries;
    }

    std::unordered_map<std::string, Entry> entries_{};
    std::size_t next_registration_index_{ 0 };
};

} // namespace detail

} // namespace utility
