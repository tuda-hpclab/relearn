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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <bit>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A monotone integer bucket queue (the queue of Dial's algorithm).
 *		Keys must never be smaller than the smallest key popped so far ("monotone"), which is
 *		exactly the access pattern of Dijkstra's algorithm with non-negative arc lengths.
 *		The buckets form a circular array that is grown on demand, so the maximum arc length
 *		does not need to be known up front; it only has to fit the invariant that all live keys
 *		lie in a window of the current array size above the current minimum, which growth restores.
 *		push is O(1) amortized; the pops are O(1) amortized plus, over the queue's whole lifetime,
 *		one empty-bucket step per skipped key value, i.e. O(largest popped key) in total.
 *		The queue is therefore best suited for small integer keys/weights.
 * @tparam key_type An unsigned integral type other than bool for the non-negative keys;
 *      keys must never be smaller than current_key
 * @tparam value_type The type of the payload attached to every key
 */
template <std::unsigned_integral key_type, typename value_type>
    requires(!std::same_as<key_type, bool>)
class BucketQueue {
public:
    [[nodiscard]] bool empty() const noexcept {
        return number_entries == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return number_entries;
    }

    /**
     * @brief Inserts the key with the associated value. O(1) amortized.
     * @param key The key, must not be smaller than the smallest live key (monotonicity)
     * @param value The associated value
     * @exception Throws an Exception if key is smaller than the current minimum key
     */
    void push(const key_type key, const value_type value) {
        Exception::check(key >= current_key, "BucketQueue::push: key {} is smaller than the current minimum key {}", key, current_key);

        if (static_cast<std::size_t>(key - current_key) >= buckets.size()) {
            grow(key - current_key + 1);
        }

        buckets[to_slot(key)].push_back(Entry{ key, value });
        ++number_entries;
    }

    /**
     * @brief Removes one entry with the smallest key and returns its key and value.
     * @exception Throws an Exception if the queue is empty
     * @return The key and the value of the removed entry
     */
    [[nodiscard]] std::pair<key_type, value_type> pop() {
        Exception::check(number_entries > 0, "BucketQueue::pop: the queue is empty");

        // All live keys are at least current_key and within one window, so this stops
        // after at most buckets.size() steps at the bucket of the smallest live key.
        while (buckets[to_slot(current_key)].empty()) {
            ++current_key;
        }

        auto& bucket = buckets[to_slot(current_key)];
        auto entry = std::move(bucket.back());
        bucket.pop_back();
        --number_entries;

        return { entry.key, std::move(entry.value) };
    }

private:
    struct Entry {
        key_type key{};
        value_type value{};
    };

    constexpr static std::size_t initial_number_buckets = 1024;

    // buckets.size() is always a power of two, so the modulo is a bit mask
    [[nodiscard]] std::size_t to_slot(const key_type key) const noexcept {
        return static_cast<std::size_t>(key) & (buckets.size() - 1);
    }

    /**
     * @brief Enlarges the circular array to the next power of two that covers the required
     *		window size and redistributes all live entries to their new slots.
     */
    void grow(const key_type required_window) {
        const auto new_number_buckets = std::bit_ceil(safe_cast<std::size_t>(required_window));

        auto old_buckets = std::exchange(buckets, std::vector<std::vector<Entry>>(new_number_buckets));
        for (auto& bucket : old_buckets) {
            for (auto& entry : bucket) {
                buckets[to_slot(entry.key)].push_back(std::move(entry));
            }
        }
    }

    std::vector<std::vector<Entry>> buckets = std::vector<std::vector<Entry>>(initial_number_buckets);
    key_type current_key{ 0 };
    std::size_t number_entries{ 0 };
};

} // namespace utility
