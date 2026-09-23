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

#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A two-level cache that stores a std::vector<DataType> per (partition, key) pair and hands the
 *      stored data back as a std::span<const DataType>.
 *
 * The cache is indexed by two separate coordinates:
 *      - the partition, a dense and contiguous index in the range [0, num_partitions()). It is stored as a
 *        flat std::vector, so partitions must be a small, known-in-advance set (e.g. a source id, a bucket,
 *        or a worker index). Use init() (or the constructor) to fix the number of partitions up front.
 *      - the key, a sparse identifier stored in a std::unordered_map per partition. Any number of keys can be
 *        added on demand.
 *
 * @tparam DataType The type of the cached elements.
 * @tparam PartitionIndex The integral type used to address a partition. Defaults to std::size_t.
 * @tparam Key The type used to identify an entry within a partition. Must be usable as an unordered_map key.
 * @tparam Hash The hash functor for Key. Defaults to std::hash<Key>.
 */
template <typename DataType, std::integral PartitionIndex = std::size_t, typename Key = std::size_t, typename Hash = std::hash<Key>>
class DataCache {
public:
    /**
     * @brief Constructs an empty cache with zero partitions. init() must be called before inserting data.
     */
    DataCache() = default;

    /**
     * @brief Constructs a cache that can hold the given number of partitions.
     * @param number_partitions The number of partitions
     * @exception Throws an Exception if number_partitions is negative
     */
    explicit DataCache(const PartitionIndex number_partitions) {
        init(number_partitions);
    }

    /**
     * @brief (Re-)initializes the cache to hold the given number of partitions. Discards all previously cached data.
     * @param number_partitions The number of partitions
     * @exception Throws an Exception if number_partitions is negative
     */
    void init(const PartitionIndex number_partitions) {
        cache.assign(safe_cast<std::size_t>(number_partitions), {});
    }

    /**
     * @brief Clears the cached keys and data of every partition. The number of partitions is left unchanged.
     */
    void clear() noexcept {
        for (auto& partition : cache) {
            partition.clear();
        }
    }

    /**
     * @brief Returns the number of partitions the cache currently holds.
     * @return The number of partitions
     */
    [[nodiscard]] std::size_t num_partitions() const noexcept {
        return cache.size();
    }

    /**
     * @brief Checks whether the cache holds data for the given partition and key.
     * @param partition The partition index, must be in [0, num_partitions())
     * @param key The key
     * @exception Throws an Exception if partition is out of range
     * @return True iff there is data cached for the key in the partition
     */
    [[nodiscard]] bool contains(const PartitionIndex partition, const Key& key) const {
        const auto& partition_map = cache[to_index(partition)];
        return partition_map.find(key) != partition_map.cend();
    }

    /**
     * @brief Returns the data cached for the given partition and key.
     * @param partition The partition index, must be in [0, num_partitions())
     * @param key The key
     * @exception Throws an Exception if partition is out of range or if no data is cached for the key
     * @return A view of the cached data
     */
    [[nodiscard]] std::span<const DataType> get_value(const PartitionIndex partition, const Key& key) const {
        const auto& partition_map = cache[to_index(partition)];
        const auto where_it = partition_map.find(key);

        Exception::check(where_it != partition_map.cend(), "DataCache::get_value: There is no cached data for the requested key");

        return where_it->second;
    }

    /**
     * @brief Returns the data cached for the given partition and key, or an empty optional if none is cached.
     * @param partition The partition index, must be in [0, num_partitions())
     * @param key The key
     * @exception Throws an Exception if partition is out of range
     * @return An optional view of the cached data, empty if the key is not cached
     */
    [[nodiscard]] std::optional<std::span<const DataType>> get_value_opt(const PartitionIndex partition, const Key& key) const {
        const auto& partition_map = cache[to_index(partition)];
        const auto where_it = partition_map.find(key);

        if (where_it == partition_map.cend()) {
            return std::nullopt;
        }

        return where_it->second;
    }

    /**
     * @brief Caches the given data for the partition and key. Overwrites already cached data, if present.
     * @param partition The partition index, must be in [0, num_partitions())
     * @param key The key
     * @param values The data to cache
     * @exception Throws an Exception if partition is out of range
     */
    void insert(const PartitionIndex partition, const Key& key, std::vector<DataType>&& values) {
        cache[to_index(partition)].insert_or_assign(key, std::move(values));
    }

private:
    /**
     * @brief Converts a partition index to a validated std::vector index.
     * @param partition The partition index
     * @exception Throws an Exception if partition is negative or not in [0, num_partitions())
     * @return The partition index as a std::size_t
     */
    [[nodiscard]] std::size_t to_index(const PartitionIndex partition) const {
        const auto index = safe_cast<std::size_t>(partition);
        Exception::check(index < cache.size(), "DataCache: The partition index {} is out of range, only {} partitions are available", index, cache.size());
        return index;
    }

    std::vector<std::unordered_map<Key, std::vector<DataType>, Hash>> cache{};
};

} // namespace utility
