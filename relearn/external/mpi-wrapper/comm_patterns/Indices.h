#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/comm_patterns/Types.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace mpiPP {

namespace comm_patterns {

template <typename identifier_type>
class Indices {
public:
    static_assert(std::is_integral_v<identifier_type>);
    static_assert(std::is_unsigned_v<identifier_type>);

    /**
     * @brief Initializes the indices with the number of ranks and the number of local values.
     * @param num_ranks The number of MPI ranks, must be > 0
     * @param number_local_values The number of local values
     */
    Indices(const std::size_t num_ranks, const identifier_type number_local_values)
        : indices{ number_local_values }
        , number_ranks{ num_ranks } {
        utility::Exception::check(num_ranks > 0, "Indices::Indices: Number of ranks must be larger than 0");
    }

    /**
     * @brief Returns the number of ranks.
     * @return The number of ranks.
     */
    [[nodiscard]] std::size_t get_number_ranks() const noexcept {
        return number_ranks;
    }

    /**
     * @brief Returns the number of local values.
     * @return The number of local values
     */
    [[nodiscard]] identifier_type get_number_local_values() const noexcept {
        return utility::save_cast<identifier_type>(indices.size());
    }

    /**
     * @brief Sets the index range for a specific local index.
     *      Must only be called at most once for each local index.
     *      Assumes that the indices are contiguous and monotonically increasing.
     * @param local_index The local index, must be smaller than the number of local values (from the constructor)
     * @param index_ranges The indices to set, must be the size of num_ranks (from the constructor)
     * @exception Throws an exception if the local index is too large, if the index was already set (to a non-empty value),
     *      or if the number of index ranges does not match the number of ranks
     */
    void set_indices(const identifier_type local_index, std::vector<IndexRange> index_ranges) {
        utility::Exception::check(local_index < indices.size(), "Indices::set_indices: Index {} was too large ({})", local_index, indices.size());
        utility::Exception::check(indices[local_index].empty(), "Indices::set_indices: Index {} was already set", local_index);
        utility::Exception::check(index_ranges.size() == number_ranks, "Indices::set_indices: Number of index ranges ({}) does not match the number of ranks ({})", index_ranges.size(), number_ranks);

        indices[local_index] = std::move(index_ranges);
    }

    /**
     * @brief Returns the index ranges for a specific local index.
     * @param local_index The local index, must be smaller than the number of local values (from the constructor)
     * @exception Throws an exception if the local index is too large
     * @return The index range (can be empty)
     */
    [[nodiscard]] std::span<const IndexRange> get_indices(const identifier_type local_index) const {
        utility::Exception::check(local_index < indices.size(), "Indices::get_indices: Index {} was too large ({})", local_index, indices.size());
        return indices[local_index];
    }

    /**
     * @brief Normalizes the indices, i.e., before the call, the indices look like:
     *      number_local_values: >     0       1       2       ...
     *      v rank_id v
     *          0                   [0, 4)  [4, 6)  [6, 9)
     *          1                   [0, 0)  [0, 3)  [3, 5)
     *          2                   [0, 3)  [3, 3)  [3, 9)
     *          ...
     *
     *  After the call, the indices look like:
     *      number_local_values: >      0         1         2       ...
     *      v rank_id v
     *          0                   [ 0,  4)  [ 4,  6)  [ 6,  9)
     *          1                   [ 9,  9)  [ 9, 12)  [12, 14)
     *          2                   [14, 18)  [18, 18)  [18, 23)
     *
     * That is, it adds the last index of the previous rank to the indices of the current rank (inductively).
     */
    void normalize() {
        auto offsets = std::vector<std::size_t>(number_ranks, 0);

        for (auto rank = std::size_t{ 0 }; rank < number_ranks; rank++) {
            for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
                const auto& range = *it;
                if (range.empty()) {
                    continue;
                }

                const auto off = range[rank].second;
                offsets[rank] = std::max<std::size_t>(offsets[rank], off);
            }
        }

        for (auto rank = std::size_t{ 1 }; rank < number_ranks; rank++) {
            offsets[rank] += offsets[rank - 1];
        }

        for (auto& ranges : indices) {
            if (ranges.empty()) {
                continue;
            }

            for (auto i = std::size_t{ 0 }; i < number_ranks; i++) {
                if (i == 0) {
                    continue;
                }

                ranges[i].first += offsets[i - 1];
                ranges[i].second += offsets[i - 1];
            }
        }
    }

private:
    std::vector<std::vector<IndexRange>> indices{};
    std::size_t number_ranks{};
};

} // namespace comm_patterns

} // namespace mpiPP
