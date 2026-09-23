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

#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPIRankRange.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace mpiPP {

/**
 * This type accumulates multiple values that should be exchanged between different MPI ranks.
 * It does not perform MPI communication on its own.
 * It saves the RequestData in a vector with the MPIRank as index.
 *
 * @tparam RequestType The type of the values that should be exchanged
 */
template <typename RequestType>
class CommunicationVector {

public:
    using container_type = std::vector<std::vector<RequestType>>;
    using size_type = typename container_type::size_type;
    using requests_size_type = typename std::vector<RequestType>::size_type;
    using sizes_type = std::unordered_map<MPIRank, requests_size_type>;

    template <typename iterator_type, bool _b>
    struct IndexedIterator {
        using iterator_category = std::forward_iterator_tag;

        using rank_type = const MPIRank;
        using ref_type = std::vector<RequestType>&;
        using const_ref_type = const std::vector<RequestType>&;

        using val_type = std::conditional_t<_b, const_ref_type, ref_type>;

        using value_type = std::pair<rank_type, val_type>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        IndexedIterator(const iterator_type iterator, const int rank) noexcept
            : _iterator(iterator)
            , _rank(rank) { }

        value_type& operator*() {
            _pair.emplace(MPIRank{ _rank }, *_iterator);
            return _pair.value();
        }

        const value_type& operator*() const {
            _pair.emplace(MPIRank{ _rank }, *_iterator);
            return _pair.value();
        }

        IndexedIterator& operator++() noexcept {
            ++_iterator;
            ++_rank;

            return *this;
        }

        IndexedIterator operator++(int) noexcept {
            IndexedIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const IndexedIterator& other) const noexcept {
            return _iterator == other._iterator;
        }

        bool operator!=(const IndexedIterator& other) const noexcept {
            return !(*this == other);
        }

    private:
        iterator_type _iterator;
        int _rank;

        mutable std::optional<value_type> _pair{};
    };

    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    /**
     * @brief Constructs a new communication vector
     * @param num_ranks The number of MPI ranks. Is used to check later on for correct usage
     * @param size_hint UNUSED; exists for compatibility with CommunicationMap
     * @exception Throws an Exception if number_ranks is smaller than 1
     */
    explicit CommunicationVector(const int num_ranks, [[maybe_unused]] const size_type size_hint = 1)
        : number_ranks(num_ranks)
        , requests(utility::safe_cast<std::size_t>(num_ranks), std::vector<RequestType>{}) {
        utility::Exception::check(number_ranks > 0, "CommunicationVector::CommunicationVector: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Checks if there is data for the specified rank present
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks
     * @return True iff there is data for the MPI rank
     */
    [[nodiscard]] bool contains(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::contains: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::contains: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        return !requests[mpi_rank.get_rank_cast()].empty();
    }

    /**
     * @brief Returns the number of data packages for MPI ranks
     * @return The number of ranks
     */
    [[nodiscard]] size_type size() const noexcept {
        const auto counted = std::ranges::count_if(requests, [](const auto& rank_requests) {
            return !rank_requests.empty();
        });

        return utility::safe_cast<size_type>(counted);
    }

    /**
     * @brief Returns the number of ranks that this vector can hold
     * @return The number of ranks
     */
    [[nodiscard]] int get_number_ranks() const noexcept {
        return number_ranks;
    }

    /**
     * @brief Returns the total number of requests
     * @return The total number of requests
     */
    [[nodiscard]] requests_size_type get_total_number_requests() const noexcept {
        return ranges::accumulate(
            requests
                | ranges::views::transform(ranges::size),
            requests_size_type{ 0U });
    }

    /**
     * @brief Checks if there is data at all
     * @return True iff there is some data
     */
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    /**
     * @brief Appends the request to the data for the specified MPI rank, inserts the requests for that rank if not yet present
     * @param mpi_rank The MPI rank
     * @param request The data for the MPI rank
     * @exception Throws an Exception if mpi_rank is not initialized or too large with respect to the number of ranks
     */
    void append(const MPIRank mpi_rank, const RequestType& request) {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::append: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::append: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        requests[mpi_rank.get_rank_cast()].emplace_back(request);
    }

    /**
     * @brief Emplaces a newly created element in the communication vector
     * @tparam ...ValueType The type for the constructor of the element
     * @param mpi_rank The MPI rank
     * @param ...Val The values for the constructor of the element
     * @exception Throws an exception if mpi_rank is negative or too large with respect to the number of ranks, if the memory allocation fails, or the constructor of the element throws
     * @return A reference to the newly created element
     */
    template <class... ValueType>
    constexpr decltype(auto) emplace_back(const MPIRank mpi_rank, ValueType&&... Val) {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::emplace_back: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::emplace_back: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        return requests[mpi_rank.get_rank_cast()].emplace_back(std::forward<ValueType>(Val)...);
    }

    /**
     * @brief Sets the request for the specified position
     * @param mpi_rank The MPI rank
     * @param request_index The index of the data package
     * @param request The data for the MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks,
     *      if the index is too large within the requests for that rank, or if there is no data for the MPI rank at all
     */
    void set_request(const MPIRank mpi_rank, const requests_size_type request_index, const RequestType& request) {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::set_request: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::set_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::set_request: Does not contain a buffer for rank {}", mpi_rank);
        utility::Exception::check(request_index < size(mpi_rank), "CommunicationVector::set_request: The index was too large: {} vs {}", request_index, requests[mpi_rank.get_rank_cast()].size());

        requests[mpi_rank.get_rank_cast()][request_index] = request;
    }

    /**
     * @brief Returns the data for the specified rank and the specified index
     * @param mpi_rank The MPI rank
     * @param request_index The index of the data package
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks,
     *      if the index is too large within the requests for that rank, or if there is no data for the MPI rank at all
     * @return The data package
     */
    [[nodiscard]] RequestType get_request(const MPIRank mpi_rank, const requests_size_type request_index) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_request: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_request: There are no requests for rank {}", mpi_rank);

        const auto& requests_for_rank = requests[mpi_rank.get_rank_cast()];
        utility::Exception::check(request_index < requests_for_rank.size(), "CommunicationVector::get_request: index out of bounds: {} vs {}", request_index, requests_for_rank.size());

        return requests_for_rank[request_index];
    }

    /**
     * @brief Returns all data for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks,
     *      or if there is no data for the MPI rank at all
     * @return All data for the specified rank
     */
    [[nodiscard]] const std::vector<RequestType>& get_requests(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_requests: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_requests: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_requests: There are no requests for rank {}", mpi_rank);

        return requests[mpi_rank.get_rank_cast()];
    }

    /**
     * @brief Returns all data for the specified MPI rank wrapped in an std::optional. If the MPI rank is not saved, returns the empty state.
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks
     * @return All data for the specified rank (might be empty)
     */
    [[nodiscard]] std::optional<std::reference_wrapper<const std::vector<RequestType>>> get_optional_requests(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_optional_requests: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_optional_requests: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

        const auto& ref = requests[mpi_rank.get_rank_cast()];
        if (ref.empty()) {
            return std::nullopt;
        }

        return {
            std::reference_wrapper{ ref }
        };
    }

    /**
     * @brief Resizes the buffer for the data packages for a specified MPI rank, inserting the rank if needed
     * @param mpi_rank The MPI rank
     * @param size_for_rank The number of elements the buffer should be able to hold
     * @exception Throws an Exception if mpi_rank is uninitialized or too large with respect to the number of ranks
     */
    void resize(const MPIRank mpi_rank, const requests_size_type size_for_rank) {
        utility::Exception::check(mpi_rank.is_initialized(),
                                  "CommunicationVector::resize: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks,
                                  "CommunicationVector::resize: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

        requests[mpi_rank.get_rank_cast()].resize(size_for_rank);
    }

    /**
     * @brief Resizes the buffers for all MPI ranks
     * @param sizes One requested buffer size for every MPI rank, in rank order
     * @exception Throws an Exception if sizes.size() differs from number_ranks
     */
    void resize(const std::span<const requests_size_type> sizes) {
        utility::Exception::check(sizes.size() == static_cast<size_type>(number_ranks),
                                  "CommunicationVector::resize: number of sizes {} differs from the number of ranks {}", sizes.size(), number_ranks);

        for (const auto mpi_rank : MPIRankRange::range(number_ranks)) {
            const auto size_for_rank = sizes[utility::safe_cast<std::size_t>(mpi_rank.get_rank())];
            requests[mpi_rank.get_rank_cast()].resize(size_for_rank);
        }
    }

    /**
     * @brief Resizes the buffers for the specified MPI ranks
     * @param sizes Requested sizes keyed by rank; ranks not present retain an empty buffer
     * @exception Throws an Exception if a contained rank is invalid
     */
    void resize(sizes_type sizes) {
        requests.clear();
        requests.resize(utility::safe_cast<std::size_t>(number_ranks));

        for (const auto& [mpi_rank, size_for_rank] : sizes) {
            utility::Exception::check(mpi_rank.get_rank() < number_ranks,
                                      "CommunicationVector::resize: The rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

            if (size_for_rank == 0) {
                continue;
            }

            requests[mpi_rank.get_rank_cast()].resize(size_for_rank);
        }
    }

    /**
     * @brief Clears the requests
     */
    void clear() {
        requests.clear();
        requests.resize(utility::safe_cast<std::size_t>(number_ranks));
    }

    /**
     * @brief Returns the number of packages for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks
     * @return The number of packages for the specified MPI rank. Is 0 if there is no data present
     */
    [[nodiscard]] requests_size_type size(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::size: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::size: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

        return requests[mpi_rank.get_rank_cast()].size();
    }

    /**
     * @brief Returns the number of bytes for the packages for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks
     * @return The number of bytes for the packages for the specified MPI rank. Is 0 if there is no data present
     */
    [[nodiscard]] requests_size_type get_size_in_bytes(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_size_in_bytes: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_size_in_bytes: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

        return requests[mpi_rank.get_rank_cast()].size() * sizeof(RequestType);
    }

    /**
     * @brief Returns a non-owning pointer to the buffer for the specified MPI rank.
     *      The pointer is invalidated by calls to resize or append.
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks,
     *      or if there is no data for the specified rank
     * @return A non-owning pointer to the buffer
     */
    [[nodiscard]] RequestType* get_data(const MPIRank mpi_rank) {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_data: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_data: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_data: There are no requests for rank {}", mpi_rank);

        return requests[mpi_rank.get_rank_cast()].data();
    }

    /**
     * @brief Returns a non-owning pointer to the buffer for the specified MPI rank.
     *      The pointer is invalidated by calls to resize or append.
     * @param mpi_rank The MPI rank
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks,
     *      or if there is no data for the specified rank
     * @return A non-owning pointer to the buffer
     */
    [[nodiscard]] const RequestType* get_data(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_data const: mpi_rank is not initialized.");
        utility::Exception::check(mpi_rank.get_rank() < number_ranks, "CommunicationVector::get_data const: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_data const: There are no requests for rank {}", mpi_rank);

        return requests[mpi_rank.get_rank_cast()].data();
    }

    /**
     * @brief Returns a span on the buffer for the specified rank
     * @param mpi_rank The MPI rank whose buffer should be queried
     * @exception Throws an Exception if mpi_rank is negative or too large with respect to the number of ranks
     * @exception Throws an Exception if mpi_rank is negative, the rank does not have saved requests, or the value is too large with respect to the number of ranks
     */
    [[nodiscard]] std::span<RequestType> get_span(const MPIRank mpi_rank) {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_span: mpi_rank is not initialized.");
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_span: There are no requests for rank {}", mpi_rank);
        return std::span<RequestType>{ requests[mpi_rank.get_rank_cast()] };
    }

    /**
     * @brief Returns a constant span on the buffer for the specified rank
     * @param mpi_rank The MPI rank whose buffer should be queried
     * @exception Throws an Exception if mpi_rank is negative, the rank does not have saved requests, or the value is too large with respect to the number of ranks
     */
    [[nodiscard]] std::span<const RequestType> get_span(const MPIRank mpi_rank) const {
        utility::Exception::check(mpi_rank.is_initialized(), "CommunicationVector::get_span const: mpi_rank is not initialized.");
        utility::Exception::check(contains(mpi_rank), "CommunicationVector::get_span const: There are no requests for rank {}", mpi_rank);
        return std::span<const RequestType>{ requests[mpi_rank.get_rank_cast()] };
    }

    /**
     * @brief Returns the number of requests for each MPI rank (includes those without requests with size 0)
     * @return Returns the number of requests for each MPI rank, i.e.,
     *      <return>[i] = k indicates that there are k requests for rank i
     */
    [[nodiscard]] std::vector<requests_size_type> get_request_sizes_vector() const noexcept {
        return requests
               | ranges::views::transform([](const std::vector<RequestType>& req) -> requests_size_type {
                     return req.size();
                 })
               | ranges::to_vector;
    }

    /**
     * @brief Returns the number of requests for each stored MPI rank (leaves out those that are not stored)
     * @return Returns the number of requests for the stored MPI rank, i.e.,
     *      <return>[i] = k indicates that there are k requests for rank i
     */
    [[nodiscard]] sizes_type get_request_sizes() const {
        auto number_requests = std::unordered_map<MPIRank, requests_size_type>{};
        number_requests.reserve(utility::safe_cast<std::size_t>(number_ranks));

        auto counter = 0;

        for (const auto& req : requests) {
            if (req.empty()) {
                counter++;
                continue;
            }

            number_requests[MPIRank{ counter }] = req.size();
            counter++;
        }

        return number_requests;
    }

    /**
     * @brief Returns the begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] IndexedIterator<iterator, false> begin() noexcept {
        return IndexedIterator<iterator, false>{ requests.begin(), 0 };
    }

    /**
     * @brief Returns the end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] IndexedIterator<iterator, false> end() noexcept {
        return IndexedIterator<iterator, false>{ requests.end(), number_ranks };
    }

    /**
     * @brief Returns the constant begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] IndexedIterator<const_iterator, true> begin() const noexcept {
        return IndexedIterator<const_iterator, true>{ requests.begin(), 0 };
    }

    /**
     * @brief Returns the constant end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] IndexedIterator<const_iterator, true> end() const noexcept {
        return IndexedIterator<const_iterator, true>{ requests.end(), number_ranks };
    }

    /**
     * @brief Returns the constant begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] IndexedIterator<const_iterator, true> cbegin() const noexcept {
        return IndexedIterator<const_iterator, true>{ requests.begin(), 0 };
    }

    /**
     * @brief Returns the constant end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] IndexedIterator<const_iterator, true> cend() const noexcept {
        return IndexedIterator<const_iterator, true>{ requests.end(), number_ranks };
    }

private:
    int number_ranks{};
    container_type requests{};
};

} // namespace mpiPP
