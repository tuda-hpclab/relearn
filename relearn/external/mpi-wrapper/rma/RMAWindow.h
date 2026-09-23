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

#include "mpi-wrapper/core/MPICounters.h"
#include "mpi-wrapper/core/MPIInfo.h"
#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPISynchronization.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace mpiPP {

/**
 * @brief The different statuses how an RMAWindow can be locked
 */
enum class LockStatus {
    Unlocked,
    Shared,
    Exclusive,
};

/**
 * @brief Owns a collectively created MPI_COMM_WORLD RMA window whose local storage contains T values.
 * @tparam T The type of data stored in the window, not void
 */
template <typename T>
    requires MPICompatible<T> && std::is_trivially_copyable_v<T> && (!std::is_same_v<void, T>)
struct RMAWindow {
    /**
     * @brief Collectively allocates local storage and creates an MPI window on MPI_COMM_WORLD.
     *      Every world rank must call the constructor, although local sizes may differ or be zero.
     * @param window_size The number of local T elements
     * @exception Throws an Exception on every rank if any local byte size exceeds MPI_Aint, or if MPI reports an error
     */
    RMAWindow(const std::uint64_t window_size)
        : size(window_size) {
        auto maximum_window_size = std::uint64_t{};
        const auto error_code_max = MPI_Allreduce(&size, &maximum_window_size, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        utility::Exception::check(error_code_max == MPI_SUCCESS, "Determining the maximum RMA window size returned the error: {}", error_code_max);

        constexpr auto maximum_element_count = utility::safe_cast<std::uint64_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(T);
        utility::Exception::check(maximum_window_size <= maximum_element_count,
                                  "RMAWindow: The largest local window size {} exceeds the representable MPI_Aint byte size", maximum_window_size);

        const auto window_size_byte = utility::safe_cast<MPI_Aint>(size) * utility::safe_cast<MPI_Aint>(sizeof(T));

        if (window_size_byte > 0) {
            const auto error_code_alloc = MPI_Alloc_mem(window_size_byte, MPI_INFO_NULL, &my_base_pointer);
            utility::Exception::check(error_code_alloc == 0, "Allocating the shared memory returned the error: {}", error_code_alloc);
        }

        const auto error_code_create = MPI_Win_create(my_base_pointer, window_size_byte, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window);
        utility::Exception::check(error_code_create == 0, "Creating the shared window returned the error: {}", error_code_create);

        const auto error_handler_code = MPI_Win_set_errhandler(window, MPI_ERRORS_RETURN);
        utility::Exception::check(error_handler_code == MPI_SUCCESS, "Setting MPI_ERRORS_RETURN on the RMA window returned the error: {}", error_handler_code);
    }

    RMAWindow(const RMAWindow& other) = delete;
    RMAWindow& operator=(const RMAWindow& other) = delete;

    /**
     * @brief Move-constructs from another RMAWindow by taking ownership of the resources
     * @param other The other RMAWindow
     */
    RMAWindow(RMAWindow&& other) noexcept {
        std::swap(my_base_pointer, other.my_base_pointer);
        std::swap(size, other.size);
        std::swap(window, other.window);
        std::swap(window_status, other.window_status);
    }

    /**
     * @brief Move-assigns from another RMAWindow by taking ownership of the resources
     * @param other The other RMAWindow
     */
    RMAWindow& operator=(RMAWindow&& other) noexcept {
        std::swap(my_base_pointer, other.my_base_pointer);
        std::swap(size, other.size);
        std::swap(window, other.window);
        std::swap(window_status, other.window_status);

        return *this;
    }

    /**
     * @brief Collectively frees the MPI window and its local allocation. An active lock-all epoch is
     *      closed first; cleanup errors are suppressed because the destructor is non-throwing.
     */
    ~RMAWindow() noexcept {
        auto may_free_memory = true;
        if (window != MPI_WIN_NULL) {
            if (window_status == LockStatus::Shared) {
                may_free_memory = MPI_Win_unlock_all(window) == MPI_SUCCESS;
                window_status = LockStatus::Unlocked;
            }

            const auto error_code_free_win = MPI_Win_free(&window);
            may_free_memory = may_free_memory && error_code_free_win == MPI_SUCCESS;
        }

        if (may_free_memory && my_base_pointer != nullptr) {
            static_cast<void>(MPI_Free_mem(my_base_pointer));
        }
    }

    /**
     * @brief Returns the number of elements in the local rma window
     * @return The number of elements
     */
    [[nodiscard]] std::uint64_t get_local_size() const noexcept {
        return size;
    }

    /**
     * @brief Returns the current status of the window (unlocked, sharedly locked, exclusively locked)
     * @return The current status
     */
    [[nodiscard]] LockStatus get_lock_status() const noexcept {
        return window_status;
    }

    /**
     * @brief Locks all windows sharedly
     * @exception Throws an Exception if MPI returns an error code or if the window is already locked
     */
    void lock_each_rank() {
        if (window_status != LockStatus::Unlocked) {
            utility::Exception::fail("RMAWindow::lock_each_rank: RMAWindow is already locked");
        }

        const auto error_code = MPI_Win_lock_all(0, window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::lock_each_rank: Failed with error: {}", error_code);

        window_status = LockStatus::Shared;
    }

    /**
     * @brief Unlocks all windows
     * @exception Throws an Exception if MPI returns an error code or if the window is not locked
     */
    void unlock_each_rank() {
        if (window_status == LockStatus::Unlocked) {
            utility::Exception::fail("RMAWindow::unlock_each_rank: RMAWindow is already unlocked");
        }

        const auto error_code = MPI_Win_unlock_all(window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::unlock_each_rank: Failed with error: {}", error_code);

        window_status = LockStatus::Unlocked;
    }

    /**
     * @brief Completes all outstanding RMA operations issued during an active lock-all epoch
     * @exception Throws an Exception if no lock-all epoch is active or MPI returns an error code
     */
    void flush_each_rank() {
        const auto error_code = MPI_Win_flush_all(window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::flush_each_rank: Failed with error: {}", error_code);
    }

    /**
     * @brief Downloads a specified range of values with remote memory access.
     * 	    Requirements:
     * 		- [dest_addr, dest_addr + count) must be a valid range of T to write to
     * 		- elements [src_disp, src_disp + count) must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     * @param dest_addr Pointer to destination memory
     * @param count Number of transferred elements of type T/datatype, must fit in an int
     * @param src_disp Zero-based element index of the first remote value; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     */
    [[gnu::nonnull]] void get(T* const dest_addr, const std::unsigned_integral auto count, const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        const auto source_mpi_rank = src_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(source_mpi_rank < number_ranks, "RMAWindow::get: src_rank is not in the rank of MPI ranks: {} vs {}", src_rank, number_ranks);

        if (window_status == LockStatus::Unlocked) {
            lock_window_shared(src_rank);
        }

        const auto my_rank = MPIInfo::get_my_rank();

        if (src_rank == my_rank) {
            const auto* src_ptr = my_base_pointer + src_disp;
            std::ranges::copy(std::span{ src_ptr, utility::safe_cast<std::size_t>(count) }, dest_addr);
        } else {
            auto request = MPI_Request{};

            const auto datatype = MPITypes::convert_type_to_mpi_type<T>();

            const auto count_int = utility::safe_cast<int>(count);
            const auto displ_int = utility::safe_cast<int>(src_disp * sizeof(T));
            const auto error_code = MPI_Rget(dest_addr, count_int, datatype, source_mpi_rank, displ_int, count_int, datatype, window, &request);
            utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::get: Fetching a remote value returned the error code: {}", error_code);

            MPISynchronization::wait(request);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(src_rank);
        }

        MPICounters::add_to_remotely_accessed(count * sizeof(T));
    }

    /**
     * @brief Downloads a specified range of values with remote memory access.
     *      Downloads the range as bytes instead of T.
     * 	    Requirements:
     * 		- [dest_addr, dest_addr + count) must be a valid range of T to write to
     * 		- elements [src_disp, src_disp + count) must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     * @param dest_addr Pointer to destination memory
     * @param count Number of transferred elements of type T/datatype, must fit in an int
     * @param src_disp Zero-based element index of the first remote value; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     */
    [[gnu::nonnull]] void get_as_bytes(T* const dest_addr, const std::unsigned_integral auto count, const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        const auto source_mpi_rank = src_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(source_mpi_rank < number_ranks, "RMAWindow::get_as_bytes: src_rank is not in the rank of MPI ranks: {} vs {}", src_rank, number_ranks);

        if (window_status == LockStatus::Unlocked) {
            lock_window_shared(src_rank);
        }

        const auto my_rank = MPIInfo::get_my_rank();

        if (src_rank == my_rank) {
            const auto* src_ptr = my_base_pointer + src_disp;
            std::ranges::copy(std::span{ src_ptr, utility::safe_cast<std::size_t>(count) }, dest_addr);
        } else {
            auto request = MPI_Request{};

            const auto count_int = utility::safe_cast<int>(count * sizeof(T));
            const auto displ_int = utility::safe_cast<int>(src_disp * sizeof(T));
            const auto error_code = MPI_Rget(dest_addr, count_int, MPI_BYTE, source_mpi_rank, displ_int, count_int, MPI_BYTE, window, &request);
            utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::get_as_bytes: Fetching a remote value returned the error code: {}", error_code);

            MPISynchronization::wait(request);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(src_rank);
        }

        MPICounters::add_to_remotely_accessed(count * sizeof(T));
    }

    /**
     * @brief Downloads a single value with remote memory access.
     * 	    Requirements:
     * 		- element src_disp must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     *
     * @param src_disp Zero-based remote element index; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     *
     * @return The downloaded value
     */
    [[gnu::nonnull]] T get(const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        auto value = T{};
        get(&value, 1U, src_disp, src_rank);
        return value;
    }

    /**
     * @brief Downloads a single value with remote memory access.
     *      Downloads the range as bytes instead of T.
     * 	    Requirements:
     * 		- element src_disp must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     *
     * @param src_disp Zero-based remote element index; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     *
     * @return The downloaded value
     */
    [[gnu::nonnull]] T get_as_bytes(const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        auto value = T{};
        get_as_bytes(&value, 1U, src_disp, src_rank);
        return value;
    }

    /**
     * @brief Downloads a specified range of values with remote memory access.
     *      During an existing lock-all epoch, completion must be enforced separately with
     *      flush_each_rank() or unlock_each_rank(); otherwise the temporary per-rank epoch is
     *      completed before this function returns.
     * 	    Requirements:
     * 		- [dest_addr, dest_addr + count) must be a valid range of T to write to
     * 		- elements [src_disp, src_disp + count) must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     * @param dest_addr Pointer to destination memory
     * @param count Number of transferred elements of type T/datatype, must fit in an int
     * @param src_disp Zero-based element index of the first remote value; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     */
    [[gnu::nonnull]] void get_unsynchronized(T* const dest_addr, const std::unsigned_integral auto count, const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        const auto source_mpi_rank = src_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(source_mpi_rank < number_ranks, "RMAWindow::get_unsynchronized: src_rank is not in the rank of MPI ranks: {} vs {}", src_rank, number_ranks);

        if (window_status == LockStatus::Unlocked) {
            lock_window_shared(src_rank);
        }

        const auto my_rank = MPIInfo::get_my_rank();

        if (src_rank == my_rank) {
            const auto* src_ptr = my_base_pointer + src_disp;
            std::ranges::copy(std::span{ src_ptr, utility::safe_cast<std::size_t>(count) }, dest_addr);
        } else {
            const auto datatype = MPITypes::convert_type_to_mpi_type<T>();

            const auto count_int = utility::safe_cast<int>(count);
            const auto displ_int = utility::safe_cast<int>(src_disp * sizeof(T));
            const auto error_code = MPI_Get(dest_addr, count_int, datatype, source_mpi_rank, displ_int, count_int, datatype, window);
            utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::get_unsynchronized: Fetching a remote value returned the error code: {}", error_code);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(src_rank);
        }

        MPICounters::add_to_remotely_accessed(count * sizeof(T));
    }

    /**
     * @brief Downloads a specified range of values with remote memory access.
     *      During an existing lock-all epoch, completion must be enforced separately with
     *      flush_each_rank() or unlock_each_rank(); otherwise the temporary per-rank epoch is
     *      completed before this function returns.
     * 	    Requirements:
     * 		- [dest_addr, dest_addr + count) must be a valid range of T to write to
     * 		- elements [src_disp, src_disp + count) must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     * @param dest_addr Pointer to destination memory
     * @param count Number of transferred elements of type T/datatype, must fit in an int
     * @param src_disp Zero-based element index of the first remote value; its byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     */
    [[gnu::nonnull]] void get_unsynchronized_as_bytes(T* const dest_addr, const std::unsigned_integral auto count, const std::unsigned_integral auto src_disp, const MPIRank src_rank) const {
        const auto source_mpi_rank = src_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(source_mpi_rank < number_ranks, "RMAWindow::get_unsynchronized_as_bytes: src_rank is not in the rank of MPI ranks: {} vs {}", src_rank, number_ranks);

        if (window_status == LockStatus::Unlocked) {
            lock_window_shared(src_rank);
        }

        const auto my_rank = MPIInfo::get_my_rank();

        if (src_rank == my_rank) {
            const auto* src_ptr = my_base_pointer + src_disp;
            std::ranges::copy(std::span{ src_ptr, utility::safe_cast<std::size_t>(count) }, dest_addr);
        } else {
            const auto count_int = utility::safe_cast<int>(count * sizeof(T));
            const auto displ_int = utility::safe_cast<int>(src_disp * sizeof(T));
            const auto error_code = MPI_Get(dest_addr, count_int, MPI_BYTE, source_mpi_rank, displ_int, count_int, MPI_BYTE, window);
            utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::get_unsynchronized_as_bytes: Fetching a remote value returned the error code: {}", error_code);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(src_rank);
        }

        MPICounters::add_to_remotely_accessed(count * sizeof(T));
    }

    /**
     * @brief Downloads the specified elements with remote memory access.
     *      Requirements:
     * 		- every element index in src_disp must exist in src_rank's window
     * 		- src_rank in [0, number_of_ranks)
     *
     * @tparam count_type The type that specified the number of elements
     *
     * @param src_disp Zero-based remote element indices; each byte offset must fit in an int
     * @param src_rank The rank from where to get the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     *
     * @return The downloaded values
     */
    template <std::unsigned_integral count_type>
    [[gnu::nonnull]] std::vector<T> get_multiple(const std::span<const count_type> src_disp, const MPIRank src_rank) const {
        const auto number_elements = src_disp.size();

        auto results = std::vector<T>{};
        results.resize(number_elements);

        const auto source_mpi_rank = src_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(source_mpi_rank < number_ranks, "RMAWindow::get_multiple: src_rank is not in the rank of MPI ranks: {} vs {}", src_rank, number_ranks);

        const auto my_rank = MPIInfo::get_my_rank();
        const auto datatype = MPITypes::convert_type_to_mpi_type<T>();

        if (window_status == LockStatus::Unlocked) {
            lock_window_shared(src_rank);
        }

        if (src_rank == my_rank) {
            auto current_iterator = std::size_t(0);

            for (const auto displacement : src_disp) {
                results[current_iterator] = my_base_pointer[displacement];
                current_iterator++;
            }
        } else {
            auto request_items = std::vector<MPI_Request>{};
            request_items.resize(number_elements);

            auto current_iterator = std::size_t(0);

            for (const auto displacement : src_disp) {
                auto* request_item = &request_items[current_iterator];

                const auto displ_int = utility::safe_cast<int>(displacement * sizeof(T));
                const auto error_code = MPI_Rget(results.data() + current_iterator, 1, datatype, source_mpi_rank, displ_int, 1, datatype, window, request_item);
                utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::get_multiple: Fetching a remote value returned the error code: {}", error_code);
                current_iterator++;
            }

            MPISynchronization::wait_all(request_items);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(src_rank);
        }

        MPICounters::add_to_remotely_accessed(number_elements * sizeof(T));

        return results;
    }

    /**
     * @brief Uploads a specified range of values with remote memory access.
     * 	    Requirements:
     * 		- [source_addr, source_addr + count) must be a valid range of T to read from
     * 		- elements [target_disp, target_disp + count) must exist in target_rank's window
     * 		- target_rank in [0, number_of_ranks)
     *
     * @tparam T A convertible type that can be moved via MPI
     * @tparam count_type_1 The type that specified the number of elements
     * @tparam count_type_2 The type that specified the displacement
     *
     * @param source_addr Pointer to source memory
     * @param count Number of transferred elements of type T/datatype, must fit in an int
     * @param target_disp Zero-based element index of the first remote value; its byte offset must fit in an int
     * @param target_rank The rank on where to write the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible, if the window is locked sharedly, or MPI returned an error code
     */
    [[gnu::nonnull]] void put(const T* const source_addr, const std::unsigned_integral auto count, const std::unsigned_integral auto target_disp, const MPIRank target_rank) {
        const auto target_mpi_rank = target_rank.get_rank();
        const auto number_ranks = MPIInfo::get_number_ranks();
        utility::Exception::check(target_mpi_rank < number_ranks, "RMAWindow::put: target_rank is not in the rank of MPI ranks: {} vs {}", target_rank, number_ranks);
        utility::Exception::check(window_status != LockStatus::Shared, "RMAWindow::put: Window is locked sharedly");

        if (window_status == LockStatus::Unlocked) {
            lock_window_exclusive(target_rank);
        }

        const auto my_rank = MPIInfo::get_my_rank();

        if (target_rank == my_rank) {
            T* const target_pointer = my_base_pointer + target_disp;
            std::ranges::copy(std::span{ source_addr, utility::safe_cast<std::size_t>(count) }, target_pointer);
        } else {
            const auto datatype = MPITypes::convert_type_to_mpi_type<T>();

            const auto count_int = utility::safe_cast<int>(count);
            const auto displ_int = utility::safe_cast<int>(target_disp * sizeof(T));
            const auto error_code = MPI_Put(source_addr, count_int, datatype, target_mpi_rank, displ_int, count_int, datatype, window);
            utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::put: Putting a remote value returned the error code: {}", error_code);
        }

        if (window_status == LockStatus::Unlocked) {
            unlock_window(target_rank);
        }

        MPICounters::add_to_remotely_accessed(count * sizeof(T));
    }

    /**
     * @brief Uploads a single value with remote memory access.
     * 	    Requirements:
     * 		- element target_disp must exist in target_rank's window
     * 		- target_rank in [0, number_of_ranks)
     *
     * @tparam count_type The type that specified the displacement
     *
     * @param value The value to store
     * @param target_disp Zero-based remote element index; its byte offset must fit in an int
     * @param target_rank The rank on where to write the values
     *
     * @exception Throws an Exception if MPIRank is larger than possible or MPI returned an error code
     */
    [[gnu::nonnull]] void put(const T value, const std::unsigned_integral auto target_disp, const MPIRank target_rank) {
        put(&value, 1U, target_disp, target_rank);
    }

    /**
     * @brief Fills the local window with the specified value
     * @param value The value to fill
     */
    void fill(const T value) {
        utility::Exception::check(window_status != LockStatus::Shared, "RMAWindow::fill: Window is locked sharedly");

        const auto my_rank = MPIInfo::get_my_rank();

        if (window_status == LockStatus::Unlocked) {
            lock_window_exclusive(my_rank);
        }

        std::fill(my_base_pointer, my_base_pointer + size, value);

        if (window_status == LockStatus::Unlocked) {
            unlock_window(my_rank);
        }

        MPICounters::add_to_remotely_accessed(size * sizeof(T));
    }

    /**
     * @brief Returns the pointer to the base memory of the RMAWindow
     * @return The pointer
     */
    [[nodiscard]] T* get_pointer() noexcept {
        return my_base_pointer;
    }

    /**
     * @brief Returns the pointer to the base memory of the RMAWindow
     * @return The pointer
     */
    [[nodiscard]] const T* get_pointer() const noexcept {
        return my_base_pointer;
    }

private:
    /**
     * @brief Locks this RMAWindow on the specified rank sharedly
     * @param mpi_rank The specified rank
     * @exception Throws an Exception if MPI returned an error code
     */
    void lock_window_shared(const MPIRank mpi_rank) const {
        const auto error_code = MPI_Win_lock(MPI_LOCK_SHARED, mpi_rank.get_rank(), 0, window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::lock_window_shared: Shared-locking the RMA window returned the error: {}", error_code);
    }

    /**
     * @brief Locks this RMAWindow on the specified rank exclusively
     * @param mpi_rank The specified rank
     * @exception Throws an Exception if MPI returned an error code
     */
    void lock_window_exclusive(const MPIRank mpi_rank) const {
        const auto error_code = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_rank.get_rank(), 0, window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::lock_window_exclusive: Exclusive-locking the RMA window returned the error: {}", error_code);
    }

    /**
     * @brief Unlocks this RMAWindow on the specified rank
     * @param mpi_rank The specified rank
     * @exception Throws an Exception if MPI returned an error code
     */
    void unlock_window(const MPIRank mpi_rank) const {
        const auto error_code = MPI_Win_unlock(mpi_rank.get_rank(), window);
        utility::Exception::check(error_code == MPI_SUCCESS, "RMAWindow::unlock_window: Unlocking the RMA window returned the error: {}", error_code);
    }

    T* my_base_pointer{ nullptr };
    std::uint64_t size{ 0 };

    MPI_Win window{ MPI_WIN_NULL };

    LockStatus window_status{ LockStatus::Unlocked };
};

} // namespace mpiPP
