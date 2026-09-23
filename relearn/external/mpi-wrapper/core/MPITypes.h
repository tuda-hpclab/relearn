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

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Concepts.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/zip.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpiPP {

class MPIWrapper;

namespace detail {
template <std::size_t Index, utility::detail::has_adl_get T>
[[nodiscard]] constexpr auto get(T&& val) -> decltype(auto) {
    return get<Index>(std::forward<T>(val));
}

template <std::size_t Index, utility::detail::has_member_get T>
[[nodiscard]] constexpr auto get(T&& val) -> decltype(auto) {
    return std::forward<T>(val).template get<Index>();
}

template <typename T>
struct is_std_array : std::false_type { };
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type { };
template <typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;

template <typename>
struct is_std_vector : std::false_type { };
template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type { };
template <typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename T>
struct false_constant : std::false_type { };
template <typename T>
inline constexpr bool false_constant_v = false_constant<T>::value;

template <std::ranges::contiguous_range T>
    requires(!utility::detail::has_tuple_size<T>)
[[nodiscard]] constexpr static std::size_t get_size() {
    if constexpr (utility::detail::has_extent<T>) {
        if constexpr (T::extent != std::numeric_limits<std::size_t>::max()) {
            return T::extent;
        }
    } else {
        static_assert(detail::false_constant_v<T>, "cannot determine size of argument T at compile time");
        return 0;
    }
}

template <utility::detail::has_tuple_size T>
[[nodiscard]] constexpr static std::size_t get_size() {
    return std::tuple_size_v<T>;
}

template <typename T>
[[nodiscard]] consteval bool has_statically_sized_contiguous_storage() {
    if constexpr (!std::ranges::contiguous_range<T> || utility::detail::has_tuple_size<T> || !utility::detail::has_extent<T>) {
        return false;
    } else {
        return T::extent != std::numeric_limits<std::size_t>::max();
    }
}
} // namespace detail

/** Converts supported C++ value types into reusable MPI datatypes. */
class MPITypes {
public:
    // MPIWrapper should be able to call init(...), but no other class
    friend class MPIWrapper;

    /**
     * @brief Returns the predefined or lazily registered MPI datatype for T.
     *      Predefined arithmetic types and enums use their corresponding MPI types. Fixed-size contiguous
     *      ranges, tuple-like types, and eligible standard-layout value types are registered on first use and
     *      cached until MPIWrapper is finalized.
     * @tparam T A semiregular, trivially destructible value type supported by one of the conversions above
     * @return A predefined or committed MPI datatype valid until MPIWrapper finalization
     */
    template <typename T>
        requires std::semiregular<T> && std::is_trivially_destructible_v<T>
    [[nodiscard]] static MPI_Datatype convert_type_to_mpi_type() {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            return MPI_UINT8_T;
        } else if constexpr (std::is_same_v<T, std::uint16_t>) {
            return MPI_UINT16_T;
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
            return MPI_UINT32_T;
        } else if constexpr (std::is_same_v<T, std::uint64_t>) {
            return MPI_UINT64_T;
        } else if constexpr (std::is_same_v<T, std::int8_t>) {
            return MPI_INT8_T;
        } else if constexpr (std::is_same_v<T, std::int16_t>) {
            return MPI_INT16_T;
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            return MPI_INT32_T;
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
            return MPI_INT64_T;
        } else if constexpr (std::is_same_v<T, signed char>) {
            return MPI_SIGNED_CHAR;
        } else if constexpr (std::is_same_v<T, unsigned char>) {
            return MPI_UNSIGNED_CHAR;
        } else if constexpr (std::is_same_v<T, short>) {
            return MPI_SHORT;
        } else if constexpr (std::is_same_v<T, unsigned short>) {
            return MPI_UNSIGNED_SHORT;
        } else if constexpr (std::is_same_v<T, int>) {
            return MPI_INT;
        } else if constexpr (std::is_same_v<T, unsigned int>) {
            return MPI_UNSIGNED;
        } else if constexpr (std::is_same_v<T, float>) {
            return MPI_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            return MPI_DOUBLE;
        } else if constexpr (std::is_same_v<T, long double>) {
            return MPI_LONG_DOUBLE;
        } else if constexpr (std::is_same_v<T, long long>) {
            return MPI_LONG_LONG;
        } else if constexpr (std::is_same_v<T, unsigned long long>) {
            return MPI_UNSIGNED_LONG_LONG;
        } else if constexpr (std::is_same_v<T, long>) {
            return MPI_LONG;
        } else if constexpr (std::is_same_v<T, unsigned long>) {
            return MPI_UNSIGNED_LONG;
        } else if constexpr (std::is_same_v<T, char>) {
            return MPI_CHAR;
        } else {
            return get_mpi_type<T>();
        }
    }

private:
    static void init() { }

    static void finalize() {
        const auto lock = std::scoped_lock{ MPI_datatypes_mutex };
        for (auto [_, mpi_data_type] : MPI_datatypes) {
            const auto error_code = MPI_Type_free(&mpi_data_type);
            utility::Exception::check(error_code == MPI_SUCCESS, "MPITypes::finalize: Freeing a registered datatype returned error code {}", error_code);
        }
        MPI_datatypes.clear();
    }

    template <typename T>
    [[nodiscard]] static MPI_Datatype correct_type_size(MPI_Datatype registered_data_type) {
        static constexpr auto type_size = sizeof(T);

        auto lb = MPI_Aint{};
        auto extent = MPI_Aint{};

        const auto extent_error = MPI_Type_get_extent(registered_data_type, &lb, &extent);
        utility::Exception::check(extent_error == MPI_SUCCESS, "MPITypes::correct_type_size: Failed to get the extent for type T.");

        if (utility::safe_cast<std::size_t>(extent) == type_size) {
            // MPI and C++ agree on the size of the type
            return registered_data_type;
        }

        if (utility::safe_cast<std::size_t>(extent) < type_size) {
            // MPI thinks the type is smaller than it actually is
            // This is a problem, as MPI will only send the data up to the extent
            // We need to create a new MPI_Datatype that is large enough
            auto new_mpi_type = MPI_Datatype{};

            const auto resize_error = MPI_Type_create_resized(registered_data_type, lb, type_size, &new_mpi_type);
            utility::Exception::check(resize_error == MPI_SUCCESS, "MPITypes::correct_type_size: Resizing a datatype returned error code {}", resize_error);
            // We are replacing `registered_data_type` inside this branch. To
            // avoid leaking the committed type we are replacing, free it.
            const auto free_error = MPI_Type_free(&registered_data_type);
            utility::Exception::check(free_error == MPI_SUCCESS, "MPITypes::correct_type_size: Freeing the replaced datatype returned error code {}", free_error);

            return new_mpi_type;
        }

        // Supported registrations only compose member datatypes whose extents fit inside T. Keep
        // this defensive invariant guard, but exclude the unreachable path from coverage.
        utility::Exception::fail("MPITypes::correct_type_size: The extent of the MPI_Datatype is larger than size of the type: {} vs {}.", extent, type_size); // GCOVR_EXCL_LINE
    }

    template <std::ranges::contiguous_range T>
    [[nodiscard]] static MPI_Datatype register_type() {
        static constexpr auto size = detail::get_size<T>();
        auto mpi_type = MPI_Datatype{};
        const auto create_error = MPI_Type_contiguous(size, convert_type_to_mpi_type<std::ranges::range_value_t<T>>(), &mpi_type);
        utility::Exception::check(create_error == MPI_SUCCESS, "MPITypes::register_type: Creating a contiguous datatype returned error code {}", create_error);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        const auto commit_error = MPI_Type_commit(&corrected_mpi_type);
        utility::Exception::check(commit_error == MPI_SUCCESS, "MPITypes::register_type: Committing a contiguous datatype returned error code {}", commit_error);
        MPI_datatypes.try_emplace(std::type_index{ typeid(T) }, corrected_mpi_type);

        return corrected_mpi_type;
    }

    template <std::semiregular T>
        requires std::is_trivially_destructible_v<T> && utility::detail::has_tuple_size<T> && (!detail::is_std_array_v<T>)
    [[nodiscard]] static MPI_Datatype register_type() {
        static constexpr auto size = detail::get_size<T>();

        static constexpr auto block_lengths = []() {
            auto res = std::array<int, size>{};
            std::ranges::fill(res, 1);
            return res;
        }();

        using detail::get;
        using std::get;
        auto displacements = []<std::size_t... I>(std::index_sequence<I...>) {
            if constexpr (sizeof...(I) == 0) {
                return std::array<MPI_Aint, 0>{};
            } else {
                auto value = T{};
                auto base_address = MPI_Aint{};
                const auto base_error = MPI_Get_address(std::addressof(value), &base_address);
                utility::Exception::check(base_error == MPI_SUCCESS, "MPITypes::register_type: Getting the base address returned error code {}", base_error);

                const auto member_displacement = [&value, base_address]<std::size_t MemberIndex>() {
                    auto member_address = MPI_Aint{};
                    const auto address_error = MPI_Get_address(std::addressof(get<MemberIndex>(value)), &member_address);
                    utility::Exception::check(address_error == MPI_SUCCESS, "MPITypes::register_type: Getting a member address returned error code {}", address_error);
                    return member_address - base_address;
                };
                return std::array<MPI_Aint, size>{ member_displacement.template operator()<I>()... };
            }
        }(std::make_index_sequence<size>());

        auto types = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<MPI_Datatype, size>{ convert_type_to_mpi_type<std::tuple_element_t<I, T>>()... };
        }(std::make_index_sequence<size>());

        ranges::sort(ranges::views::zip(displacements, types));

        auto mpi_type = MPI_Datatype{};
        const auto create_error = MPI_Type_create_struct(size, block_lengths.data(), displacements.data(), types.data(), &mpi_type);
        utility::Exception::check(create_error == MPI_SUCCESS, "MPITypes::register_type: Creating a structured datatype returned error code {}", create_error);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        const auto commit_error = MPI_Type_commit(&corrected_mpi_type);
        utility::Exception::check(commit_error == MPI_SUCCESS, "MPITypes::register_type: Committing a structured datatype returned error code {}", commit_error);
        MPI_datatypes.try_emplace(std::type_index{ typeid(T) }, corrected_mpi_type);

        return corrected_mpi_type;
    }

    template <utility::detail::Enum T>
    [[nodiscard]] static MPI_Datatype register_type() noexcept {
        return convert_type_to_mpi_type<std::underlying_type_t<T>>();
    }

    template <typename T>
        requires std::is_standard_layout_v<T> && (!std::is_enum_v<T>) && (!utility::detail::has_tuple_size<T>) && (!detail::is_std_array_v<T>)
    [[nodiscard]] static MPI_Datatype register_type() {
        // This catches MPIRank
        static constexpr auto size = sizeof(T);
        auto mpi_type = MPI_Datatype{};
        const auto create_error = MPI_Type_contiguous(size, MPI_BYTE, &mpi_type);
        utility::Exception::check(create_error == MPI_SUCCESS, "MPITypes::register_type: Creating a byte datatype returned error code {}", create_error);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        const auto commit_error = MPI_Type_commit(&corrected_mpi_type);
        utility::Exception::check(commit_error == MPI_SUCCESS, "MPITypes::register_type: Committing a byte datatype returned error code {}", commit_error);
        MPI_datatypes.try_emplace(std::type_index{ typeid(T) }, corrected_mpi_type);

        return corrected_mpi_type;
    }

    template <typename T>
    [[nodiscard]] static MPI_Datatype register_type() noexcept {
        static_assert(detail::false_constant_v<T>, "MPITypes::register_type: The MPI_Datatype for type T is not statically known and could not be dynamically constructed.");

        // This is here for the gcc warning
        return MPI_DATATYPE_NULL;
    }

    template <typename T>
    [[nodiscard]] static MPI_Datatype get_mpi_type() {
        const auto lock = std::scoped_lock{ MPI_datatypes_mutex };
        const auto iter = MPI_datatypes.find(std::type_index{ typeid(T) });
        if (iter == MPI_datatypes.end()) {
            return register_type<T>();
        }
        return iter->second;
    }

    inline static std::unordered_map<std::type_index, MPI_Datatype> MPI_datatypes{};
    inline static std::recursive_mutex MPI_datatypes_mutex{};
};

/** A value type for which MPITypes can provide a finite, compile-time MPI memory layout. */
template <typename T>
concept MPICompatible = std::semiregular<T> && std::is_trivially_destructible_v<T>
                        && (std::is_enum_v<T> || utility::detail::has_tuple_size<T>
                            || detail::has_statically_sized_contiguous_storage<T>()
                            || (!std::ranges::contiguous_range<T> && std::is_standard_layout_v<T>));

/** An MPI-compatible integer type accepted by MPI's predefined reduction operations. */
template <typename T>
concept MPIReductionInteger = MPICompatible<T> && (std::same_as<T, char> || std::same_as<T, signed char> || std::same_as<T, unsigned char> || std::same_as<T, short> || std::same_as<T, unsigned short> || std::same_as<T, int> || std::same_as<T, unsigned int> || std::same_as<T, long> || std::same_as<T, unsigned long> || std::same_as<T, long long> || std::same_as<T, unsigned long long>);

/** An MPI-compatible integer or floating-point type accepted by arithmetic reductions. */
template <typename T>
concept MPIReductionArithmetic = MPIReductionInteger<T> || (MPICompatible<T> && (std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, long double>));

/** A contiguous range whose element type is MPI-compatible. */
template <typename T>
concept MPICompatibleRange = std::ranges::contiguous_range<T> && MPICompatible<std::ranges::range_value_t<T>>;

/** A contiguous range whose element type is accepted by arithmetic MPI reductions. */
template <typename T>
concept MPIReductionArithmeticRange = std::ranges::contiguous_range<T> && MPIReductionArithmetic<std::ranges::range_value_t<T>>;

/** An MPI-compatible contiguous range with exactly ValueType as its element type. */
template <typename T, typename ValueType>
concept MPICompatibleRangeOfType = MPICompatibleRange<T> && std::same_as<ValueType, std::ranges::range_value_t<T>>;

} // namespace mpiPP
