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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Concepts.hpp"

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
} // namespace detail

/**
 * @brief This class offers the functionality to convert from C++ types to MPI types.
 */
class MPITypes {
public:
    // MPIWrapper should call able to call init(...), but no other class
    friend class MPIWrapper;

    /**
     * @brief Convertes a type paraneter T to a data type that can be send via MPI.
     *      Works for trivial types, as well as compount types and custom types,
     *      as long as those implement the tuple interface
     * @tparam T The type to convert
     * @return An MPI_Datatype that can then be used with MPI
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
        for (auto [_, mpi_data_type] : MPI_datatypes) {
            MPI_Type_free(&mpi_data_type);
        }
    }

    template <typename T>
    [[nodiscard]] static MPI_Datatype correct_type_size(MPI_Datatype registered_data_type) {
        static constexpr auto type_size = sizeof(T);

        auto lb = MPI_Aint{};
        auto extent = MPI_Aint{};

        const auto extent_error = MPI_Type_get_extent(registered_data_type, &lb, &extent);
        utility::Exception::check(extent_error == MPI_SUCCESS, "MPITypes::correct_type_size: Failed to get the extent for type T.");

        if (utility::save_cast<std::size_t>(extent) == type_size) {
            // MPI and C++ agree on the size of the type
            return registered_data_type;
        }

        if (utility::save_cast<std::size_t>(extent) < type_size) {
            // MPI thinks the type is smaller than it actually is
            // This is a problem, as MPI will only send the data up to the extent
            // We need to create a new MPI_Datatype that is large enough
            auto new_mpi_type = MPI_Datatype{};

            MPI_Type_create_resized(registered_data_type, lb, type_size, &new_mpi_type);
            MPI_Type_commit(&new_mpi_type);
            // We are replacing `registered_data_type` inside this branch. To
            // avoid leaking the committed type we are replacing, free it.
            MPI_Type_free(&registered_data_type);

            return new_mpi_type;
        }

        utility::Exception::fail("MPITypes::correct_type_size: The extent of the MPI_Datatype is larger than size of the type: {} vs {}.", extent, type_size);
        return registered_data_type;
    }

    template <std::ranges::contiguous_range T>
    [[nodiscard]] static MPI_Datatype register_type() {
        static constexpr auto size = detail::get_size<T>();
        auto mpi_type = MPI_Datatype{};
        MPI_Type_contiguous(size, convert_type_to_mpi_type<std::ranges::range_value_t<T>>(), &mpi_type);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        MPI_Type_commit(&corrected_mpi_type);
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
            auto value = T{};
            return std::array<MPI_Aint, size>{ (reinterpret_cast<const std::byte*>(&get<I>(value)) - reinterpret_cast<const std::byte*>(&value))... };
        }(std::make_index_sequence<size>());

        auto types = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<MPI_Datatype, size>{ convert_type_to_mpi_type<std::tuple_element_t<I, T>>()... };
        }(std::make_index_sequence<size>());

        ranges::sort(ranges::views::zip(displacements, types));

        auto mpi_type = MPI_Datatype{};
        MPI_Type_create_struct(size, block_lengths.data(), displacements.data(), types.data(), &mpi_type);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        MPI_Type_commit(&corrected_mpi_type);
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
        MPI_Type_contiguous(size, MPI_BYTE, &mpi_type);

        auto corrected_mpi_type = correct_type_size<T>(mpi_type);
        MPI_Type_commit(&corrected_mpi_type);
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
        const auto iter = MPI_datatypes.find(std::type_index{ typeid(T) });
        if (iter == MPI_datatypes.end()) {
            return register_type<T>();
        }
        return iter->second;
    }

    inline static std::unordered_map<std::type_index, MPI_Datatype> MPI_datatypes{};
};

template <typename T>
concept MPICompatible = requires(T t) { MPITypes::convert_type_to_mpi_type<T>(); };

template <typename T>
concept MPICompatibleRange = std::ranges::contiguous_range<T> && MPICompatible<std::ranges::range_value_t<T>>;

template <typename T, typename ValueType>
concept MPICompatibleRangeOfType = MPICompatibleRange<T> && std::same_as<ValueType, std::ranges::range_value_t<T>>;

} // namespace mpiPP
