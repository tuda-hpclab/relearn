#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnException.h"

#include "cpp-utility/ranges/Functional.hpp"

#include <fmt/ostream.h>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <compare>
#include <concepts>
#include <cstdint>
#include <ostream>
#include <type_traits>

template <typename U>
class TaggedIDTest;

namespace detail {
template <std::integral T>
[[nodiscard]] constexpr T get_max_size(const std::size_t& bit_count) {
    auto res = std::size_t{ 1 };

    for (auto i = std::size_t{ 0 }; i < bit_count - std::size_t{ 1 }; ++i) {
        res <<= 1U;
        ++res;
    }

    return static_cast<T>(res);
}

template <std::integral T, std::size_t num_bits>
struct TaggedIDNumericalLimitsUnsigned {
    using value_type = T;
    static constexpr value_type min = 0;
    static constexpr value_type max = get_max_size<value_type>(num_bits);
};
} // namespace detail

/**
 * @brief ID class to represent a neuron id with flags as a bitfield.
 *
 * Flag members include is_virtual and is_initialized.
 * The limits type can be used to query the range of id values the tagged id can represent.
 *
 * The flag is_virtual is false by default and can only be specified in the constructor.
 * The is_initialized flag is true when the id was explicitly initialized with an id value,
 * or the id object gets an id assigned.
 */
class NeuronID {
public:
    using value_type = std::uint64_t;
    static constexpr auto num_flags = 2;
    static constexpr auto id_bit_count = (sizeof(value_type) * 8) - num_flags;
    using limits = detail::TaggedIDNumericalLimitsUnsigned<value_type, id_bit_count>;

    /**
     * @brief Get an uninitialized id
     *
     * @return constexpr NeuronID uninitialized id
     */
    [[nodiscard]] static constexpr NeuronID uninitialized_id() noexcept {
        return NeuronID{};
    }

    /**
     * @brief Get a virtual id (is initialized, but virtual)
     * @return constexpr NeuronID virtual id
     */
    [[nodiscard]] static constexpr NeuronID virtual_id() noexcept {
        return NeuronID{ true, 0 };
    }

    /**
     * @brief Get a virtual id (is initialized, but virtual)
     * @param hijacked_value The offset in the RMA window/index of the branch node
     * @return constexpr NeuronID virtual id
     */
    [[nodiscard]] static constexpr NeuronID virtual_id(const std::integral auto hijacked_value) noexcept {
        return NeuronID{ true, hijacked_value };
    }

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end)
     *
     * @param begin begin of the range
     * @param end end of the range
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const value_type begin, const value_type end) {
        return ranges::views::iota(begin, end) | ranges::views::transform(utility::construct<NeuronID>);
    }

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end)
     *
     * @param begin begin of the range
     * @param end end of the range
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const NeuronID begin, const NeuronID end) {
        return range(begin.get_neuron_id(), end.get_neuron_id());
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size)
     *
     * @param size size of the range
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const value_type size) {
        return range(0U, size);
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size)
     *
     * @param size size of the range
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const NeuronID size) {
        return range(0U, size.get_neuron_id());
    }

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end) but as ids of type value_type
     *
     * @param begin begin of the range
     * @param end end of the range
     * @return range of NeuronIDs of type value_type
     */
    [[nodiscard]] static auto range_id(const value_type begin, const value_type end) {
        return ranges::views::iota(begin, end);
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size) but as ids of type value_type
     *
     * @param size size of the range
     * @return range of NeuronIDs of type value_type
     */
    [[nodiscard]] static auto range_id(const value_type size) {
        return range_id(0U, size);
    }

    /**
     * @brief Construct a new NeuronID object where the flag is_initialized is false
     *
     */
    constexpr NeuronID() = default;

    /**
     * @brief Construct a new initialized NeuronID object with the given id
     *
     * @param id the id value
     */
    constexpr explicit NeuronID(const std::integral auto id) noexcept
        : is_initialized_{ true }
        , id_{ (static_cast<value_type>(id) & 0x3FFFFFFFFFFFFFFF) } {
    }

    /**
     * @brief Construct a new initialized NeuronID object with the given flags and id
     *
     * @param is_virtual flag if the id should be marked virtual
     * @param id the id value
     */
    constexpr explicit NeuronID(const bool is_virtual, const std::integral auto id) noexcept
        : is_initialized_{ true }
        , is_virtual_{ is_virtual }
        , id_{ (static_cast<value_type>(id) & 0x3FFFFFFFFFFFFFFF) } {
    }

    constexpr NeuronID(const NeuronID&) noexcept = default;
    constexpr NeuronID& operator=(const NeuronID&) noexcept = default;

    constexpr NeuronID(NeuronID&&) noexcept = default;
    constexpr NeuronID& operator=(NeuronID&&) noexcept = default;

    constexpr ~NeuronID() = default;

    /**
     * @brief Get the id
     *
     * @return value_type id
     */
    [[nodiscard]] constexpr explicit operator value_type() const noexcept {
        return id_;
    }

    /**
     * @brief Check if the id is initialized
     *
     * The same as calling is_initialized()
     * @return true iff the id is initialized
     */
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return is_initialized();
    }

    /**
     * @brief Get the neuron id
     *
     * @exception RelearnException if the id is not initialized or is virtual
     * @return constexpr value_type id
     */
    [[nodiscard]] constexpr value_type get_neuron_id() const {
        RelearnException::check(is_initialized(), "NeuronID::get_neuron_id: Is not initialized {:s}", *this);
        RelearnException::check(!is_virtual(), "NeuronID::get_neuron_id: Is virtual {:s}", *this);
        return id_;
    }

    /**
     * @brief Get the offset in the RMA window. The neuron id must be virtual
     * @exception RelearnException if the id is not initialized or is not virtual
     * @return constexpr value_type The virtual id
     */
    [[nodiscard]] constexpr value_type get_rma_offset() const {
        RelearnException::check(is_initialized(), "NeuronID::get_rma_offset: Is not initialized {:s}", *this);
        RelearnException::check(is_virtual(), "NeuronID::get_rma_offset: Is not virtual {:s}", *this);
        return id_;
    }

    /**
     * @brief Check if the id is initialized
     *
     * @return true iff the id is initialized
     */
    [[nodiscard]] constexpr bool is_initialized() const noexcept {
        return is_initialized_;
    }

    /**
     * @brief Check if the id is virtual
     *
     * @return true iff the id is virtual
     */
    [[nodiscard]] constexpr bool is_virtual() const noexcept {
        return is_virtual_;
    }

    /**
     * @brief Check if there is an actual id in here
     *
     * @return true iff the id is valid
     */
    [[nodiscard]] constexpr bool is_actual_id() const noexcept {
        return is_initialized_ && !is_virtual_;
    }

    /**
     * @brief Compare two NeuronIDs
     *
     * Compares the members in order of declaration
     * @return std::strong_ordering ordering
     */
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const NeuronID&, const NeuronID&) noexcept = default;

    /**
     * @brief Returns the hash_value of the neuron ID *this. Is a perfect hash function
     * @return The hash value
     */
    [[nodiscard]] constexpr std::size_t hash_value() const noexcept {
        // The size of the stored value inside NeuronID has two bits less than value_type

        constexpr auto max = std::numeric_limits<std::size_t>::max();
        if (!is_initialized()) {
            // All bits are set
            return max;
        }

        if (is_virtual()) {
            // Shift the RMA offset by +1 and subtract from max by using XOR
            // The highest bit is set, but some others are not
            const auto offset = get_rma_offset();
            const auto hash_value = max ^ (offset + 1);
            return hash_value;
        }

        // The highest bit is cleared, but some are set
        return get_neuron_id();
    }

private:
    // the ordering of members is important for the defaulted <=> comparison
    bool is_initialized_ : 1 = false;
    bool is_virtual_ : 1 = false;
    value_type id_ : id_bit_count = 0;
};

/**
 * @brief Formatter for NeuronID
 *
 * NeuronID is represented as follows:
 * is_initialized is_virtual : id
 * printing the flags is optional
 *
 * Formatting options are:
 * - i (default): id only   -> 123456
 * - s: small               -> 00:123456
 * - m: medium              -> i0v0:123456
 * - l: large               -> initialized: bool, virtual: bool, id: 123456
 *
 * The id can be formatted with the appropriate
 * formatting for its type.
 * Requirement: NeuronID formatting has to be specified
 * before the formatting of the id.
 * Example: "{:s>20}"
 */
template <>
class fmt::formatter<NeuronID> : public fmt::formatter<typename NeuronID::value_type> {
public:
    [[nodiscard]] constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        const auto* it = ctx.begin();
        const auto* const end = ctx.end();
        if (it != end && (*it == 'i' || *it == 's' || *it == 'm' || *it == 'l')) {
            presentation = *it++; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            ctx.advance_to(it);
        }
        if (it != end && *it != '}') {
            throw format_error("unrecognized format for NeuronID");
        }

        return fmt::formatter<typename NeuronID::value_type>::parse(ctx);
    }

    template <typename FormatContext>
    [[nodiscard]] auto format(const NeuronID& id, FormatContext& ctx) const -> decltype(ctx.out()) {
        switch (presentation) {
        case 'i':
            break;
        case 's':
            fmt::format_to(
                ctx.out(),
                "{:1b}{:1b}:",
                id.is_initialized(), id.is_virtual());
            break;
        case 'm':
            fmt::format_to(
                ctx.out(),
                "i{:1b}v{:1b}:",
                id.is_initialized(), id.is_virtual());
            break;
        case 'l':
            fmt::format_to(
                ctx.out(),
                "initialized: {}, virtual: {}, id: ",
                id.is_initialized(), id.is_virtual());
            break;
        default:
            // unreachable
            throw format_error("unrecognized format for NeuronID");
        }

        using type = typename NeuronID::value_type;
        constexpr static auto offset = type{ 10000000000000000000ULL };

        auto id_ = type{ 0 };

        if (!id.is_initialized()) {
            id_ = std::numeric_limits<type>::max();
        } else if (id.is_virtual()) {
            id_ = offset + id.get_rma_offset();
        } else if (id.is_actual_id()) {
            id_ = id.get_neuron_id();
        } else {
            RelearnException::fail("Format of neuron id failed!");
        }

        return fmt::formatter<type>::format(id_, ctx);
    }

private:
    char presentation = 'i';
};

inline std::ostream& operator<<(std::ostream& os, const NeuronID& id) {
    return os << fmt::format("{}", id);
}

/**
 * @brief Returns the hash_value of the neuron ID. Is a perfect hash function.
 * @param neuron_id The neuron id, can be virtual, can be unitialized
 * @return The hash value
 */
[[nodiscard]] constexpr std::size_t hash_value(const NeuronID neuron_id) noexcept {
    return neuron_id.hash_value();
}

namespace std {
template <>
struct hash<NeuronID> {
    using argument_type = NeuronID;
    using result_type = std::size_t;

    result_type operator()(const argument_type& neuron_id) const noexcept {
        return neuron_id.hash_value();
    }
};
} // namespace std
