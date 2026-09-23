#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnException.h"

#include <cpp-utility/data-structure/TaggedID.hpp>

#include <fmt/ostream.h>

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <string_view>

/**
 * @brief The TaggedID customization that carries the semantics of NeuronID.
 *
 * An id built from these traits behaves exactly like the hand-written NeuronID did: it is uninitialized when
 * default constructed, initialized when constructed from a value, and the virtual flag is set on top of that.
 * As flag 0 is the most significant bit of the underlying type, the defaulted comparison of the id orders by
 * is_initialized first, by is_virtual second and by the value last, i.e., exactly as the member-wise comparison
 * of the previous bitfield-based implementation did.
 */
struct NeuronIDTraits : utility::TaggedIDTraits {
    /** @brief The index of the flag that marks an id as carrying a meaningful value, i.e., as being initialized */
    static constexpr std::size_t initialized_flag = 0;

    /** @brief The index of the flag that marks an id as virtual, i.e., as an offset into the RMA window */
    static constexpr std::size_t virtual_flag = 1;

    /**
     * @brief The underlying id prints as "NeuronID: <value>", which is what shows up in the messages of the
     *      exceptions that the id itself throws. NeuronID has its own formatter with the i/s/m/l presentations
     */
    static constexpr std::string_view name = "NeuronID";

    /** @brief Constructing an id from a value marks it as initialized, default constructing one does not */
    static constexpr std::size_t constructed_flags = std::size_t{ 1 } << initialized_flag;

    /** @brief An uninitialized id prints as "NeuronID: uninitialized" instead of its (meaningless) value */
    static constexpr std::size_t unset_flag = initialized_flag;
};

/**
 * @brief The TaggedID that carries NeuronID's semantics, see NeuronIDTraits.
 *
 * 62 value bits and the two flags, which makes it the same size as the previous bitfield-based NeuronID.
 */
using NeuronIDData = utility::TaggedID<std::uint64_t, 2, NeuronIDTraits>;

static_assert(NeuronIDData::value_bit_count == 62, "NeuronIDTraits: The number of id bits does not match the one of NeuronID");
static_assert(NeuronIDData::min_value == 0, "NeuronIDTraits: The smallest admissible id does not match the one of NeuronID");
static_assert(NeuronIDData::max_value == 0x3FFFFFFFFFFFFFFFULL, "NeuronIDTraits: The largest admissible id does not match the one of NeuronID");

// The ordering that the previous member-wise comparison of (is_initialized, is_virtual, id) produced
static_assert(NeuronIDData{} < NeuronIDData{ 0 }, "NeuronIDTraits: An uninitialized id must order before an initialized one");
static_assert(NeuronIDData{ 0 } < NeuronIDData{ 0 }.with_flag<NeuronIDTraits::virtual_flag>(), "NeuronIDTraits: A local id must order before a virtual one");
static_assert(NeuronIDData{ 0 } < NeuronIDData{ 1 }, "NeuronIDTraits: Ids with equal flags must be ordered by their value");

/**
 * @brief ID class to represent a neuron id with flags as a bitfield.
 *
 * It is a thin wrapper around NeuronIDData, i.e., around a utility::TaggedID configured with NeuronIDTraits:
 * the bit layout, the admissible values, the ordering and the hash all come from there, this class adds the
 * names of the neuron domain and the checks that guard the access to an uninitialized or virtual id.
 *
 * The factories for ranges of ids are static methods of NeuronIDRange in util/NeuronIDRange.h,
 * so that this header stays free of range-v3.
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
    using value_type = NeuronIDData::value_type;

    /** @brief The tagged id that carries the neuron id, see NeuronIDTraits */
    using id_type = NeuronIDData;

    static constexpr auto num_flags = NeuronIDData::num_flags;
    static constexpr auto id_bit_count = NeuronIDData::value_bit_count;

    /** @brief The range of id values that a NeuronID can represent */
    struct limits {
        using value_type = NeuronID::value_type;
        static constexpr value_type min = NeuronIDData::min_value;
        static constexpr value_type max = NeuronIDData::max_value;
    };

    /**
     * @brief Construct a new NeuronID object where the flag is_initialized is false
     *
     */
    constexpr NeuronID() noexcept = default;

    /**
     * @brief Construct a new initialized NeuronID object with the given id
     *
     * @param id the id value
     * @exception RelearnException if id < 0 or id > limits::max
     */
    constexpr explicit NeuronID(const std::integral auto id)
        : id_{ id } {
    }

    /**
     * @brief Construct a new initialized NeuronID object with the given flags and id
     *
     * @param is_virtual flag if the id should be marked virtual
     * @param id the id value
     * @exception RelearnException if id < 0 or id > limits::max
     */
    constexpr explicit NeuronID(const bool is_virtual, const std::integral auto id)
        : id_{ NeuronIDData{ id }.with_flag<NeuronIDTraits::virtual_flag>(is_virtual) } {
    }

    /**
     * @brief Construct a NeuronID from an already built tagged id, which is initialized iff the id's flag is set
     *
     * @param id The tagged id that carries the neuron id
     */
    constexpr explicit NeuronID(const NeuronIDData id) noexcept
        : id_{ id } {
    }

    constexpr NeuronID(const NeuronID&) noexcept = default;
    constexpr NeuronID& operator=(const NeuronID&) noexcept = default;

    constexpr NeuronID(NeuronID&&) noexcept = default;
    constexpr NeuronID& operator=(NeuronID&&) noexcept = default;

    constexpr ~NeuronID() = default;

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
    [[nodiscard]] static constexpr NeuronID virtual_id() {
        return NeuronID{ true, limits::min };
    }

    /**
     * @brief Get a virtual id (is initialized, but virtual)
     * @param hijacked_value The offset in the RMA window/index of the branch node
     * @exception RelearnException if hijacked_value < 0 or hijacked_value > limits::max
     * @return constexpr NeuronID virtual id
     */
    [[nodiscard]] static constexpr NeuronID virtual_id(const std::integral auto hijacked_value) {
        return NeuronID{ true, hijacked_value };
    }

    /**
     * @brief Get the id
     *
     * @return value_type id
     */
    [[nodiscard]] constexpr explicit operator value_type() const noexcept {
        return id_.get_value();
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
        return id_.get_value();
    }

    /**
     * @brief Get the offset in the RMA window. The neuron id must be virtual
     * @exception RelearnException if the id is not initialized or is not virtual
     * @return constexpr value_type The virtual id
     */
    [[nodiscard]] constexpr value_type get_rma_offset() const {
        RelearnException::check(is_initialized(), "NeuronID::get_rma_offset: Is not initialized {:s}", *this);
        RelearnException::check(is_virtual(), "NeuronID::get_rma_offset: Is not virtual {:s}", *this);
        return id_.get_value();
    }

    /**
     * @brief Check if the id is initialized
     *
     * @return true iff the id is initialized
     */
    [[nodiscard]] constexpr bool is_initialized() const noexcept {
        return id_.get_flag<NeuronIDTraits::initialized_flag>();
    }

    /**
     * @brief Check if the id is virtual
     *
     * @return true iff the id is virtual
     */
    [[nodiscard]] constexpr bool is_virtual() const noexcept {
        return id_.get_flag<NeuronIDTraits::virtual_flag>();
    }

    /**
     * @brief Check if there is an actual id in here
     *
     * @return true iff the id is valid
     */
    [[nodiscard]] constexpr bool is_actual_id() const noexcept {
        return is_initialized() && !is_virtual();
    }

    /**
     * @brief Returns the tagged id that carries the neuron id, e.g., to reuse the algorithms of utility
     * @return The tagged id
     */
    [[nodiscard]] constexpr NeuronIDData get_id() const noexcept {
        return id_;
    }

    /**
     * @brief Compare two NeuronIDs
     *
     * Compares is_initialized first, is_virtual second and the id last
     * @return std::strong_ordering ordering
     */
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const NeuronID&, const NeuronID&) noexcept = default;

    /**
     * @brief Returns the hash_value of the neuron ID *this. Is a perfect hash function
     * @return The hash value
     */
    [[nodiscard]] constexpr std::size_t hash_value() const noexcept {
        return id_.hash_value();
    }

private:
    NeuronIDData id_{};
};

static_assert(sizeof(NeuronID) == sizeof(NeuronIDData), "NeuronID grew beyond the tagged id it wraps");

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
