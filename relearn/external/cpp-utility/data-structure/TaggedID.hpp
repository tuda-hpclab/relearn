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

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief The customization points of a TaggedID, bundled in one type so that the id itself keeps a short signature.
 *
 * A concrete id customizes it by inheriting from this struct and re-declaring only the members it wants to change;
 * every member that is not re-declared keeps the default below, i.e., the behaviour of a plain TaggedID.
 *
 * Example: an MPI rank, i.e., an id whose flag 0 means "was constructed from a value"
 *      struct MPIRankTraits : TaggedIDTraits {
 *          static constexpr std::string_view name = "MPIRank";
 *          static constexpr std::size_t constructed_flags = 1U << 0U;
 *          static constexpr std::size_t unset_flag = 0;
 *          static constexpr std::uint64_t max_value_limit = std::numeric_limits<int>::max() / 2;
 *      };
 *      using MPIRank = TaggedID<std::uint32_t, 1, MPIRankTraits>;
 *
 *      MPIRank{ 3 }        flag 0 set, prints as "MPIRank: 3"
 *      MPIRank{}           flag 0 unset, prints as "MPIRank: uninitialized"
 */
struct TaggedIDTraits {
    /** @brief The value of unset_flag that means "there is no such flag" */
    static constexpr std::size_t no_flag = std::numeric_limits<std::size_t>::max();

    /**
     * @brief The name that is used when printing an id. If it is empty, the generic format
     *      [value: ..., flags: 010...] is used; otherwise the id prints as "<name>: <value>"
     */
    static constexpr std::string_view name{};

    /**
     * @brief The flags that the value constructor sets, as a bit mask in which bit i (counted from the least
     *      significant bit) corresponds to the flag with index i. The default constructor never sets a flag,
     *      which is what makes a default constructed id distinguishable from one that was constructed from a value
     */
    static constexpr std::size_t constructed_flags = 0;

    /**
     * @brief The index of the flag that marks an id as carrying a meaningful value. If it is set to a flag index,
     *      printing an id whose flag is unset yields "<name>: <unset_text>" instead of the value.
     *      Must be no_flag or a valid flag index
     */
    static constexpr std::size_t unset_flag = no_flag;

    /** @brief The text that is printed instead of the value if unset_flag is a flag index and that flag is unset */
    static constexpr std::string_view unset_text = "uninitialized";

    /**
     * @brief An additional upper bound for the value, on top of the one that the number of value bits imposes.
     *      The effective maximum is the smaller of the two. The default imposes no additional bound
     */
    static constexpr std::uint64_t max_value_limit = std::numeric_limits<std::uint64_t>::max();
};

namespace detail {

/** Returns the mask of the value bits, i.e., of all bits that the given number of flags leaves over. */
template <std::unsigned_integral Type>
[[nodiscard]] constexpr Type tagged_id_value_mask(const std::size_t num_flags) noexcept {
    const auto digits = static_cast<std::size_t>(std::numeric_limits<Type>::digits);
    if (num_flags == 0 || num_flags >= digits) {
        // an ill-formed layout, diagnosed by the static_asserts of TaggedID. Avoids an out-of-range shift here
        return Type{ 0 };
    }
    return static_cast<Type>((Type{ 1 } << (digits - num_flags)) - Type{ 1 });
}

/** Returns the mask of the single flag with the given index; flag 0 is the most significant bit of Type. */
template <std::unsigned_integral Type>
[[nodiscard]] constexpr Type tagged_id_flag_mask(const std::size_t flag_index) noexcept {
    const auto digits = static_cast<std::size_t>(std::numeric_limits<Type>::digits);
    if (flag_index >= digits) {
        return Type{ 0 };
    }
    return static_cast<Type>(Type{ 1 } << (digits - std::size_t{ 1 } - flag_index));
}

/** Translates the flag indices in the bit mask of TaggedIDTraits::constructed_flags into a mask of the id's bits. */
template <std::unsigned_integral Type, std::size_t NumFlags>
[[nodiscard]] constexpr Type tagged_id_flag_mask_of(const std::size_t flag_indices) noexcept {
    constexpr auto index_bound
        = (NumFlags < static_cast<std::size_t>(std::numeric_limits<std::size_t>::digits)) ? NumFlags : std::size_t{ 0 };

    auto mask = Type{ 0 };
    for (auto flag_index = std::size_t{ 0 }; flag_index < index_bound; flag_index++) {
        if (((flag_indices >> flag_index) & std::size_t{ 1 }) != 0) {
            mask = static_cast<Type>(mask | tagged_id_flag_mask<Type>(flag_index));
        }
    }
    return mask;
}

} // namespace detail

/**
 * @brief A tagged id: an id of an integral type from which a fixed number of 1-bit flags is carved out.
 *      The flags carry no built-in meaning, their interpretation (initialized, virtual, ...) is up to the user.
 *
 * The bit layout from the most to the least significant bit is: flag 0 | flag 1 | ... | flag NumFlags-1 | value.
 * The defaulted comparison therefore orders by the flags first (in order of their indices) and by the value last.
 *
 * Example: using NeuronID = TaggedID<std::uint64_t, 2>; // 62 value bits, e.g., flag 0 = "is initialized", flag 1 = "is virtual"
 *      NeuronID{ 5 }                   an id with value 5 and all flags unset
 *      NeuronID{ 5 }.with_flag<0>()    an id with value 5 and flag 0 set
 *      NeuronID{}                      an id with value 0 and all flags unset
 *
 * Which flags the value constructor sets, how an id prints, and how large its value may become is configured via
 * Traits, see TaggedIDTraits. Ids that differ in their traits are distinct types even if their bit layout matches.
 *
 * This header deliberately does not depend on range-v3. The factories for ranges of ids and of plain values
 * are static methods of TaggedIDRange in cpp-utility/data-structure/TaggedIDRange.hpp.
 *
 * @tparam T The underlying type. Must be an unsigned integral type
 * @tparam NumFlags The number of flags. Must be at least 1 and must leave at least one bit for the value
 * @tparam Traits The customization points, see TaggedIDTraits
 */
template <std::unsigned_integral T, std::size_t NumFlags, typename Traits = TaggedIDTraits>
class TaggedID {
private:
    static constexpr T value_mask = detail::tagged_id_value_mask<T>(NumFlags);

    static constexpr T constructed_flag_mask = detail::tagged_id_flag_mask_of<T, NumFlags>(Traits::constructed_flags);

public:
    using value_type = T;
    using traits_type = Traits;

    static constexpr std::size_t num_flags = NumFlags;
    static constexpr std::size_t value_bit_count = static_cast<std::size_t>(std::numeric_limits<T>::digits) - NumFlags;

    static_assert(NumFlags >= 1, "TaggedID: There must be at least one flag");
    static_assert(NumFlags < static_cast<std::size_t>(std::numeric_limits<T>::digits),
        "TaggedID: The flags must leave at least one bit for the value");
    static_assert(NumFlags >= static_cast<std::size_t>(std::numeric_limits<std::size_t>::digits)
            || (Traits::constructed_flags >> NumFlags) == 0,
        "TaggedID: The traits set a flag that does not exist");
    static_assert(Traits::unset_flag == TaggedIDTraits::no_flag || Traits::unset_flag < NumFlags,
        "TaggedID: The traits name an unset flag that does not exist");

    /** @brief True iff the traits give the id a name, i.e., iff it prints as "<name>: <value>" */
    static constexpr bool has_name = !Traits::name.empty();

    /** @brief True iff the traits name a flag that marks an id as carrying a meaningful value */
    static constexpr bool has_unset_flag = Traits::unset_flag < NumFlags;

    static constexpr value_type min_value = 0;
    static constexpr value_type max_value = (Traits::max_value_limit < static_cast<std::uint64_t>(value_mask))
        ? static_cast<value_type>(Traits::max_value_limit)
        : value_mask;

    /**
     * @brief Constructs a new id with value 0 and all flags unset, in particular also those that the traits
     *      declare as constructed_flags
     */
    constexpr TaggedID() noexcept = default;

    /**
     * @brief Constructs a new id with the given value and exactly the flags of Traits::constructed_flags set
     * @param value The value
     * @exception Throws an Exception if value < 0 or value > max_value
     */
    constexpr explicit TaggedID(const std::integral auto value)
        : data_{ static_cast<T>(static_cast<T>(static_cast<T>(value) & value_mask) | constructed_flag_mask) } {
        Exception::check(std::cmp_greater_equal(value, min_value), "TaggedID::TaggedID: The value must be >= 0, was {}", value);
        Exception::check(std::cmp_less_equal(value, max_value), "TaggedID::TaggedID: The value must be <= {}, was {}", max_value, value);
    }

    /**
     * @brief Returns a copy of *this in which the specified flag is set to the given value
     * @tparam FlagIndex The index of the flag, must be < NumFlags
     * @param flag_value The new value of the flag
     * @return The copy with the changed flag
     */
    template <std::size_t FlagIndex>
    [[nodiscard]] constexpr TaggedID with_flag(const bool flag_value = true) const noexcept {
        static_assert(FlagIndex < NumFlags, "TaggedID::with_flag: The flag index is out of range");

        auto result = *this;
        if (flag_value) {
            result.data_ |= get_flag_mask(FlagIndex);
        } else {
            result.data_ &= static_cast<T>(~get_flag_mask(FlagIndex));
        }
        return result;
    }

    /**
     * @brief Returns the value of the specified flag
     * @tparam FlagIndex The index of the flag, must be < NumFlags
     * @return True iff the flag is set
     */
    template <std::size_t FlagIndex>
    [[nodiscard]] constexpr bool get_flag() const noexcept {
        static_assert(FlagIndex < NumFlags, "TaggedID::get_flag: The flag index is out of range");
        return (data_ & get_flag_mask(FlagIndex)) != 0;
    }

    /**
     * @brief Returns the value (without the flags)
     * @return The value
     */
    [[nodiscard]] constexpr value_type get_value() const noexcept {
        return static_cast<value_type>(data_ & value_mask);
    }

    /**
     * @brief Returns the value (without the flags) converted to the requested integral type. Saves the call sites
     *      the cast that the unsigned value_type would otherwise force upon them, e.g., to int for MPI
     * @tparam U The requested integral type
     * @exception Throws an Exception if the value is not representable in U. Cannot happen, and is therefore not
     *      checked at all, if every value up to max_value fits into U
     * @return The value as U
     */
    template <std::integral U>
    [[nodiscard]] constexpr U get_value_as() const noexcept(std::in_range<U>(max_value)) {
        const auto value = get_value();
        if constexpr (!std::in_range<U>(max_value)) {
            Exception::check(std::in_range<U>(value), "TaggedID::get_value_as: The value {} is not representable in the requested type", value);
        }
        return static_cast<U>(value);
    }

    /**
     * @brief Returns the value (without the flags), but only if the specified flag is set. Guards those accesses
     *      whose invariant is that the id actually carries a meaningful value, e.g., that it was not default constructed
     * @tparam FlagIndex The index of the flag that must be set, must be < NumFlags
     * @tparam U The requested integral type, defaults to value_type
     * @exception Throws an Exception if the flag is unset or if the value is not representable in U
     * @return The value as U
     */
    template <std::size_t FlagIndex, std::integral U = value_type>
    [[nodiscard]] constexpr U get_value_checked() const {
        static_assert(FlagIndex < NumFlags, "TaggedID::get_value_checked: The flag index is out of range");

        Exception::check(get_flag<FlagIndex>(), "TaggedID::get_value_checked: The flag {} is not set for {}", FlagIndex, *this);
        return get_value_as<U>();
    }

    /**
     * @brief Returns the value (without the flags). The same as calling get_value()
     * @return The value
     */
    [[nodiscard]] constexpr explicit operator value_type() const noexcept {
        return get_value();
    }

    /**
     * @brief Returns the hash value of *this. The mapping is injective (a perfect hash function)
     *      as long as value_type does not have more bits than std::size_t
     * @return The hash value
     */
    [[nodiscard]] constexpr std::size_t hash_value() const noexcept {
        return static_cast<std::size_t>(data_);
    }

    /**
     * @brief Checks if two ids are equal, i.e., all flags and the values are equal
     * @return True iff they are equal
     */
    [[nodiscard]] friend constexpr bool operator==(const TaggedID&, const TaggedID&) noexcept = default;

    /**
     * @brief Compares two ids three-way: first the flags in order of their indices, then the values
     * @return The three-way comparison result
     */
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const TaggedID&, const TaggedID&) noexcept = default;

    /**
     * @brief Prints the id to the ostream, identical to the fmt output: in the format [value: ..., flags: 010...]
     *      with the flags in order of their indices, or as "<name>: <value>" if the traits give the id a name
     * @param output_stream The stream to which the object should be printed
     * @param id The id that should be printed
     * @return A reference to output_stream that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const TaggedID& id) {
        // delegates to the fmt formatter below so that both outputs cannot drift apart
        return output_stream << fmt::format("{}", id);
    }

private:
    [[nodiscard]] static constexpr T get_flag_mask(const std::size_t flag_index) noexcept {
        return detail::tagged_id_flag_mask<T>(flag_index);
    }

    T data_{ 0 };
};

namespace detail {

template <typename Type>
inline constexpr bool is_tagged_id = false;

template <std::unsigned_integral Type, std::size_t NumFlags, typename Traits>
inline constexpr bool is_tagged_id<TaggedID<Type, NumFlags, Traits>> = true;

} // namespace detail

/**
 * @brief Satisfied by every specialization of TaggedID, ignoring cv-qualifiers and references.
 *      Type is the type that should be checked
 */
template <typename Type>
concept TaggedIDType = detail::is_tagged_id<std::remove_cvref_t<Type>>;

} // namespace utility

/**
 * @brief Formats a TaggedID via fmt. Without a name in its traits the format is [value: ..., flags: 010...],
 *      with a name it is "<name>: <value>", respectively "<name>: <unset_text>" if the traits' unset flag is unset
 * @tparam T The underlying type of the TaggedID
 * @tparam NumFlags The number of flags of the TaggedID
 * @tparam Traits The traits of the TaggedID
 */
template <std::unsigned_integral T, std::size_t NumFlags, typename Traits>
struct fmt::formatter<utility::TaggedID<T, NumFlags, Traits>> {
    constexpr auto parse(fmt::format_parse_context& ctx) {
        return ctx.begin();
    }

    /**
     * @brief Formats the id into the output of the format context
     * @param id The id that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted id
     */
    auto format(const utility::TaggedID<T, NumFlags, Traits>& id, fmt::format_context& ctx) const {
        using IDType = utility::TaggedID<T, NumFlags, Traits>;

        if constexpr (IDType::has_name) {
            if constexpr (IDType::has_unset_flag) {
                if (!id.template get_flag<Traits::unset_flag>()) {
                    return fmt::format_to(ctx.out(), "{}: {}", Traits::name, Traits::unset_text);
                }
            }
            return fmt::format_to(ctx.out(), "{}: {}", Traits::name, static_cast<std::uint64_t>(id.get_value()));
        } else {
            auto out = fmt::format_to(ctx.out(), "[value: {}, flags: ", static_cast<std::uint64_t>(id.get_value()));
            [&out, &id]<std::size_t... Indices>(std::index_sequence<Indices...>) {
                ((*out++ = (id.template get_flag<Indices>() ? '1' : '0')), ...);
            }(std::make_index_sequence<NumFlags>{});
            *out++ = ']';
            return out;
        }
    }
};

/**
 * @brief Uses the id's packed value, including its flags, as its hash. Specializing std::hash lets a TaggedID
 *      be used in std::unordered_map and std::unordered_set without naming a hasher explicitly
 * @tparam T The underlying type of the TaggedID
 * @tparam NumFlags The number of flags of the TaggedID
 * @tparam Traits The traits of the TaggedID
 */
template <std::unsigned_integral T, std::size_t NumFlags, typename Traits>
struct std::hash<utility::TaggedID<T, NumFlags, Traits>> {
    [[nodiscard]] std::size_t operator()(const utility::TaggedID<T, NumFlags, Traits>& id) const noexcept {
        return id.hash_value();
    }
};
