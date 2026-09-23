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
#include "cpp-utility/data-structure/Vec3.hpp"

#include <fmt/ranges.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief This class defines an axis-aligned bounding box based on two Vec3<T> values.
 *      The box is closed, i.e., it spans [minimum, maximum] in every dimension.
 *      The invariant minimum <= maximum (componentwise) is checked at construction.
 *      Mutable structured bindings expose both boundaries; callers using them must preserve the invariant.
 * @tparam T The type that shall be stored within the Vec3. Is required to fulfill std::is_arithmetic_v<T>
 */
template <typename T>
class BoundingBox {
public:
    using value_type = Vec3<T>;

    /**
     * @brief Constructs a new bounding box with minimum and maximum initialized to (0, 0, 0)
     */
    constexpr BoundingBox() = default;

    /**
     * @brief Constructs a new bounding box with the specified minimum and maximum
     * @param _minimum The minimum of the bounding box
     * @param _maximum The maximum of the bounding box
     * @exception Throws an Exception if any component of minimum is larger than the respective component of maximum
     */
    constexpr BoundingBox(const value_type& _minimum, const value_type& _maximum)
        : minimum(_minimum)
        , maximum(_maximum) {

        const auto& [min_x, min_y, min_z] = _minimum;
        const auto& [max_x, max_y, max_z] = _maximum;

        Exception::check(min_x <= max_x, "BoundingBox::BoundingBox: minimum.x ({}) is larger than maximum.x ({})", min_x, max_x);
        Exception::check(min_y <= max_y, "BoundingBox::BoundingBox: minimum.y ({}) is larger than maximum.y ({})", min_y, max_y);
        Exception::check(min_z <= max_z, "BoundingBox::BoundingBox: minimum.z ({}) is larger than maximum.z ({})", min_z, max_z);
    }

    /**
     * @brief Returns the minimum of the bounding box
     * @return The minimum
     */
    [[nodiscard]] constexpr const value_type& get_minimum() const noexcept {
        return minimum;
    }

    /**
     * @brief Returns the maximum of the bounding box
     * @return The maximum
     */
    [[nodiscard]] constexpr const value_type& get_maximum() const noexcept {
        return maximum;
    }

    /**
     * @brief Checks if a position is within the bounding box (the boundary counts as inside)
     * @param position The position to check
     * @return True iff the position is in the bounding box
     */
    [[nodiscard]] constexpr bool contains(const value_type& position) const noexcept {
        const auto in_x_range = minimum.get_x() <= position.get_x() && position.get_x() <= maximum.get_x();
        const auto in_y_range = minimum.get_y() <= position.get_y() && position.get_y() <= maximum.get_y();
        const auto in_z_range = minimum.get_z() <= position.get_z() && position.get_z() <= maximum.get_z();

        return in_x_range && in_y_range && in_z_range;
    }

    /**
     * @brief Checks if the other bounding box is completely within this bounding box
     *      (touching boundaries count as inside)
     * @param other The other bounding box
     * @return True iff the other bounding box is completely contained in this one
     */
    [[nodiscard]] constexpr bool contains(const BoundingBox<T>& other) const noexcept {
        return contains(other.minimum) && contains(other.maximum);
    }

    /**
     * @brief Checks if this bounding box and the other one share at least one point.
     *      As the boxes are closed, boxes that merely touch intersect as well
     * @param other The other bounding box
     * @return True iff the bounding boxes intersect
     */
    [[nodiscard]] constexpr bool intersects(const BoundingBox<T>& other) const noexcept {
        const auto disjoint_x = maximum.get_x() < other.minimum.get_x() || other.maximum.get_x() < minimum.get_x();
        const auto disjoint_y = maximum.get_y() < other.minimum.get_y() || other.maximum.get_y() < minimum.get_y();
        const auto disjoint_z = maximum.get_z() < other.minimum.get_z() || other.maximum.get_z() < minimum.get_z();

        return !(disjoint_x || disjoint_y || disjoint_z);
    }

    /**
     * @brief Returns the dimensions of the bounding box, i.e., maximum - minimum
     * @return The lengths of the bounding box in the three dimensions
     */
    [[nodiscard]] constexpr value_type get_dimensions() const noexcept {
        return maximum - minimum;
    }

    /**
     * @brief Returns the (unsigned) volume of the bounding box
     * @return The volume
     */
    [[nodiscard]] constexpr T get_volume() const noexcept {
        return get_dimensions().get_volume();
    }

    /**
     * @brief Returns the maximum difference of the boundaries, i.e., the largest length in one dimension
     * @return The maximum difference
     */
    [[nodiscard]] constexpr T get_maximum_difference() const noexcept {
        return get_dimensions().get_maximum();
    }

    /**
     * @brief Returns the midpoint of the bounding box
     * @return The midpoint
     */
    [[nodiscard]] constexpr value_type get_midpoint() const noexcept {
        return minimum.get_midpoint(maximum);
    }

    /**
     * @brief Checks if two bounding boxes are equal up to an absolute tolerance,
     *      i.e., whether no component of the boundaries differs by more than epsilon.
     *      Is only available for floating point types
     * @param other The other bounding box
     * @param epsilon The non-negative maximum allowed absolute difference per component
     * @return True iff the two boxes are equal up to the tolerance
     */
    [[nodiscard]] bool almost_equal(const BoundingBox<T>& other, const T epsilon) const noexcept
        requires std::is_floating_point_v<T>
    {
        const auto diff_min = (other.minimum - minimum).abs();
        const auto diff_max = (other.maximum - maximum).abs();

        return epsilon >= T{ 0 }
               && diff_min.get_x() <= epsilon && diff_min.get_y() <= epsilon && diff_min.get_z() <= epsilon
               && diff_max.get_x() <= epsilon && diff_max.get_y() <= epsilon && diff_max.get_z() <= epsilon;
    }

    /**
     * @brief Checks if two bounding boxes are equal (with actually equal components!)
     * @return True iff they are equal
     */
    [[nodiscard]] friend constexpr bool operator==(const BoundingBox<T>&, const BoundingBox<T>&) noexcept = default;

    /**
     * @brief Compares two bounding boxes lexicographically three-way (first the minimum, then the maximum)
     * @return The three-way comparison result. The comparison category depends on type T
     */
    [[nodiscard]] friend constexpr auto operator<=>(const BoundingBox<T>&, const BoundingBox<T>&) noexcept = default;

    /**
     * @brief Prints the bounding box to the ostream in the format [(min_x, min_y, min_z), (max_x, max_y, max_z)]
     * @param output_stream The stream to which the object should be printed
     * @param box The bounding box that should be printed
     * @return A reference to output_stream that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const BoundingBox<T>& box) {
        return output_stream << '[' << box.get_minimum() << ", " << box.get_maximum() << ']';
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        static_assert(Index < 2);

        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        static_assert(Index < 2);

        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        static_assert(Index < 2);

        if constexpr (Index == 0) {
            return std::move(minimum);
        }
        if constexpr (Index == 1) {
            return std::move(maximum);
        }
    }

private:
    value_type minimum{};
    value_type maximum{};

    static_assert(std::is_arithmetic_v<T>);
};

template <std::size_t I, typename T>
const Vec3<T>& get(const BoundingBox<T>& box) { return box.template get<I>(); }

template <std::size_t I, typename T>
Vec3<T>& get(BoundingBox<T>& box) { return box.template get<I>(); }

template <std::size_t I, typename T>
Vec3<T>&& get(BoundingBox<T>&& box) { return std::move(box).template get<I>(); }

} // namespace utility

namespace std {
template <typename T>
struct tuple_size<::utility::BoundingBox<T>> {
    static constexpr size_t value = 2;
};

template <typename T>
struct tuple_element<0, ::utility::BoundingBox<T>> {
    using type = typename utility::BoundingBox<T>::value_type;
};

template <typename T>
struct tuple_element<1, ::utility::BoundingBox<T>> {
    using type = typename utility::BoundingBox<T>::value_type;
};

} // namespace std

/**
 * @brief Disables fmt's tuple-like formatting for BoundingBox<T> (fmt/ranges.h),
 *      so that the dedicated formatter below is chosen unambiguously
 * @tparam T The component type of the BoundingBox
 */
template <typename T>
struct fmt::is_tuple_like<utility::BoundingBox<T>> {
    static constexpr bool value = false;
};

/**
 * @brief Formats a BoundingBox<T> via fmt in the format [(min_x, min_y, min_z), (max_x, max_y, max_z)].
 *      The format specification is applied to each component, e.g., "{:.2f}" formats
 *      every component with two decimal places
 * @tparam T The component type of the BoundingBox
 */
template <typename T>
struct fmt::formatter<utility::BoundingBox<T>> : fmt::formatter<utility::Vec3<T>> {
    /**
     * @brief Formats the bounding box into the output of the format context
     * @param box The bounding box that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted bounding box
     */
    auto format(const utility::BoundingBox<T>& box, fmt::format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '[';
        ctx.advance_to(out);
        out = fmt::formatter<utility::Vec3<T>>::format(box.get_minimum(), ctx);
        *out++ = ',';
        *out++ = ' ';
        ctx.advance_to(out);
        out = fmt::formatter<utility::Vec3<T>>::format(box.get_maximum(), ctx);
        *out++ = ']';
        return out;
    }
};
