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

#include "Config.h"

#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <fmt/ranges.h> // IWYU pragma: keep
#include <fmt/std.h>

#include <compare>
#include <cstddef>
#include <ostream>
#include <tuple>
#include <type_traits>

/**
 * @brief This class defines a bounding box based on two Vec3<T> values
 * @tparam T The type that shall be stored within the Vec3. Is required to fulfill std::is_arithmetic_v<T>
 */
template <typename T>
class BoundingBox {
public:
    using value_type = Vec3<T>;

    constexpr BoundingBox() = default;

    /**
     * @brief Constructs a new bounding box with the specified minimum and maximum
     * @param _minimum The minimum of the bounding box
     * @param _maximum The maximum of the bounding box
     * @exception Throws a RelearnException if any component of minimum is larger than the respective component of maximum
     */
    constexpr BoundingBox(const value_type& _minimum, const value_type& _maximum)
        : minimum(_minimum)
        , maximum(_maximum) {

        const auto& [min_x, min_y, min_z] = _minimum;
        const auto& [max_x, max_y, max_z] = _maximum;

        RelearnException::check(min_x <= max_x, "BoundingBox::BoundingBox: minimum.x ({}) is larger than maximum.x ({})", min_x, max_x);
        RelearnException::check(min_y <= max_y, "BoundingBox::BoundingBox: minimum.y ({}) is larger than maximum.y ({})", min_y, max_y);
        RelearnException::check(min_z <= max_z, "BoundingBox::BoundingBox: minimum.z ({}) is larger than maximum.z ({})", min_z, max_z);
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
     * @brief Checks if a position is within the bounding box
     * @param position The position to check
     * @return True iff the position is in the bounding box
     */
    [[nodiscard]] constexpr bool check_in_box(const Vec3<T>& position) const {
        return position.check_in_box(minimum, maximum);
    }

    /**
     * @brief Returns the maximum difference of the boundaries, i.e., the largest length in one dimesion
     * @return The maximum difference
     */
    [[nodiscard]] constexpr T get_maximum_difference() const noexcept {
        const auto diff_vector = maximum - minimum;
        const auto diff = diff_vector.get_maximum();
        return diff;
    }

    /**
     * @brief Returns the midpoint of the boundary box
     * @return The midpoint
     */
    [[nodiscard]] constexpr value_type get_midpoint() const noexcept {
        return minimum.get_midpoint(maximum);
    }

    /**
     * @brief Checks if two bounding boxes are equal up to Constants::eps
     * @param other The other bounding box
     * @return True iff the two boxes are comparably equal
     */
    [[nodiscard]] bool equals_eps(const BoundingBox<T>& other) const noexcept {
        const auto diff_min = (other.minimum - minimum).abs();
        const auto diff_max = (other.maximum - maximum).abs();
        return diff_max.calculate_1_norm() + diff_min.calculate_1_norm() < Constants::eps;
    }

    /**
     * @brief Checks if two bounding boxes are equal (with actually equal components!)
     * @param other The other bounding box
     * @return True iff they are equal
     */
    [[nodiscard]] constexpr bool operator==(const BoundingBox<T>& other) const noexcept = default;

    /**
     * @brief Compares two bounding boxes component wise.
     * @return The comparison result. Comparison type depends on type T
     */
    [[nodiscard]] friend constexpr auto operator<=>(const BoundingBox<T>& lhs, const BoundingBox<T>& rhs) noexcept = default;

    /**
     * @brief Prints the bounding box in the format "print(minimum) - print(maximum)" to the outstream
     * @param output_stream Where to print to
     * @param bb The bounding box
     * @return output_stream for chaining
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const BoundingBox<T>& bb) {
        const auto& [min, max] = bb;
        output_stream << '[' << min << ", " << max << ']';

        return output_stream;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return minimum;
        }
        if constexpr (Index == 1) {
            return maximum;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(minimum);
        }
        if constexpr (Index == 1) {
            return std::move(maximum);
        }
    }

private:
    value_type minimum{ Constants::uninitialized };
    value_type maximum{ Constants::uninitialized };
};

template <std::size_t I, typename T>
const BoundingBox<T>::value_type& get(const BoundingBox<T>& box) { return box.template get<I>(); }
template <std::size_t I, typename T>
BoundingBox<T>::value_type& get(BoundingBox<T>& box) { return box.template get<I>(); }
template <std::size_t I, typename T>
BoundingBox<T>::value_type&& get(BoundingBox<T>&& box) { return std::move(box).template get<I>(); }

namespace std {
template <typename T>
struct tuple_size<::BoundingBox<T>> {
    static constexpr size_t value = 2;
};

template <typename T>
struct tuple_element<0, ::BoundingBox<T>> {
    using type = typename BoundingBox<T>::value_type;
};

template <typename T>
struct tuple_element<1, ::BoundingBox<T>> {
    using type = typename BoundingBox<T>::value_type;
};
} // namespace std
