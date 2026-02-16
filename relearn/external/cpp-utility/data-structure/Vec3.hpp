#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Exception.hpp"
#include "cpp-utility/Math.hpp"

#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A Vec3 holds three different values of type T and allows computations via operators.
 * @tparam T The type that shall be stored inside this class. Is required to fulfill std::is_arithmetic_v<T>
 */
template <typename T>
class Vec3 {
public:
    using value_type = T;

    /**
     * @brief Constructs a new instance and initializes all values with 0
     */
    constexpr Vec3() = default;
    constexpr ~Vec3() = default;

    /**
     * @brief Constructs a new instance and initializes all values with val
     * @param val The value that is used to initialize all values
     */
    constexpr explicit Vec3(const T& val) noexcept
        : x(val)
        , y(val)
        , z(val) {
    }

    /**
     * @brief Constructs a new instance and initializes all values
     * @param _x The value for x
     * @param _y The value for y
     * @param _z The value for z
     */
    constexpr Vec3(const T& _x, const T& _y, const T& _z) noexcept
        : x(_x)
        , y(_y)
        , z(_z) {
    }

    constexpr Vec3(const Vec3<T>& other) = default;
    constexpr Vec3<T>& operator=(const Vec3<T>& other) = default;

    constexpr Vec3(Vec3<T>&& other) noexcept = default;
    constexpr Vec3<T>& operator=(Vec3<T>&& other) noexcept = default;

    /**
     * @brief Returns a constant reference to the x component. The reference is only invalidated by destruction of the object
     * @return The x value
     */
    [[nodiscard]] constexpr const T& get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns a constant reference to the y component. The reference is only invalidated by destruction of the object
     * @return The y value
     */
    [[nodiscard]] constexpr const T& get_y() const noexcept {
        return y;
    }

    /**
     * @brief Returns a constant reference to the z component. The reference is only invalidated by destruction of the object
     * @return The z value
     */
    [[nodiscard]] constexpr const T& get_z() const noexcept {
        return z;
    }

    /**
     * @brief Sets the x component to the new value
     * @param _x The new value
     */
    constexpr void set_x(const T& _x) noexcept {
        x = _x;
    }

    /**
     * @brief Sets the y component to the new value
     * @param _y The new value
     */
    constexpr void set_y(const T& _y) noexcept {
        y = _y;
    }

    /**
     * @brief Sets the z component to the new value
     * @param _z The new value
     */
    constexpr void set_z(const T& _z) noexcept {
        z = _z;
    }

    /**
     * @brief Casts to current object to an object of type Vec3<K>. Uses static_cast<K> componentwise
     * @tparam K The new type of the components
     * @return A casted version of the current object
     */
    template <typename K>
    [[nodiscard]] constexpr explicit operator Vec3<K>() const noexcept {
        Vec3<K> res{ static_cast<K>(x), static_cast<K>(y), static_cast<K>(z) };
        return res;
    }

    /**
     * @brief Compares the vectors for equality componentwise
     * @return True iff all components are equal
     */
    [[nodiscard]] friend constexpr bool operator==(const Vec3<T>&, const Vec3<T>&) noexcept = default;

    /**
     * @brief Compares the vectors componentwise three-way
     * @return The three-way comparison result
     */
    [[nodiscard]] friend constexpr auto operator<=>(const Vec3<T>&, const Vec3<T>&) noexcept = default;

    /**
     * @brief Calculates the difference between two vectors and returns it as a newly created object
     * @param lhs The vector from which should be subtracted
     * @param rhs The vector that should be subtracted
     * @return The difference of both vectors as a new object
     */
    [[nodiscard]] constexpr friend Vec3<T> operator-(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
    }

    /**
     * @brief Calculates the sum of two vectors and returns it as a newly created object
     * @param lhs The vector to which should be summed
     * @param rhs The vector that should be summed
     * @return The sum of both vectors as a new object
     */
    [[nodiscard]] constexpr friend Vec3<T> operator+(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
    }

    /**
     * @brief Componentwise adds the scalar value and returns the sum as a newly created object
     * @param scalar The value that should be added to each component
     * @return The sum as a new object
     */
    [[nodiscard]] constexpr Vec3<T> operator+(const T& scalar) const noexcept {
        auto res = *this;
        res += scalar;
        return res;
    }

    /**
     * @brief Componentwise subtracts the scalar value and returns the difference as a newly created object
     * @param scalar The value that should be subtracted from each component
     * @return The difference as a new object
     */
    [[nodiscard]] constexpr Vec3<T> operator-(const T& scalar) const noexcept {
        auto res = *this;
        res -= scalar;
        return res;
    }

    /**
     * @brief Componentwise multiplies the scalar value and returns the product as a newly created object
     * @param scalar The value that should be multiplied to each component
     * @return The product as a new object
     */
    [[nodiscard]] constexpr Vec3<T> operator*(const T& scalar) const noexcept {
        auto res = *this;
        res *= scalar;
        return res;
    }

    /**
     * @brief Componentwise divides by the scalar value and returns the quotient as a newly created object
     * @param scalar The value that should be divided by, is not checked for 0
     * @return The quotient as a new object
     */
    [[nodiscard]] constexpr Vec3<T> operator/(const T& scalar) const noexcept {
        auto res = *this;
        res /= scalar;
        return res;
    }

    /**
     * @brief Rounds the current object componentwise to a larger multiple of value.
     *      This is effectively an ugly function and should only be used with enough care
     * @param value The value of which the components should be rounded to (a multiple of)
     */
    constexpr void round_to_larger_multiple(const T& value) noexcept {
        x = ceil((x - 0.00001) / value) * value;
        y = ceil((y - 0.00001) / value) * value;
        z = ceil((z - 0.00001) / value) * value;
    }

    /**
     * @brief Floors the current vector and returns the results in a newly created object.
     *      Can only be used if the values are non-negative
     * @exception Throws a Exception if any of the components is < 0
     * @return A newly created object with the floored values
     */
    [[nodiscard]] constexpr Vec3<std::size_t> floor_componentwise() const {
        Exception::check(x >= 0, "Vec3::floor_componentwise: x was negative: {}", x);
        Exception::check(y >= 0, "Vec3::floor_componentwise: y was negative: {}", y);
        Exception::check(z >= 0, "Vec3::floor_componentwise: z was negative: {}", z);

        const auto floored_x = static_cast<std::size_t>(floor(x));
        const auto floored_y = static_cast<std::size_t>(floor(y));
        const auto floored_z = static_cast<std::size_t>(floor(z));

        return Vec3<std::size_t>(floored_x, floored_y, floored_z);
    }

    /**
     * @brief Calculates the (signed) volume of the cube with the side length of the current object
     * @return The volume, calculated by x * y * z
     */
    [[nodiscard]] constexpr T get_volume() const noexcept {
        return x * y * z;
    }

    /**
     * @brief Componentwise multiplies the scalar value and changes the current object
     * @param scalar The value that should be multiplied to each component
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator*=(const T& scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    /**
     * @brief Componentwise divides by the scalar value and changes the current object
     * @param scalar The value that should be divided by for each component, is not checked for 0
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator/=(const T& scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /**
     * @brief Componentwise adds the scalar value and changes the current object
     * @param scalar The value that should be added to each component
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator+=(const T& scalar) noexcept {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }

    /**
     * @brief Componentwise adds the other vector and changes the current object
     * @param other The other vector that should be added componentwise
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator+=(const Vec3<T>& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /**
     * @brief Componentwise subtracts the scalar value and changes the current object
     * @param scalar The value that should be subtracted from each component
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator-=(const T& scalar) noexcept {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }

    /**
     * @brief Componentwise subtracts the other vector and changes the current object
     * @param other The other vector that should be subtracted componentwise
     * @return A reference to the current object
     */
    constexpr Vec3<T>& operator-=(const Vec3<T>& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    /**
     * @brief Returns a copy of this vector with the absolute values as components
     * @return The componentwise absolute values
     */
    [[nodiscard]] Vec3<T> abs() const noexcept {
        if constexpr (std::is_unsigned_v<T>) {
            return *this;
        } else {
            const auto abs_x = std::abs(x);
            const auto abs_y = std::abs(y);
            const auto abs_z = std::abs(z);

            return Vec3<T>{ abs_x, abs_y, abs_z };
        }
    }

    /**
     * @brief Calculates the p-norm of the (absolute value of the) current object
     * @param p The exponent of the norm, must be >= 1.0
     * @exception Throws a Exception if p < 1.0
     * @return The calculated p-norm
     */
    [[nodiscard]] double calculate_p_norm(const double p) const {
        Exception::check(p >= 1.0, "Vec3::calculate_p_norm: p-norm is only valid for p >= 1.0, but it was: {}", p);

        const auto xx = std::pow(std::abs(static_cast<double>(x)), p);
        const auto yy = std::pow(std::abs(static_cast<double>(y)), p);
        const auto zz = std::pow(std::abs(static_cast<double>(z)), p);

        const auto sum = xx + yy + zz;
        const auto norm = std::pow(sum, 1.0 / p);
        return norm;
    }

    /**
     * @brief Calculates the 1-norm of the vector (the sum of absolutes)
     * @return The calculated 1-norm
     */
    [[nodiscard]] T calculate_1_norm() const noexcept {
        // Visual studio reports multiple defined symbols for T=unsigned int if this is not included
        if constexpr (std::is_unsigned_v<T>) {
            const auto sum = x + y + z;
            return sum;
        } else {
            const auto abs_x = std::abs(x);
            const auto abs_y = std::abs(y);
            const auto abs_z = std::abs(z);

            const auto sum = abs_x + abs_y + abs_z;
            return sum;
        }
    }

    /**
     * @brief Calculates the 2-norm of the vector
     * @return The calculated 2-norm
     */
    [[nodiscard]] double calculate_2_norm() const noexcept {
        const auto xx = x * x;
        const auto yy = y * y;
        const auto zz = z * z;

        const auto sum = xx + yy + zz;
        const auto norm = std::sqrt(sum);
        return norm;
    }

    /**
     * @brief Calculates the squared 2-norm of the vector, i.e., ||this||^2_2
     * @return The squared calculated 2-norm
     */
    [[nodiscard]] constexpr double calculate_squared_2_norm() const noexcept {
        const auto xx = x * x;
        const auto yy = y * y;
        const auto zz = z * z;

        const auto sum = xx + yy + zz;
        return sum;
    }

    /**
     * @brief Calculates the maximum of both vectors componentwise and changes the current object
     * @param other The other vector
     */
    constexpr void calculate_componentwise_maximum(const Vec3<T>& other) noexcept {
        if (other.x > x) {
            x = other.x;
        }
        if (other.y > y) {
            y = other.y;
        }
        if (other.z > z) {
            z = other.z;
        }
    }

    /**
     * @brief Calculates the minimum of both vectors componentwise and changes the current object
     * @param other The other vector
     */
    constexpr void calculate_componentwise_minimum(const Vec3<T>& other) noexcept {
        if (other.x < x) {
            x = other.x;
        }
        if (other.y < y) {
            y = other.y;
        }
        if (other.z < z) {
            z = other.z;
        }
    }

    /**
     * @brief Returns the maximum out of x, y, and z
     * @return The maximum out of x, y, and z
     */
    [[nodiscard]] constexpr T get_maximum() const noexcept {
        return std::max({ x, y, z });
    }

    /**
     * @brief Returns the minimum out of x, y, and z
     * @return The minimum out of x, y, and z
     */
    [[nodiscard]] constexpr T get_minimum() const noexcept {
        return std::min({ x, y, z });
    }

    /**
     * @brief Calculates the factorial of each component and multiplies them.
     *      Is only available if std::is_integral_v<T>. Casts to the unsigned version of T first
     * @return Returns the product of the factorials
     */
    [[nodiscard]] constexpr auto get_componentwise_factorial() const noexcept {
        static_assert(std::is_integral_v<T>);

        using unsigned_type_T = std::make_unsigned_t<T>;

        const auto fac_x = factorial(static_cast<unsigned_type_T>(x));
        const auto fac_y = factorial(static_cast<unsigned_type_T>(y));
        const auto fac_z = factorial(static_cast<unsigned_type_T>(z));

        const auto product = fac_x * fac_y * fac_z;
        return product;
    }

    /**
     * @brief Calculates this^exponent componentwise and returns the product.
     *      Casts the components to double first
     * @param exponent The exponents for this
     * @return The product of the componentwise power
     */
    [[nodiscard]] double get_componentwise_power(const Vec3<unsigned int>& exponent) const {
        const auto pow_x = std::pow(static_cast<double>(x), exponent.get_x());
        const auto pow_y = std::pow(static_cast<double>(y), exponent.get_y());
        const auto pow_z = std::pow(static_cast<double>(z), exponent.get_z());

        const auto product = pow_x * pow_y * pow_z;
        return product;
    }

    /**
     * @brief Returns the midpoint between this and other, effectively the same as (*this + other) / 2.
     * @param other The other vector
     * @return The middle between this and other
     */
    [[nodiscard]] constexpr Vec3 get_midpoint(const Vec3& other) const noexcept {
        const auto mid_x = std::midpoint(x, other.x);
        const auto mid_y = std::midpoint(y, other.y);
        const auto mid_z = std::midpoint(z, other.z);

        return Vec3{ mid_x, mid_y, mid_z };
    }

    /**
     * @brief Checks if *this is in [lower, upper] component-wise, required lower <= upper component-wise, and returns a flag indicating the result
     * @param lower The lower bound for each component
     * @param upper The upper bound for each component
     * @exception Throws a Exception if lower <= upper is violated
     * @return True iff *this is in [lower, upper]
     */
    [[nodiscard]] constexpr bool check_in_box(const Vec3<T>& lower, const Vec3<T>& upper) const {
        Exception::check(lower.x <= upper.x, "Vec3::check_in_box: lower.x ({}) is larger than upper.x ({})", lower.x, upper.x);
        Exception::check(lower.y <= upper.y, "Vec3::check_in_box: lower.y ({}) is larger than upper.y ({})", lower.y, upper.y);
        Exception::check(lower.z <= upper.z, "Vec3::check_in_box: lower.z ({}) is larger than upper.z ({})", lower.z, upper.z);

        const auto is_in_x_range = lower.x <= x && x <= upper.x;
        const auto is_in_y_range = lower.y <= y && y <= upper.y;
        const auto is_in_z_range = lower.z <= z && z <= upper.z;

        const auto is_in_box = is_in_x_range && is_in_y_range && is_in_z_range;

        return is_in_box;
    }

    /**
     * @brief Clips *this into [lower, upper] component-wise, required lower <= upper component-wise
     * @param lower The lower bound for each component
     * @param upper The upper bound for each component
     * @exception Throws a Exception if lower <= upper is violated
     */
    constexpr void clip_to_box(const Vec3<T>& lower, const Vec3<T>& upper) {
        Exception::check(lower.x <= upper.x, "Vec3::clip_to_box: lower.x ({}) is larger than upper.x ({})", lower.x, upper.x);
        Exception::check(lower.y <= upper.y, "Vec3::clip_to_box: lower.y ({}) is larger than upper.y ({})", lower.y, upper.y);
        Exception::check(lower.z <= upper.z, "Vec3::clip_to_box: lower.z ({}) is larger than upper.z ({})", lower.z, upper.z);

        x = std::min(upper.x, std::max(lower.x, x));
        y = std::min(upper.y, std::max(lower.y, y));
        z = std::min(upper.z, std::max(lower.z, z));
    }

    /**
     * @brief This struct is here to allow Vec3<T> in std::map, etc.
     */
    struct less {
        /**
         * @brief Calls lhs < rhs
         * @param lhs One Vec3 that should be compared
         * @param rhs The second Vec3 that should be compared
         * @return True iff lhs < rhs
         */
        [[nodiscard]] constexpr bool operator()(const Vec3<T>& lhs, const Vec3<T>& rhs) const noexcept {
            return lhs < rhs;
        }
    };

    /**
     * @brief Prints the object to the ostream in the format (x, y, z)
     * @param output_stream The stream to which the object should be printed
     * @param vector The object that should be printed
     * @return A reference to output_stream that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Vec3<T>& vector) {
        return output_stream << '(' << vector.get_x() << ", " << vector.get_y() << ", " << vector.get_z() << ')';
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return x;
        }
        if constexpr (Index == 1) {
            return y;
        }
        if constexpr (Index == 2) {
            return z;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return x;
        }
        if constexpr (Index == 1) {
            return y;
        }
        if constexpr (Index == 2) {
            return z;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(x);
        }
        if constexpr (Index == 1) {
            return std::move(y);
        }
        if constexpr (Index == 2) {
            return std::move(z);
        }
    }

private:
    T x{ 0 };
    T y{ 0 };
    T z{ 0 };

    static_assert(std::is_arithmetic_v<T>);
};

template <std::size_t I, typename T>
const T& get(const Vec3<T>& vec) { return vec.template get<I>(); }

template <std::size_t I, typename T>
T& get(Vec3<T>& vec) { return vec.template get<I>(); }

template <std::size_t I, typename T>
T&& get(Vec3<T>&& vec) { return std::move(vec).template get<I>(); }

} // namespace utility

namespace std {
template <typename T>
struct tuple_size<::utility::Vec3<T>> {
    static constexpr size_t value = 3;
};

template <typename T>
struct tuple_element<0, ::utility::Vec3<T>> {
    using type = typename utility::Vec3<T>::value_type;
};

template <typename T>
struct tuple_element<1, ::utility::Vec3<T>> {
    using type = typename utility::Vec3<T>::value_type;
};

template <typename T>
struct tuple_element<2, ::utility::Vec3<T>> {
    using type = typename utility::Vec3<T>::value_type;
};

} // namespace std
