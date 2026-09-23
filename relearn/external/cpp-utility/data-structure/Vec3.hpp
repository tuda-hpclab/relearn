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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
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
     * @brief Constructs a new instance and initializes all values by converting val from type K
     *      to T via utility::safe_cast. Only participates in overload resolution if that conversion is
     *      well-formed, i.e., if K and T are both integral or both floating-point
     * @tparam K The type of the values to convert from
     * @param val The value for x, y, and z
     * @exception Throws an Exception if a value is not exactly representable as T (see utility::safe_cast)
     */
    template <typename K>
        requires detail::safely_castable<T, K>
    constexpr explicit Vec3(const K& val) {
        if (std::is_constant_evaluated()) {
            x = static_cast<T>(val);
            y = static_cast<T>(val);
            z = static_cast<T>(val);
        } else {
            x = detail::invoke_safe_cast<T>(val);
            y = detail::invoke_safe_cast<T>(val);
            z = detail::invoke_safe_cast<T>(val);
        }
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

    /**
     * @brief Constructs a new instance and initializes all values by converting _x, _y, and _z from type K
     *      to T via utility::safe_cast. Only participates in overload resolution if that conversion is
     *      well-formed, i.e., if K and T are both integral or both floating-point
     * @tparam K The type of the values to convert from
     * @param _x The value for x
     * @param _y The value for y
     * @param _z The value for z
     * @exception Throws an Exception if a value is not exactly representable as T (see utility::safe_cast)
     */
    template <typename K>
        requires detail::safely_castable<T, K>
    constexpr explicit Vec3(const K& _x, const K& _y, const K& _z) {
        if (std::is_constant_evaluated()) {
            x = static_cast<T>(_x);
            y = static_cast<T>(_y);
            z = static_cast<T>(_z);
        } else {
            x = detail::invoke_safe_cast<T>(_x);
            y = detail::invoke_safe_cast<T>(_y);
            z = detail::invoke_safe_cast<T>(_z);
        }
    }

    constexpr Vec3(const Vec3<T>& other) = default;
    constexpr Vec3<T>& operator=(const Vec3<T>& other) = default;

    constexpr Vec3(Vec3<T>&& other) noexcept = default;
    constexpr Vec3<T>& operator=(Vec3<T>&& other) noexcept = default;

    /**
     * @brief Returns the zero vector, i.e., (0, 0, 0)
     * @return The zero vector
     */
    [[nodiscard]] static constexpr Vec3<T> zero() noexcept {
        return Vec3<T>{ T{ 0 }, T{ 0 }, T{ 0 } };
    }

    /**
     * @brief Returns the unit vector in x direction, i.e., (1, 0, 0)
     * @return The unit vector in x direction
     */
    [[nodiscard]] static constexpr Vec3<T> unit_x() noexcept {
        return Vec3<T>{ T{ 1 }, T{ 0 }, T{ 0 } };
    }

    /**
     * @brief Returns the unit vector in y direction, i.e., (0, 1, 0)
     * @return The unit vector in y direction
     */
    [[nodiscard]] static constexpr Vec3<T> unit_y() noexcept {
        return Vec3<T>{ T{ 0 }, T{ 1 }, T{ 0 } };
    }

    /**
     * @brief Returns the unit vector in z direction, i.e., (0, 0, 1)
     * @return The unit vector in z direction
     */
    [[nodiscard]] static constexpr Vec3<T> unit_z() noexcept {
        return Vec3<T>{ T{ 0 }, T{ 0 }, T{ 1 } };
    }

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
     * @brief Casts the current object to an object of type Vec3<K>. Uses static_cast<K> componentwise
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
     * @brief Compares the vectors lexicographically three-way
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
     * @brief Negates the vector componentwise and returns the result as a newly created object.
     *      Is only available if T is a signed type, as negating an unsigned value would silently wrap around
     * @return The componentwise negation as a new object
     */
    [[nodiscard]] constexpr Vec3<T> operator-() const noexcept
        requires std::is_signed_v<T>
    {
        return Vec3<T>{ static_cast<T>(-x), static_cast<T>(-y), static_cast<T>(-z) };
    }

    /**
     * @brief Rounds the current object componentwise to a larger multiple of value, calculating in FloatType
     *      and converting the results back to T.
     *      This is effectively an ugly function and should only be used with enough care.
     *      Uses 0.00001 as a magic constant. PROCEED WITH CARE!
     * @tparam FloatType The floating-point type in which the rounding is calculated, i.e., float, double, or long double.
     *      Defaults to the type the componentwise arithmetic is promoted to anyway, i.e., to double for every T but long double
     * @param value The positive value of which the components should be rounded to a multiple
     * @exception Throws an Exception if value is not positive
     */
    template <std::floating_point FloatType = std::common_type_t<T, double>>
    void round_to_larger_multiple(const T& value) {
        Exception::check(value > T{ 0 }, "Vec3::round_to_larger_multiple: value must be positive, was {}", value);

        const auto epsilon = static_cast<FloatType>(0.00001);
        const auto multiple = static_cast<FloatType>(value);
        const auto round_component = [&epsilon, &multiple](const T component) {
            return static_cast<T>(std::ceil((static_cast<FloatType>(component) - epsilon) / multiple) * multiple);
        };

        x = round_component(x);
        y = round_component(y);
        z = round_component(z);
    }

    /**
     * @brief Floors the current vector and converts the results to std::size_t componentwise
     * @exception Throws an Exception if a component is negative, non-finite, or outside the range of std::size_t
     * @return A newly created object with the floored values
     */
    [[nodiscard]] Vec3<std::size_t> floor_componentwise() const {
        const auto floor_component = [](const T component, const char* const name) {
            if constexpr (std::is_floating_point_v<T>) {
                Exception::check(std::isfinite(component), "Vec3::floor_componentwise: {} was not finite: {}", name, component);
            }
            Exception::check(component >= T{ 0 }, "Vec3::floor_componentwise: {} was negative: {}", name, component);

            if constexpr (std::is_floating_point_v<T>) {
                const auto floored = std::floor(component);
                const auto upper_exclusive = std::ldexp(static_cast<long double>(1), std::numeric_limits<std::size_t>::digits);
                Exception::check(static_cast<long double>(floored) < upper_exclusive,
                                 "Vec3::floor_componentwise: {} was too large for std::size_t: {}", name, component);
                return static_cast<std::size_t>(floored);
            } else if constexpr (std::same_as<std::remove_cv_t<T>, bool>) {
                return component ? std::size_t{ 1 } : std::size_t{ 0 };
            } else {
                Exception::check(std::in_range<std::size_t>(component),
                                 "Vec3::floor_componentwise: {} was too large for std::size_t: {}", name, component);
                return static_cast<std::size_t>(component);
            }
        };

        const auto floored_x = floor_component(x, "x");
        const auto floored_y = floor_component(y, "y");
        const auto floored_z = floor_component(z, "z");

        return Vec3<std::size_t>(floored_x, floored_y, floored_z);
    }

    /**
     * @brief Calculates the (signed) volume of the cuboid with the side length of the current object
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
     * @exception Throws an Exception for a signed integral component whose absolute value is not representable by T
     * @return The componentwise absolute values
     */
    [[nodiscard]] Vec3<T> abs() const noexcept(!std::signed_integral<T>) {
        if constexpr (std::is_unsigned_v<T>) {
            return *this;
        } else {
            if constexpr (std::signed_integral<T>) {
                Exception::check(x != std::numeric_limits<T>::min(), "Vec3::abs: abs(x) is not representable");
                Exception::check(y != std::numeric_limits<T>::min(), "Vec3::abs: abs(y) is not representable");
                Exception::check(z != std::numeric_limits<T>::min(), "Vec3::abs: abs(z) is not representable");
            }

            const auto abs_x = std::abs(x);
            const auto abs_y = std::abs(y);
            const auto abs_z = std::abs(z);

            using T0 = std::remove_cvref_t<T>;
            using U0 = std::remove_cvref_t<decltype(abs_x)>;

            if constexpr (!std::same_as<T0, U0>) {
                return Vec3<T>{ static_cast<T>(abs_x), static_cast<T>(abs_y), static_cast<T>(abs_z) };
            }

            return Vec3<T>{ abs_x, abs_y, abs_z };
        }
    }

    /**
     * @brief Calculates the p-norm of the current object in the precision of FloatType,
     *      using scaling to avoid avoidable intermediate overflow.
     *      FloatType is deduced from p, so passing a float, double, or long double exponent
     *      calculates the norm in exactly that type
     * @tparam FloatType The floating-point type in which the norm is calculated, i.e., float, double, or long double
     * @param p The finite exponent of the norm, must be >= 1
     * @exception Throws an Exception if p is not finite or is less than 1
     * @return The calculated p-norm in the precision of FloatType
     */
    template <std::floating_point FloatType>
    [[nodiscard]] FloatType calculate_p_norm(const FloatType p) const {
        Exception::check(std::isfinite(p) && p >= FloatType{ 1 },
                         "Vec3::calculate_p_norm: p must be finite and >= 1.0, but it was: {}", p);

        const auto abs_x = std::abs(static_cast<FloatType>(x));
        const auto abs_y = std::abs(static_cast<FloatType>(y));
        const auto abs_z = std::abs(static_cast<FloatType>(z));
        const auto scale = std::max({ abs_x, abs_y, abs_z });
        if (scale == FloatType{ 0 } || !std::isfinite(scale)) {
            return scale;
        }

        const auto sum = std::pow(abs_x / scale, p) + std::pow(abs_y / scale, p) + std::pow(abs_z / scale, p);
        return scale * std::pow(sum, FloatType{ 1 } / p);
    }

    /**
     * @brief Calculates the p-norm of the current object in double precision using scaling to avoid
     *      avoidable intermediate overflow. This is the default overload that is also selected for
     *      an integral exponent; use the templated overload above to calculate in another precision
     * @param p The finite exponent of the norm, must be >= 1.0
     * @exception Throws an Exception if p is not finite or is less than 1.0
     * @return The calculated p-norm
     */
    [[nodiscard]] double calculate_p_norm(const double p) const {
        return calculate_p_norm<double>(p);
    }

    /**
     * @brief Calculates the 1-norm of the vector (the sum of absolute component values)
     * @exception Throws an Exception if the result is not representable by T
     * @return The calculated 1-norm
     */
    [[nodiscard]] T calculate_1_norm() const noexcept(std::is_floating_point_v<T>) {
        if constexpr (std::same_as<std::remove_cv_t<T>, bool>) {
            const auto sum = static_cast<unsigned int>(x) + static_cast<unsigned int>(y) + static_cast<unsigned int>(z);
            Exception::check(sum <= 1U, "Vec3::calculate_1_norm: result {} is not representable by bool", sum);
            return sum != 0U;
        } else if constexpr (std::integral<T>) {
            using unsigned_type = std::make_unsigned_t<T>;
            const auto magnitude = [](const T value) constexpr -> unsigned_type {
                const auto converted = static_cast<unsigned_type>(value);
                if constexpr (std::signed_integral<T>) {
                    return value < T{ 0 } ? static_cast<unsigned_type>(unsigned_type{ 0 } - converted) : converted;
                } else {
                    return converted;
                }
            };

            constexpr auto max_result = static_cast<unsigned_type>(std::numeric_limits<T>::max());
            auto sum = unsigned_type{ 0 };
            for (const auto value : { x, y, z }) {
                const auto absolute = magnitude(value);
                Exception::check(absolute <= max_result - sum,
                                 "Vec3::calculate_1_norm: result is not representable by the component type");
                sum = static_cast<unsigned_type>(sum + absolute);
            }
            return static_cast<T>(sum);
        } else {
            return static_cast<T>(std::abs(x) + std::abs(y) + std::abs(z));
        }
    }

    /**
     * @brief Calculates the 2-norm in the precision of FloatType without squaring in T.
     *      Defaults to double, i.e., calculate_2_norm() calculates in double precision,
     *      while calculate_2_norm<float>() calculates in float
     * @tparam FloatType The floating-point type in which the norm is calculated, i.e., float, double, or long double
     * @return The calculated 2-norm in the precision of FloatType
     */
    template <std::floating_point FloatType = double>
    [[nodiscard]] FloatType calculate_2_norm() const noexcept {
        return std::hypot(static_cast<FloatType>(x), static_cast<FloatType>(y), static_cast<FloatType>(z));
    }

    /**
     * @brief Calculates the squared 2-norm in the precision of FloatType, i.e., ||this||^2_2.
     *      Defaults to double, i.e., calculate_squared_2_norm() calculates in double precision,
     *      while calculate_squared_2_norm<float>() calculates in float
     * @tparam FloatType The floating-point type in which the norm is calculated, i.e., float, double, or long double
     * @return The squared calculated 2-norm in the precision of FloatType
     */
    template <std::floating_point FloatType = double>
    [[nodiscard]] constexpr FloatType calculate_squared_2_norm() const noexcept {
        const auto x_as_float = static_cast<FloatType>(x);
        const auto y_as_float = static_cast<FloatType>(y);
        const auto z_as_float = static_cast<FloatType>(z);
        const auto xx = x_as_float * x_as_float;
        const auto yy = y_as_float * y_as_float;
        const auto zz = z_as_float * z_as_float;

        const auto sum = xx + yy + zz;
        return sum;
    }

    /**
     * @brief Calculates the dot product of *this and other, i.e., x*other.x + y*other.y + z*other.z
     * @param other The other vector
     * @return The dot product
     */
    [[nodiscard]] constexpr T calculate_dot_product(const Vec3<T>& other) const noexcept {
        const auto xx = x * other.x;
        const auto yy = y * other.y;
        const auto zz = z * other.z;

        const auto sum = xx + yy + zz;
        return sum;
    }

    /**
     * @brief Calculates the cross product of *this and other
     * @param other The other vector
     * @return The cross product as a new Vec3<T>
     */
    [[nodiscard]] constexpr Vec3<T> calculate_cross_product(const Vec3<T>& other) const noexcept {
        const auto xx = y * other.z - z * other.y;
        const auto yy = z * other.x - x * other.z;
        const auto zz = x * other.y - y * other.x;
        return Vec3{ xx, yy, zz };
    }

    /**
     * @brief Normalizes *this to a unit vector using the 2-norm. Casts the components to FloatType first.
     *      Defaults to double, i.e., normalize() normalizes in double precision,
     *      while normalize<float>() normalizes in float
     * @tparam FloatType The floating-point type in which the normalization is calculated, i.e., float, double, or long double
     * @exception Throws an Exception if *this is the zero vector or its norm is not finite in FloatType
     * @return A new Vec3<FloatType> with 2-norm equal to 1
     */
    template <std::floating_point FloatType = double>
    [[nodiscard]] Vec3<FloatType> normalize() const {
        const auto norm = calculate_2_norm<FloatType>();
        Exception::check(norm > FloatType{ 0 } && std::isfinite(norm),
                         "Vec3::normalize: norm must be positive and finite, was {}", norm);

        return Vec3<FloatType>{ static_cast<FloatType>(x) / norm, static_cast<FloatType>(y) / norm,
                                static_cast<FloatType>(z) / norm };
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
     * @brief Calculates the factorial of each component and multiplies the three results
     * @exception Throws an Exception if a component is negative, an intermediate factorial overflows,
     *      or the final product is not representable by the unsigned version of T
     * @return The product x! * y! * z! in the unsigned version of T
     */
    [[nodiscard]] constexpr auto get_componentwise_factorial() const {
        static_assert(std::is_integral_v<T>);
        static_assert(!std::same_as<std::remove_cv_t<T>, bool>);

        using unsigned_type_T = std::make_unsigned_t<T>;

        Exception::check(x >= 0, "Vec3::get_componentwise_factorial: x was < 0");
        Exception::check(y >= 0, "Vec3::get_componentwise_factorial: y was < 0");
        Exception::check(z >= 0, "Vec3::get_componentwise_factorial: z was < 0");

        const auto checked_multiply = [](const unsigned_type_T lhs, const unsigned_type_T rhs) constexpr {
            Exception::check(rhs == unsigned_type_T{ 0 } || lhs <= std::numeric_limits<unsigned_type_T>::max() / rhs,
                             "Vec3::get_componentwise_factorial: result overflows the component type");
            return static_cast<unsigned_type_T>(lhs * rhs);
        };
        const auto checked_factorial = [&checked_multiply](const unsigned_type_T value) constexpr {
            auto result = unsigned_type_T{ 1 };
            for (auto factor = unsigned_type_T{ 2 }; factor <= value; ++factor) {
                result = checked_multiply(result, factor);
            }
            return result;
        };

        const auto fac_x = checked_factorial(static_cast<unsigned_type_T>(x));
        const auto fac_y = checked_factorial(static_cast<unsigned_type_T>(y));
        const auto fac_z = checked_factorial(static_cast<unsigned_type_T>(z));

        return checked_multiply(checked_multiply(fac_x, fac_y), fac_z);
    }

    /**
     * @brief Calculates this^exponent componentwise and returns the product.
     *      Casts the components and the exponents to FloatType first, which defaults to double,
     *      i.e., get_componentwise_power<float>(exponent) calculates in float
     * @tparam FloatType The floating-point type in which the power is calculated, i.e., float, double, or long double
     * @param exponent The exponents for this
     * @return The product of the componentwise power in the precision of FloatType
     */
    template <std::floating_point FloatType = double>
    [[nodiscard]] FloatType get_componentwise_power(const Vec3<unsigned int>& exponent) const {
        const auto pow_x = std::pow(static_cast<FloatType>(x), static_cast<FloatType>(exponent.get_x()));
        const auto pow_y = std::pow(static_cast<FloatType>(y), static_cast<FloatType>(exponent.get_y()));
        const auto pow_z = std::pow(static_cast<FloatType>(z), static_cast<FloatType>(exponent.get_z()));

        const auto product = pow_x * pow_y * pow_z;
        return product;
    }

    /**
     * @brief Returns the midpoint between this and other, effectively the same as (*this + other) / 2,
     *      but without divison issues.
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
     * @exception Throws an Exception if lower <= upper is violated
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
     * @exception Throws an Exception if lower <= upper is violated
     */
    constexpr void clip_to_box(const Vec3<T>& lower, const Vec3<T>& upper) {
        Exception::check(lower.x <= upper.x, "Vec3::clip_to_box: lower.x ({}) is larger than upper.x ({})", lower.x, upper.x);
        Exception::check(lower.y <= upper.y, "Vec3::clip_to_box: lower.y ({}) is larger than upper.y ({})", lower.y, upper.y);
        Exception::check(lower.z <= upper.z, "Vec3::clip_to_box: lower.z ({}) is larger than upper.z ({})", lower.z, upper.z);

        x = std::clamp(x, lower.x, upper.x);
        y = std::clamp(y, lower.y, upper.y);
        z = std::clamp(z, lower.z, upper.z);
    }

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
        static_assert(Index < 3);

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
        static_assert(Index < 3);

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
        static_assert(Index < 3);

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

/**
 * @brief Disables fmt's tuple-like formatting for Vec3<T> (fmt/ranges.h),
 *      so that the dedicated formatter below is chosen unambiguously
 * @tparam T The component type of the Vec3
 */
template <typename T>
struct fmt::is_tuple_like<utility::Vec3<T>> {
    static constexpr bool value = false;
};

/**
 * @brief Formats a Vec3<T> via fmt in the format (x, y, z).
 *      The format specification is applied to each component, e.g.,
 *      fmt::format("{:.2f}", Vec3<double>{ 1.0, 2.0, 3.0 }) yields "(1.00, 2.00, 3.00)"
 * @tparam T The component type of the Vec3
 */
template <typename T>
struct fmt::formatter<utility::Vec3<T>> : fmt::formatter<T> {
    /**
     * @brief Formats the vector into the output of the format context
     * @param vector The vector that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted vector
     */
    auto format(const utility::Vec3<T>& vector, fmt::format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '(';
        ctx.advance_to(out);
        out = fmt::formatter<T>::format(vector.get_x(), ctx);
        *out++ = ',';
        *out++ = ' ';
        ctx.advance_to(out);
        out = fmt::formatter<T>::format(vector.get_y(), ctx);
        *out++ = ',';
        *out++ = ' ';
        ctx.advance_to(out);
        out = fmt::formatter<T>::format(vector.get_z(), ctx);
        *out++ = ')';
        return out;
    }
};
