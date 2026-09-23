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

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <range/v3/range/conversion.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

class RelearnTest : public ::testing::Test {
protected:
    void SetUp() override; // Called immediately after the constructor for each test

    void TearDown() override; // Called immediately before the destructor for each test

    static size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
        auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
        auto rounded_exp = std::ceil(log);
        auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
        return static_cast<size_t>(new_val);
    }

    /**
     * @brief Returns the tolerance with which two values of a floating point type may differ once they were
     *      accumulated in a different order or round-tripped through a decimal representation. The numeric
     *      aliases of this project are switchable between double and float, and a float resolves a value only
     *      relative to its magnitude, so an absolute epsilon does not suffice for the larger ones.
     * @tparam T The floating point type the compared values are stored in
     * @param magnitude The magnitude of the compared values
     * @return The tolerance, never less than RelearnTest::eps
     */
    template <typename T>
    [[nodiscard]] static double tolerance_for(const double magnitude) {
        static_assert(std::is_floating_point_v<T>);
        constexpr auto relative_tolerance = double{ 128 } * static_cast<double>(std::numeric_limits<T>::epsilon());
        return std::max(eps, std::abs(magnitude) * relative_tolerance);
    }

    constexpr static int number_neurons_out_of_scope = 100;

    std::mt19937 mt;

    static std::size_t iterations;
    static double eps;

private:
    static bool use_predetermined_seed;
    static std::mt19937::result_type predetermined_seed;
};

class RelearnMemoryTest : public RelearnTest {
protected:
    RelearnMemoryTest();
};

/**
 * Macro that performs ASSERT_THROW of the google test library but suppresses the messages of the RelearnException to the console.
 * Call this macro only within a google test. This macro checks if a given function throws an exception of the given class.
 * If not, the test fails.
 * @param fun Function that shall throw the exception
 * @param clazz Class of the exception that should be thrown
 * @param msg Message that will be printed if the function does not throw an exception
 */
#define ASSERT_THROW_NO_PRINT_MSG(fun, clazz, msg) \
    RelearnException::hide_messages = true;        \
    ASSERT_THROW(fun, clazz) << (msg);             \
    RelearnException::hide_messages = false;

/**
 * Macro that performs ASSERT_THROW of the google test library but suppresses the messages of the RelearnException to the console.
 * Call this macro only within a google test. This macro checks if a given function throws an exception of the given class.
 * If not, the test fails.
 * @param fun Function that shall throw the exception
 * @param clazz Class of the exception that should be thrown
 * @param msg Message that will be printed if the function does not throw an exception
 */
#define ASSERT_THROW_NO_PRINT(fun, clazz) ASSERT_THROW_NO_PRINT_MSG(fun, clazz, "");

#define ASSERT_NEAR_EPS(x, y) ASSERT_NEAR(x, y, eps)