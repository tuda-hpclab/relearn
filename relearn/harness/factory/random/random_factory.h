#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Macros.h"

#include "util/RelearnException.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#ifdef HOST_COMPILER
#include "cpp-utility/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#pragma GCC diagnostic pop

#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/transform.hpp>
#endif

class RandomFactory {
public:
    template <typename T>
    static T get_random_double(T min, T max, std::mt19937& mt) {
        const boost::random::uniform_real_distribution<double> urd(static_cast<double>(min), static_cast<double>(max));
        return static_cast<T>(urd(mt));
    }

    static double get_random_double(std::mt19937& mt) {
        return get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
    }

    template <typename T>
    static T get_random_integer(T min, T max, std::mt19937& mt) {
        const boost::random::uniform_int_distribution<T> uid(min, max);
        return uid(mt);
    }

    template <typename T>
    static T get_random_integer(T min, T max, T avoid, std::mt19937& mt) {
        const boost::random::uniform_int_distribution<T> uid(min, max);
        while (true) {
            const auto val = uid(mt);
            if (val != avoid) {
                return val;
            }
        }
    }

    template <typename T>
    static T get_random_element(const std::vector<T>& elements, std::mt19937& mt) {
        const auto index = RandomFactory::get_random_integer(std::size_t{ 0 }, elements.size() - 1, mt);
        return elements[index];
    }

    template <typename T>
    static T get_random_integer(std::mt19937& mt) {
        return get_random_integer<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), mt);
    }

    template <typename T>
    static T get_random_percentage(std::mt19937& mt) {
        return get_random_double<T>(T{ 0 }, std::nextafter(T{ 1 }, T{ 2 }), mt);
    }

    static bool get_random_bool(std::mt19937& mt) {
        const auto val = get_random_integer(0, 1, mt);
        return val == 0;
    }

    static bool get_random_bool(double probability,std::mt19937& mt) {
        const auto val = get_random_double(0.0, 1.0, mt);
        return val < probability;
    }

    static std::string get_random_string(size_t length, std::mt19937& mt) {
        auto randchar = [&mt]() -> char {
            const std::string_view charset = "0123456789"
                                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                             "abcdefghijklmnopqrstuvwxyz";
            const std::size_t max_index = charset.size() - 1;
            return charset[get_random_integer(size_t{ 0 }, max_index, mt)];
        };
        std::string str(length, 0);
        std::generate_n(str.begin(), length, randchar);
        return str;
    }

#ifdef HOST_COMPILER
    static std::vector<size_t> get_random_derangement(size_t size, std::mt19937& mt) {
        auto derangement = ranges::views::indices(size) | ranges::to_vector;

        auto check = [](const std::vector<size_t>& vec) {
            const auto index_equals_value = [](const auto& IndexValuePair) {
                const auto& [Index, Value] = IndexValuePair;
                return Index == Value;
            };

            return ranges::any_of(vec | ranges::views::enumerate, index_equals_value);
        };

        if (size <= 1) {
            return derangement;
        }

        shuffle(derangement, mt);
        while (!check(derangement)) {
            shuffle(derangement, mt);
        }

        return derangement;
    }

    template <typename Iterator>
    static void shuffle(Iterator begin, Iterator end, std::mt19937& mt) {
        ::shuffle(begin, end, mt);
    }

    template <typename Range>
    static void shuffle(Range& range, std::mt19937& mt) {
        ::shuffle(range, mt);
    }

    template <typename T>
    static std::unordered_set<T> sample_from_integer_range(T min_inclusive, T max_inclusive, size_t sample_size, std::mt19937& mt) {
        RelearnException::check(max_inclusive - min_inclusive + 1 >= sample_size, "RandomFactory::sample_from_integer_range: Sample size must be larger than integer range");
        std::unordered_set<T> set{};
        for (auto i = 0ULL; i < sample_size; i++) {
            T random_number = get_random_integer(min_inclusive, max_inclusive, mt);
            while (set.contains(random_number)) {
                random_number = get_random_integer(min_inclusive, max_inclusive, mt);
            }
            set.insert(random_number);
        }
        return set;
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, size_t sample_size, std::mt19937& mt) {
        std::unordered_set<size_t> indices{};
        while (indices.size() != sample_size) {
            auto index = get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
            while (indices.contains(index)) {
                index = get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
            }
            indices.insert(index);
        }

        return indices
               | ranges::views::transform(utility::lookup(vector))
               | ranges::to_vector
               | actions::shuffle(mt);
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, std::mt19937& mt) {
        const auto sample_size = get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
        return sample(vector, sample_size, mt);
    }
#endif
};
