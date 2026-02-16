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

#include <string>
#include <string_view>
#include <vector>

/**
 * This class provides a static interface to load interrupts from files, i.e., when during the simulation the neurons should be altered.
 */
class StringUtil {
public:
    /**
     * @brief Split a string based on a delimiter character in a list of substrings.
     *      Empty strings within two delimiters or at the beginning are kept while one at the end is discarded
     * @param string The string to split
     * @param delim Single char used as delimiter
     * @return Vector of substrings
     */
    static std::vector<std::string> split_string(const std::string& string, char delim);

    /**
     * @brief Split a string based on a delimiter character in a list of substrings.
     *      Empty strings within two delimiters or at the beginning are kept while one at the end is discarded
     * @param string_view The string to split, passed as a view
     * @param delim Single char used as delimiter
     * @return Vector of substrings
     */
    static std::vector<std::string> split_string(std::string_view string_view, char delim);

    /**
     * @brief Checks if the string contains only digits (i.e., cannot handle "-12321")
     * @param s The string to check, passed as a view
     * @return true iff string is a number
     */
    static bool is_number(std::string_view s);

    /**
     * @brief Changes a string such that all characters are in lower-case
     * @param str The string to change, a reference, i.e., is changed in-place
     */
    static void to_lower(std::string& str);

    /**
     * Converts an integer to a string with leading zeros, having at least the number of specified digits
     * @param number The number will be converted to a string
     * @param nr_of_digits Number of digits including the leading zeros
     * @exception Throws a RelearnException if nr_of_digits == 0
     * @return string with the number and leading zeros if necessary
     */
    static std::string format_int_with_leading_zeros(int number, unsigned int nr_of_digits);
};
