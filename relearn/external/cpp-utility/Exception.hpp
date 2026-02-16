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

#include <fmt/core.h>
#include <fmt/format.h>

#include <exception>
#include <string>
#include <utility>

namespace utility {

/**
 * This class serves as a collective exception class that can check for conditions,
 * and in case of the condition evaluating to false, it logs the message and then fails.
 * Log messages can be disabled via Exception::hide_messages.
 * In case a condition evaluated to false and it logs the message, it calls MPIWrapper::get_number_ranks and MPIWrapper::get_my_rank.
 */
class Exception : public std::exception {
public:
    /**
     * @brief Allows to hide the messages, i.e., not print the messages to std::cerr
     */
    static inline bool hide_messages{ false };

    /**
     * @brief Returns the cause of the exception, i.e., the stored message
     * @return A constant char pointer to the content of the message
     */
    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }

    /**
     * @brief Checks the condition and in case of false, logs the message and throws an Exception
     * @tparam FormatString A string-like type
     * @tparam Args Different types that can be substituted into the placeholders
     * @param condition The condition to evaluate
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a Exception if the condition evaluates to false
     */
    template <typename FormatString, typename... Args>
    static constexpr void check(const bool condition, FormatString&& format, Args&&... args) {
        if (condition) {
            return;
        }

        fail(std::forward<FormatString>(format), std::forward<Args>(args)...);
    }

    /**
     * @brief Prints the log message and throws an Exception afterwards
     * @tparam FormatString A string-like type
     * @tparam Args Different types that can be substituted into the placeholders
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a Exception
     */
    template <typename FormatString, typename... Args>
    [[noreturn]] static constexpr void fail(FormatString&& format, Args&&... args) {
        if (hide_messages) {
            throw Exception{};
        }

        auto message = fmt::format(fmt::runtime(std::forward<FormatString>(format)), std::forward<Args>(args)...);
        throw Exception{ std::move(message) };
    }

private:
    std::string message;

    /**
     * @brief Default constructs an instance with empty message
     */
    Exception() = default;

    /**
     * @brief Constructs an instance with the associated message
     * @param mes The message of the exception
     */
    explicit Exception(std::string&& mes)
        : message(std::move(mes)) {
    }
};

} // namespace utility
