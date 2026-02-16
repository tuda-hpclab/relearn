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

#include <fmt/core.h>
#include <fmt/format.h>

#include <exception>
#include <string>
#include <string_view>

/**
 * This class serves as a collective exception class that can check for conditions,
 * and in case of the condition evaluating to false, it logs the message and then fails.
 * Log messages can be disabled via RelearnException::hide_messages.
 * In case a condition evaluated to false and it logs the message, it calls MPIWrapper::get_number_ranks and MPIWrapper::get_my_rank.
 */
class RelearnException : public std::exception {
public:
    /**
     * @brief Allows to hide the messages, i.e., not print the messages to std::
     */
    static inline bool hide_messages{ false };

    /**
     * @brief Returns the cause of the exception, i.e., the stored message
     * @return A constant char pointer to the content of the message
     */
    [[nodiscard]] const char* what() const noexcept override;

    /**
     * @brief Checks the condition and in case of false, logs the message and throws an RelearnException
     * @tparam Args Different types that can be substituted into the placeholders
     * @param condition The condition to evaluate
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a RelearnException if the condition evaluates to false
     */
    template <typename... Args>
    static constexpr void check(bool condition, const std::string_view format, Args&&... args) {
        if (condition) {
            return;
        }

        fail(format, std::forward<Args>(args)...);
    }

    /**
     * @brief Prints the log message and throws a RelearnException afterwards
     * @tparam Args Different types that can be substituted into the placeholders
     * @param format The format string. Placeholders can used: "{}"
     * @param args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a RelearnException
     */
    template <typename... Args>
    [[noreturn]] static constexpr void fail(const std::string_view format, Args&&... args) {
        if (hide_messages) {
            throw RelearnException{};
        }

        auto message = fmt::format(fmt::runtime(format), std::forward<Args>(args)...);
        log_message(message);
        throw RelearnException{ std::move(message) };
    }

private:
    std::string message{};

    /**
     * @brief Default constructs an instance with empty message
     */
    RelearnException() = default;

    /**
     * @brief Constructs an instance with the associated message
     * @param mes The message of the exception
     */
    explicit RelearnException(std::string&& mes)
        : message(std::move(mes)) {
    }

    static void log_message(const std::string& message);
};
