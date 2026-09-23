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

#include <utility>
#include <vector>

namespace utility {

/**
 * This class provides a queue-like (FIFO) interface, uses an std::vector as container,
 * and allows to reserve space before.
 * Popped elements are cleaned up lazily: popping advances an internal index and compacts the
 * underlying vector once at least half of it consists of already popped elements,
 * which keeps popping amortized O(1).
 * References returned by front() and back() are invalidated by push, emplace, pop, and pop_front.
 * @tparam T The type of elements in the queue
 */
template <typename T>
class Queue {
public:
    using size_type = typename std::vector<T>::size_type;

    /**
     * @brief Constructs a new Queue with the specified reserved capacity
     * @param reserved_size The reserved capacity
     * @exception Throws an exception if the memory allocation fails
     */
    explicit Queue(const size_type reserved_size = 0) {
        container.reserve(reserved_size);
    }

    /**
     * @brief Emplaces a newly created element at the end of the queue
     * @tparam ValueType The type for the constructor of the element
     * @param Val The values for the constructor of the element
     * @exception Throws an exception if the memory allocation fails or the constructor of the element throws
     * @return A reference to the newly created element
     */
    template <class... ValueType>
    constexpr decltype(auto) emplace(ValueType&&... Val) {
        return container.emplace_back(std::forward<ValueType>(Val)...);
    }

    /**
     * @brief Pushes a copy of the element to the end of the queue
     * @param value The element that should be copied into the queue
     * @exception Throws an exception if the memory allocation fails or the copy constructor of the element throws
     */
    constexpr void push(const T& value) {
        container.push_back(value);
    }

    /**
     * @brief Moves the element to the end of the queue
     * @param value The element that should be moved into the queue
     * @exception Throws an exception if the memory allocation fails or the move constructor of the element throws
     */
    constexpr void push(T&& value) {
        container.push_back(std::move(value));
    }

    /**
     * @brief Returns a mutable reference to the first element, i.e., the element that entered the queue first
     * @exception Throws an Exception if the queue was empty
     * @return A mutable reference to the first element
     */
    [[nodiscard]] constexpr T& front() {
        Exception::check(!empty(), "Queue::front(): The queue was empty!");
        return container[head];
    }

    /**
     * @brief Returns an immutable reference to the first element, i.e., the element that entered the queue first
     * @exception Throws an Exception if the queue was empty
     * @return An immutable reference to the first element
     */
    [[nodiscard]] constexpr const T& front() const {
        Exception::check(!empty(), "Queue::front(): The queue was empty!");
        return container[head];
    }

    /**
     * @brief Returns a mutable reference to the last element, i.e., the element that entered the queue last
     * @exception Throws an Exception if the queue was empty
     * @return A mutable reference to the last element
     */
    [[nodiscard]] constexpr T& back() {
        Exception::check(!empty(), "Queue::back(): The queue was empty!");
        return container.back();
    }

    /**
     * @brief Returns an immutable reference to the last element, i.e., the element that entered the queue last
     * @exception Throws an Exception if the queue was empty
     * @return An immutable reference to the last element
     */
    [[nodiscard]] constexpr const T& back() const {
        Exception::check(!empty(), "Queue::back(): The queue was empty!");
        return container.back();
    }

    /**
     * @brief Removes the first element
     * @exception Throws an Exception if the queue was empty
     * @exception Propagates exceptions from moving elements when lazy compaction is triggered
     */
    constexpr void pop() {
        Exception::check(!empty(), "Queue::pop(): The queue was empty!");
        head++;
        compact();
    }

    /**
     * @brief Returns the first element stored in the queue and pops it as well.
     * @exception Throws an Exception if the queue was empty
     * @exception Propagates exceptions from moving the result or from moving elements during lazy compaction
     * @return The first element
     */
    [[nodiscard]] constexpr T pop_front() {
        Exception::check(!empty(), "Queue::pop_front(): The queue was empty!");

        auto result = std::move(container[head]);
        head++;
        compact();
        return result;
    }

    /**
     * @brief Reserves the specified capacity for the underlying vector.
     *      Does nothing if the current capacity is larger than the specified one
     * @param new_capacity The to-be-reserved capacity
     * @exception Throws an exception if allocating the memory fails
     */
    constexpr void reserve(const size_type new_capacity) {
        container.reserve(new_capacity);
    }

    /**
     * @brief Returns the number of currently stored elements
     * @return The number of currently stored elements
     */
    [[nodiscard]] constexpr size_type size() const noexcept {
        return container.size() - head;
    }

    /**
     * @brief Returns the current capacity of the underlying vector.
     *      Slots of already popped but not yet compacted elements count towards it
     * @return The current capacity
     */
    [[nodiscard]] constexpr size_type capacity() const noexcept {
        return container.capacity();
    }

    /**
     * @brief Returns whether the queue is empty
     * @return True iff the queue is empty
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return head == container.size();
    }

    /**
     * @brief Deletes all elements
     */
    constexpr void clear() noexcept {
        container.clear();
        head = 0;
    }

private:
    /**
     * @brief Reclaims the slots of already popped elements once they make up
     *      at least half of the underlying vector
     */
    constexpr void compact() {
        if (head == container.size()) {
            container.clear();
            head = 0;
            return;
        }

        if (head >= container.size() - head) {
            using difference_type = typename std::vector<T>::difference_type;
            container.erase(container.begin(), container.begin() + static_cast<difference_type>(head));
            head = 0;
        }
    }

    std::vector<T> container{};
    size_type head{ 0 };
};

} // namespace utility
