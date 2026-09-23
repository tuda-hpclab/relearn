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

#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A fixed-capacity FIFO buffer backed by circular storage.
 *
 * Elements are constructed on insertion and destroyed immediately on removal. Once the configured capacity
 * is reached, insertion throws instead of overwriting the oldest element. Insertions and removals are O(1)
 * and never allocate after construction. References are invalidated when their element is removed, when clear()
 * is called, or when the buffer is assigned to, moved from, or destroyed; operations on other elements keep them valid.
 *
 * @tparam T The type of the buffered elements
 */
template <typename T>
class RingBuffer {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    /** @brief Constructs a buffer with zero capacity */
    RingBuffer() = default;

    /**
     * @brief Constructs an empty buffer with the specified capacity
     * @param new_capacity The maximum number of elements
     */
    explicit RingBuffer(const size_type new_capacity)
        : slots(new_capacity) {
    }

    RingBuffer(const RingBuffer&) = default;
    RingBuffer& operator=(const RingBuffer&) = default;

    RingBuffer(RingBuffer&& other) noexcept(std::is_nothrow_move_constructible_v<decltype(slots)>)
        : slots(std::move(other.slots))
        , head(std::exchange(other.head, 0))
        , number_elements(std::exchange(other.number_elements, 0)) {
    }

    RingBuffer& operator=(RingBuffer&& other) noexcept(std::is_nothrow_move_assignable_v<decltype(slots)>) {
        if (this == &other) {
            return *this;
        }

        slots = std::move(other.slots);
        head = std::exchange(other.head, 0);
        number_elements = std::exchange(other.number_elements, 0);
        return *this;
    }

    ~RingBuffer() = default;

    /**
     * @brief Constructs an element at the back of the buffer
     * @tparam Args The element constructor's argument types
     * @param args The element constructor's arguments
     * @exception Throws an Exception if the buffer is full
     * @return A reference to the inserted element
     */
    template <typename... Args>
        requires std::constructible_from<value_type, Args...>
    reference emplace(Args&&... args) {
        Exception::check(!full(), "RingBuffer::emplace: the buffer is full, capacity is {}", capacity());

        auto& slot = slots[to_physical_index(number_elements)];
        auto& value = slot.emplace(std::forward<Args>(args)...);
        ++number_elements;
        return value;
    }

    /**
     * @brief Appends a copy of value
     * @param value The value to copy
     * @exception Throws an Exception if the buffer is full
     */
    void push(const value_type& value) {
        static_cast<void>(emplace(value));
    }

    /**
     * @brief Appends value by moving it
     * @param value The value to move
     * @exception Throws an Exception if the buffer is full
     */
    void push(value_type&& value) {
        static_cast<void>(emplace(std::move(value)));
    }

    /**
     * @brief Returns the oldest element
     * @exception Throws an Exception if the buffer is empty
     * @return A mutable reference to the oldest element
     */
    [[nodiscard]] reference front() {
        Exception::check(!empty(), "RingBuffer::front: the buffer is empty");
        return *slots[head];
    }

    /**
     * @brief Returns the oldest element
     * @exception Throws an Exception if the buffer is empty
     * @return An immutable reference to the oldest element
     */
    [[nodiscard]] const_reference front() const {
        Exception::check(!empty(), "RingBuffer::front: the buffer is empty");
        return *slots[head];
    }

    /**
     * @brief Returns the newest element
     * @exception Throws an Exception if the buffer is empty
     * @return A mutable reference to the newest element
     */
    [[nodiscard]] reference back() {
        Exception::check(!empty(), "RingBuffer::back: the buffer is empty");
        return *slots[to_physical_index(number_elements - 1)];
    }

    /**
     * @brief Returns the newest element
     * @exception Throws an Exception if the buffer is empty
     * @return An immutable reference to the newest element
     */
    [[nodiscard]] const_reference back() const {
        Exception::check(!empty(), "RingBuffer::back: the buffer is empty");
        return *slots[to_physical_index(number_elements - 1)];
    }

    /**
     * @brief Removes and destroys the oldest element
     * @exception Throws an Exception if the buffer is empty
     */
    void pop() {
        Exception::check(!empty(), "RingBuffer::pop: the buffer is empty");
        slots[head].reset();
        --number_elements;
        head = number_elements == 0 ? 0 : increment(head);
    }

    /**
     * @brief Moves the oldest element out of the buffer and removes it
     * @exception Throws an Exception if the buffer is empty
     * @return The removed element
     */
    [[nodiscard]] value_type pop_front()
        requires std::move_constructible<value_type>
    {
        Exception::check(!empty(), "RingBuffer::pop_front: the buffer is empty");
        auto result = std::move(*slots[head]);
        pop();
        return result;
    }

    /** @brief Removes and destroys all elements while preserving the capacity */
    void clear() noexcept {
        for (auto i = size_type{ 0 }; i < number_elements; ++i) {
            slots[to_physical_index(i)].reset();
        }
        head = 0;
        number_elements = 0;
    }

    /** @return The number of stored elements */
    [[nodiscard]] size_type size() const noexcept {
        return number_elements;
    }

    /** @return The maximum number of elements */
    [[nodiscard]] size_type capacity() const noexcept {
        return slots.size();
    }

    /** @return True iff the buffer holds no elements */
    [[nodiscard]] bool empty() const noexcept {
        return number_elements == 0;
    }

    /** @return True iff the number of elements equals the capacity */
    [[nodiscard]] bool full() const noexcept {
        return number_elements == capacity();
    }

    /**
     * @brief Exchanges the contents and capacities of two buffers
     * @param other The other buffer
     */
    void swap(RingBuffer& other) noexcept(noexcept(slots.swap(other.slots))) {
        slots.swap(other.slots);
        std::swap(head, other.head);
        std::swap(number_elements, other.number_elements);
    }

    friend void swap(RingBuffer& lhs, RingBuffer& rhs) noexcept(noexcept(lhs.swap(rhs))) {
        lhs.swap(rhs);
    }

private:
    [[nodiscard]] size_type increment(const size_type index) const noexcept {
        return index + 1 == capacity() ? 0 : index + 1;
    }

    [[nodiscard]] size_type to_physical_index(const size_type logical_index) const noexcept {
        const auto tail_space = capacity() - head;
        return logical_index < tail_space ? head + logical_index : logical_index - tail_space;
    }

    std::vector<std::optional<value_type>> slots{};
    size_type head{ 0 };
    size_type number_elements{ 0 };
};

} // namespace utility
