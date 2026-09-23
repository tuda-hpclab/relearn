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

#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A vector with a fixed capacity whose storage lives inline in the object, i.e., it never allocates on the heap.
 *
 * Elements are constructed and destroyed on demand via placement new,
 * so T does not need to be default constructible.
 * Exceeding the capacity does not grow the vector but throws an Exception.
 * Appending does not invalidate references, pointers, or iterators; removing an element invalidates
 * references and pointers to that element, while clear() invalidates all of them.
 *
 * @tparam T The type of the elements
 * @tparam Capacity The maximum number of elements
 */
template <typename T, std::size_t Capacity>
class StaticVector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    /**
     * @brief Constructs a new empty instance
     */
    StaticVector() noexcept = default;

    /**
     * @brief Constructs a new instance with the elements from the list
     * @param list The initial elements
     * @exception Throws an Exception if the list holds more than Capacity elements, or propagates
     *      an exception from T's copy constructor without leaking already constructed elements
     */
    StaticVector(std::initializer_list<T> list) {
        Exception::check(list.size() <= Capacity,
                         "StaticVector::StaticVector(): The initializer list was too long: {} > {}", list.size(), Capacity);

        try {
            for (const auto& value : list) {
                std::construct_at(data() + number_elements, value);
                number_elements++;
            }
        } catch (...) {
            clear();
            throw;
        }
    }

    /**
     * @brief Constructs a new instance by copying all elements from the other instance
     * @param other The other instance
     * @exception Propagates an exception from T's copy constructor without leaking already constructed elements
     */
    StaticVector(const StaticVector& other) {
        try {
            for (const auto& value : other) {
                std::construct_at(data() + number_elements, value);
                number_elements++;
            }
        } catch (...) {
            clear();
            throw;
        }
    }

    /**
     * @brief Replaces the elements with copies of the elements from the other instance
     * @param other The other instance
     * @exception Propagates an exception from T's copy constructor; if this happens, this object remains valid
     *      and contains the successfully copied prefix
     * @return A reference to the current object
     */
    StaticVector& operator=(const StaticVector& other) {
        if (this == &other) {
            return *this;
        }

        clear();
        for (const auto& value : other) {
            std::construct_at(data() + number_elements, value);
            number_elements++;
        }
        return *this;
    }

    /**
     * @brief Constructs a new instance by moving all elements out of the other instance.
     *      The other instance is empty afterwards
     * @param other The other instance
     */
    StaticVector(StaticVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if constexpr (std::is_nothrow_move_constructible_v<T>) {
            for (auto& value : other) {
                std::construct_at(data() + number_elements, std::move(value));
                number_elements++;
            }
        } else {
            try {
                for (auto& value : other) {
                    std::construct_at(data() + number_elements, std::move(value));
                    number_elements++;
                }
            } catch (...) {
                clear();
                throw;
            }
        }
        other.clear();
    }

    /**
     * @brief Replaces the elements by moving all elements out of the other instance.
     *      The other instance is empty afterwards
     * @param other The other instance
     * @exception Propagates an exception from T's move constructor; if this happens, both objects remain valid,
     *      this object contains the successfully moved prefix, and other may contain moved-from elements
     * @return A reference to the current object
     */
    StaticVector& operator=(StaticVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this == &other) {
            return *this;
        }

        clear();
        for (auto& value : other) {
            std::construct_at(data() + number_elements, std::move(value));
            number_elements++;
        }
        other.clear();
        return *this;
    }

    ~StaticVector() {
        clear();
    }

    /**
     * @brief Emplaces a newly created element at the end of the vector
     * @tparam Ts The types for the constructor of the element
     * @param args The values for the constructor of the element
     * @exception Throws an Exception if the vector was full, or an exception if the constructor of the element throws
     * @return A reference to the newly created element
     */
    template <typename... Ts>
    reference emplace_back(Ts&&... args)
        requires std::constructible_from<T, Ts...>
    {
        Exception::check(!full(), "StaticVector::emplace_back(): The vector was full, capacity is {}!", Capacity);

        auto* const element = std::construct_at(data() + number_elements, std::forward<Ts>(args)...);
        number_elements++;
        return *element;
    }

    /**
     * @brief Pushes a copy of the element to the end of the vector
     * @param value The element that should be copied into the vector
     * @exception Throws an Exception if the vector was full, or an exception if the copy constructor of the element throws
     */
    void push_back(const T& value) {
        Exception::check(!full(), "StaticVector::push_back(): The vector was full, capacity is {}!", Capacity);

        std::construct_at(data() + number_elements, value);
        number_elements++;
    }

    /**
     * @brief Moves the element to the end of the vector
     * @param value The element that should be moved into the vector
     * @exception Throws an Exception if the vector was full, or an exception if the move constructor of the element throws
     */
    void push_back(T&& value) {
        Exception::check(!full(), "StaticVector::push_back(): The vector was full, capacity is {}!", Capacity);

        std::construct_at(data() + number_elements, std::move(value));
        number_elements++;
    }

    /**
     * @brief Removes the last element and destroys it
     * @exception Throws an Exception if the vector was empty
     */
    void pop_back() {
        Exception::check(!empty(), "StaticVector::pop_back(): The vector was empty!");

        number_elements--;
        std::destroy_at(data() + number_elements);
    }

    /**
     * @brief Returns a mutable reference to the element at the specified position. The position is not checked
     * @param pos The position of the element
     * @return A mutable reference to the element
     */
    [[nodiscard]] reference operator[](const size_type pos) noexcept {
        return data()[pos];
    }

    /**
     * @brief Returns an immutable reference to the element at the specified position. The position is not checked
     * @param pos The position of the element
     * @return An immutable reference to the element
     */
    [[nodiscard]] const_reference operator[](const size_type pos) const noexcept {
        return data()[pos];
    }

    /**
     * @brief Returns a mutable reference to the element at the specified position
     * @param pos The position of the element
     * @exception Throws an Exception if pos is out of range
     * @return A mutable reference to the element
     */
    [[nodiscard]] reference at(const size_type pos) {
        Exception::check(pos < number_elements, "StaticVector::at(): The position {} was out of range, size is {}!", pos, number_elements);
        return data()[pos];
    }

    /**
     * @brief Returns an immutable reference to the element at the specified position
     * @param pos The position of the element
     * @exception Throws an Exception if pos is out of range
     * @return An immutable reference to the element
     */
    [[nodiscard]] const_reference at(const size_type pos) const {
        Exception::check(pos < number_elements, "StaticVector::at(): The position {} was out of range, size is {}!", pos, number_elements);
        return data()[pos];
    }

    /**
     * @brief Returns a mutable reference to the first element
     * @exception Throws an Exception if the vector was empty
     * @return A mutable reference to the first element
     */
    [[nodiscard]] reference front() {
        Exception::check(!empty(), "StaticVector::front(): The vector was empty!");
        return data()[0];
    }

    /**
     * @brief Returns an immutable reference to the first element
     * @exception Throws an Exception if the vector was empty
     * @return An immutable reference to the first element
     */
    [[nodiscard]] const_reference front() const {
        Exception::check(!empty(), "StaticVector::front(): The vector was empty!");
        return data()[0];
    }

    /**
     * @brief Returns a mutable reference to the last element
     * @exception Throws an Exception if the vector was empty
     * @return A mutable reference to the last element
     */
    [[nodiscard]] reference back() {
        Exception::check(!empty(), "StaticVector::back(): The vector was empty!");
        return data()[number_elements - 1];
    }

    /**
     * @brief Returns an immutable reference to the last element
     * @exception Throws an Exception if the vector was empty
     * @return An immutable reference to the last element
     */
    [[nodiscard]] const_reference back() const {
        Exception::check(!empty(), "StaticVector::back(): The vector was empty!");
        return data()[number_elements - 1];
    }

    /**
     * @brief Returns a pointer to the contiguous storage of the elements
     * @return A pointer to the first element
     */
    [[nodiscard]] T* data() noexcept {
        return reinterpret_cast<T*>(storage.data());
    }

    /**
     * @brief Returns a constant pointer to the contiguous storage of the elements
     * @return A constant pointer to the first element
     */
    [[nodiscard]] const T* data() const noexcept {
        return reinterpret_cast<const T*>(storage.data());
    }

    [[nodiscard]] iterator begin() noexcept { return data(); }
    [[nodiscard]] iterator end() noexcept { return number_elements == 0 ? data() : data() + number_elements; }
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }
    [[nodiscard]] const_iterator end() const noexcept { return number_elements == 0 ? data() : data() + number_elements; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }

    /**
     * @brief Returns the number of currently stored elements
     * @return The number of currently stored elements
     */
    [[nodiscard]] size_type size() const noexcept {
        return number_elements;
    }

    /**
     * @brief Returns the capacity, i.e., the maximum number of elements
     * @return The capacity
     */
    [[nodiscard]] static constexpr size_type capacity() noexcept {
        return Capacity;
    }

    /**
     * @brief Returns whether the vector is empty
     * @return True iff the vector is empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return number_elements == 0;
    }

    /**
     * @brief Returns whether the vector is full, i.e., whether it holds Capacity elements
     * @return True iff the vector is full
     */
    [[nodiscard]] bool full() const noexcept {
        return number_elements == Capacity;
    }

    /**
     * @brief Destroys all elements
     */
    void clear() noexcept {
        while (number_elements > 0) {
            number_elements--;
            std::destroy_at(data() + number_elements);
        }
    }

private:
    alignas(T) std::array<std::byte, Capacity * sizeof(T)> storage;
    size_type number_elements{ 0 };
};

} // namespace utility
