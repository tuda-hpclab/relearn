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

#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A contiguous vector that stores its first InlineCapacity elements inside the vector object.
 *
 * The vector allocates heap storage only when its size grows beyond InlineCapacity. Its capacity then grows
 * geometrically. References, pointers, and iterators stay valid while an insertion fits into the current capacity;
 * reserve(), shrink_to_fit(), and a growing insertion invalidate all of them. Removing the last element invalidates
 * references, pointers, and iterators to that element. Moving a heap-backed vector preserves pointers to its elements,
 * while moving an inline-backed vector does not.
 *
 * @tparam T The element type
 * @tparam InlineCapacity The number of elements that fit into the object without a heap allocation
 */
template <typename T, std::size_t InlineCapacity>
class SmallVector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_assert(InlineCapacity <= std::numeric_limits<size_type>::max() / sizeof(value_type),
                  "SmallVector: InlineCapacity * sizeof(T) must be representable by size_t");

    /** @brief Constructs an empty vector backed by its inline storage */
    SmallVector() noexcept = default;

    /**
     * @brief Constructs a vector from an initializer list
     * @param values The values to copy
     */
    SmallVector(std::initializer_list<value_type> values)
        requires std::is_copy_constructible_v<value_type>
    {
        try {
            reserve(values.size());
            for (const auto& value : values) {
                std::construct_at(data() + number_elements, value);
                ++number_elements;
            }
        } catch (...) {
            destroy_elements();
            release_heap();
            throw;
        }
    }

    /** @brief Constructs an independent copy of other */
    SmallVector(const SmallVector& other)
        requires std::is_copy_constructible_v<value_type>
    {
        try {
            reserve(other.size());
            for (const auto& value : other) {
                std::construct_at(data() + number_elements, value);
                ++number_elements;
            }
        } catch (...) {
            destroy_elements();
            release_heap();
            throw;
        }
    }

    /**
     * @brief Replaces the elements with copies of other's elements
     * @return A reference to this vector
     */
    SmallVector& operator=(const SmallVector& other)
        requires(std::is_copy_constructible_v<value_type> && std::is_move_constructible_v<value_type>)
    {
        if (this == &other) {
            return *this;
        }

        auto copy = SmallVector{ other };
        *this = std::move(copy);
        return *this;
    }

    /**
     * @brief Moves other into this vector. Heap storage is transferred without moving its elements;
     *      inline elements are move-constructed. other is empty afterwards
     */
    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<value_type>)
        requires std::is_move_constructible_v<value_type>
    {
        move_construct_from(other);
    }

    /**
     * @brief Replaces the elements by moving from other. other is empty afterwards
     * @return A reference to this vector
     */
    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<value_type>)
        requires std::is_move_constructible_v<value_type>
    {
        if (this == &other) {
            return *this;
        }

        destroy_elements();
        release_heap();
        move_construct_from(other);
        return *this;
    }

    ~SmallVector() {
        destroy_elements();
        release_heap();
    }

    /**
     * @brief Constructs an element at the end
     * @tparam Args The element constructor's argument types
     * @param args The element constructor's arguments
     * @return A reference to the new element
     */
    template <typename... Args>
        requires std::constructible_from<value_type, Args...>
    reference emplace_back(Args&&... args) {
        if (number_elements == current_capacity) {
            return grow_and_emplace(std::forward<Args>(args)...);
        }

        auto* const element = std::construct_at(data() + number_elements, std::forward<Args>(args)...);
        ++number_elements;
        return *element;
    }

    /** @brief Appends a copy of value */
    void push_back(const value_type& value)
        requires std::is_copy_constructible_v<value_type>
    {
        static_cast<void>(emplace_back(value));
    }

    /** @brief Appends value by moving it */
    void push_back(value_type&& value)
        requires std::is_move_constructible_v<value_type>
    {
        static_cast<void>(emplace_back(std::move(value)));
    }

    /**
     * @brief Removes and destroys the last element
     * @exception Throws an Exception if the vector is empty
     */
    void pop_back() {
        Exception::check(!empty(), "SmallVector::pop_back: the vector is empty");
        --number_elements;
        std::destroy_at(data() + number_elements);
    }

    /** @return The element at pos without bounds checking */
    [[nodiscard]] reference operator[](const size_type pos) noexcept {
        return data()[pos];
    }

    /** @return The element at pos without bounds checking */
    [[nodiscard]] const_reference operator[](const size_type pos) const noexcept {
        return data()[pos];
    }

    /**
     * @brief Returns the element at pos
     * @exception Throws an Exception if pos is outside [0, size())
     */
    [[nodiscard]] reference at(const size_type pos) {
        Exception::check(pos < number_elements, "SmallVector::at: position {} is out of range, size is {}", pos, size());
        return data()[pos];
    }

    /**
     * @brief Returns the element at pos
     * @exception Throws an Exception if pos is outside [0, size())
     */
    [[nodiscard]] const_reference at(const size_type pos) const {
        Exception::check(pos < number_elements, "SmallVector::at: position {} is out of range, size is {}", pos, size());
        return data()[pos];
    }

    /** @exception Throws an Exception if the vector is empty */
    [[nodiscard]] reference front() {
        Exception::check(!empty(), "SmallVector::front: the vector is empty");
        return data()[0];
    }

    /** @exception Throws an Exception if the vector is empty */
    [[nodiscard]] const_reference front() const {
        Exception::check(!empty(), "SmallVector::front: the vector is empty");
        return data()[0];
    }

    /** @exception Throws an Exception if the vector is empty */
    [[nodiscard]] reference back() {
        Exception::check(!empty(), "SmallVector::back: the vector is empty");
        return data()[number_elements - 1];
    }

    /** @exception Throws an Exception if the vector is empty */
    [[nodiscard]] const_reference back() const {
        Exception::check(!empty(), "SmallVector::back: the vector is empty");
        return data()[number_elements - 1];
    }

    /** @return A pointer to the contiguous element storage */
    [[nodiscard]] pointer data() noexcept {
        return heap_data == nullptr ? inline_data() : heap_data;
    }

    /** @return A pointer to the contiguous element storage */
    [[nodiscard]] const_pointer data() const noexcept {
        return heap_data == nullptr ? inline_data() : heap_data;
    }

    [[nodiscard]] iterator begin() noexcept { return data(); }
    [[nodiscard]] iterator end() noexcept { return number_elements == 0 ? data() : data() + number_elements; }
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }
    [[nodiscard]] const_iterator end() const noexcept { return number_elements == 0 ? data() : data() + number_elements; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }
    [[nodiscard]] reverse_iterator rbegin() noexcept { return reverse_iterator{ end() }; }
    [[nodiscard]] reverse_iterator rend() noexcept { return reverse_iterator{ begin() }; }
    [[nodiscard]] const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator{ end() }; }
    [[nodiscard]] const_reverse_iterator rend() const noexcept { return const_reverse_iterator{ begin() }; }
    [[nodiscard]] const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator{ cend() }; }
    [[nodiscard]] const_reverse_iterator crend() const noexcept { return const_reverse_iterator{ cbegin() }; }

    /** @return The number of elements */
    [[nodiscard]] size_type size() const noexcept {
        return number_elements;
    }

    /** @return The number of elements that fit without another allocation */
    [[nodiscard]] size_type capacity() const noexcept {
        return current_capacity;
    }

    /** @return The maximum number of elements supported by the allocator */
    [[nodiscard]] size_type max_size() const noexcept {
        return allocator_traits::max_size(allocator);
    }

    /** @return True iff the vector contains no elements */
    [[nodiscard]] bool empty() const noexcept {
        return number_elements == 0;
    }

    /**
     * @brief Ensures capacity for at least new_capacity elements
     * @exception Throws an Exception if new_capacity exceeds max_size()
     */
    void reserve(const size_type new_capacity) {
        if (new_capacity <= current_capacity) {
            return;
        }
        Exception::check(new_capacity <= max_size(),
                         "SmallVector::reserve: requested capacity {} exceeds max_size {}", new_capacity, max_size());
        relocate_to_heap(new_capacity);
    }

    /**
     * @brief Reduces capacity to size(), or returns to inline storage when all elements fit there
     */
    void shrink_to_fit() {
        if (heap_data == nullptr) {
            return;
        }
        if (number_elements <= InlineCapacity) {
            relocate_to_inline();
        } else if (number_elements < current_capacity) {
            relocate_to_heap(number_elements);
        }
    }

    /** @brief Destroys all elements without releasing allocated capacity */
    void clear() noexcept {
        destroy_elements();
    }

    /**
     * @brief Exchanges two vectors
     * @param other The other vector
     */
    void swap(SmallVector& other) noexcept(std::is_nothrow_move_constructible_v<value_type>)
        requires std::is_move_constructible_v<value_type>
    {
        if (this == &other) {
            return;
        }
        auto temporary = SmallVector{ std::move(other) };
        other = std::move(*this);
        *this = std::move(temporary);
    }

    friend void swap(SmallVector& lhs, SmallVector& rhs) noexcept(noexcept(lhs.swap(rhs))) {
        lhs.swap(rhs);
    }

    [[nodiscard]] friend bool operator==(const SmallVector& lhs, const SmallVector& rhs)
        requires std::equality_comparable<value_type>
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    [[nodiscard]] friend auto operator<=>(const SmallVector& lhs, const SmallVector& rhs)
        requires std::three_way_comparable<value_type>
    {
        return std::lexicographical_compare_three_way(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

private:
    using allocator_type = std::allocator<value_type>;
    using allocator_traits = std::allocator_traits<allocator_type>;

    [[nodiscard]] pointer inline_data() noexcept {
        return reinterpret_cast<pointer>(inline_storage.data());
    }

    [[nodiscard]] const_pointer inline_data() const noexcept {
        return reinterpret_cast<const_pointer>(inline_storage.data());
    }

    [[nodiscard]] size_type recommend_capacity(const size_type required_capacity) const {
        Exception::check(required_capacity <= max_size(),
                         "SmallVector: required capacity {} exceeds max_size {}", required_capacity, max_size());

        if (current_capacity >= max_size() - current_capacity) {
            return max_size();
        }
        return std::max(required_capacity, std::max(size_type{ 1 }, current_capacity * 2));
    }

    template <typename... Args>
    reference grow_and_emplace(Args&&... args) {
        const auto new_capacity = recommend_capacity(number_elements + 1);
        auto* const new_data = allocator_traits::allocate(allocator, new_capacity);
        auto transferred = size_type{ 0 };
        auto end_constructed = false;

        try {
            std::construct_at(new_data + number_elements, std::forward<Args>(args)...);
            end_constructed = true;
            for (; transferred < number_elements; ++transferred) {
                std::construct_at(new_data + transferred, std::move_if_noexcept(data()[transferred]));
            }
        } catch (...) {
            destroy_range(new_data, transferred);
            if (end_constructed) {
                std::destroy_at(new_data + number_elements);
            }
            allocator_traits::deallocate(allocator, new_data, new_capacity);
            throw;
        }

        const auto old_size = number_elements;
        destroy_elements();
        release_heap();
        heap_data = new_data;
        current_capacity = new_capacity;
        number_elements = old_size + 1;
        return heap_data[old_size];
    }

    void relocate_to_heap(const size_type new_capacity) {
        auto* const new_data = allocator_traits::allocate(allocator, new_capacity);
        auto transferred = size_type{ 0 };

        try {
            for (; transferred < number_elements; ++transferred) {
                std::construct_at(new_data + transferred, std::move_if_noexcept(data()[transferred]));
            }
        } catch (...) {
            destroy_range(new_data, transferred);
            allocator_traits::deallocate(allocator, new_data, new_capacity);
            throw;
        }

        const auto old_size = number_elements;
        destroy_elements();
        release_heap();
        heap_data = new_data;
        current_capacity = new_capacity;
        number_elements = old_size;
    }

    void relocate_to_inline() {
        auto transferred = size_type{ 0 };
        try {
            for (; transferred < number_elements; ++transferred) {
                std::construct_at(inline_data() + transferred, std::move_if_noexcept(heap_data[transferred]));
            }
        } catch (...) {
            destroy_range(inline_data(), transferred);
            throw;
        }

        const auto old_size = number_elements;
        destroy_range(heap_data, number_elements);
        allocator_traits::deallocate(allocator, heap_data, current_capacity);
        heap_data = nullptr;
        current_capacity = InlineCapacity;
        number_elements = old_size;
    }

    void move_construct_from(SmallVector& other) {
        if (other.heap_data != nullptr) {
            heap_data = std::exchange(other.heap_data, nullptr);
            current_capacity = std::exchange(other.current_capacity, InlineCapacity);
            number_elements = std::exchange(other.number_elements, 0);
            return;
        }

        try {
            for (auto& value : other) {
                std::construct_at(inline_data() + number_elements, std::move(value));
                ++number_elements;
            }
        } catch (...) {
            destroy_elements();
            throw;
        }
        other.clear();
    }

    static void destroy_range(pointer values, size_type count) noexcept {
        while (count > 0) {
            --count;
            std::destroy_at(values + count);
        }
    }

    void destroy_elements() noexcept {
        destroy_range(data(), number_elements);
        number_elements = 0;
    }

    void release_heap() noexcept {
        if (heap_data != nullptr) {
            allocator_traits::deallocate(allocator, heap_data, current_capacity);
            heap_data = nullptr;
            current_capacity = InlineCapacity;
        }
    }

    alignas(value_type) std::array<std::byte, InlineCapacity * sizeof(value_type)> inline_storage;
    [[no_unique_address]] allocator_type allocator{};
    pointer heap_data{ nullptr };
    size_type current_capacity{ InlineCapacity };
    size_type number_elements{ 0 };
};

} // namespace utility
