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
#include <deque>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A semi stable vector
 *
 * Considered semi stable as iterators are invalidated when inserting or removing elements,
 * but references and pointers to the remaining elements are not (see std::deque).
 * Restricts insertion and removal to the front and back, provides no erase(...) in the middle, only clear().
 * Implemented as a wrapper around std::deque, all functions are forwarding calls to deque.
 *
 * @tparam T value type
 */
template <typename T>
class SemiStableVector : private std::deque<T> {
private:
    using Base = std::deque<T>;

public:
    using value_type = typename Base::value_type;
    using reference = typename Base::reference;
    using const_reference = typename Base::const_reference;
    using size_type = typename Base::size_type;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;
    using container_type = Base;

    template <typename... Ts>
    explicit SemiStableVector(Ts&&... args)
        requires std::constructible_from<Base, Ts...>
        : Base(std::forward<Ts>(args)...) {
    }

    SemiStableVector(const SemiStableVector&) = default;

    SemiStableVector& operator=(const SemiStableVector&) = default;

    SemiStableVector(SemiStableVector&&) noexcept(std::is_nothrow_move_constructible_v<Base>) = default;

    SemiStableVector& operator=(SemiStableVector&&) noexcept(std::is_nothrow_move_assignable_v<Base>) = default;

    ~SemiStableVector() = default;

    [[nodiscard]] iterator begin() { return Base::begin(); }
    [[nodiscard]] iterator end() { return Base::end(); }
    [[nodiscard]] const_iterator begin() const { return Base::begin(); }
    [[nodiscard]] const_iterator end() const { return Base::end(); }
    [[nodiscard]] const_iterator cbegin() const { return Base::cbegin(); }
    [[nodiscard]] const_iterator cend() const { return Base::cend(); }

    [[nodiscard]] bool empty() const { return Base::empty(); }

    [[nodiscard]] size_type size() const { return Base::size(); }

    void clear() { Base::clear(); }

    [[nodiscard]] reference operator[](const size_type pos) { return Base::operator[](pos); }
    [[nodiscard]] const_reference operator[](const size_type pos) const { return Base::operator[](pos); }

    [[nodiscard]] reference at(size_type pos) { return Base::at(pos); }
    [[nodiscard]] const_reference at(size_type pos) const { return Base::at(pos); }

    /**
     * @brief Returns a mutable reference to the first element
     * @exception Throws an Exception if *this is empty
     * @return A mutable reference to the first element
     */
    [[nodiscard]] reference front() {
        Exception::check(!empty(), "SemiStableVector::front(): The vector was empty!");
        return Base::front();
    }

    /**
     * @brief Returns an immutable reference to the first element
     * @exception Throws an Exception if *this is empty
     * @return An immutable reference to the first element
     */
    [[nodiscard]] const_reference front() const {
        Exception::check(!empty(), "SemiStableVector::front(): The vector was empty!");
        return Base::front();
    }

    /**
     * @brief Returns a mutable reference to the last element
     * @exception Throws an Exception if *this is empty
     * @return A mutable reference to the last element
     */
    [[nodiscard]] reference back() {
        Exception::check(!empty(), "SemiStableVector::back(): The vector was empty!");
        return Base::back();
    }

    /**
     * @brief Returns an immutable reference to the last element
     * @exception Throws an Exception if *this is empty
     * @return An immutable reference to the last element
     */
    [[nodiscard]] const_reference back() const {
        Exception::check(!empty(), "SemiStableVector::back(): The vector was empty!");
        return Base::back();
    }

    /**
     * @brief Removes the first element. Invalidates all iterators and the references
     *      and pointers to the removed element; references and pointers to the remaining elements stay valid
     * @exception Throws an Exception if *this is empty
     */
    void pop_front() {
        Exception::check(!empty(), "SemiStableVector::pop_front(): The vector was empty!");
        Base::pop_front();
    }

    /**
     * @brief Removes the last element. Invalidates all iterators and the references
     *      and pointers to the removed element; references and pointers to the remaining elements stay valid
     * @exception Throws an Exception if *this is empty
     */
    void pop_back() {
        Exception::check(!empty(), "SemiStableVector::pop_back(): The vector was empty!");
        Base::pop_back();
    }

    void push_front(const value_type& val) { Base::push_front(val); }
    void push_front(value_type&& val) { Base::push_front(std::move(val)); }

    void push_back(const value_type& val) { Base::push_back(val); }
    void push_back(value_type&& val) { Base::push_back(std::move(val)); }

    template <typename... Ts>
    reference emplace_front(Ts&&... args)
        requires std::constructible_from<value_type, Ts...>
    {
        return Base::emplace_front(std::forward<Ts>(args)...);
    }

    template <typename... Ts>
    reference emplace_back(Ts&&... args)
        requires std::constructible_from<value_type, Ts...>
    {
        return Base::emplace_back(std::forward<Ts>(args)...);
    }

    void resize(const size_type size) { Base::resize(size); }
    void resize(const size_type size, const value_type& value) { Base::resize(size, value); }
};

} // namespace utility
