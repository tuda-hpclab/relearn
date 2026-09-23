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

#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A min-ordered Fibonacci heap.
 *		push and decrease_key run in O(1) amortized, extract_min in O(log n) amortized.
 *		push returns a handle to the inserted node that stays valid until the heap is destroyed;
 *		decrease_key must not be called for a node that has already been extracted.
 *		The memory of extracted nodes is kept until the heap is destroyed, so the total memory
 *		usage is proportional to the number of pushes, not to the current size.
 *		The heap is movable but not copyable. Moving or melding transfers ownership of all handles
 *		to the destination without invalidating them; the source is empty afterwards.
 * @tparam key_type The type of the keys, must be totally ordered via operator<
 * @tparam value_type The type of the payload attached to every key
 */
template <typename key_type, typename value_type>
class FibonacciHeap {
public:
    struct Node {
        key_type key{};
        value_type value{};

        Node* parent{ nullptr };
        Node* child{ nullptr };
        Node* left{ nullptr };
        Node* right{ nullptr };
        std::uint32_t degree{ 0 };
        bool marked{ false };
    };

    FibonacciHeap() = default;

    FibonacciHeap(const FibonacciHeap&) = delete;
    FibonacciHeap& operator=(const FibonacciHeap&) = delete;

    /**
     * @brief Moves all nodes and handles from other into a new heap. other is empty afterwards
     * @param other The heap to move from
     */
    FibonacciHeap(FibonacciHeap&& other) noexcept
        : node_arenas(std::move(other.node_arenas))
        , minimum(other.minimum)
        , number_nodes(other.number_nodes) {
        other.minimum = nullptr;
        other.number_nodes = 0;
        other.degree_table.clear();
        other.roots_buffer.clear();
    }

    /**
     * @brief Replaces this heap by all nodes and handles from other. Existing handles into this heap are invalidated;
     *      handles into other stay valid and belong to this heap afterwards. other is empty afterwards
     * @param other The heap to move from
     * @return A reference to this heap
     */
    FibonacciHeap& operator=(FibonacciHeap&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        node_arenas = std::move(other.node_arenas);
        degree_table.clear();
        roots_buffer.clear();
        minimum = other.minimum;
        number_nodes = other.number_nodes;

        other.minimum = nullptr;
        other.number_nodes = 0;
        other.degree_table.clear();
        other.roots_buffer.clear();
        return *this;
    }

    [[nodiscard]] bool empty() const noexcept {
        return minimum == nullptr;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return number_nodes;
    }

    /**
     * @brief Returns the smallest key
     * @exception Throws an Exception if the heap is empty
     * @return An immutable reference to the smallest key
     */
    [[nodiscard]] const key_type& min() const {
        Exception::check(minimum != nullptr, "FibonacciHeap::min: the heap is empty");
        return minimum->key;
    }

    /**
     * @brief Returns the value associated with the smallest key
     * @exception Throws an Exception if the heap is empty
     * @return A mutable reference to the value associated with the smallest key
     */
    [[nodiscard]] value_type& top() {
        Exception::check(minimum != nullptr, "FibonacciHeap::top: the heap is empty");
        return minimum->value;
    }

    /**
     * @brief Returns the value associated with the smallest key
     * @exception Throws an Exception if the heap is empty
     * @return An immutable reference to the value associated with the smallest key
     */
    [[nodiscard]] const value_type& top() const {
        Exception::check(minimum != nullptr, "FibonacciHeap::top: the heap is empty");
        return minimum->value;
    }

    /**
     * @brief Inserts the key with the associated value. O(1) amortized.
     * @param key The key
     * @param value The associated value
     * @return A handle to the inserted node, usable with decrease_key; stays valid until the heap is destroyed
     */
    Node* push(const key_type key, const value_type value) {
        if (node_arenas.empty()) {
            node_arenas.emplace_back();
        }

        auto* const node = &node_arenas.front().emplace_back(Node{ key, value });
        insert_into_root_list(node);
        ++number_nodes;
        return node;
    }

    /**
     * @brief Melds other into this heap in O(1). Live handles into either heap stay valid and belong to this heap;
     *      other is empty afterwards
     * @param other The heap to consume
     */
    void meld(FibonacciHeap& other) {
        if (this == &other) {
            return;
        }

        const auto other_is_smaller = minimum != nullptr && other.minimum != nullptr && other.minimum->key < minimum->key;

        node_arenas.splice(node_arenas.end(), other.node_arenas);
        degree_table.clear();
        roots_buffer.clear();

        if (minimum == nullptr) {
            minimum = other.minimum;
        } else if (other.minimum != nullptr) {
            auto* const this_right = minimum->right;
            auto* const other_left = other.minimum->left;

            minimum->right = other.minimum;
            other.minimum->left = minimum;
            other_left->right = this_right;
            this_right->left = other_left;

            if (other_is_smaller) {
                minimum = other.minimum;
            }
        }

        number_nodes += other.number_nodes;
        other.minimum = nullptr;
        other.number_nodes = 0;
        other.degree_table.clear();
        other.roots_buffer.clear();
    }

    /**
     * @brief Melds other into this heap in O(1), see meld(FibonacciHeap&)
     * @param other The heap to consume
     */
    void meld(FibonacciHeap&& other) {
        meld(other);
    }

    /**
     * @brief Removes and destroys all nodes. Invalidates every handle into the heap
     */
    void clear() noexcept {
        node_arenas.clear();
        degree_table.clear();
        roots_buffer.clear();
        minimum = nullptr;
        number_nodes = 0;
    }

    /**
     * @brief Removes the minimum node and returns its key and value. O(log n) amortized.
     * @exception Throws an Exception if the heap is empty
     * @return The key and the value of the removed minimum
     */
    [[nodiscard]] std::pair<key_type, value_type> extract_min() {
        Exception::check(minimum != nullptr, "FibonacciHeap::extract_min: the heap is empty");

        auto* const smallest = minimum;

        // Promote all children of the minimum to the root list
        while (smallest->child != nullptr) {
            auto* const child = smallest->child;

            if (child->right == child) {
                smallest->child = nullptr;
            } else {
                smallest->child = child->right;
                child->left->right = child->right;
                child->right->left = child->left;
            }

            child->parent = nullptr;
            child->marked = false;

            child->left = smallest;
            child->right = smallest->right;
            smallest->right->left = child;
            smallest->right = child;
        }

        // Remove the minimum itself from the root list
        if (smallest->right == smallest) {
            minimum = nullptr;
        } else {
            smallest->left->right = smallest->right;
            smallest->right->left = smallest->left;
            minimum = smallest->right;
            consolidate();
        }

        --number_nodes;
        return { smallest->key, smallest->value };
    }

    /**
     * @brief Lowers the key of the node behind the handle. O(1) amortized.
     *		Must not be called for a node that has already been extracted.
     * @param node The handle returned by push
     * @param new_key The new key, must not be larger than the node's current key
     * @exception Throws an Exception if new_key is larger than the current key or the heap is empty
     */
    void decrease_key(Node* const node, const key_type new_key) {
        Exception::check(minimum != nullptr, "FibonacciHeap::decrease_key: the heap is empty");
        Exception::check(!(node->key < new_key), "FibonacciHeap::decrease_key: the new key is larger than the current key");

        node->key = new_key;

        auto* const parent = node->parent;
        if (parent != nullptr && node->key < parent->key) {
            cut(node, parent);
            cascading_cut(parent);
        }

        if (node->key < minimum->key) {
            minimum = node;
        }
    }

private:
    /**
     * @brief Splices a node into the root list and updates the minimum pointer.
     *		Overwrites the node's left/right pointers, so it must already be detached.
     */
    void insert_into_root_list(Node* const node) noexcept {
        node->parent = nullptr;

        if (minimum == nullptr) {
            node->left = node;
            node->right = node;
            minimum = node;
            return;
        }

        node->left = minimum;
        node->right = minimum->right;
        minimum->right->left = node;
        minimum->right = node;

        if (node->key < minimum->key) {
            minimum = node;
        }
    }

    /**
     * @brief Removes the root child_root from the root list and makes it a child of parent_root.
     *		Requires that parent_root's key is not larger than child_root's key.
     */
    void link(Node* const child_root, Node* const parent_root) noexcept {
        child_root->left->right = child_root->right;
        child_root->right->left = child_root->left;

        child_root->parent = parent_root;
        child_root->marked = false;

        if (parent_root->child == nullptr) {
            parent_root->child = child_root;
            child_root->left = child_root;
            child_root->right = child_root;
        } else {
            child_root->left = parent_root->child;
            child_root->right = parent_root->child->right;
            parent_root->child->right->left = child_root;
            parent_root->child->right = child_root;
        }

        ++parent_root->degree;
    }

    /**
     * @brief Merges roots of equal degree until all roots have pairwise distinct degrees,
     *		then rebuilds the root list and the minimum pointer.
     */
    void consolidate() {
        // The maximum degree of any node is bounded by log_phi(number_nodes),
        // so twice the bit width plus some slack is a safe table size.
        const auto table_size = static_cast<std::size_t>(std::bit_width(number_nodes) * 2 + 2);
        degree_table.assign(table_size, nullptr);

        // Collect the current roots first; linking below restructures the list while we iterate
        roots_buffer.clear();
        auto* current = minimum;
        do {
            roots_buffer.push_back(current);
            current = current->right;
        } while (current != minimum);

        for (auto* root : roots_buffer) {
            auto degree = static_cast<std::size_t>(root->degree);
            if (degree >= degree_table.size()) {
                degree_table.resize(degree + 1, nullptr);
            }

            while (degree_table[degree] != nullptr) {
                auto* other = degree_table[degree];
                if (other->key < root->key) {
                    std::swap(root, other);
                }

                link(other, root);
                degree_table[degree] = nullptr;

                ++degree;
                if (degree >= degree_table.size()) {
                    degree_table.resize(degree + 1, nullptr);
                }
            }

            degree_table[degree] = root;
        }

        minimum = nullptr;
        for (auto* const node : degree_table) {
            if (node != nullptr) {
                insert_into_root_list(node);
            }
        }
    }

    /**
     * @brief Detaches the node from its parent and moves it to the root list.
     */
    void cut(Node* const node, Node* const parent) noexcept {
        if (node->right == node) {
            parent->child = nullptr;
        } else {
            if (parent->child == node) {
                parent->child = node->right;
            }
            node->left->right = node->right;
            node->right->left = node->left;
        }
        --parent->degree;

        node->marked = false;

        node->left = minimum;
        node->right = minimum->right;
        minimum->right->left = node;
        minimum->right = node;
        node->parent = nullptr;
    }

    /**
     * @brief Walks up from the given node, cutting marked ancestors and marking the first unmarked one.
     */
    void cascading_cut(Node* const node) noexcept {
        auto* current = node;

        while (true) {
            auto* const parent = current->parent;
            if (parent == nullptr) {
                return;
            }

            if (!current->marked) {
                current->marked = true;
                return;
            }

            cut(current, parent);
            current = parent;
        }
    }

    // Owns all nodes. Every deque keeps its node addresses (i.e., the handles) stable,
    // and the list lets meld() transfer whole arenas without moving a node.
    std::list<std::deque<Node>> node_arenas{};

    // Scratch buffers for consolidate, kept as members to avoid re-allocating them on every extract_min
    std::vector<Node*> degree_table{};
    std::vector<Node*> roots_buffer{};

    Node* minimum{ nullptr };
    std::size_t number_nodes{ 0 };
};

} // namespace utility
