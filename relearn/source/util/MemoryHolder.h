#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "cpp-utility/data-structure/SemiStableVector.hpp"
#include "util/RelearnException.h"

#include <fmt/format.h>

#include <mpi-wrapper/rma/RMAWindow.h>

#include <range/v3/algorithm/for_each.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <unordered_map>

template <typename T>
class OctreeNode;

/**
 * This class manages a portion of memory and can hand out OctreeNodes as long as there is space left.
 * Hands out pointers via get_available(), which have to be reclaimed with make_all_available() later on.
 * get_available() makes sure that memory is really handled in portions, and all children are next to each other.
 *
 * In effect calls OctreeNode<AdditionalCellAttributes>::reset()
 *
 * @tparam AdditionalCellAttributes The template parameter of the objects
 */
template <typename AdditionalCellAttributes>
class MemoryHolder {
public:
    MemoryHolder() = default;

    MemoryHolder(const MemoryHolder&) = delete;
    MemoryHolder(MemoryHolder&&) = default;

    MemoryHolder& operator=(const MemoryHolder&) = delete;
    MemoryHolder& operator=(MemoryHolder&&) = default;

    virtual ~MemoryHolder() = default;

    /**
     * @brief Initializes the class to hold the specified span of memory.
     * @param size_hint The size of the memory
     */
    virtual void init(const std::size_t size_hint) noexcept {
        parent_to_offset.clear();
        parent_to_offset.reserve(size_hint);
        offset_to_parent.clear();
        offset_to_parent.reserve(size_hint);
    }

    /**
     * @brief Returns the currently held memory
     * @return The currently held memory
     */
    [[nodiscard]] virtual std::span<const OctreeNode<AdditionalCellAttributes>> get_current_memory() const noexcept = 0;

    /**
     * @brief Returns the number of objects that fit into the memory portion
     * @return The number of objects that fit into the memory portion
     */
    [[nodiscard]] virtual typename std::span<OctreeNode<AdditionalCellAttributes>>::size_type get_size() const noexcept = 0;

    /**
     * @brief Returns the number of objects that are currently held
     * @return The number of objects that are currently held
     */
    [[nodiscard]] virtual std::uint64_t get_current_filling() const noexcept = 0;

    /**
     * @brief Destroys all objects that were handed out via get_available. All pointers are invalidated.
     */
    virtual void make_all_available() noexcept {
        parent_to_offset.clear();
        offset_to_parent.clear();
    }

    /**
     * @brief Returns the pointer for the octant-th child of parent.
     *      Is deterministic if called repeatedly without calls to make_all_available inbetween.
     * @param parent The OctreeNode whose child the newly created node shall be
     * @param octant The octant of the newly created child
     * @exception Throws a RelearnException if parent == nullptr, octant >= Constants::number_oct, or if there is no more space left
     * @return Returns a pointer to the newly created child
     */
    [[nodiscard]] virtual OctreeNode<AdditionalCellAttributes>* get_available(OctreeNode<AdditionalCellAttributes>* const parent, const unsigned int octant) = 0;

    /**
     * @brief Returns the offset of the specified node's children with respect to the base pointer in bytes
     * @param parent_node The node for whose children we want to have the offset
     * @exception Throws a RelearnException if parent_node does not have an associated children array
     * @return The offset of node wrt. the base pointer
     */
    [[nodiscard]] std::uint64_t get_offset_from_parent(OctreeNode<AdditionalCellAttributes>* const parent_node) const {
        const auto iterator = parent_to_offset.find(parent_node);

        RelearnException::check(iterator != parent_to_offset.end(), "MemoryHolder::get_offset_from_parent: parent_node {} does not have an offset.", fmt::ptr(parent_node));

        const auto offset = iterator->second;
        return offset;
    }

    /**
     * @brief Returns the parent node for a node specified by the offset
     * @param offset The offset, must be a multple of Constants::number_oct
     * @exception Throws a RelearnException if the offset is not saved for a parent or if offset % Constants::number_oct != 0
     * @return A pointer to the parent of the node stored at the offset
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_parent_from_offset(const std::uint64_t offset) const {
        RelearnException::check(offset % Constants::number_oct == 0, "MemoryHolder::get_parent_from_offset: offset {} is not a multiple of {}.", offset, Constants::number_oct);
        const auto iterator = offset_to_parent.find(offset);

        RelearnException::check(iterator != offset_to_parent.end(), "MemoryHolder::get_parent_from_offset: offset {} does not have a parent node.", offset);

        const auto parent = iterator->second;
        return parent;
    }

    /**
     * @brief Returns the OctreeNode at the specified offset
     * @param offset The offset at which the OctreeNode shall be returned
     * @exception Throws a RelearnException if offset is larger or equal to the total number of objects or to the current filling
     * @return The OctreeNode with the specified offset
     */
    [[nodiscard]] virtual const OctreeNode<AdditionalCellAttributes>* get_node_from_offset(std::uint64_t offset) const = 0;

protected:
    const std::unordered_map<OctreeNode<AdditionalCellAttributes>*, std::uint64_t>& get_parent_to_offset() const {
        return parent_to_offset;
    }

    const std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*>& get_offset_to_parent() const {
        return offset_to_parent;
    }

    std::unordered_map<OctreeNode<AdditionalCellAttributes>*, std::uint64_t> parent_to_offset{};
    std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> offset_to_parent{};
};

template <typename AdditionalCellAttributes>
class RMAMemoryHolder : public MemoryHolder<AdditionalCellAttributes> {
public:
    RMAMemoryHolder() = default;

    RMAMemoryHolder(const RMAMemoryHolder&) = delete;
    RMAMemoryHolder(RMAMemoryHolder&&) = default;

    RMAMemoryHolder& operator=(const RMAMemoryHolder&) = delete;
    RMAMemoryHolder& operator=(RMAMemoryHolder&&) = default;

    ~RMAMemoryHolder() = default;

    void init(const std::size_t size_hint) noexcept override {
        MemoryHolder<AdditionalCellAttributes>::init(size_hint);
        rma_window = mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>>(size_hint);
        memory_holder = { rma_window.get_pointer(), size_hint };
        current_filling = 0;
        std::ranges::uninitialized_default_construct(memory_holder);
    }

    [[nodiscard]] std::span<const OctreeNode<AdditionalCellAttributes>> get_current_memory() const noexcept override {
        return memory_holder;
    }

    [[nodiscard]] typename std::span<OctreeNode<AdditionalCellAttributes>>::size_type get_size() const noexcept override {
        return memory_holder.size();
    }

    [[nodiscard]] std::uint64_t get_current_filling() const noexcept override {
        return current_filling;
    }

    void make_all_available() noexcept override {
        MemoryHolder<AdditionalCellAttributes>::make_all_available();
        ranges::for_each(memory_holder, &OctreeNode<AdditionalCellAttributes>::reset);
        current_filling = 0;
    }

    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_available(OctreeNode<AdditionalCellAttributes>* const parent, const unsigned int octant) override {
        RelearnException::check(parent != nullptr, "RMAMemoryHolder::get_available: parent is nullptr");
        RelearnException::check(octant < Constants::number_oct, "RMAMemoryHolder::get_available: octant is too large: {} vs {}", octant, Constants::number_oct);

        if (!this->parent_to_offset.contains(parent)) {
            this->parent_to_offset[parent] = current_filling;
            this->offset_to_parent[current_filling] = parent;
            current_filling += Constants::number_oct;
        }

        const auto offset = this->parent_to_offset[parent];
        RelearnException::check(offset + Constants::number_oct <= memory_holder.size(),
                                "RMAMemoryHolder::get_available: The offset is too large: {} + {} vs {}", offset, Constants::number_oct, memory_holder.size());

        return &memory_holder[offset + octant];
    }

    /**
     * @brief Returns the OctreeNode at the specified offset
     * @param offset The offset at which the OctreeNode shall be returned
     * @exception Throws a RelearnException if offset is larger or equal to the total number of objects or to the current filling
     * @return The OctreeNode with the specified offset
     */
    [[nodiscard]] const OctreeNode<AdditionalCellAttributes>* get_node_from_offset(const std::uint64_t offset) const override {
        RelearnException::check(offset < memory_holder.size(), "RMAMemoryHolder::get_node_from_offset(): offset ({}) is too large. The total size is: {}.", offset, memory_holder.size());
        RelearnException::check(offset < current_filling, "RMAMemoryHolder::get_node_from_offset(): offset ({}) is too large. I only contain: {} elements.", offset, current_filling);
        return &memory_holder[offset];
    }

    [[nodiscard]] mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>>* get_rma_window() {
        return &rma_window;
    }

private:
    // NOLINTNEXTLINE
    std::span<OctreeNode<AdditionalCellAttributes>> memory_holder{};
    mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>> rma_window = mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>>(0);
    std::uint64_t current_filling{ 0 };
};

template <typename AdditionalCellAttributes>
class SemiStableVectorMemoryHolder : public MemoryHolder<AdditionalCellAttributes> {
public:
    SemiStableVectorMemoryHolder() = default;

    SemiStableVectorMemoryHolder(const SemiStableVectorMemoryHolder&) = delete;
    SemiStableVectorMemoryHolder(SemiStableVectorMemoryHolder&&) = default;

    SemiStableVectorMemoryHolder& operator=(const SemiStableVectorMemoryHolder&) = delete;
    SemiStableVectorMemoryHolder& operator=(SemiStableVectorMemoryHolder&&) = default;

    ~SemiStableVectorMemoryHolder() = default;

    void init(const std::size_t size_hint) noexcept override {
        MemoryHolder<AdditionalCellAttributes>::init(size_hint);
        // memory_holder.reserve(size_hint);
    }

    [[nodiscard]] std::span<const OctreeNode<AdditionalCellAttributes>> get_current_memory() const noexcept override {
        return {};
    }

    [[nodiscard]] typename std::span<OctreeNode<AdditionalCellAttributes>>::size_type get_size() const noexcept override {
        return memory_holder.size();
    }

    [[nodiscard]] std::uint64_t get_current_filling() const noexcept override {
        return memory_holder.size();
    }

    void make_all_available() noexcept override {
        MemoryHolder<AdditionalCellAttributes>::make_all_available();
        memory_holder.clear();
    }

    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_available(OctreeNode<AdditionalCellAttributes>* const parent, const unsigned int octant) override {
        RelearnException::check(parent != nullptr, "SemiStableVectorMemoryHolder::get_available: parent is nullptr");
        RelearnException::check(octant < Constants::number_oct, "SemiStableVectorMemoryHolder::get_available: octant is too large: {} vs {}", octant, Constants::number_oct);

        if (!this->parent_to_offset.contains(parent)) {
            const auto current_filling = memory_holder.size();
            this->parent_to_offset[parent] = current_filling;
            this->offset_to_parent[current_filling] = parent;
        }

        const auto parent_offset = this->parent_to_offset[parent];
        const auto offset = parent_offset + octant;

        const auto new_min_size = parent_offset + Constants::number_oct;
        if (new_min_size > memory_holder.size()) {
            memory_holder.resize(new_min_size);
        }
        return &memory_holder[offset];
    }

    /**
     * @brief Returns the OctreeNode at the specified offset
     * @param offset The offset at which the OctreeNode shall be returned
     * @exception Throws a RelearnException if offset is larger or equal to the total number of objects or to the current filling
     * @return The OctreeNode with the specified offset
     */
    [[nodiscard]] const OctreeNode<AdditionalCellAttributes>* get_node_from_offset(const std::uint64_t offset) const override {
        RelearnException::check(offset < memory_holder.size(), "SemiStableVectorMemoryHolder::get_node_from_offset(): offset ({}) is too large. The total size is: {}.", offset, memory_holder.size());
        return &memory_holder[offset];
    }

private:
    // NOLINTNEXTLINE
    utility::SemiStableVector<OctreeNode<AdditionalCellAttributes>> memory_holder{};
};
