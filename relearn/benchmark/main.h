#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "types/SpaceTypes.h"
#include "util/MemoryHolder.h"

#include "factory/octree/octree_factory.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>

constexpr bool excessive_testing = false;

constexpr auto small_number_neurons = 400;  // 4'000;
constexpr auto medium_number_neurons = 400; // 64'000;
constexpr auto large_number_neurons = 400;  // 512'000;

constexpr auto small_number_iterations = 10;
constexpr auto medium_number_iterations = 10; // 40;
constexpr auto large_number_iterations = 10;  // 500;

constexpr auto static_number_neurons = 1000000;
constexpr auto static_number_synapses = 20;

constexpr auto static_many_iterations = 500;
constexpr auto static_few_iterations = 10;
static std::mt19937 _mt{};

namespace {
template <typename T>
auto octrees = std::unordered_map<std::size_t, OctreeNode<T>>{};

template <typename T>
auto memory_holders = std::vector<std::shared_ptr<MemoryHolder<T>>>{};

template <typename T>
auto cell_holders = std::vector<std::vector<OctreeNode<T>>>{};

auto mt = std::mt19937{};
} // namespace

template <typename T>
class OctreeNode;

template <typename T>
const OctreeNode<T>& get_octree(std::size_t number_nodes) {
    const auto iterator = octrees<T>.find(number_nodes);
    if (iterator != octrees<T>.end()) {
        return iterator->second;
    }

    auto& cells = cell_holders<T>.emplace_back();
    cells.resize((number_nodes * std::size_t{ 16 }) + std::size_t{ 1024 });

    auto& memory_holder = memory_holders<T>.emplace_back(std::make_shared<SemiStableVectorMemoryHolder<T>>());
    memory_holder->init(cells.size());

    auto root = OctreeFactory::get_standard_tree<T>(number_nodes, memory_holder, RelearnTypes::position_type{ 0, 0, 0 }, RelearnTypes::position_type{ 1, 1, 1 }, mt);
    OctreeNodeUpdater<T>::update_tree(&root);

    const auto [emplace_iterator, _] = octrees<T>.emplace(number_nodes, root);

    return emplace_iterator->second;
}
