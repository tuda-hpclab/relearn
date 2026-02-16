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

#include "algorithm/Internal/octree/OctreeNode.h"
#include "util/MemoryHolder.h"

#include <tuple>
#include <vector>

class MemoryHolderFactory {
public:
    constexpr static inline std::size_t default_size = 1024UL * 1024UL;

    template <typename AdditionalCellAttributes>
    [[nodiscard]] static std::tuple<MemoryHolder<AdditionalCellAttributes>, std::vector<OctreeNode<AdditionalCellAttributes>>>
    get_memory_holder(const std::size_t size = default_size) {
        auto cells = std::vector<OctreeNode<AdditionalCellAttributes>>{};
        cells.resize(size);

        auto memory_holder = MemoryHolder<AdditionalCellAttributes>{};
        memory_holder.init(cells);

        return std::make_tuple(std::move(memory_holder), std::move(cells));
    }
};
