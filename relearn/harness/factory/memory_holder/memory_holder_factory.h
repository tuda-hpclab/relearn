#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
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
    [[nodiscard]] static std::shared_ptr<MemoryHolder<AdditionalCellAttributes>>
    get_default_memory_holder(const std::size_t size = default_size) {
        return get_local_memory_holder<AdditionalCellAttributes>(size);
    }

    template <typename AdditionalCellAttributes>
    [[nodiscard]] static std::shared_ptr<MemoryHolder<AdditionalCellAttributes>>
    get_rma_memory_holder(const std::size_t size = default_size) {

        auto memory_holder = std::make_shared<RMAMemoryHolder<AdditionalCellAttributes>>();
        memory_holder->init(size);

        return memory_holder;
    }

    template <typename AdditionalCellAttributes>
    [[nodiscard]] static std::shared_ptr<MemoryHolder<AdditionalCellAttributes>>
    get_local_memory_holder(const std::size_t size = default_size) {

        auto memory_holder = std::make_shared<SemiStableVectorMemoryHolder<AdditionalCellAttributes>>();
        memory_holder->init(size);

        return memory_holder;
    }
};
