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

#include "RelearnTest.hpp"

#include "algorithm/Internal/octree/NodeCache.h"
#include "mpi-wrapper/RMAWindow.h"
#include "util/MemoryHolder.h"

#include <memory>
#include <span>

template <typename AdditionalCellAttributes>
class NodeCacheTest : public RelearnMemoryTest {

    void SetUp() override {
        RelearnMemoryTest::SetUp();

        rma_window = std::make_shared<mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>>>(Constants::mpi_alloc_mem);
    }

protected:
    std::shared_ptr<mpiPP::RMAWindow<OctreeNode<AdditionalCellAttributes>>> rma_window;
};
