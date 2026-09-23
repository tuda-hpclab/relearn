/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NetworkGraphGPU.h"
#include "Views.cuh"

NetworkGraphGPUBase::NetworkGraphGPUBase(const mpi_rank_type _number_ranks, const mpi_rank_type _my_rank)
    : number_ranks(_number_ranks)
    , my_rank(_my_rank) {
    dummy_view = new DummyView{};
}

NetworkGraphGPUBase::~NetworkGraphGPUBase() {
    delete static_cast<DummyView*>(dummy_view);
}