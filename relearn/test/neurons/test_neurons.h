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

#include "RelearnTest.hpp"

#include "types/BasicTypes.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <memory>
#include <utility>

class Neurons;
class NetworkGraph;
class Partition;

class NeuronsTest : public RelearnTest {
protected:
    static std::unique_ptr<Neurons> create_neurons_object(std::shared_ptr<Partition>& partition, mpiPP::MPIRank rank, int number_ranks, RelearnTypes::number_neurons_type number_neurons);
};
