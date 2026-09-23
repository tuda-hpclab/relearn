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

#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <utility>

class LocalGroupTranslator;
class NeuronID;

namespace mpiPP {
class MPIRank;
}

/**
 * Provides the functionality to load the target calcium values and the initial calcium values
 * on a per-neuron basis (currently, the MPI rank is ignored). The format is as follows (lines starting with # are ignored):
 * <neuron_id + 1> <initial_calcium_value> <target_calcium_value>
 * or once in the file:
 * 0 <default_initial_calcium_value> <target_calcium_value>
 */
class CalciumIO {
public:
    using calcium_type = RelearnTypes::calcium_type;

    using initial_value_calculator = std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)>;
    using target_value_calculator = std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)>;

    /**
     * @brief Loads from a file the functions that map the current MPI rank and a neuron's id to its initial and target calcium value
     * @param path_to_file The file that contains the description
     * @return A pair of (a) the initial calcium calculator and (b) the target calcium calculator
     */
    static std::pair<initial_value_calculator, target_value_calculator> load_initial_and_target_function(const std::filesystem::path& path_to_file, const std::shared_ptr<const LocalGroupTranslator>& local_group_translator, mpiPP::MPIRank my_rank);
};
