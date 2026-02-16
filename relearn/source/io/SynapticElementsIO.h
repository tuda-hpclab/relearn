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

#include "Types.h"
#include "Types2.h"

#include "mpi-wrapper/MPIRank.h"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>

class LocalGroupTranslator;

/**
 * Provides the functionality to load individual parameters of synaptic elements for each file
 * on a per-neuron basis. The format is as follows (lines starting with # are ignored):
 * <rank>:<neuron_id + 1> min_calcium_axons nu_axons vacant_retract_ratio_axons min_elements_axons max_elements_axons min_calcium_den_exc nu_den_exc vacant_retract_ratio_den_exc min_elements_den_exc max_elements_den_exc min_calcium_den_inh nu_den_inh vacant_retract_ratio_den_inh min_elements_den_inh max_elements_den_inh
 * or once in the file:
 * default <default_initial_calcium_value> <target_calcium_value>
 * Instead of a rank-neuron id pair, you can specify an group name
 */
class SynapticElementsIO {
public:
    using neuron_id_to_calcium_calculator = RelearnTypes::neuron_id_to_calcium_calculator;

    /**
     * @brief Loads from a file the functions that map the current MPI rank and a neuron's id to its initial and target calcium value
     * @param path_to_file The file that contains the description
     * @return A pair of (a) the initial calcium calculator and (b) the target calcium calculator
     */
    [[nodiscard]] static std::array<SynapticElementsIO::neuron_id_to_calcium_calculator, 15>
    load_function_from_file(const std::filesystem::path& path_to_file, mpiPP::MPIRank my_rank,
                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator);
};