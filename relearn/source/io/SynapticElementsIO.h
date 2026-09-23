#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "types/CalciumTypes.h"
#include "types/SynapticElementsTypes.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <filesystem>
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
    using neuron_id_to_grown_calculator = RelearnTypes::neuron_id_to_grown_calculator;

    /**
     * The per-neuron parameters of one synaptic element type, each of them a calculator that maps a neuron's id to its value.
     * Only the minimum calcium is a calcium concentration, the others are continuous numbers of synaptic elements.
     */
    struct ElementCalculators {
        neuron_id_to_calcium_calculator min_calcium;
        neuron_id_to_grown_calculator nu;
        neuron_id_to_grown_calculator vacant_retract_ratio;
        neuron_id_to_grown_calculator min_elements;
        neuron_id_to_grown_calculator max_elements;
    };

    /** The per-neuron parameters of all three synaptic element types */
    struct Calculators {
        ElementCalculators axons;
        ElementCalculators dendrites_excitatory;
        ElementCalculators dendrites_inhibitory;
    };

    /**
     * @brief Loads from a file the functions that map a neuron's id to its individual synaptic element parameters
     * @param path_to_file The file that contains the description
     * @return The calculators for the axons, the excitatory dendrites, and the inhibitory dendrites
     */
    [[nodiscard]] static Calculators
    load_function_from_file(const std::filesystem::path& path_to_file, mpiPP::MPIRank my_rank,
                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator);
};
