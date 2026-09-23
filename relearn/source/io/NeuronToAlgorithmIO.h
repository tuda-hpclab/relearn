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

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"
#include "neurons/LocalGroupTranslator.h"
#include "types/AlgorithmTypes.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class LocalGroupTranslator;

/**
 * This class reads from a file the mapping from algorithm to neurons for the combined algorithms method.
 */
class NeuronToAlgorithmIO {
public:
    /**
     * @brief Reads the descriptions of the neurons corresponding to algorithms in the combined algorithms method.
     * 		Returns vectors that the CombinedAlgorithms algorithm uses.
     * @param file_path The path to the file containing the mapping
     * @param my_rank The current MPI rank
     * @param local_group_translator The local group translator
     * @return A pair of two vectors. (1) The vector of tuples containing the index of an algorithm in the algorithm configs vector
     * 		and the neurons that should use the respective algorithm. (2) The vector of the algorithm configs used by the first vector.
     */
    [[nodiscard]] static std::pair<RelearnTypes::AlgorithmIndexWithNeuronsType, RelearnTypes::AlgorithmConfigs> read_descriptions(const std::filesystem::path& file_path, mpiPP::MPIRank my_rank, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Converts a string description of an algorithm with kernel into an AlgorithmConfig object
     * @param description The string describing the configuration (of form algorithm:kernel([kernel params, comma seperated];theta (kernel params and ";theta" are optional)
     * @return The algorithm config that the description represents
     */
    [[nodiscard]] static AlgorithmConfig get_algorithm_config(std::string description);
};