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

#include "util/NeuronID.h"

#include "mpi-wrapper/MPIRank.h"

#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

class ActivityInput;
class ChoiceFunction;
class LocalGroupTranslator;

class BackgroundActivityIO {
public:
    /**
     * Read the activity input for the flexible background activity calculator from a text file
     * @param file_path Path to the text file
     * @param my_rank Current mpi rank
     * @param local_group_translator Local group translator
     * @return Pair of (1) a vector of activity inputs and (2) a choice function that takes the current step and a neuron id as input and returns a set of indices to the activity input list that are currently active
     */
    [[nodiscard]] static std::pair<std::vector<std::shared_ptr<ActivityInput>>, std::unique_ptr<ChoiceFunction>>
    load_background_activity(const std::filesystem::path& file_path, mpiPP::MPIRank my_rank, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * Create a new ActivityInput object based on the string description
     * @param description String describing the type of the activity input
     * @return Shared pointer to the new activity input
     */
    [[nodiscard]] static std::shared_ptr<ActivityInput> create_input(std::string description);
};
