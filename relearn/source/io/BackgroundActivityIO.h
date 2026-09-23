#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/helper/ChoiceFunction.h"
#include "neurons/input/ActivityInput.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class LocalGroupTranslator;

/**
 * Result of BackgroundActivityIO::load_background_activity(): the activity inputs, and a choice
 * function that takes the current step and a neuron id and returns the set of indices into
 * `inputs` that are currently active for that neuron.
 */
struct LoadedBackgroundActivity {
    std::vector<std::shared_ptr<ActivityInput>> inputs{};
    std::unique_ptr<ChoiceFunction> choice_function{};
};

class BackgroundActivityIO {
public:
    /**
     * Read the activity input for the flexible background activity calculator from a text file
     * @param file_path Path to the text file
     * @param my_rank Current mpi rank
     * @param local_group_translator Local group translator
     * @return The activity inputs and a choice function that takes the current step and a neuron id as input and returns a set of indices to the activity input list that are currently active
     */
    [[nodiscard]] static LoadedBackgroundActivity
    load_background_activity(const std::filesystem::path& file_path, mpiPP::MPIRank my_rank, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * Create a new ActivityInput object based on the string description
     * @param description String describing the type of the activity input
     * @return Shared pointer to the new activity input
     */
    [[nodiscard]] static std::shared_ptr<ActivityInput> create_input(std::string description);
};
