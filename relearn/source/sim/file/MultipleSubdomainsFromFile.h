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

#include "Config.h"

#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/file/AdditionalPositionInformation.h"

#include <filesystem>
#include <memory>
#include <optional>

class Partition;

/**
 * This class inherits form NeuronToSubdomainAssignment.
 * It implements reading multiple files on different MPI ranks. Those files need to fulfill their header requirements.
 */
class MultipleSubdomainsFromFile : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object, all MPI ranks simultaneously read their respective position file.
     *      Optionally all read their respective synapses
     * @param path_to_neurons The directory that includes all files with the neurons to load
     * @param path_to_synapses The directory that includes all files with the synapses, can be empty if none should be loaded
     * @param _partition The partition
     * @exception Throws a RelearnException if some errors occurred while processing the file,
     *      if there is only 1 MPI rank, or not every MPI rank has a designated file.
     */
    MultipleSubdomainsFromFile(const std::filesystem::path& path_to_neurons, std::optional<std::filesystem::path> path_to_synapses,
                               const std::shared_ptr<Partition>& _partition);

    MultipleSubdomainsFromFile(const MultipleSubdomainsFromFile& other) = delete;
    MultipleSubdomainsFromFile(MultipleSubdomainsFromFile&& other) = delete;

    MultipleSubdomainsFromFile& operator=(const MultipleSubdomainsFromFile& other) = delete;
    MultipleSubdomainsFromFile& operator=(MultipleSubdomainsFromFile&& other) = delete;

    ~MultipleSubdomainsFromFile() override = default;

    /**
     * @brief Prints relevant metrics to the essentials
     * @param essentials The essentials
     */
    void print_essentials(const std::unique_ptr<Essentials>& essentials) override;

protected:
    void fill_all_subdomains() override;

private:
    void read_neurons_from_file(const std::filesystem::path& path_to_neurons);

    AdditionalPositionInformation additional_position_information;
};
