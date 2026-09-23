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

#include "util/RelearnException.h"

#include <cpp-utility/Conversion.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <filesystem>
#include <sstream>
#include <string_view>

namespace Util {

/**
 * @brief Looks for a given file in a directory. Path: directory / prefix rank suffix. Tries different formats for the rank.
 *      Supports up to a billion MPI ranks.
 * @param directory The directory where it looks for the file
 * @param rank The mpi rank, must be initialized
 * @param prefix Filename part before the mpi rank
 * @param suffix Filename after the mpi rank
 * @throws Throws a RelearnException if the MPI rank is not initialized or the file is not found
 * @return The file path for the found file
 */
static std::filesystem::path find_file_for_rank(const std::filesystem::path& directory, const mpiPP::MPIRank rank,
                                                const std::string_view prefix, const std::string_view suffix) {

    RelearnException::check(std::filesystem::exists(directory), "Utility::find_file_for_rank: Path '{}' does not exist");
    RelearnException::check(!directory.empty(), "Utility::find_file_for_rank: Path is empty");

    if (std::filesystem::is_regular_file(directory)) {
        RelearnException::check(rank == mpiPP::MPIRank::root_rank(), "Utility::find_file_for_rank: Single file is only allowed for a single mpi rank");
        return directory;
    }

    const auto num_files_in_directory = std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{});
    if (num_files_in_directory == 1) {
        RelearnException::check(rank == mpiPP::MPIRank::root_rank(), "Utility::find_file_for_rank: Single file is only allowed for a single mpi rank");
        const auto ret_value = directory / std::filesystem::directory_iterator(directory)->path().filename();

        RelearnException::check(std::filesystem::is_regular_file(ret_value), "Utility::find_file_for_rank: There is only one directory");
        return ret_value;
    }

    RelearnException::check(rank.is_initialized(), "Util::find_file_for_rank: rank is not initialized");

    auto builder = std::stringstream{};
    for (auto nr_digits = 1U; nr_digits <= 9; nr_digits++) {
        builder << prefix << utility::format_int_with_leading_zeros(rank.get_rank(), nr_digits) << suffix;

        const auto path_to_file = directory / builder.str();
        if (std::filesystem::exists(path_to_file)) {
            return path_to_file;
        }

        builder.clear();
        builder.str(std::string());
    }

    RelearnException::fail("Util::find_file_for_rank: No file found for {}{}{}", prefix, rank, suffix);
}

} // namespace Util
