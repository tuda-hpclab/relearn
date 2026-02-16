/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <iostream>

[[nodiscard]] const char* RelearnException::what() const noexcept {
    return message.c_str();
}

void RelearnException::log_message(const std::string& message) {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    const auto num_ranks = mpiPP::MPIInfo::get_number_ranks();

    std::cerr << message << std::flush;
    fflush(stderr);

    spdlog::error("There was an error at rank: {} of {}!\n{}", my_rank, num_ranks, message);
}
