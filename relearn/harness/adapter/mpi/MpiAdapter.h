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

#include "util/RelearnException.h"

#include "mpi-wrapper/CommunicationMap.h"
#include "mpi-wrapper/CommunicationVector.h"
#include "mpi-wrapper/MPIRank.h"

#include <cmath>
#include <random>

class MPIAdapter {
public:
    /**
     * @brief Simulates a MPI exchange_requests function call for tests without actual mpi
     * @tparam T Request type
     * @param requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <class T>
    static std::vector<mpiPP::CommunicationMap<T>> exchange_requests(std::vector<mpiPP::CommunicationMap<T>>& requests) {
        const auto num_ranks = requests.size();
        std::vector<mpiPP::CommunicationMap<T>> answers;
        answers.resize(num_ranks, mpiPP::CommunicationMap<T>(static_cast<int>(num_ranks)));

        for (auto sending_rank = 0U; sending_rank < num_ranks; sending_rank++) {
            const auto sending_mpi_rank = mpiPP::MPIRank(static_cast<int>(sending_rank));

            for (auto receiving_rank = 0U; receiving_rank < num_ranks; receiving_rank++) {
                const auto receiving_mpi_rank = mpiPP::MPIRank(static_cast<int>(receiving_rank));
                if (!requests[sending_rank].contains(receiving_mpi_rank)) {
                    continue;
                }

                RelearnException::check(sending_rank != receiving_rank, "Sending != receiving");
                const auto& request_block = requests[sending_rank].get_requests(receiving_mpi_rank);
                auto& map = answers[receiving_rank];
                for (auto& request : request_block) {
                    map.append(sending_mpi_rank, request);
                }
            }
        }
        return answers;
    }

    /**
     * @brief Simulates a MPI exchange_requests function call for tests without actual mpi
     * @tparam T Request type
     * @param requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <class T>
    static std::vector<mpiPP::CommunicationVector<T>> exchange_requests(std::vector<mpiPP::CommunicationVector<T>>& requests) {
        const auto num_ranks = requests.size();
        std::vector<mpiPP::CommunicationVector<T>> answers;
        answers.resize(num_ranks, mpiPP::CommunicationVector<T>(static_cast<int>(num_ranks)));

        for (auto sending_rank = 0U; sending_rank < num_ranks; sending_rank++) {
            const auto sending_mpi_rank = mpiPP::MPIRank(static_cast<int>(sending_rank));

            for (auto receiving_rank = 0U; receiving_rank < num_ranks; receiving_rank++) {
                const auto receiving_mpi_rank = mpiPP::MPIRank(static_cast<int>(receiving_rank));
                if (!requests[sending_rank].contains(receiving_mpi_rank)) {
                    continue;
                }

                RelearnException::check(sending_rank != receiving_rank, "Sending != receiving");
                const auto& request_block = requests[sending_rank].get_requests(receiving_mpi_rank);
                auto& map = answers[receiving_rank];
                for (auto& request : request_block) {
                    map.append(sending_mpi_rank, request);
                }
            }
        }
        return answers;
    }
};
