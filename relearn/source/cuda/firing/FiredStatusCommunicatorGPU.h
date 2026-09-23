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

#include "cuda/memory/DeviceArray.h"
#include "neurons/firing/FiredStatusCommunicator.h"

template <typename TransferDataType>
class FiredStatusCommunicatorGPU : public FiredStatusCommunicator {
public:
    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for pre-allocating memory)
     * @param _my_rank The MPI rank of this process
     * @param num_ranks The number of MPI ranks
     * @exception Throws a RelearnException if num_ranks <= 0
     */
    explicit FiredStatusCommunicatorGPU(const mpiPP::MPIRank _my_rank, const int num_ranks)
        : FiredStatusCommunicator(_my_rank, num_ranks) {
        RelearnException::check(num_ranks > 0, "FiredStatusCommunicationMap::FiredStatusCommunicationMap: num_ranks is too small: {}", num_ranks);
    }

    [[nodiscard]] const DeviceArray<TransferDataType>& get_incoming_data() const {
        return d_incoming_data;
    }
    [[nodiscard]] const DeviceArray<TransferDataType>& get_outgoing_data() const {
        return d_outgoing_data;
    }
    [[nodiscard]] const std::vector<int>& get_outgoing_data_size_per_rank() const {
        return h_outgoing_sizes;
    }

    void set_incoming_data(DeviceArray<TransferDataType>&& _incoming_bitvector, DeviceArray<int>&& _d_incoming_displ) { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - both moved into members below
        d_incoming_data = std::move(_incoming_bitvector);
        d_incoming_displ = std::move(_d_incoming_displ);
    }

protected:
    DeviceArray<TransferDataType> d_incoming_data{ 0 };
    DeviceArray<TransferDataType> d_outgoing_data{ 0 };
    std::vector<int> h_outgoing_sizes{ 0 };
    DeviceArray<int> d_incoming_displ{ 0 };
};