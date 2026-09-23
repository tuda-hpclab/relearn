#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfoBase.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "neurons/enums/UpdateStatus.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <span>

/**
 * GPU implementation of NeuronsExtraInfo: get_disable_flags() lazily syncs the device mirror back
 * to the host, and get_gpu_handle() hands out the device pointer for kernels to read.
 */
class NeuronsExtraInfoGPU : public NeuronsExtraInfoBase<LazySyncedArray> {
public:
    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] std::span<const UpdateStatus> get_disable_flags() const noexcept {
        // LazySyncedArray in the non-CUDA build is a plain std::vector alias (no .host()), so
        // this sync-aware read is only needed -- and only compiles -- when CUDA is enabled.
        return update_status.host();
    }

    [[nodiscard]] NeuronsExtraInfoGPUHandleConst get_gpu_handle() const {
        return {
            static_cast<CudaConfig::number_neurons_type>(size), mpiPP::MPIInfo::get_my_rank().get_rank(), mpiPP::MPIInfo::get_number_ranks(),
            update_status.get_device_ptr_const()
        };
    }

    void update_gpu() const;
};
