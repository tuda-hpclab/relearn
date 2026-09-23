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

#include "NeuronsExtraInfoBase.h"

#include "neurons/enums/UpdateStatus.h"

#include <span>
#include <vector>

/**
 * CPU implementation of NeuronsExtraInfo: get_disable_flags() reads the host array directly.
 */
class NeuronsExtraInfoCPU : public NeuronsExtraInfoBase<std::vector> {
public:
    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return update_status;
    }
};
