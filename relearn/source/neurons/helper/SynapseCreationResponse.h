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

/**
 * The response for a SynapseCreationRequest can be that it failed or succeeded
 */
enum class SynapseCreationResponse : char {
    Failed = 0,
    Succeeded = 1,
};
