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

enum class RandomHolderKey : unsigned char {
    Algorithm = 0,
    Partition = 1,
    Subdomain = 2,
    PoissonModel = 3,
    SynapseDeletion = 4,
    SynapticElements = 5,
    NeuronsExtraInformation = 6,
    Connector = 7,
    BackgroundActivity = 8,
    FiringStatusApproximator = 9,
};
