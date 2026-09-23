#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "algorithm/FMMInternal/FastMultipoleMethodInverted.h"
#include "neurons/helper/SynapseCreationRequests.h"

#include <tuple>

class FMMTest : public RelearnMemoryTest {
protected:
    // FastMultipoleMethodInverted declares `friend class FMMTest;`, but gtest's TEST_F expands
    // into a *subclass* of FMMTest (not FMMTest itself), and friendship isn't inherited -- so the
    // TEST_F body can't call these protected members directly. Route through these thin,
    // inherited wrapper methods instead, which run as genuine FMMTest members and therefore keep
    // the friend access.
    static RelearnTypes::comm_map_creation<SynapseCreationRequest> call_find_target_neurons(FastMultipoleMethodInverted& algorithm, const RelearnTypes::number_neurons_type number_neurons) {
        return algorithm.find_target_neurons(number_neurons);
    }

    static BackwardProcessRequestsResult<SynapseCreationResponse>
    call_process_requests(FastMultipoleMethodInverted& algorithm, const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
        return algorithm.process_requests(creation_requests);
    }

    static PlasticDistantInSynapses call_process_responses(FastMultipoleMethodInverted& algorithm, const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                           const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
        return algorithm.process_responses(creation_requests, creation_responses);
    }
};
