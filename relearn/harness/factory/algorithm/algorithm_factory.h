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

#include "algorithm/Algorithm.h"
#include "types/SpaceTypes.h"
#include "util/BoundingBox.h"

#include <cstdint>
#include <memory>

class AlgorithmFactory {
public:
    static std::shared_ptr<Algorithm> construct_naive_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);

    static std::shared_ptr<Algorithm> construct_barnes_hut_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);

    static std::shared_ptr<Algorithm> construct_barnes_hut_inverted_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);

    static std::shared_ptr<Algorithm> construct_barnes_hut_location_aware_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);

    static std::shared_ptr<Algorithm> construct_fmm_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);

    static std::shared_ptr<Algorithm> construct_fmm_inverted_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes);
};
