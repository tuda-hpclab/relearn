/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm_factory.h"

#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutLocationAware.h"
#include "algorithm/FMMInternal/FastMultipoleMethod.h"
#include "algorithm/FMMInternal/FastMultipoleMethodInverted.h"
#include "algorithm/NaiveInternal/Naive.h"
#include "structure/Morton.h"

#include <cstdint>
#include <memory>

std::shared_ptr<Algorithm> AlgorithmFactory::construct_naive_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<Naive>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}

std::shared_ptr<Algorithm> AlgorithmFactory::construct_barnes_hut_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<BarnesHut>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}

std::shared_ptr<Algorithm> AlgorithmFactory::construct_barnes_hut_inverted_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<BarnesHutInverted>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}

std::shared_ptr<Algorithm> AlgorithmFactory::construct_barnes_hut_location_aware_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<BarnesHutLocationAware>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}

std::shared_ptr<Algorithm> AlgorithmFactory::construct_fmm_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<FastMultipoleMethod>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}

std::shared_ptr<Algorithm> AlgorithmFactory::construct_fmm_inverted_algorithm(const RelearnTypes::bounding_box_type& simulation_box, RelearnTypes::level_type level_of_branch_nodes) {
    return std::make_shared<FastMultipoleMethodInverted>(simulation_box, std::make_shared<Morton>(level_of_branch_nodes));
}
