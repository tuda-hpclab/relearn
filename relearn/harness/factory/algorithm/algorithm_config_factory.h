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

#include "Types2.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "algorithm/Kernel/KernelBase.h"
#include "algorithm/Kernel/KernelType.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

class AlgorithmConfigFactory {
public:
	static AlgorithmConfig create_random_algorithm_config(std::mt19937& mt);

	static RelearnTypes::AlgorithmConfigs create_random_algorithm_configs(std::uint64_t number, std::mt19937& mt);

	static std::uint64_t get_random_number_algorithms(std::mt19937& mt);

	static std::uint64_t get_random_number_algorithms(std::uint64_t min, std::uint64_t max, std::mt19937& mt);

	static AlgorithmEnum get_random_algorithm_enum(std::mt19937& mt);

	static AlgorithmEnum get_random_algorithm_enum_for_combined_algorithms(std::mt19937& mt);

	static KernelType get_random_kernel_type(std::mt19937& mt);
	 
	static std::unique_ptr<KernelBase> build_kernel(const KernelType& kernel_type, bool default_parameters, std::mt19937& mt);
	 
	static std::vector<double> get_random_gamma_parameters(std::mt19937& mt);
	 
	static std::vector<double> get_random_gaussian_parameters(std::mt19937& mt);
	 
	static std::vector<double> get_random_linear_parameters(std::mt19937& mt);
	 
	static std::vector<double> get_random_weibull_parameters(std::mt19937& mt);
};