/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm_config_factory.h"

#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"

#include "factory/random/random_factory.h"

#include <algorithm>
#include <limits>

AlgorithmConfig AlgorithmConfigFactory::create_random_algorithm_config(std::mt19937& mt) {
	const auto algorithm_enum = get_random_algorithm_enum_for_combined_algorithms(mt);
	const auto& kernel_type = get_random_kernel_type(mt);
	const auto use_default_kernel_parameters = RandomFactory::get_random_bool(mt);
	auto kernel = build_kernel(kernel_type, use_default_kernel_parameters, mt);
	const auto use_default_theta = RandomFactory::get_random_bool(mt);
	if (algorithm_enum == AlgorithmEnum::Naive || use_default_theta) {
		return AlgorithmConfig(algorithm_enum, std::move(kernel));
	}
	const auto theta = RandomFactory::get_random_double(std::numeric_limits<double>::min(), 0.5, mt);

	return AlgorithmConfig(algorithm_enum, std::move(kernel), theta);
}

RelearnTypes::AlgorithmConfigs AlgorithmConfigFactory::create_random_algorithm_configs(std::uint64_t number, std::mt19937& mt) {
	auto configs = RelearnTypes::AlgorithmConfigs{};
	for (std::uint64_t i = 0; i < number; i++) {
		configs.push_back(create_random_algorithm_config(mt));
	}
	return configs;
}

std::uint64_t AlgorithmConfigFactory::get_random_number_algorithms(std::mt19937& mt) {
	return RandomFactory::get_random_integer<std::uint64_t>(1, 10, mt);
}

std::uint64_t AlgorithmConfigFactory::get_random_number_algorithms(std::uint64_t min, std::uint64_t max, std::mt19937& mt) {
	return RandomFactory::get_random_integer<std::uint64_t>(min, max, mt);
}

AlgorithmEnum AlgorithmConfigFactory::get_random_algorithm_enum(std::mt19937& mt) {
	return AlgorithmEnum{ RandomFactory::get_random_integer<uint8_t>(0, 5, mt) };
}

AlgorithmEnum AlgorithmConfigFactory::get_random_algorithm_enum_for_combined_algorithms(std::mt19937& mt) {
	return AlgorithmEnum{ RandomFactory::get_random_integer<uint8_t>(0, 3, mt) };
}

KernelType AlgorithmConfigFactory::get_random_kernel_type(std::mt19937& mt) {
	return KernelType{ RandomFactory::get_random_integer<uint8_t>(0, 3, mt) };
}

std::unique_ptr<KernelBase> AlgorithmConfigFactory::build_kernel(const KernelType& kernel_type, bool default_parameters, std::mt19937& mt) {
	switch (kernel_type) {
	case KernelType::Gamma: {
		if (default_parameters) {
			return std::make_unique<GammaDistributionKernel>();
		}
		const auto parameters = get_random_gamma_parameters(mt);
		return std::make_unique<GammaDistributionKernel>(parameters[0], parameters[1]);
	}
	case KernelType::Gaussian: {
		if (default_parameters) {
			return std::make_unique<GaussianDistributionKernel>();
		}
		const auto parameters = get_random_gaussian_parameters(mt);
		return std::make_unique<GaussianDistributionKernel>(parameters[0], parameters[1]);
	}
	case KernelType::Linear: {
		if (default_parameters) {
			return std::make_unique<GaussianDistributionKernel>();
		}
		const auto parameters = get_random_linear_parameters(mt);
		return std::make_unique<LinearDistributionKernel>(parameters[0]);
	}
	case KernelType::Weibull: {
		if (default_parameters) {
			return std::make_unique<WeibullDistributionKernel>();
		}
		const auto parameters = get_random_weibull_parameters(mt);
		return std::make_unique<WeibullDistributionKernel>(parameters[0], parameters[1]);
	}
	default:
		RelearnException::fail("AlgorithmConfigFactory::build_kernel: kernel_type unknown! ({})", kernel_type);
	}
}

std::vector<double> AlgorithmConfigFactory::get_random_gamma_parameters(std::mt19937& mt) {
	const auto k = RandomFactory::get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
	const auto theta = RandomFactory::get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
	return { k, theta };
}

std::vector<double> AlgorithmConfigFactory::get_random_gaussian_parameters(std::mt19937& mt) {
	const auto mu = RandomFactory::get_random_double(0.0, std::numeric_limits<double>::max(), mt);
	const auto sigma = RandomFactory::get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
	return { mu, sigma };
}

std::vector<double> AlgorithmConfigFactory::get_random_linear_parameters(std::mt19937& mt) {
	const auto cutoff = RandomFactory::get_random_double(0.0, std::numeric_limits<double>::max(), mt);
	return { cutoff };
}

std::vector<double> AlgorithmConfigFactory::get_random_weibull_parameters(std::mt19937& mt) {
	const auto k = RandomFactory::get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
	const auto b = RandomFactory::get_random_double(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), mt);
	return { k, b };
}
