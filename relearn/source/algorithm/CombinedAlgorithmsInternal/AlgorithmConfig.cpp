/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "AlgorithmConfig.h"

#include <sstream>

std::string AlgorithmConfig::to_string() const {
    auto ss = std::stringstream{};
    ss << algorithm_type_to_string();
    ss << ":";
    const auto kernel_type = get_kernel_type();
    ss << kernel_type_to_string(kernel_type);
    const auto parameters = get_kernel_parameters(kernel_type);
    const auto check_params_default = check_parameters_default(parameters, kernel_type);
    if (!check_params_default) {
        ss << "(";
        ss << kernel_parameters_to_string(parameters);
        ss << ")";
    }
    if (has_theta()) {
        ss << ";";
        ss << boost::lexical_cast<std::string>(_theta.value());
    }
    return ss.str();
}

std::string AlgorithmConfig::algorithm_type_to_string() const {
    switch(_algorithm_type) {
    case AlgorithmEnum::BarnesHut:
        return "barnes-hut";
    case AlgorithmEnum::BarnesHutInverted:
        return "barnes-hut-inverted";
    case AlgorithmEnum::BarnesHutLocationAware:
        return "barnes-hut-location-aware";
    case AlgorithmEnum::Naive:
        return "naive";
    default:
        return "unimplemented";
    }
}

std::string AlgorithmConfig::kernel_type_to_string(const KernelType kernel_type) const {
    switch(kernel_type) {
    case KernelType::Gamma:
        return "gamma";
    case KernelType::Gaussian:
        return "gaussian";
    case KernelType::Linear:
        return "linear";
    case KernelType::Weibull:
        return "weibull";
    default:
        return "unimplemented";
    }
}

std::vector<double> AlgorithmConfig::get_kernel_parameters(const KernelType kernel_type) const {
    switch(kernel_type) {
    case KernelType::Gamma: {
        auto* kernel_ptr = dynamic_cast<GammaDistributionKernel*>(_kernel.get());
        if (!kernel_ptr) {
            RelearnException::fail("AlgorithmConfig::get_kernel_parameters: The following kernel type was given: {}, but casting failed!", kernel_type);
        }
        const auto k = kernel_ptr->get_k();
        const auto gamma_theta = kernel_ptr->get_theta();
        return {k, gamma_theta};
    }
    case KernelType::Gaussian: {
        auto* kernel_ptr = dynamic_cast<GaussianDistributionKernel*>(_kernel.get());
        if (!kernel_ptr) {
            RelearnException::fail("AlgorithmConfig::get_kernel_parameters: The following kernel type was given: {}, but casting failed!", kernel_type);
        }
        const auto mu = kernel_ptr->get_mu();
        const auto sigma = kernel_ptr->get_sigma();
        return {mu, sigma};
    }
    case KernelType::Linear:{
        auto* kernel_ptr = dynamic_cast<LinearDistributionKernel*>(_kernel.get());
        if (!kernel_ptr) {
            RelearnException::fail("AlgorithmConfig::get_kernel_parameters: The following kernel type was given: {}, but casting failed!", kernel_type);
        }
        const auto cutoff = kernel_ptr->get_cutoff();
        return {cutoff};
    }
    case KernelType::Weibull: {
        auto* kernel_ptr = dynamic_cast<WeibullDistributionKernel*>(_kernel.get());
        if (!kernel_ptr) {
            RelearnException::fail("AlgorithmConfig::get_kernel_parameters: The following kernel type was given: {}, but casting failed!", kernel_type);
        }
        const auto k = kernel_ptr->get_k();
        const auto b = kernel_ptr->get_b();
        return {k, b};
    }
    default:
        RelearnException::fail("AlgorithmConfig::get_kernel_parameters: kernel type unknown! ({})", kernel_type);
    }
}

bool AlgorithmConfig::check_parameters_default(const std::vector<double>& parameters, const KernelType kernel_type) const {
    switch(kernel_type) {
        case KernelType::Gamma: {
            const auto k = parameters[0];
            const auto gamma_theta = parameters[1];
            return k == GammaDistributionKernel::default_k && gamma_theta == GammaDistributionKernel::default_theta;
        }
        case KernelType::Gaussian: { 
            const auto mu = parameters[0];
            const auto sigma = parameters[1];
            return mu == GaussianDistributionKernel::default_mu && sigma == GaussianDistributionKernel::default_sigma;
        }
        case KernelType::Linear: {
            const auto cutoff = parameters[0];
            return cutoff == LinearDistributionKernel::default_cutoff;
        }
        case KernelType::Weibull: {
            const auto k = parameters[0];
            const auto b = parameters[1];
            return k == WeibullDistributionKernel::default_k && b == WeibullDistributionKernel::default_b;
        }
        default:
            RelearnException::fail("AlgorithmConfig::check_parameters_default: kernel type unknown! ({})", kernel_type);
    }
}

std::string AlgorithmConfig::kernel_parameters_to_string(const std::vector<double>& parameters) const {
    auto result = std::string{};
    for (const auto& parameter : parameters) {
        result.append(boost::lexical_cast<std::string>(parameter));
        result.append(",");
    }
    return result;
}