/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "kernel_factory.h"

#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"

#include "factory/random/random_factory.h"

std::unique_ptr<KernelBase> KernelFactory::get_standard_gamma() noexcept {
    return std::make_unique<GammaDistributionKernel>();
}

std::unique_ptr<KernelBase> KernelFactory::get_standard_gaussian() noexcept {
    return std::make_unique<GaussianDistributionKernel>();
}

std::unique_ptr<KernelBase> KernelFactory::get_standard_linear() noexcept {
    return std::make_unique<LinearDistributionKernel>();
}

std::unique_ptr<KernelBase> KernelFactory::get_standard_weibull() noexcept {
    return std::make_unique<WeibullDistributionKernel>();
}

std::unique_ptr<KernelBase> KernelFactory::get_random_standard_kernel(std::mt19937& mt) noexcept {
    const auto kernel_number = RandomFactory::get_random_integer(0, 3, mt);

    switch (kernel_number) {
    case 0:
        return get_standard_gamma();
    case 1:
        return get_standard_gaussian();
    case 2:
        return get_standard_linear();
    case 3:
        return get_standard_weibull();
    default:
        return get_standard_gamma(); // this should not happen
    }
}
