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

#include "algorithm/Kernel/KernelBase.h"

#include <memory>
#include <random>

class KernelFactory {
public:
    [[nodiscard]] static std::unique_ptr<KernelBase> get_standard_gamma() noexcept;

    [[nodiscard]] static std::unique_ptr<KernelBase> get_standard_gaussian() noexcept;

    [[nodiscard]] static std::unique_ptr<KernelBase> get_standard_linear() noexcept;

    [[nodiscard]] static std::unique_ptr<KernelBase> get_standard_weibull() noexcept;

    [[nodiscard]] static std::unique_ptr<KernelBase> get_random_standard_kernel(std::mt19937& mt) noexcept;
};
