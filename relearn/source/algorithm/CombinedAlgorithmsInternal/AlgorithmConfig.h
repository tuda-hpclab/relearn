#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/KernelBase.h"
#include "algorithm/Kernel/KernelType.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "types/BasicTypes.h"

#include <boost/lexical_cast.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief This class models a configuration of an algorithm used in the combined algorithms method. It incorporates
 * the type of the algorithm, its kernel (with parameters) and possibly an acceptance criterion for barnes-hut algorithms.
 */
class AlgorithmConfig {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using acceptance_criterion_type = RelearnTypes::acceptance_criterion_type;

    /**
     * Constructs a new AlgorithmConfig object
     * @param algorithm_type The type of the algorithm
     * @param kernel The kernel that the algorithm should use
     * @param theta Optionally a theta value if the algorithm implements barnes-hut. If none is given but the algorithm is barnes-hut,
     *              the theta given through --theta will be used in usage with CombinedAlgorithms. If that is not given, the default value will be used.
     */
    AlgorithmConfig(AlgorithmEnum algorithm_type, std::unique_ptr<KernelBase> kernel, std::optional<acceptance_criterion_type> theta = std::nullopt)
        : _algorithm_type{ algorithm_type }
        , _kernel{ std::move(kernel) }
        , _theta{ theta } { }

    AlgorithmConfig(AlgorithmConfig&& other) = default;

    AlgorithmConfig& operator=(AlgorithmConfig&& other) = default;

    AlgorithmConfig(const AlgorithmConfig&) = delete;
    AlgorithmConfig& operator=(const AlgorithmConfig&) = delete;

    /**
     * @brief Checks whether this AlgorithmConfig is approximately equal to the other one. This is not transitive! It accounts for
     *        small deviations in the parameters by using Constants::eps
     * @param other The other AlgorithmConfig
     * @param epsilon The epsilon used to compare the parameters
     * @return Whether this and the other AlgorithmConfig are approximately equal
     */
    [[nodiscard]] bool is_approximately_equal(const AlgorithmConfig& other, acceptance_criterion_type epsilon = static_cast<acceptance_criterion_type>(Constants::eps)) const noexcept {
        if (_algorithm_type != other.get_algorithm_type() || has_theta() != other.has_theta()) {
            return false;
        }
        if (has_theta()) {
            if (std::fabs(_theta.value() - other.get_theta().value()) >= epsilon) { // checks inequality by testing if the difference is greater epsilon
                return false;
            }
        }
        if (!_kernel || !other.get_kernel()) {
            return _kernel.get() == other.get_kernel();
        }
        return _kernel->is_approximately_equal(*(other.get_kernel()), epsilon);
    }

    /**
     * @brief Returns whether this algorithm config has a theta
     * @return A bool indicating whether the algorithm config has a theta
     */
    [[nodiscard]] bool has_theta() const {
        return _theta.has_value();
    }

    /**
     * @brief Returns the algorithm type
     * @return The algorithm type of the config
     */
    [[nodiscard]] AlgorithmEnum get_algorithm_type() const {
        return _algorithm_type;
    }

    /**
     * @brief Returns a regular pointer on the kernel of the config
     * @return A regular pointer on the kernel of the config
     */
    [[nodiscard]] KernelBase* get_kernel() const {
        return _kernel.get();
    }

    /**
     * @brief Returns the theta optional
     * @return The theta optional of the config (can be empty)
     */
    [[nodiscard]] std::optional<acceptance_criterion_type> get_theta() const {
        return _theta;
    }

    /**
     * @brief Moves the kernel to prepare for transfer of the kernel (it won't be accessible from the config afterwards)
     * @return The kernel
     */
    [[nodiscard]] std::unique_ptr<KernelBase>&& transfer_kernel() && {
        return std::move(_kernel);
    }

    /**
     * @brief Returns the type of the kernel
     * @return The type of the kernel
     */
    [[nodiscard]] KernelType get_kernel_type() const {
        return _kernel->get_kernel_type();
    }

    /**
     * @brief Returns a string representation of the config as it could be used in a combined algorithms config file
     * @return A string representation of the config as it could be used in a config file
     */
    [[nodiscard]] std::string to_string() const;

private:
    /**
     * @brief Returns a string represantion of the algorithm type as it could be used in a combined algorithms config file
     * @return A string represantion of the algorithm type as it could be used in a combined algorithms config file
     */
    [[nodiscard]] std::string algorithm_type_to_string() const;

    /**
     * @brief Returns a string represantion of the kernel type as it could be used in a combined algorithms config file
     * @return A string represantion of the kernel type as it could be used in a combined algorithms config file
     */
    [[nodiscard]] std::string kernel_type_to_string(const KernelType kernel_type) const;

    /**
     * @brief Returns the kernel parameters of the kernel
     * @return The kernel parameters of the kernel in a vector
     */
    [[nodiscard]] std::vector<attraction_type> get_kernel_parameters(const KernelType kernel_type) const;

    /**
     * @brief Returns whether the kernel parameters are default parameters
     * @return A bool indicating whether the kernel parameters are default
     */
    [[nodiscard]] bool check_parameters_default(const std::vector<attraction_type>& parameters, const KernelType kernel_type) const;

    /**
     * @brief Returns a string representation of the kernel parameters as they could be used in a combined algorithms config file
     * @return A string representation of the kernel parameters as they could be used in a combined algorithms config file
     */
    [[nodiscard]] std::string kernel_parameters_to_string(const std::vector<attraction_type>& parameters) const;

    AlgorithmEnum _algorithm_type;
    std::unique_ptr<KernelBase> _kernel;
    std::optional<acceptance_criterion_type> _theta;
};