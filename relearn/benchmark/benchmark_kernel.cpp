/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "types/BasicTypes.h"
#include "util/Vec3.h"

#include <benchmark/benchmark.h>

#include <cpp-utility/Cast.hpp>

#include <range/v3/numeric/accumulate.hpp>

#include <utility>
#include <vector>

namespace {
// The positions are computed in the type they are stored in; a Vec3 of space_type rejects a double that it
// cannot represent exactly, and the factors below are not exactly representable once space_type is float.
using space_type = RelearnTypes::space_type;

template <typename KernelType>
void BM_Kernel(benchmark::State& state) {
    const auto number_pairs = static_cast<std::size_t>(state.range(0));

    auto pairs = std::vector<std::pair<RelearnTypes::position_type, RelearnTypes::position_type>>{};
    pairs.reserve(number_pairs);

    for (auto i = 0U; i < number_pairs; i++) {
        const auto index = static_cast<space_type>(i);

        const auto source = RelearnTypes::position_type{ index, index * utility::as<space_type>(13.5), index * utility::as<space_type>(18.4) };
        const auto target = RelearnTypes::position_type{ index + utility::as<space_type>(82.3), index * utility::as<space_type>(472.4), index * utility::as<space_type>(-1.3) };

        pairs.emplace_back(source, target);
    }

    auto attractivenesses = std::vector<RelearnTypes::attraction_type>{};
    attractivenesses.resize(number_pairs, 0.0);

    const auto kernel = KernelType{};

    for (auto _ : state) {
        for (auto i = 0U; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = kernel.get_probability(source, target, 3);
            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        benchmark::DoNotOptimize(ranges::accumulate(attractivenesses, 0.0));

        state.ResumeTiming();
    }
}

void BM_Gaussian_Kernel(benchmark::State& state) {
    BM_Kernel<GaussianDistributionKernel>(state);
}

void BM_Gamma_Kernel(benchmark::State& state) {
    BM_Kernel<GammaDistributionKernel>(state);
}

void BM_Linear_Kernel(benchmark::State& state) {
    BM_Kernel<LinearDistributionKernel>(state);
}

void BM_Weibull_Kernel(benchmark::State& state) {
    BM_Kernel<WeibullDistributionKernel>(state);
}
} // namespace

BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);

BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);

BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);

BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(small_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(medium_number_neurons)->Iterations(large_number_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(large_number_neurons)->Iterations(large_number_iterations);
