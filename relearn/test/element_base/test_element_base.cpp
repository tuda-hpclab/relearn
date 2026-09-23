/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_element_base.h"

#include "neurons/synaptic_elements/ElementBase.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include "factory/extra_info/extra_info_factory.h"

#include <cpp-utility/Cast.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <array>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <vector>

// The deltas below are written as decimal literals; this spells them in the type the elements are stored in.
[[nodiscard]] inline std::vector<RelearnTypes::grown_type> grown_values(const std::initializer_list<double> values) {
    auto result = std::vector<RelearnTypes::grown_type>{};
    result.reserve(values.size());
    for (const auto value : values) {
        result.emplace_back(static_cast<RelearnTypes::grown_type>(value));
    }
    return result;
}

// The expected values below are written as decimal literals, too; this spells a whole array of them
// in the type the elements are stored in, so that the comparisons stay exact.
template <typename... Values>
[[nodiscard]] consteval auto grown_array(const Values... values) {
    return std::array{ utility::as<RelearnTypes::grown_type>(values)... };
}

TEST_F(ElementBaseTest, testGetZero) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < RelearnTypes::number_neurons_type{ 2049 }; i++) {
        const auto zero_double = details::get_zero<double>(i);
        ASSERT_EQ(zero_double, 0.0);

        const auto zero_float = details::get_zero<float>(i);
        ASSERT_EQ(zero_float, 0.0F);

        const auto zero_unsigned_int = details::get_zero<unsigned int>(i);
        ASSERT_EQ(zero_unsigned_int, 0U);

        const auto zero_int = details::get_zero<int>(i);
        ASSERT_EQ(zero_int, 0);

        const auto zero_number_neurons_type = details::get_zero<RelearnTypes::number_neurons_type>(i);
        ASSERT_EQ(zero_number_neurons_type, RelearnTypes::number_neurons_type{ 0 });
    }
}

TEST_F(ElementBaseTest, testConstruction) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();

    ASSERT_EQ(grown_elements.size(), 0);
    ASSERT_EQ(deltas.size(), 0);
    ASSERT_EQ(vacant_elements.size(), 0);
    ASSERT_EQ(connected_elements.size(), 0);
    ASSERT_EQ(retract_ratio.size(), 0);
}

TEST_F(ElementBaseTest, testEmptyInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 13 };
    const auto number_neurons_create = RelearnTypes::number_neurons_type{ 15 };
    const auto number_neurons = number_neurons_init + number_neurons_create;

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

#ifndef RELEARN_CUDA_ENABLED
    ASSERT_EQ(element_base.get_total_additions(), 0.0);
    ASSERT_EQ(element_base.get_total_deletions(), 0.0);
#endif
    ASSERT_EQ(grown_elements.size(), number_neurons_init);
    ASSERT_EQ(deltas.size(), number_neurons_init);
    ASSERT_EQ(vacant_elements.size(), number_neurons_init);
    ASSERT_EQ(connected_elements.size(), number_neurons_init);
    ASSERT_EQ(retract_ratio.size(), number_neurons_init);
    ASSERT_EQ(minimum_calcum.size(), number_neurons_init);

    for (const auto val : grown_elements) {
        ASSERT_EQ(val, 0.0);
    }

    for (const auto delta : deltas) {
        ASSERT_EQ(delta, 0.0);
    }

    for (const auto vacant : vacant_elements) {
        ASSERT_EQ(vacant, 0U);
    }

    for (const auto connected : connected_elements) {
        ASSERT_EQ(connected, 0U);
    }

    for (const auto ratio : retract_ratio) {
        ASSERT_EQ(ratio, 0.0);
    }

    for (const auto calcium : minimum_calcum) {
        ASSERT_EQ(calcium, 0.0);
    }

    element_base.create_neurons(number_neurons_create);

    const auto grown_elements_2 = element_base.get_grown_elements();
    const auto deltas_2 = element_base.get_deltas();
    const auto vacant_elements_2 = element_base.get_vacant_elements();
    const auto connected_elements_2 = element_base.get_connected_elements();
    const auto retract_ratio_2 = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum_2 = element_base.get_minimum_calcium();

#ifndef RELEARN_CUDA_ENABLED
    ASSERT_EQ(element_base.get_total_additions(), 0.0);
    ASSERT_EQ(element_base.get_total_deletions(), 0.0);
#endif
    ASSERT_EQ(grown_elements_2.size(), number_neurons);
    ASSERT_EQ(deltas_2.size(), number_neurons);
    ASSERT_EQ(vacant_elements_2.size(), number_neurons);
    ASSERT_EQ(connected_elements_2.size(), number_neurons);
    ASSERT_EQ(retract_ratio_2.size(), number_neurons);
    ASSERT_EQ(minimum_calcum_2.size(), number_neurons);

    for (const auto val : grown_elements_2) {
        ASSERT_EQ(val, 0.0);
    }

    for (const auto delta : deltas_2) {
        ASSERT_EQ(delta, 0.0);
    }

    for (const auto vacant : vacant_elements_2) {
        ASSERT_EQ(vacant, 0U);
    }

    for (const auto connected : connected_elements_2) {
        ASSERT_EQ(connected, 0U);
    }

    for (const auto ratio : retract_ratio_2) {
        ASSERT_EQ(ratio, 0.0);
    }

    for (const auto calcium : minimum_calcum_2) {
        ASSERT_EQ(calcium, 0.0);
    }
}

TEST_F(ElementBaseTest, testInitAndCreateException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 13 };
    const auto number_neurons_create = RelearnTypes::number_neurons_type{ 15 };
    const auto number_neurons = number_neurons_init + number_neurons_create;

    ASSERT_THROW_NO_PRINT(element_base.init(0), RelearnException);

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    ASSERT_THROW_NO_PRINT(element_base.init(number_neurons_init), RelearnException);

    ASSERT_THROW_NO_PRINT(element_base.create_neurons(0), RelearnException);

    element_base.create_neurons(number_neurons_create);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

#ifndef RELEARN_CUDA_ENABLED
    ASSERT_EQ(element_base.get_total_additions(), 0.0);
    ASSERT_EQ(element_base.get_total_deletions(), 0.0);
#endif
    ASSERT_EQ(grown_elements.size(), number_neurons);
    ASSERT_EQ(deltas.size(), number_neurons);
    ASSERT_EQ(vacant_elements.size(), number_neurons);
    ASSERT_EQ(connected_elements.size(), number_neurons);
    ASSERT_EQ(retract_ratio.size(), number_neurons);
    ASSERT_EQ(minimum_calcum.size(), number_neurons);

    for (const auto val : grown_elements) {
        ASSERT_EQ(val, 0.0);
    }

    for (const auto delta : deltas) {
        ASSERT_EQ(delta, 0.0);
    }

    for (const auto vacant : vacant_elements) {
        ASSERT_EQ(vacant, 0U);
    }

    for (const auto connected : connected_elements) {
        ASSERT_EQ(connected, 0U);
    }

    for (const auto ratio : retract_ratio) {
        ASSERT_EQ(ratio, 0.0);
    }

    for (const auto calcium : minimum_calcum) {
        ASSERT_EQ(calcium, 0.0);
    }
}

TEST_F(ElementBaseTest, testSetCalculatorsInit) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto grown_elements_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return static_cast<RelearnTypes::grown_type>(i);
    };

    const auto delta_since_last_update_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return static_cast<RelearnTypes::grown_type>(i * 2);
    };

    const auto connected_elements_calculator = [](const RelearnTypes::number_neurons_type i) -> unsigned int {
        return static_cast<unsigned int>(i * 3);
    };

    const auto vacant_retract_ratio_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return RelearnTypes::grown_type{ 1 } / (RelearnTypes::grown_type{ 1 } + static_cast<RelearnTypes::grown_type>(i));
    };

    const auto minimum_calcium_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::calcium_type {
        return static_cast<RelearnTypes::calcium_type>(i) * utility::as<RelearnTypes::calcium_type>(0.1);
    };

    element_base.set_grown_elements_calculator(grown_elements_calculator);
    element_base.set_delta_since_last_update_calculator(delta_since_last_update_calculator);
    element_base.set_connected_elements_calculator(connected_elements_calculator);
    element_base.set_vacant_retract_ratio_calculator(vacant_retract_ratio_calculator);
    element_base.set_minimum_calcium_calculator(minimum_calcium_calculator);

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 13 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

#ifndef RELEARN_CUDA_ENABLED
    ASSERT_EQ(element_base.get_total_additions(), 0.0);
    ASSERT_EQ(element_base.get_total_deletions(), 0.0);
#endif
    ASSERT_EQ(grown_elements.size(), number_neurons_init);
    ASSERT_EQ(deltas.size(), number_neurons_init);
    ASSERT_EQ(vacant_elements.size(), number_neurons_init);
    ASSERT_EQ(connected_elements.size(), number_neurons_init);
    ASSERT_EQ(retract_ratio.size(), number_neurons_init);
    ASSERT_EQ(minimum_calcum.size(), number_neurons_init);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], grown_elements_calculator(i));
        ASSERT_EQ(deltas[i], delta_since_last_update_calculator(i));
        ASSERT_EQ(connected_elements[i], connected_elements_calculator(i));
        ASSERT_NEAR_EPS(retract_ratio[i], vacant_retract_ratio_calculator(i));
        ASSERT_NEAR_EPS(minimum_calcum[i], minimum_calcium_calculator(i));

        if (static_cast<RelearnTypes::grown_type>(connected_elements[i]) >= grown_elements[i]) {
            ASSERT_EQ(vacant_elements[i], 0U);
        } else {
            ASSERT_EQ(vacant_elements[i], static_cast<unsigned int>(grown_elements[i] - static_cast<RelearnTypes::grown_type>(connected_elements[i])));
        }
    }
}

TEST_F(ElementBaseTest, testSetCalculatorsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    ASSERT_THROW(element_base.set_grown_elements_calculator(nullptr), RelearnException);
    ASSERT_THROW(element_base.set_delta_since_last_update_calculator(nullptr), RelearnException);
    ASSERT_THROW(element_base.set_connected_elements_calculator(nullptr), RelearnException);
    ASSERT_THROW(element_base.set_vacant_retract_ratio_calculator(nullptr), RelearnException);
    ASSERT_THROW(element_base.set_minimum_calcium_calculator(nullptr), RelearnException);
}

TEST_F(ElementBaseTest, testSetCalculatorsInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto grown_elements_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return static_cast<RelearnTypes::grown_type>(i);
    };

    const auto delta_since_last_update_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return static_cast<RelearnTypes::grown_type>(i * 2);
    };

    const auto connected_elements_calculator = [](const RelearnTypes::number_neurons_type i) -> unsigned int {
        return static_cast<unsigned int>(i * 3);
    };

    const auto vacant_retract_ratio_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::grown_type {
        return RelearnTypes::grown_type{ 1 } / (RelearnTypes::grown_type{ 1 } + static_cast<RelearnTypes::grown_type>(i));
    };

    const auto minimum_calcium_calculator = [](const RelearnTypes::number_neurons_type i) -> RelearnTypes::calcium_type {
        return static_cast<RelearnTypes::calcium_type>(i) * utility::as<RelearnTypes::calcium_type>(0.1);
    };

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 13 };
    const auto number_neurons_create = RelearnTypes::number_neurons_type{ 15 };
    const auto number_neurons = number_neurons_init + number_neurons_create;

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    element_base.set_grown_elements_calculator(grown_elements_calculator);
    element_base.set_delta_since_last_update_calculator(delta_since_last_update_calculator);
    element_base.set_connected_elements_calculator(connected_elements_calculator);
    element_base.set_vacant_retract_ratio_calculator(vacant_retract_ratio_calculator);
    element_base.set_minimum_calcium_calculator(minimum_calcium_calculator);

    element_base.create_neurons(number_neurons_create);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

#ifndef RELEARN_CUDA_ENABLED
    ASSERT_EQ(element_base.get_total_additions(), 0.0);
    ASSERT_EQ(element_base.get_total_deletions(), 0.0);
#endif
    ASSERT_EQ(grown_elements.size(), number_neurons);
    ASSERT_EQ(deltas.size(), number_neurons);
    ASSERT_EQ(vacant_elements.size(), number_neurons);
    ASSERT_EQ(connected_elements.size(), number_neurons);
    ASSERT_EQ(retract_ratio.size(), number_neurons);
    ASSERT_EQ(minimum_calcum.size(), number_neurons);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], 0.0);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_EQ(retract_ratio[i], 0.0);
        ASSERT_EQ(minimum_calcum[i], 0.0);
    }

    for (auto i = number_neurons_init; i < number_neurons; i++) {
        ASSERT_EQ(grown_elements[i], grown_elements_calculator(i));
        ASSERT_EQ(deltas[i], delta_since_last_update_calculator(i));
        ASSERT_EQ(connected_elements[i], connected_elements_calculator(i));
        ASSERT_NEAR_EPS(retract_ratio[i], vacant_retract_ratio_calculator(i));
        ASSERT_NEAR_EPS(minimum_calcum[i], minimum_calcium_calculator(i));

        if (static_cast<RelearnTypes::grown_type>(connected_elements[i]) >= grown_elements[i]) {
            ASSERT_EQ(vacant_elements[i], 0U);
        } else {
            ASSERT_EQ(vacant_elements[i], static_cast<unsigned int>(grown_elements[i] - static_cast<RelearnTypes::grown_type>(connected_elements[i])));
        }
    }
}

TEST_F(ElementBaseTest, testAddToDelta) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto deltas_to_add = grown_values({ 0.0, -4.2, 6.01, 1.0 });

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.add_to_delta(deltas_to_add[i], i);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], 0.0);
        ASSERT_EQ(deltas[i], deltas_to_add[i]);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    element_base.add_to_delta(deltas_to_add);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], 0.0);
        ASSERT_EQ(deltas[i], RelearnTypes::grown_type{ 2 } * deltas_to_add[i]);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testAddToDeltaException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto deltas_to_add = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    auto deltas_to_add_2 = grown_values({ 0.0, -4.2, 6.01, 1.0, 9.5 });

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.add_to_delta(deltas_to_add[i], i);
    }

    for (auto i = number_neurons_init; i < number_neurons_init + number_neurons_init; i++) {
        ASSERT_THROW_NO_PRINT(element_base.add_to_delta(utility::as<RelearnTypes::grown_type>(0.3), i), RelearnException);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], 0.0);
        ASSERT_EQ(deltas[i], deltas_to_add[i]);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    element_base.add_to_delta(deltas_to_add);
    ASSERT_THROW_NO_PRINT(element_base.add_to_delta(deltas_to_add_2);, RelearnException);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        ASSERT_EQ(grown_elements[i], 0.0);
        ASSERT_EQ(deltas[i], RelearnTypes::grown_type{ 2 } * deltas_to_add[i]);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testAddConnectedElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto newly_added = std::vector<unsigned int>{ 0U, 4U, 6U, 1U };

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.add_connected_elements(newly_added[i], i);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(0.0, 4.0, 6.0, 1.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], newly_added[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    element_base.add_connected_elements(newly_added);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(0.0, 4.0, 6.0, 1.0);

        ASSERT_EQ(grown_elements[i], 2 * expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 2 * newly_added[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testAddConnectedElementsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto newly_added = std::vector<unsigned int>{ 0U, 4U, 6U, 1U };
    auto newly_added_2 = std::vector<unsigned int>{ 0U, 4U, 6U, 1U, 9U };

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.add_connected_elements(newly_added[i], i);
    }

    for (auto i = number_neurons_init; i < number_neurons_init + number_neurons_init; i++) {
        ASSERT_THROW_NO_PRINT(element_base.add_connected_elements(3U, i), RelearnException);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(0.0, 4.0, 6.0, 1.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], newly_added[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    element_base.add_connected_elements(newly_added);
    ASSERT_THROW_NO_PRINT(element_base.add_connected_elements(newly_added_2);, RelearnException);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(0.0, 4.0, 6.0, 1.0);

        ASSERT_EQ(grown_elements[i], 2 * expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], 0U);
        ASSERT_EQ(connected_elements[i], 2 * newly_added[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testRemoveConnectedElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto newly_added = std::vector<unsigned int>{ 10U, 10U, 10U, 10U };
    element_base.add_connected_elements(newly_added);

    auto newly_removed = std::vector<unsigned int>{ 0U, 4U, 3U, 1U };

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.remove_connected_elements(newly_removed[i], i);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(10.0, 10.0, 10.0, 10.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], newly_removed[i]);
        ASSERT_EQ(connected_elements[i], newly_added[i] - newly_removed[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    element_base.remove_connected_elements(newly_removed);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(10.0, 10.0, 10.0, 10.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], newly_removed[i] + newly_removed[i]);
        ASSERT_EQ(connected_elements[i], newly_added[i] - newly_removed[i] - newly_removed[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testRemoveConnectedElementsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto newly_added = std::vector<unsigned int>{ 10U, 10U, 10U, 10U };
    element_base.add_connected_elements(newly_added);

    auto newly_removed = std::vector<unsigned int>{ 0U, 4U, 3U, 1U };
    auto newly_removed_2 = std::vector<unsigned int>{ 0U, 4U, 3U, 1U, 2U };

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.remove_connected_elements(newly_removed[i], i);
    }

    for (auto i = number_neurons_init; i < number_neurons_init + number_neurons_init; i++) {
        ASSERT_THROW_NO_PRINT(element_base.remove_connected_elements(3U, i), RelearnException);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(10.0, 10.0, 10.0, 10.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], newly_removed[i]);
        ASSERT_EQ(connected_elements[i], newly_added[i] - newly_removed[i]);
        ASSERT_EQ(retract_ratio[i], 0.0);
        ASSERT_EQ(minimum_calcum[i], 0.0);
    }

    element_base.remove_connected_elements(newly_removed);
    ASSERT_THROW_NO_PRINT(element_base.remove_connected_elements(newly_removed_2);, RelearnException);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(10.0, 10.0, 10.0, 10.0);

        ASSERT_EQ(grown_elements[i], expected_grown[i]);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], newly_removed[i] + newly_removed[i]);
        ASSERT_EQ(connected_elements[i], newly_added[i] - newly_removed[i] - newly_removed[i]);
        ASSERT_EQ(retract_ratio[i], 0.0);
        ASSERT_EQ(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testConnectElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add = grown_values({ 21.5, 12.5, 16.01, 72.0 });
    element_base.add_to_delta(deltas_to_add);
    std::ignore = element_base.commit_updates();

    auto to_connect = std::vector<unsigned int>{ 4U, 3U, 0U, 16U };
    element_base.connect_elements(to_connect);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 17U, 9U, 16U, 56U };
        constexpr auto expected_connected = std::array{ 4U, 3U, 0U, 16U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.connect_elements(to_connect[i], i);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 13U, 6U, 16U, 40U };
        constexpr auto expected_connected = std::array{ 8U, 6U, 0U, 32U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testConnectElementsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add = grown_values({ 21.5, 12.5, 16.01, 72.0 });
    element_base.add_to_delta(deltas_to_add);
    std::ignore = element_base.commit_updates();

    for (auto i = number_neurons_init; i < number_neurons_init + number_neurons_init; i++) {
        ASSERT_THROW_NO_PRINT(element_base.connect_elements(3U, i), RelearnException);
    }

    auto to_connect_1 = std::vector<unsigned int>{ 2U, 3U, 0U, 16U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.connect_elements(to_connect_1);, RelearnException);

    auto to_connect_2 = std::vector<unsigned int>{ 200U, 1U, 1U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.connect_elements(to_connect_2);, RelearnException);

    auto to_connect_3 = std::vector<unsigned int>{ 0U, 200U, 1U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.connect_elements(to_connect_3);, RelearnException);

    auto to_connect_4 = std::vector<unsigned int>{ 0U, 0U, 100U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.connect_elements(to_connect_4);, RelearnException);

    auto to_connect_5 = std::vector<unsigned int>{ 0U, 0U, 0U, 704U };
    ASSERT_THROW_NO_PRINT(element_base.connect_elements(to_connect_5);, RelearnException);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 21U, 12U, 16U, 72U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testDisconnectElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add = grown_values({ 21.5, 12.5, 16.01, 72.0 });
    element_base.add_to_delta(deltas_to_add);
    std::ignore = element_base.commit_updates();

    auto to_connect = std::vector<unsigned int>{ 17U, 11U, 6U, 45U };
    element_base.connect_elements(to_connect);

    auto to_disconnect = std::vector<unsigned int>{ 4U, 3U, 0U, 16U };
    element_base.disconnect_elements(to_disconnect);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 8U, 4U, 10U, 43U };
        constexpr auto expected_connected = std::array{ 13U, 8U, 6U, 29U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        element_base.disconnect_elements(to_disconnect[i], i);
    }

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 12U, 7U, 10U, 59U };
        constexpr auto expected_connected = std::array{ 9U, 5U, 6U, 13U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testDisconnectElementsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add = grown_values({ 21.5, 12.5, 16.01, 72.0 });
    element_base.add_to_delta(deltas_to_add);
    std::ignore = element_base.commit_updates();

    auto to_connect = std::vector<unsigned int>{ 17U, 11U, 6U, 45U };
    element_base.connect_elements(to_connect);

    auto to_disconnect_1 = std::vector<unsigned int>{ 2U, 3U, 0U, 16U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.disconnect_elements(to_disconnect_1);, RelearnException);

    auto to_disconnect_2 = std::vector<unsigned int>{ 200U, 1U, 1U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.disconnect_elements(to_disconnect_2);, RelearnException);

    auto to_disconnect_3 = std::vector<unsigned int>{ 0U, 200U, 1U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.disconnect_elements(to_disconnect_3);, RelearnException);

    auto to_disconnect_4 = std::vector<unsigned int>{ 0U, 0U, 100U, 7U };
    ASSERT_THROW_NO_PRINT(element_base.disconnect_elements(to_disconnect_4);, RelearnException);

    auto to_disconnect_5 = std::vector<unsigned int>{ 0U, 0U, 0U, 704U };
    ASSERT_THROW_NO_PRINT(element_base.disconnect_elements(to_disconnect_5);, RelearnException);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr auto expected_grown = grown_array(21.5, 12.5, 16.01, 72.0);
        constexpr auto expected_vacant = std::array{ 4U, 1U, 10U, 27U };
        constexpr auto expected_connected = std::array{ 17U, 11U, 6U, 45U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testCommitUpdates1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    element_base.add_to_delta(deltas_to_add);

    const auto deletions = element_base.commit_updates();

    ASSERT_EQ(deletions.size(), number_neurons_init);
    for (const auto deletion : deletions) {
        ASSERT_EQ(deletion, 0U);
    }

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(0.0, 0.0, 6.01, 1.0);
        constexpr static auto expected_vacant = std::array{ 0U, 0U, 6U, 1U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testCommitUpdates2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add_1 = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    element_base.add_to_delta(deltas_to_add_1);

    const auto deletions_1 = element_base.commit_updates();

    ASSERT_EQ(deletions_1.size(), number_neurons_init);
    for (const auto deletion : deletions_1) {
        ASSERT_EQ(deletion, 0U);
    }

    auto grown_elements = element_base.get_grown_elements();
    auto deltas = element_base.get_deltas();
    auto vacant_elements = element_base.get_vacant_elements();
    auto connected_elements = element_base.get_connected_elements();
    auto retract_ratio = element_base.get_vacant_retract_ratio();
    auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(0.0, 0.0, 6.01, 1.0);
        constexpr static auto expected_vacant = std::array{ 0U, 0U, 6U, 1U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    auto deltas_to_add_2 = grown_values({ 0.4, 1.3, -4.002, -0.2 });
    element_base.add_to_delta(deltas_to_add_2);

    const auto deletions_2 = element_base.commit_updates();

    ASSERT_EQ(deletions_2.size(), number_neurons_init);
    for (const auto deletion : deletions_2) {
        ASSERT_EQ(deletion, 0U);
    }

    grown_elements = element_base.get_grown_elements();
    deltas = element_base.get_deltas();
    vacant_elements = element_base.get_vacant_elements();
    connected_elements = element_base.get_connected_elements();
    retract_ratio = element_base.get_vacant_retract_ratio();
    minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(0.4, 1.3, 2.008, 0.8);
        constexpr static auto expected_vacant = std::array{ 0U, 1U, 2U, 0U };

        // The third value is the sum of two deltas, which grown_type does not round to the literal below
        ASSERT_NEAR(grown_elements[i], expected_grown[i], eps);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testCommitUpdates3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto connected_to_add = std::vector<RelearnTypes::counter_type>{ 10U, 10U, 10U, 9U };
    auto connected_to_remove = std::vector<RelearnTypes::counter_type>{ 4U, 8U, 2U, 0U };
    auto delta_to_add = grown_values({ 0.2, 0.3, 0.9, 0.7 });

    element_base.add_to_delta(delta_to_add);

    const auto deletions = element_base.commit_updates();
    ASSERT_EQ(deletions.size(), number_neurons_init);
    for (const auto deletion : deletions) {
        ASSERT_EQ(deletion, 0U);
    }

    element_base.add_connected_elements(connected_to_add);
    element_base.remove_connected_elements(connected_to_remove);

    auto grown_elements = element_base.get_grown_elements();
    auto deltas = element_base.get_deltas();
    auto vacant_elements = element_base.get_vacant_elements();
    auto connected_elements = element_base.get_connected_elements();
    auto retract_ratio = element_base.get_vacant_retract_ratio();
    auto minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(10.2, 10.3, 10.9, 9.7);
        constexpr static auto expected_vacant = std::array{ 4U, 8U, 2U, 0U };
        constexpr static auto expected_connected = std::array{ 6U, 2U, 8U, 9U };

        ASSERT_NEAR_EPS(grown_elements[i], expected_grown[i]);
        ASSERT_NEAR_EPS(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }

    auto deltas_to_add_2 = grown_values({ -3.7, 2.1, -4.5, -12.1 });
    element_base.add_to_delta(deltas_to_add_2);

    const auto deletions_2 = element_base.commit_updates();
    ASSERT_EQ(deletions_2.size(), number_neurons_init);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_deleted = std::array{ 0U, 0U, 2U, 9U };
        ASSERT_EQ(deletions_2[i], expected_deleted[i]);
    }

    grown_elements = element_base.get_grown_elements();
    deltas = element_base.get_deltas();
    vacant_elements = element_base.get_vacant_elements();
    connected_elements = element_base.get_connected_elements();
    retract_ratio = element_base.get_vacant_retract_ratio();
    minimum_calcum = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(6.5, 12.4, 6.4, 0.0);
        constexpr static auto expected_vacant = std::array{ 0U, 10U, 0U, 0U };
        constexpr static auto expected_connected = std::array{ 6U, 2U, 6U, 0U };

        ASSERT_NEAR(grown_elements[i], expected_grown[i], eps);
        ASSERT_EQ(deltas[i], 0.0);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], expected_connected[i]);
        ASSERT_EQ(retract_ratio[i], 0.0);
        ASSERT_EQ(minimum_calcum[i], 0.0);
    }
}

#ifndef RELEARN_CUDA_ENABLED
TEST_F(ElementBaseTest, testCommitUpdates4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    auto deltas_to_add_1 = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    element_base.add_to_delta(deltas_to_add_1);
    std::ignore = element_base.commit_updates();

    ASSERT_NEAR(element_base.get_total_additions(), 7.01, eps);
    ASSERT_NEAR(element_base.get_total_deletions(), 0.0, eps);

    auto deltas_to_add_2 = grown_values({ 2.3, 1.8, 0.4, 0.2 });
    element_base.add_to_delta(deltas_to_add_2);
    std::ignore = element_base.commit_updates();

    ASSERT_NEAR(element_base.get_total_additions(), 11.71, eps);
    ASSERT_NEAR(element_base.get_total_deletions(), 0.0, eps);

    auto deltas_to_add_3 = grown_values({ 0.2, -0.4, -2.43, 0.4 });
    element_base.add_to_delta(deltas_to_add_3);
    std::ignore = element_base.commit_updates();

    ASSERT_NEAR(element_base.get_total_additions(), 12.31, eps);
    ASSERT_NEAR(element_base.get_total_deletions(), 2.83, eps);

    auto deltas_to_add_4 = grown_values({ 0.0, 0.0, 0.0, 0.0 });
    element_base.add_to_delta(deltas_to_add_4);
    std::ignore = element_base.commit_updates();

    ASSERT_NEAR(element_base.get_total_additions(), 12.31, eps);
    ASSERT_NEAR(element_base.get_total_deletions(), 2.83, eps);

    auto deltas_to_add_5 = grown_values({ 0.45, 0.22, 1.9, -0.5 });
    element_base.add_to_delta(deltas_to_add_5);
    std::ignore = element_base.commit_updates();

    ASSERT_NEAR(element_base.get_total_additions(), 14.88, eps);
    ASSERT_NEAR(element_base.get_total_deletions(), 3.33, eps);

    const auto grown_elements_1 = element_base.get_grown_elements();
    const auto deltas_1 = element_base.get_deltas();
    const auto vacant_elements_1 = element_base.get_vacant_elements();
    const auto connected_elements_1 = element_base.get_connected_elements();
    const auto retract_ratio_1 = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum_1 = element_base.get_minimum_calcium();

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(2.95, 1.62, 5.88, 1.1);
        constexpr static auto expected_vacant = std::array{ 2U, 1U, 5U, 1U };

        ASSERT_NEAR(grown_elements_1[i], expected_grown[i], eps);
        ASSERT_EQ(deltas_1[i], 0.0);
        ASSERT_EQ(vacant_elements_1[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements_1[i], 0U);
        ASSERT_EQ(retract_ratio_1[i], 0.0);
        ASSERT_EQ(minimum_calcum_1[i], 0.0);
    }
}
#endif

TEST_F(ElementBaseTest, testDisableNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto grown_elements = element_base.get_grown_elements();
    const auto deltas = element_base.get_deltas();
    const auto vacant_elements = element_base.get_vacant_elements();
    const auto connected_elements = element_base.get_connected_elements();
    const auto retract_ratio = element_base.get_vacant_retract_ratio();
    const auto minimum_calcum = element_base.get_minimum_calcium();

    auto deltas_to_add_1 = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    element_base.add_to_delta(deltas_to_add_1);
    std::ignore = element_base.commit_updates();

    auto deltas_to_add_2 = grown_values({ 2.3, 1.8, 0.4, 0.2 });
    element_base.add_to_delta(deltas_to_add_2);
    std::ignore = element_base.commit_updates();

    auto deltas_to_add_3 = grown_values({ 0.2, -0.4, -2.43, 0.4 });
    element_base.add_to_delta(deltas_to_add_3);
    std::ignore = element_base.commit_updates();

    auto deltas_to_add_4 = grown_values({ 0.0, 0.0, 0.0, 0.0 });
    element_base.add_to_delta(deltas_to_add_4);
    std::ignore = element_base.commit_updates();

    auto deltas_to_add_5 = grown_values({ 0.45, 0.22, 1.9, -0.5 });
    element_base.add_to_delta(deltas_to_add_5);
    std::ignore = element_base.commit_updates();

    auto deltas_to_add_6 = grown_values({ 0.45, 0.22, 1.9, -0.5 });
    element_base.add_to_delta(deltas_to_add_6);

    const auto disabled_neuron_ids = std::vector{
        RelearnTypes::number_neurons_type{ 1 }, RelearnTypes::number_neurons_type{ 3 }
    };

    element_base.disable_neurons(disabled_neuron_ids);

    for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < number_neurons_init; i++) {
        constexpr static auto expected_grown = grown_array(2.95, 0.0, 5.88, 0.0);
        constexpr static auto expected_vacant = std::array{ 2U, 0U, 5U, 0U };
        constexpr static auto expected_delta = grown_array(0.45, 0.0, 1.9, 0.0);

        ASSERT_NEAR(grown_elements[i], expected_grown[i], eps);
        ASSERT_NEAR_EPS(deltas[i], expected_delta[i]);
        ASSERT_EQ(vacant_elements[i], expected_vacant[i]);
        ASSERT_EQ(connected_elements[i], 0U);
        ASSERT_NEAR_EPS(retract_ratio[i], 0.0);
        ASSERT_NEAR_EPS(minimum_calcum[i], 0.0);
    }
}

TEST_F(ElementBaseTest, testDisableNeuronsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto element_base = ElementBase{};

    const auto number_neurons_init = RelearnTypes::number_neurons_type{ 4 };

    element_base.init(number_neurons_init);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons_init);
    element_base.set_extra_infos(extra_infos);

    const auto deltas_to_add_1 = grown_values({ 0.0, -4.2, 6.01, 1.0 });
    element_base.add_to_delta(deltas_to_add_1);
    std::ignore = element_base.commit_updates();

    const auto disabled_neuron_ids = std::vector{
        RelearnTypes::number_neurons_type{ 4 }, RelearnTypes::number_neurons_type{ 3 }
    };

    ASSERT_THROW_NO_PRINT(element_base.disable_neurons(disabled_neuron_ids), RelearnException);
}
