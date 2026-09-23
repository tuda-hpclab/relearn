/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "io/parser/ActivityInputParser.h"
#include "neurons/firing/FiredStatusApproximator.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/input/CombinedActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "neurons/input/ScaleActivityInput.h"
#include "neurons/input/SynapticEquallyWeightedActivityInput.h"
#include "types/BasicTypes.h"
#include "types/StimulusTypes.h"
#include "util/RelearnException.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/random/random_factory.h"

#include <cpp-utility/Cast.hpp>

#include <ctpg/ctpg.hpp>

#include <fmt/ranges.h>
#include <fmt/std.h>

#include <gtest/gtest.h>

#include <range/v3/algorithm/equal.hpp>

#include <cmath>
#include <memory>
#include <random>
#include <source_location>

namespace {

} // namespace

#ifndef RELEARN_CUDA_ENABLED
class ActivityInputParser : public RelearnTest {
public:
    void SetUp() override {
        number_neurons = RandomFactory::get_random_integer(size_t{ 1 }, size_t{ 10 }, mt);
        network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

        Context.background_base = 1.0;
        Context.background_mean = 1.0;
        Context.background_stddev = 1.0;

        extra_info = NeuronsExtraInfoFactory::construct_extra_info();
        extra_info->init(number_neurons);
        Context.communicator = std::make_shared<FiredStatusApproximator>(mpiPP::MPIRank::root_rank(), 4);
        Context.communicator->init(number_neurons);
        Context.communicator->set_network_graph(network_graph);
        Context.communicator->set_extra_infos(extra_info);
        auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
        fired_status_recorder->init(number_neurons);
        Context.communicator->set_fired_status_recorder(fired_status_recorder);

        using activity_type = ActivityInputParseContext::activity_type;
        Context.linear = [](const activity_type val) { return utility::as<activity_type>(1.2) * val; };
        Context.logarithmic = [](const activity_type val) { return activity_type{ 2 } * std::log10(utility::as<activity_type>(1.2) * val + activity_type{ 1 }); };
        Context.hyperbolic_tangent = [](const activity_type val) { return activity_type{ 2 } * std::tanh(utility::as<activity_type>(1.2) * val); };

        Context.load_stimulus = []() { return [](const RelearnTypes::step_type) { return RelearnTypes::stimuli_function_type::result_type{}; }; };
    }

    void testActivityUpdate(ActivityInput& parsed_activity, ActivityInput& expected_activity) {
        const auto parsed_neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
        parsed_neurons_extra_info->init(number_neurons);
        parsed_activity.init(number_neurons);
        parsed_activity.set_extra_infos(extra_info);
        parsed_activity.update_input(1);

        expected_activity.init(number_neurons);
        expected_activity.set_extra_infos(extra_info);
        expected_activity.update_input(1);

        EXPECT_TRUE(ranges::equal(expected_activity.get_input(), parsed_activity.get_input()));
    }

protected:
    RelearnTypes::number_neurons_type number_neurons;
    std::shared_ptr<NetworkGraph> network_graph;
    ActivityInputParseContext Context = ActivityInputParseContext{};
    std::shared_ptr<NeuronsExtraInfo> extra_info;
};

TEST_F(ActivityInputParser, parseUnknown) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("asdf(1.5, \"Hello World\")", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseRandomChars) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("cn02349un)(#$%)Hd", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters0) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal()", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters1) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal(1.5)", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters3) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal(1.5, 1, 100)", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseConst) {
    const auto parsed_activity = parse_activity("const(1.5)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ConstantActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ConstantActivityInput{ 1, 1.5 };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseConstUsingVar) {
    const auto parsed_activity = parse_activity("const(background_base)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ConstantActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ConstantActivityInput{ 1, utility::cast<ActivityInput::activity_type>(Context.background_base) };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseNormal) {
    const auto parsed_activity = parse_activity("normal(1.5, 0.5)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<NormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = NormalActivityInput{ 1, 1.5, 0.5 };
}

TEST_F(ActivityInputParser, parseNormalUsingVars) {
    const auto parsed_activity = parse_activity("normal(background_mean, background_stddev)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<NormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = NormalActivityInput{ 1, utility::cast<ActivityInput::activity_type>(Context.background_mean), utility::cast<ActivityInput::activity_type>(Context.background_stddev) };
}

TEST_F(ActivityInputParser, parseFastNormal) {
    const auto parsed_activity = parse_activity("fastnormal(1.5, 5, 10)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<FastNormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = FastNormalActivityInput{ 1, 1.5, 5, 10 };
}

TEST_F(ActivityInputParser, parseCombinedSingle) {
    const auto parsed_activity = parse_activity("combined(const(10.0))", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<CombinedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = CombinedActivityInput{ 1, { std::make_shared<ConstantActivityInput>(1, ConstantActivityInput::activity_type{ 10 }) } };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseCombined) {
    const auto parsed_activity = parse_activity("combined(const(10.0), synaptic_equally_weighted)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<CombinedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    const auto synaptic_input = std::make_shared<SynapticEquallyWeightedActivityInput>(1, Context.communicator, Context.synapse_conductance);
    synaptic_input->set_extra_infos(extra_info);
    synaptic_input->set_network_graph(network_graph);
    auto expected_activity = CombinedActivityInput{ 1, {
                                                           std::make_shared<ConstantActivityInput>(1, ConstantActivityInput::activity_type{ 10 }),
                                                           synaptic_input,
                                                       } };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseSynaptic) {
    const auto parsed_activity = parse_activity("synaptic_equally_weighted", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<SynapticEquallyWeightedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = SynapticEquallyWeightedActivityInput{ 1, Context.communicator, Context.synapse_conductance };
    expected_activity.set_extra_infos(extra_info);
    expected_activity.set_network_graph(network_graph);

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseScaleSynaptic) {
    auto parsed_activity = parse_activity("scale(synaptic_equally_weighted, linear)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    {
        const auto synaptic_input = std::make_shared<SynapticEquallyWeightedActivityInput>(1, Context.communicator, Context.synapse_conductance);
        synaptic_input->set_extra_infos(extra_info);
        synaptic_input->set_network_graph(network_graph);
        auto expected_activity = ScaleActivityInput{
            1,
            synaptic_input,
            Context.linear,
        };

        testActivityUpdate(*parsed_activity, expected_activity);

        parsed_activity = parse_activity("scale(synaptic_equally_weighted, logarithmic)", Context);
        ASSERT_TRUE(parsed_activity);
        ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
        parsed_activity->set_network_graph(network_graph);
    }
    {
        const auto synaptic_input = std::make_shared<SynapticEquallyWeightedActivityInput>(1, Context.communicator, Context.synapse_conductance);
        synaptic_input->set_extra_infos(extra_info);
        synaptic_input->set_network_graph(network_graph);
        auto expected_activity = ScaleActivityInput{
            1,
            synaptic_input,
            Context.logarithmic,
        };

        testActivityUpdate(*parsed_activity, expected_activity);

        parsed_activity = parse_activity("scale(synaptic_equally_weighted, hyperbolic_tangent)", Context);
        ASSERT_TRUE(parsed_activity);
        ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
        parsed_activity->set_network_graph(network_graph);
    }
    {
        const auto synaptic_input = std::make_shared<SynapticEquallyWeightedActivityInput>(1, Context.communicator, Context.synapse_conductance);
        synaptic_input->set_extra_infos(extra_info);
        synaptic_input->set_network_graph(network_graph);
        auto expected_activity = ScaleActivityInput{
            1,
            synaptic_input,
            Context.hyperbolic_tangent,
        };

        testActivityUpdate(*parsed_activity, expected_activity);
    }
}

TEST_F(ActivityInputParser, parseScaleNormal) {
    auto parsed_activity = parse_activity("scale(normal(background_mean, background_stddev), linear)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ScaleActivityInput{
        1,
        std::make_shared<NormalActivityInput>(1, Context.background_mean, Context.background_stddev),
        Context.linear,
    };

    parsed_activity = parse_activity("scale(normal(background_mean, background_stddev), logarithmic)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    expected_activity = ScaleActivityInput{
        1,
        std::make_shared<NormalActivityInput>(1, Context.background_mean, Context.background_stddev),
        Context.logarithmic,
    };

    parsed_activity = parse_activity("scale(normal(background_mean, background_stddev), hyperbolic_tangent)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    expected_activity = ScaleActivityInput{
        1,
        std::make_shared<NormalActivityInput>(1, Context.background_mean, Context.background_stddev),
        Context.hyperbolic_tangent,
    };
}

TEST_F(ActivityInputParser, parseScaleConst) {
    auto parsed_activity = parse_activity("scale(const(0.0), linear)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ScaleActivityInput{
        1,
        std::make_shared<ConstantActivityInput>(1, ConstantActivityInput::activity_type{ 0 }),
        Context.linear,
    };

    testActivityUpdate(*parsed_activity, expected_activity);

    parsed_activity = parse_activity("scale(const(0.0), logarithmic)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    expected_activity = ScaleActivityInput{
        1,
        std::make_shared<ConstantActivityInput>(1, ConstantActivityInput::activity_type{ 0 }),
        Context.logarithmic,
    };

    testActivityUpdate(*parsed_activity, expected_activity);

    parsed_activity = parse_activity("scale(const(0.0), hyperbolic_tangent)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ScaleActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    expected_activity = ScaleActivityInput{
        1,
        std::make_shared<ConstantActivityInput>(1, ConstantActivityInput::activity_type{ 0 }),
        Context.hyperbolic_tangent,
    };

    testActivityUpdate(*parsed_activity, expected_activity);
}
#else
class ActivityInputParser : public RelearnTest {
public:
    void SetUp() override {
        number_neurons = RandomFactory::get_random_integer(size_t{ 1 }, size_t{ 10 }, mt);
        network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

        Context.background_base = 1.0;
        Context.background_mean = 1.0;
        Context.background_stddev = 1.0;

        extra_info = NeuronsExtraInfoFactory::construct_extra_info();
        extra_info->init(number_neurons);
        Context.communicator = std::make_shared<FiredStatusApproximator>(mpiPP::MPIRank::root_rank(), 4);
        Context.communicator->init(number_neurons);
        Context.communicator->set_network_graph(network_graph);
        Context.communicator->set_extra_infos(extra_info);
        auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
        fired_status_recorder->init(number_neurons);
        Context.communicator->set_fired_status_recorder(fired_status_recorder);

        Context.linear = CudaConfig::LINEAR;
        Context.logarithmic = CudaConfig::LOGARITHMIC;
        Context.hyperbolic_tangent = CudaConfig::HYPERBOLIC_TANGENT;

        Context.load_stimulus = []() { return [](const RelearnTypes::step_type) { return RelearnTypes::stimuli_function_type::result_type{}; }; };
    }

    void testActivityUpdate(ActivityInput& parsed_activity, ActivityInput& expected_activity, const std::source_location& test_loc = std::source_location::current()) {
        const auto parsed_neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
        parsed_neurons_extra_info->init(number_neurons);
        parsed_activity.init(number_neurons);
        parsed_activity.set_extra_infos(extra_info);
        parsed_activity.update_input(1);

        expected_activity.init(number_neurons);
        expected_activity.set_extra_infos(extra_info);
        expected_activity.update_input(1);

        EXPECT_TRUE(ranges::equal(expected_activity.get_input(), parsed_activity.get_input()))
            << fmt::format("Test originated at {}\nExpected: {}\n Got: {}", test_loc, expected_activity.get_input(), parsed_activity.get_input());
    }

protected:
    RelearnTypes::number_neurons_type number_neurons;
    std::shared_ptr<NetworkGraph> network_graph;
    ActivityInputParseContext Context = ActivityInputParseContext{};
    std::shared_ptr<NeuronsExtraInfo> extra_info;
};

TEST_F(ActivityInputParser, parseUnknown) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("asdf(1.5, \"Hello World\")", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseRandomChars) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("cn02349un)(#$%)Hd", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters0) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal()", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters1) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal(1.5)", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseWrongNumberOfParameters3) {
    ASSERT_THROW_NO_PRINT(std::ignore = parse_activity("normal(1.5, 1, 100)", Context), RelearnException);
}

TEST_F(ActivityInputParser, parseConst) {
    const auto parsed_activity = parse_activity("const(1.5)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ConstantActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ConstantActivityInput{ 1, 1.5 };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseConstUsingVar) {
    const auto parsed_activity = parse_activity("const(background_base)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<ConstantActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = ConstantActivityInput{ 1, utility::cast<ActivityInput::activity_type>(Context.background_base) };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseNormal) {
    const auto parsed_activity = parse_activity("normal(1.5, 0.5)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<NormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = NormalActivityInput{ 1, 1.5, 0.5 };
}

TEST_F(ActivityInputParser, parseNormalUsingVars) {
    const auto parsed_activity = parse_activity("normal(background_mean, background_stddev)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<NormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = NormalActivityInput{ 1, utility::cast<ActivityInput::activity_type>(Context.background_mean), utility::cast<ActivityInput::activity_type>(Context.background_stddev) };
}

#ifndef RELEARN_CUDA_ENABLED
TEST_F(ActivityInputParser, parseFastNormal) {
    const auto parsed_activity = parse_activity("fastnormal(1.5, 5, 10)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<FastNormalActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = FastNormalActivityInput{ 1, 1.5, 5, 10 };
}
#endif

TEST_F(ActivityInputParser, parseCombinedSingle) {
    const auto parsed_activity = parse_activity("combined(const(10.0))", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<CombinedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = CombinedActivityInput{ 1, { std::make_shared<ConstantActivityInput>(1, 10.0) } };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseCombined) {
    const auto parsed_activity = parse_activity("combined(const(10.0), synaptic_equally_weighted)", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<CombinedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    const auto synaptic_input = std::make_shared<SynapticEquallyWeightedActivityInput>(1, Context.communicator, Context.synapse_conductance);
    synaptic_input->set_extra_infos(extra_info);
    synaptic_input->set_network_graph(network_graph);
    auto expected_activity = CombinedActivityInput{ 1, {
                                                           std::make_shared<ConstantActivityInput>(1, 10.0),
                                                           synaptic_input,
                                                       } };

    testActivityUpdate(*parsed_activity, expected_activity);
}

TEST_F(ActivityInputParser, parseSynaptic) {
    const auto parsed_activity = parse_activity("synaptic_equally_weighted", Context);
    ASSERT_TRUE(parsed_activity);
    ASSERT_NE(dynamic_cast<SynapticEquallyWeightedActivityInput*>(parsed_activity.get()), nullptr);
    parsed_activity->set_network_graph(network_graph);

    auto expected_activity = SynapticEquallyWeightedActivityInput{ 1, Context.communicator, Context.synapse_conductance };
    expected_activity.set_extra_infos(extra_info);
    expected_activity.set_network_graph(network_graph);

    testActivityUpdate(*parsed_activity, expected_activity);
}

#endif