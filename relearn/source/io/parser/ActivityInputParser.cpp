/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ActivityInputParser.h"

#include "neurons/input/ActivityInput.h"
#include "neurons/input/CombinedActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/FlexibleActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "neurons/input/ScaleActivityInput.h"
#include "neurons/input/StimulationActivityInput.h"
#include "neurons/input/SynapticEquallyWeightedActivityInput.h"
#include "neurons/input/SynapticScalingActivityInput.h"
#include "util/RelearnException.h"

#include <ctpg/ctpg.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <charconv>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using activity_input_type = std::shared_ptr<ActivityInput>;
using list_type = std::vector<activity_input_type>;

constexpr ctpg::nterm<activity_input_type> activity("activity");
constexpr ctpg::nterm<list_type> activity_list("activity list");

[[maybe_unused]] constexpr char double_pattern[] = "[0-9]*(\\.[0-9]*)?";
constexpr ctpg::regex_term<double_pattern> double_literal("double literal");
constexpr ctpg::nterm<RelearnTypes::activity_type> double_number("double number");
constexpr ctpg::nterm<std::size_t> size_t_number("std::size_t number");

#ifndef RELEARN_CUDA_ENABLED
constexpr ctpg::nterm<ActivityInputParseContext::scaling_function_type> scaling_function("scaling function");
#else
constexpr ctpg::nterm<CudaConfig::scaling_function_enum> scaling_function("scaling function");
#endif

namespace kw {
constexpr ctpg::char_term lparen('(');
constexpr ctpg::char_term rparen(')');
constexpr ctpg::char_term comma(',');

// activities
constexpr ctpg::string_term const_("const");
constexpr ctpg::string_term normal("normal");
constexpr ctpg::string_term fastnormal("fastnormal");
constexpr ctpg::string_term combined("combined");
constexpr ctpg::string_term synaptic_equally_weighted("synaptic_equally_weighted");
constexpr ctpg::string_term synaptic_scaling("synaptic_scaling");
constexpr ctpg::string_term scale("scale");
constexpr ctpg::string_term flexible("flexible");
constexpr ctpg::string_term stimulated("stimulated");

// variables available to be used in `(fast-)normal`
constexpr ctpg::string_term background_base("background_base");
constexpr ctpg::string_term background_mean("background_mean");
constexpr ctpg::string_term background_stddev("background_stddev");

// variables available to be used in `scale`
constexpr ctpg::string_term linear("linear");
constexpr ctpg::string_term logarithmic("logarithmic");
constexpr ctpg::string_term hyperbolic_tangent("hyperbolic_tangent");
} // namespace kw

/**
 * @brief Parse an arithmetic value of type T from a string_view.
 *
 * @exception Throws a RelearnException when trying to parse an integral type, but the input contains a '.'
 * @exception Throws a RelearnException if the call to from_chars fails
 *
 * @tparam T target arithmetic type
 * @param arithmetic_value input string
 * @return the parsed value
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr T parse_arithmetic(const std::string_view arithmetic_value) {
    if constexpr (!std::is_floating_point_v<T>) {
        RelearnException::check(arithmetic_value.find('.') == std::string_view::npos,
                                "activity_input_parser::parse_arithmetic: trying to parse '{}' as type T, which is not a floating point type", arithmetic_value);
    }

    auto value = T(0);
    if (const auto res = std::from_chars(arithmetic_value.data(), arithmetic_value.data() + arithmetic_value.size(), value); res.ec != std::errc{}) {
        const auto distance = static_cast<std::size_t>(std::distance(arithmetic_value.data(), res.ptr));

        RelearnException::fail("activity_input_parser::parse_arithmetic: error while trying to convert '{}' to T. The input does not match T starting from '{}'",
                               arithmetic_value, arithmetic_value.substr(distance));
    }

    return value;
}

// missing: flexible

/**
 * @brief Parser definition that can parse a string describing input activity.
 *
 */
constexpr ctpg::parser activity_input_parser(
    activity,
    terms(
        kw::lparen,
        kw::rparen,
        kw::comma,
        double_literal,
        kw::const_,
        kw::normal,
        kw::fastnormal,
        kw::combined,
        kw::synaptic_equally_weighted,
        kw::synaptic_scaling,
        kw::scale,
        kw::flexible,
        kw::stimulated,
        kw::background_base,
        kw::background_mean,
        kw::background_stddev,
        kw::linear,
        kw::logarithmic,
        kw::hyperbolic_tangent),
    nterms(
        activity,
        activity_list,
        double_number,
        size_t_number,
        scaling_function),
    rules(
        activity_list(activity_list, kw::comma, activity) >= ctpg::ftors::push_back<1, 3>{},
        activity_list(activity) >= ctpg::ftors::construct<list_type>{},

        double_number(double_literal) >= parse_arithmetic<RelearnTypes::activity_type>,
        size_t_number(double_literal) >= parse_arithmetic<std::size_t>,

        double_number(kw::background_base) >>= [](const ActivityInputParseContext& context, const auto& /*background_base*/) { return context.background_base; },
        double_number(kw::background_mean) >>= [](const ActivityInputParseContext& context, const auto& /*background_mean*/) { return context.background_mean; },
        double_number(kw::background_stddev) >>= [](const ActivityInputParseContext& context, const auto& /*background_stddev*/) { return context.background_stddev; },

        scaling_function(kw::linear) >>= [](const ActivityInputParseContext& context, const auto& /*linear*/) { return context.linear; },
        scaling_function(kw::logarithmic) >>= [](const ActivityInputParseContext& context, const auto& /*logarithmic*/) { return context.logarithmic; },
        scaling_function(kw::hyperbolic_tangent) >>= [](const ActivityInputParseContext& context, const auto& /*hyperbolic_tangent*/) { return context.hyperbolic_tangent; },

        activity(kw::const_, kw::lparen, double_number, kw::rparen) >=
            [](const auto& /*const*/,
               const auto& /*(*/,
               const RelearnTypes::activity_type constant,
               const auto& /*)*/) -> activity_input_type {
            return std::make_shared<ConstantActivityInput>(mpiPP::MPIInfo::get_number_ranks(), constant);
        },

        activity(kw::normal, kw::lparen, double_number, kw::comma, double_number, kw::rparen) >=
            [](const auto& /*normal*/,
               const auto& /*(*/,
               const RelearnTypes::activity_type mean,
               const auto& /*,*/,
               const RelearnTypes::activity_type stddev,
               const auto& /*)*/) -> activity_input_type {
            return std::make_shared<NormalActivityInput>(mpiPP::MPIInfo::get_number_ranks(), mean, stddev);
        },

        activity(kw::fastnormal, kw::lparen, double_number, kw::comma, double_number, kw::comma, size_t_number, kw::rparen) >=
            [](const auto& /*fastnormal*/,
               const auto& /*(*/,
               const RelearnTypes::activity_type mean,
               const auto& /*,*/,
               const RelearnTypes::activity_type stddev,
               const auto& /*,*/,
               const std::size_t multiplier,
               const auto& /*)*/) -> activity_input_type {
            return std::make_shared<FastNormalActivityInput>(mpiPP::MPIInfo::get_number_ranks(), mean, stddev, multiplier);
        },

        activity(kw::combined, kw::lparen, activity_list, kw::rparen) >=
            [](const auto& /*combined*/,
               const auto& /*(*/,
               list_type&& activities,
               const auto& /*)*/) -> activity_input_type {
            return std::make_shared<CombinedActivityInput>(mpiPP::MPIInfo::get_number_ranks(), std::move(activities));
        },

        activity(kw::synaptic_equally_weighted) >>=
        [](const ActivityInputParseContext& context,
           const auto& /*synaptic_equally_weighted*/) -> activity_input_type {
            auto parsed_activity = std::make_shared<SynapticEquallyWeightedActivityInput>(mpiPP::MPIInfo::get_number_ranks(), context.communicator, context.synapse_conductance);
            return parsed_activity;
        },

        activity(kw::synaptic_scaling, kw::lparen, double_number, kw::rparen) >>=
        [](const ActivityInputParseContext& context,
           const auto& /*synaptic_scaling*/,
           const auto& /*(*/,
           const RelearnTypes::activity_type constant,
           const auto& /*)*/) -> activity_input_type {
            auto parsed_activity = std::make_shared<SynapticScalingActivityInput>(mpiPP::MPIInfo::get_number_ranks(), context.communicator, constant);
            return parsed_activity;
        },
#ifdef RELEARN_CUDA_ENABLED
        activity(kw::scale, kw::lparen, activity, kw::comma, scaling_function, kw::rparen)
        >>=
        [](const ActivityInputParseContext& context,
           const auto& /*scale*/,
           const auto& /*(*/,
           const activity_input_type& inner_activity,
           const auto& /*,*/,
           const CudaConfig::scaling_function_enum scale_function,
           const auto& /*)*/) {
            return std::make_shared<ScaleActivityInput>(mpiPP::MPIInfo::get_number_ranks(), inner_activity, scale_function, context.input_scale);
        },
#else

        activity(kw::scale, kw::lparen, activity, kw::comma, scaling_function, kw::rparen)
            >=
            [](const auto& /*scale*/,
               const auto& /*(*/,
               const activity_input_type& inner_activity,
               const auto& /*,*/,
               const ActivityInputParseContext::scaling_function_type& scale_function,
               const auto& /*)*/) {
                return std::make_shared<ScaleActivityInput>(mpiPP::MPIInfo::get_number_ranks(), inner_activity, scale_function);
            },
#endif

        activity(kw::flexible) >>=
        [](const ActivityInputParseContext& Context,
           const auto& /*flexible*/) -> activity_input_type {
            auto loaded = Context.flexible_background();
            auto inputs = loaded.inputs;
            auto fun = std::move(loaded.choice_function);
            return std::make_shared<FlexibleActivityInput>(mpiPP::MPIInfo::get_number_ranks(), inputs, std::move(fun));
        },

        activity(kw::stimulated) >>=
        [](const ActivityInputParseContext& Context,
           const auto& /*stimulated*/) -> activity_input_type {
            auto fun = Context.load_stimulus();
            return std::make_shared<StimulationActivityInput>(mpiPP::MPIInfo::get_number_ranks(), std::move(fun));
        }

        ));
} // namespace

std::shared_ptr<ActivityInput> parse_activity(const std::string& input, const ActivityInputParseContext& context) {
    if (const auto res = activity_input_parser.context_parse(context, ctpg::buffers::string_buffer{ std::string{ input } }, std::cerr); res.has_value()) {
        return res.value();
    }

    RelearnException::fail("parse_activity: the activity input specification '{}' does not create a valid input activity", input);
}
