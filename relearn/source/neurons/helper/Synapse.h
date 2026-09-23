#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <compare>
#include <utility>

/**
 * This class encapsulates what a synapse is made of in this program,
 *  i.e., a target, a source, and a weight.
 * @tparam Target The type of the target
 * @tparam Source The type of the source
 * @tparam Weight The type of the weight
 */
template <typename Target, typename Source, typename Weight>
class Synapse {
public:
    /**
     * @brief Constructs a new synapse with the given parameter
     * @param _target The target of the synapse
     * @param _source The source of the synapse
     * @param _weight The weight of the synapse
     */
    Synapse(const Target _target, const Source _source, const Weight _weight)
        : target(_target)
        , source(_source)
        , weight(_weight) { }

    /**
     * @brief Returns the target of the synapse
     * @return The target of the synapse
     */
    [[nodiscard]] const Target& get_target() const noexcept {
        return target;
    }

    /**
     * @brief Returns the source of the synapse
     * @return The source of the synapse
     */
    [[nodiscard]] const Source& get_source() const noexcept {
        return source;
    }

    /**
     * @brief Returns the weight of the synapse
     * @return The weight of the synapse
     */
    [[nodiscard]] const Weight& get_weight() const noexcept {
        return weight;
    }

    /**
     * @brief Provides a standard ordering on this class based on its components.
     */
    [[nodiscard]] friend constexpr auto operator<=>(const Synapse& first, const Synapse& second) noexcept = default;

    template <std::size_t Index>
    [[nodiscard]] auto& get() & {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return weight;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] const auto& get() const& {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return weight;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(target);
        }
        if constexpr (Index == 1) {
            return std::move(source);
        }
        if constexpr (Index == 2) {
            return std::move(weight);
        }
    }

private:
    Target target{};
    Source source{};
    Weight weight{};
};

namespace std {
// NOLINTBEGIN(cert-dcl58-cpp) - specializing tuple_size/tuple_element for a program-defined type is standard-sanctioned
template <typename Target, typename Source, typename Weight>
struct tuple_size<::Synapse<Target, Source, Weight>> {
    static constexpr size_t value = 3;
};

template <typename Target, typename Source, typename Weight>
struct tuple_element<0, ::Synapse<Target, Source, Weight>> {
    using type = Target;
};

template <typename Target, typename Source, typename Weight>
struct tuple_element<1, ::Synapse<Target, Source, Weight>> {
    using type = Source;
};

template <typename Target, typename Source, typename Weight>
struct tuple_element<2, ::Synapse<Target, Source, Weight>> {
    using type = Weight;
};
// NOLINTEND(cert-dcl58-cpp)

} // namespace std
