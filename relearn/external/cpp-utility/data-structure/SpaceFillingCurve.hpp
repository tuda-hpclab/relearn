#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Exception.hpp"
#include "cpp-utility/data-structure/Vec3.hpp"

#include <cstdint>
#include <limits>

namespace utility {

/**
 * @brief Base class for space filling curves that map between a 3D coordinate triple and a single
 *      coordinates_1d index by interleaving 3 bits per refinement level (one bit per axis).
 *      A curve with refinement level L only ever produces indices that fit into 3 * L bits and coordinates
 *      that fit into L bits per axis. Inputs outside that domain are handled according to the concrete curve:
 *      an implementation may reject them or map only their representable bits. Callers that require one
 *      particular behavior must use the contract of the concrete curve.
 *
 *      Copying and moving is only available to derived classes, so that a SpaceFillingCurve can never be
 *      copied or moved through a reference or pointer to this base class, which would otherwise slice
 *      away the derived part of the object.
 */
class SpaceFillingCurve {
public:
    using coordinates_1d = std::uint64_t;
    using coordinates_3d = Vec3<coordinates_1d>;

    /**
     * @brief The largest refinement level for which every coordinates_1d value produced by a mapping still
     *      fits into coordinates_1d, i.e., floor(number of bits of coordinates_1d / 3)
     */
    static constexpr std::uint8_t max_refinement_level = static_cast<std::uint8_t>(std::numeric_limits<coordinates_1d>::digits / 3);

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param new_refinement_level The desired refinement level
     * @exception Throws an Exception if new_refinement_level > max_refinement_level
     */
    constexpr explicit SpaceFillingCurve(const std::uint8_t new_refinement_level = 0) {
        set_refinement_level(new_refinement_level);
    }

    // GCC 11 rejects "= default" here ("used before its definition") for a constexpr virtual destructor
    // used to initialize a constexpr variable of a derived type; an explicit empty body avoids that bug.
    constexpr virtual ~SpaceFillingCurve() { }

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] constexpr std::uint8_t get_current_refinement_level() const noexcept {
        return refinement_level;
    }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain, at the current refinement level
     * @param idx The one dimensional index
     * @exception Implementations may throw an Exception if idx is outside the domain of the current refinement level
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr virtual coordinates_3d map_1d_to_3d(const coordinates_1d idx) const = 0;

    /**
     * @brief Maps a three dimensional index into the one dimensional domain, at the current refinement level
     * @param coords The three dimensional index
     * @exception Implementations may throw an Exception if coords is outside the domain of the current refinement level
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr virtual coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const = 0;

protected:
    constexpr SpaceFillingCurve(const SpaceFillingCurve&) = default;
    constexpr SpaceFillingCurve& operator=(const SpaceFillingCurve&) = default;
    constexpr SpaceFillingCurve(SpaceFillingCurve&&) noexcept = default;
    constexpr SpaceFillingCurve& operator=(SpaceFillingCurve&&) noexcept = default;

    /**
     * @brief Sets the new refinement level
     * @param new_refinement_level The new refinement level
     * @exception Throws an Exception if new_refinement_level > max_refinement_level
     */
    constexpr void set_refinement_level(const std::uint8_t new_refinement_level) {
        Exception::check(new_refinement_level <= max_refinement_level,
                         "SpaceFillingCurve::set_refinement_level: Refinement level {} exceeds the maximum level {}",
                         new_refinement_level, max_refinement_level);
        refinement_level = new_refinement_level;
    }

private:
    std::uint8_t refinement_level{ 0 };
};

} // namespace utility
