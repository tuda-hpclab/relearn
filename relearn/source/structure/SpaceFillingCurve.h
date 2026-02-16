#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cstdint>

/**
 * This class represents a space filling curve in 3D.
 */
class SpaceFillingCurve {
public:
    using coordinates_1d = std::uint64_t;
    using coordinates_3d = Vec3s;

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param _refinement_level The desired refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    constexpr explicit SpaceFillingCurve(const std::uint8_t _refinement_level = 0) {
        set_refinement_level(_refinement_level);
    }

    constexpr virtual ~SpaceFillingCurve() = default;

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] constexpr std::uint8_t get_current_refinement_level() const noexcept {
        return refinement_level;
    }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @excepption Can throw a RelearnException
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr virtual coordinates_3d map_1d_to_3d(const coordinates_1d idx) const = 0;

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param coords The three dimensional index
     * @excepption Can throw a RelearnException
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr virtual coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const = 0;

protected:
    /**
     * @brief Sets the new refinement level
     * @param new_refinement_level The new refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    constexpr void set_refinement_level(const std::uint8_t new_refinement_level) {
        RelearnException::check(new_refinement_level <= Constants::max_lvl_subdomains, "SpaceFillingCurve::set_refinement_level: Refinement level exceeds maximum level");
        refinement_level = new_refinement_level;
    }

private:
    std::uint8_t refinement_level{ 0 };
};
