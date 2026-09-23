#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/CudaTypes.h"
#include "types/BasicTypes.h"

// The conversions between the domain types of the simulation and their device-side counterparts in
// CudaTypes. The position is taken as a template parameter and converted with RelearnTypes::as, so that
// this header needs neither the definition of Vec3 nor the casts of cpp-utility and stays includable
// from everywhere, the device code included.

namespace CudaTypes {

/**
 * @brief Converts a position of the simulation into the type the device code expects, i.e., spells every
 *      coordinate as CudaTypes::cuda_space. The alias of the host and the one of the device are configured
 *      independently, so the conversion widens, narrows, or does nothing, depending on the configuration.
 * @tparam Position The type of the position, i.e., a Vec3 of whichever scalar the host places its objects
 *      with, which is deduced
 * @param position The position in the simulation space
 * @return The position as the device code represents it
 */
template <typename Position>
[[nodiscard]] constexpr cuda_real3 to_cuda_real3(const Position& position) noexcept {
    return cuda_real3{ RelearnTypes::as<cuda_space>(position.get_x()), RelearnTypes::as<cuda_space>(position.get_y()), RelearnTypes::as<cuda_space>(position.get_z()) };
}

} // namespace CudaTypes
