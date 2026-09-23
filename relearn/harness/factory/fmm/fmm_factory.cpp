/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "fmm_factory.h"

#include "Config.h"

#include "util/Vec3.h"

#include "factory/random/random_factory.h"

#include <random>

Vec3u FMMFactory::get_random_multi_index(std::mt19937& mt) {
    const auto x = RandomFactory::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);
    const auto y = RandomFactory::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);
    const auto z = RandomFactory::get_random_integer<Vec3u::value_type>(0, Constants::p, mt);

    return Vec3u{ x, y, z };
}
