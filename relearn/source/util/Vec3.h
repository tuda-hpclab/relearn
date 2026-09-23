#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <cpp-utility/data-structure/Vec3.hpp>

#include <cstddef>

// FIXME: remove the using decl in favor of spelling the namespace
// NOLINTNEXTLINE(google-global-names-in-headers)
using utility::Vec3;

using Vec3f = utility::Vec3<float>;
using Vec3s = utility::Vec3<std::size_t>;
using Vec3u = utility::Vec3<unsigned int>;
