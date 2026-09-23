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

#include "BasicTypes.h"

#include "util/BoundingBox.h"
#include "util/Vec3.h"

// The types that place the objects of the simulation in space, i.e., the positions of the neurons and the
// extents of the domains and the octree nodes that hold them.

namespace RelearnTypes {

/** @brief A single coordinate or length in the simulation space, i.e., the scalar the positions and the box extents are built from */
using space_type = real;

/** @brief A point in the simulation space, i.e., the position of an object such as a neuron, or the extent of a box such as a subdomain or an octree node */
using position_type = Vec3<space_type>;

/** @brief A box in the simulation space, given by its minimum and its maximum corner */
using bounding_box_type = BoundingBox<space_type>;

} // namespace RelearnTypes
