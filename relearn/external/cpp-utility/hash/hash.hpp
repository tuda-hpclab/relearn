#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <functional>

namespace utility {

template <typename Type>
struct hash {
    [[nodiscard]] std::size_t operator()(const Type& value) const {
        const auto standard_hash = std::hash<Type>{};
        return standard_hash(value);
    }
};

} // namespace utility
