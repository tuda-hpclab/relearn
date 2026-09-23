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

#include "CudaConfig.h"

#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>

using gpu_map = cuco::static_map<CudaConfig::number_neurons_type, std::size_t,
                                 cuco::extent<std::size_t>,
                                 cuda::thread_scope_device,
                                 cuda::std::equal_to<CudaConfig::number_neurons_type>,
                                 cuco::linear_probing<1, cuco::default_hash_function<CudaConfig::number_neurons_type>>>;
using gpu_map_ref_type = decltype(std::declval<gpu_map&>().ref(cuco::find));

using set_type = cuco::static_set<std::uint64_t,
                                  cuco::extent<std::size_t>,
                                  cuda::thread_scope_device,
                                  cuda::std::equal_to<std::uint64_t>,
                                  cuco::double_hashing<1, cuco::default_hash_function<std::uint64_t>>>;
using set_ref_type = decltype(std::declval<set_type&>().ref(cuco::contains));