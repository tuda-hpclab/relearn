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

#include "cuda/CudaConfig.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "util/RelearnException.h"

#include <memory>

/**
 * @brief Returns the peak GPU memory usage (in bytes) observed since the start of the program.
 */
[[nodiscard]] std::size_t get_gpu_max_memory_used();

/**
 * @brief Fills a device array of uint32 values with the given scalar.
 * @param data   Device pointer to the target array.
 * @param size   Number of elements to fill.
 * @param value  Value to write into every element.
 */
void set_memory_entry(std::uint32_t* data, std::size_t size, std::uint32_t value);

// Statement form of NOT_SUPPORTED, for use inside a function body that isn't purely the stub.
#define CUDA_NOT_SUPPORTED RelearnException::fail("Function is part of CUDA based implementations. CUDA has not been enabled on this system.");
// Statement form for the opposite direction: a CPU-only code path reached while running with CUDA enabled.
#define CPU_NOT_SUPPORTED RelearnException::fail("Function is part of cpu only based implementations but you are running with CUDA");
