/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "StreamWrapper.h"

#include "cuda/util/Util.cuh"

#include <cuda.h>

struct StreamWrapper::Impl {
    cudaStream_t stream;
};

StreamWrapper::StreamWrapper()
    : impl(new Impl)
    , owns_stream(true) {
    CUDA_CHECK(cudaStreamCreate(&impl->stream));
}
void* StreamWrapper::native_handle() const noexcept {
    return &(impl->stream);
}

StreamWrapper::~StreamWrapper() {
    // Only destroy streams this instance actually created -- default_stream() wraps the
    // null/default stream (stream 0) without owning it, and cudaStreamDestroy() on that handle
    // is invalid (cudaErrorInvalidResourceHandle), silently left in the CUDA "last error" state
    // for some unrelated later call to trip over.
    if (owns_stream) {
        cudaStreamDestroy(*reinterpret_cast<cudaStream_t*>(&(impl->stream)));
    }
    delete impl;
}

StreamWrapper::StreamWrapper(void* stream_ptr)
    : impl(new Impl)
    , owns_stream(false) {
    impl->stream = static_cast<cudaStream_t>(stream_ptr);
}

const cudaStream_t& get_cuda_stream_from_wrapper(const StreamWrapper& wrapper) {
    return *reinterpret_cast<cudaStream_t*>(wrapper.native_handle());
}