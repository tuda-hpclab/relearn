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

#include <memory>

/**
 * Owns a single CUDA stream. Non-copyable and non-movable; always used through shared_ptr.
 * All asynchronous CUDA operations in the simulation are tied to a StreamWrapper.
 */
class StreamWrapper {
public:
    StreamWrapper();
    ~StreamWrapper();

    StreamWrapper(const StreamWrapper&) = delete;
    StreamWrapper(StreamWrapper&&) = delete;
    StreamWrapper& operator=(const StreamWrapper&) = delete;
    StreamWrapper& operator=(StreamWrapper&&) = delete;

    /**
     * @brief Returns the underlying cudaStream_t as a void pointer.
     */
    [[nodiscard]] void* native_handle() const noexcept;

    /**
     * @brief Returns a shared_ptr wrapping the CUDA default stream (stream 0).
     */
    [[nodiscard]] static std::shared_ptr<StreamWrapper> default_stream() {
        return std::shared_ptr<StreamWrapper>(new StreamWrapper(nullptr));
    }

private:
#ifdef RELEARN_CUDA_ENABLED
    struct Impl;
    Impl* impl{};
    bool owns_stream{ true };

#endif

    explicit StreamWrapper(void* stream_ptr);
};
