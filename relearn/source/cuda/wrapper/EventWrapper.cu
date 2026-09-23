/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "EventWrapper.h"
#include "StreamWrapper.cuh"

#include <cuda.h>

#include <memory>

struct EventWrapper::Impl {
    cudaEvent_t event;
};

EventWrapper::EventWrapper(const std::shared_ptr<StreamWrapper>& _stream)
    : impl(std::make_unique<Impl>())
    , stream(_stream) {
    cudaEventCreate(&impl->event);
}

EventWrapper::~EventWrapper() {
    if (impl) {
        cudaEventDestroy(impl->event);
    }
}

EventWrapper::EventWrapper(EventWrapper&&) noexcept = default;
EventWrapper& EventWrapper::operator=(EventWrapper&&) noexcept = default;

void* EventWrapper::native_handle() const noexcept {
    return &(impl->event);
}

void EventWrapper::wait_for_event(const std::shared_ptr<StreamWrapper>& stream_that_waits) const {
    if (impl == nullptr) {
        return;
    }
    cudaStreamWaitEvent(get_cuda_stream_from_wrapper(*stream_that_waits), impl->event);
}

const std::shared_ptr<StreamWrapper>& EventWrapper::get_stream() const {
    return stream;
}

void record_event(EventWrapper& event_wrapper) {
    cudaEventRecord(*static_cast<cudaEvent_t*>(event_wrapper.native_handle()), get_cuda_stream_from_wrapper(*event_wrapper.get_stream()));
}
