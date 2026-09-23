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

#include "cuda/wrapper/StreamWrapper.h"

#include <memory>

/**
 * RAII wrapper around a cudaEvent_t for cross-stream synchronization.
 * An EventWrapper records a point of completion on one stream and lets other
 * streams wait on it without a full device synchronization.
 */
class EventWrapper {
public:
    /**
     * @brief Creates a CUDA event associated with the given stream.
     * @param _stream Stream to associate the event with.
     */
    explicit EventWrapper(const std::shared_ptr<StreamWrapper>& _stream);
    ~EventWrapper();

    EventWrapper(const EventWrapper&) = delete;
    EventWrapper& operator=(const EventWrapper&) = delete;

    EventWrapper(EventWrapper&&) noexcept;
    EventWrapper& operator=(EventWrapper&&) noexcept;

    /**
     * @brief Returns the underlying cudaEvent_t as a void pointer.
     */
    [[nodiscard]] void* native_handle() const noexcept;

    /**
     * @brief Makes the given stream wait until this event has been recorded.
     * @param stream_that_waits The stream that should block until this event fires.
     */
    void wait_for_event(const std::shared_ptr<StreamWrapper>& stream_that_waits) const;

    /**
     * @brief Returns the stream this event was created on.
     */
    [[nodiscard]] const std::shared_ptr<StreamWrapper>& get_stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::shared_ptr<StreamWrapper> stream;
};

/**
 * @brief Records the event on its associated stream so other streams can wait on it.
 * @param event_wrapper The event to record.
 */
void record_event(EventWrapper& event_wrapper);
