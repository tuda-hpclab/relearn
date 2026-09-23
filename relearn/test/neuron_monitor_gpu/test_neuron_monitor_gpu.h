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

#ifdef RELEARN_CUDA_ENABLED

#include "cuda/synaptic_elements/SynapticElementsHandle.h"

#include <cstdint>

// Writes deterministic, distinguishable per-neuron values directly into the device buffers behind
// handle -- grown_elements[nid] = 2*nid + grown_offset, connected_elements[nid] = nid + connected_offset
// -- mimicking what a real synapse-creation/deletion kernel does (writing through the raw device
// pointers), so the test can check that ElementBase's host-side accessors (used by
// register_neuron_monitor's callbacks) correctly observe values written this way rather than a
// stale host-side mirror. Distinct offsets per call let a test tell two handles' buffers apart
// (e.g. Dendrites' excitatory vs inhibitory ElementBase), catching a handle mix-up.
void device_write_synaptic_elements(SynapticElementsBaseCudaHandle handle, float grown_offset, std::uint32_t connected_offset);

#endif // RELEARN_CUDA_ENABLED
