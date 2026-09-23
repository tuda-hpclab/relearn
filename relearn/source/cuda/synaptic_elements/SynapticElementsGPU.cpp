/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsGPU.h"

#include "cuda/calcium/CalciumHandle.h"
#include "cuda/synaptic_elements/SynapticElements.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/Util.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/growthrate/GrowthrateCalculator.h"

void SynapticElementsGPU::update_number_elements(const calcium_type* d_calcium,
                                                 const calcium_type* d_target_calcium) {

    const auto info_handle = extra_infos->get_gpu_handle();
    const auto calcium_handle = CalciumHandleConst{ .calcium = d_calcium, .target_calcium = d_target_calcium };
    auto helper = [&info_handle, &calcium_handle](SynapticElementsBaseCudaHandle handle, grown_type growth_rate, const std::shared_ptr<StreamWrapper>& stream) {
        update_number_elements_kernel_entry(calcium_handle, handle, info_handle, growth_rate, stream);
    };

    // get_cuda_handle() below already requests a non-const device pointer for grown/delta/vacant/
    // connected, which already marks them device-modified -- nothing further to do after helper().
    helper(axons->get_cuda_handle(), growthrate_calculator_axon->get_growth_rate(0), axons_stream);
    helper(dendrites->get_cuda_handle(SignalType::Excitatory), growthrate_calculator_excitatory_dendrites->get_growth_rate(0), den_exc_stream);
    helper(dendrites->get_cuda_handle(SignalType::Inhibitory), growthrate_calculator_inhibitory_dendrites->get_growth_rate(0), den_inh_stream);

    cudaDeviceSynchronize_bridge();
    cuda_resolve_gpu_timers();
}
