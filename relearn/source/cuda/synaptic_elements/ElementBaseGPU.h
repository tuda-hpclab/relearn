#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/synaptic_elements/ElementBaseBase.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/synaptic_elements/SynapticElements.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/NeuronsExtraInfo.h"
#include "types/BasicTypes.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * GPU implementation of ElementBase: updates are committed on the device, and the device-side
 * arrays report their own extra memory footprint. get_total_additions()/get_total_deletions() are
 * only tracked on the host-loop path (ElementBaseCPU::commit_updates()) and are not meaningful
 * here, so they are unsupported.
 */
class ElementBaseGPU : public ElementBaseBase<LazySyncedArray> {
public:
    /**
     * @brief Initializes the object to contain number_neurons elements.
     *      Uses the calculators for grown elements, delta since last update, and connected elements.
     *      Calculates the number of vacant elements based on the other values.
     * @param number_neurons The number of neurons to initialize, >0
     * @exception Throws a RelearnException if the object was already initialized, if number_neurons == 0,
     *      or if the calculator for the grown elements returns a value < 0.0
     */
    void init(const RelearnTypes::number_neurons_type number_neurons) {
        RelearnException::check(size == 0, "ElementBaseGPU::init: Already initialized");
        RelearnException::check(number_neurons > 0, "ElementBaseGPU::init: number_neurons must be > 0");

        size = number_neurons;

        grown_elements.resize(size);
        delta_since_last_update.resize(size);
        vacant_elements.resize(size);
        vacant_retract_ratio.resize(size);
        minimum_calcium.resize(size);
        connected_elements.resize(size);

        init_values(RelearnTypes::number_neurons_type{ 0 }, size);

        d_to_delete = DeviceArray<CudaConfig::synaptic_count_type>(size, 0);
    }

    SynapticElementsBaseCudaHandleConst get_cuda_handle_const() const {
        return { grown_elements.get_device_ptr_const(), delta_since_last_update.get_device_ptr_const(), vacant_elements.get_device_ptr_const(), connected_elements.get_device_ptr_const(), vacant_retract_ratio.get_device_ptr_const(), minimum_calcium.get_device_ptr_const(), static_cast<CudaConfig::number_neurons_type>(size) };
    }

    SynapticElementsBaseCudaHandle get_cuda_handle() {
        return { grown_elements.get_device_ptr(), delta_since_last_update.get_device_ptr(), vacant_elements.get_device_ptr(), connected_elements.get_device_ptr(), vacant_retract_ratio.get_device_ptr(), minimum_calcium.get_device_ptr(), static_cast<CudaConfig::number_neurons_type>(size) };
    }

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates(const std::shared_ptr<StreamWrapper>& stream) {
        // get_cuda_handle() already requested a non-const device pointer for each of these arrays,
        // which already marked them device-modified -- nothing further to do here.
        commit_synaptic_elements_entry(get_cuda_handle(), d_to_delete.device_ptr(), extra_infos->get_gpu_handle(), stream);

        return d_to_delete.get_device_data();
    }

    [[nodiscard]] std::vector<counter_type> commit_updates() {
        auto def_stream = std::make_shared<StreamWrapper>();
        return commit_updates(def_stream);
    }

    /**
     * @brief Returns the total number of additions over the lifetime of this object
     * @return The total number of additions
     */
    [[nodiscard]] grown_type get_total_additions() const {
        CPU_NOT_SUPPORTED
        return total_additions;
    }

    /**
     * @brief Returns the total number of deletions over the lifetime of this object
     * @return The total number of deletions
     */
    [[nodiscard]] grown_type get_total_deletions() const {
        CPU_NOT_SUPPORTED
        return total_deletions;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     * @param key The key to use for the memory footprint
     */
    template <typename T>
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, T&& key) {
        const auto size_grown = sizeof(grown_type) * grown_elements.capacity();
        const auto size_delta = sizeof(grown_type) * delta_since_last_update.capacity();
        const auto size_vacant = sizeof(counter_type) * vacant_elements.capacity();
        const auto size_connected = sizeof(counter_type) * connected_elements.capacity();
        const auto size_vacant_retract = sizeof(grown_type) * vacant_retract_ratio.capacity();
        const auto size_minimum_calcium = sizeof(calcium_type) * minimum_calcium.capacity();

        const auto my_size = sizeof(*this);
        const auto total_size = size_grown + size_delta + size_vacant + size_connected + size_vacant_retract + size_minimum_calcium + my_size;
        footprint->emplace(key, total_size);

        footprint->emplace(std::string(key) + " GPU", minimum_calcium.get_memory_footprint() + grown_elements.get_memory_footprint() + delta_since_last_update.get_memory_footprint() + connected_elements.get_memory_footprint() + vacant_elements.get_memory_footprint() + vacant_retract_ratio.get_memory_footprint());
    }

private:
    DeviceArray<CudaConfig::synaptic_count_type> d_to_delete{ 0 };
};
