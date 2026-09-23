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
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/models/aeif/Parameters.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "neurons/models/izhikevich//Parameters.h"
#include "neurons/models/poisson/Parameters.h"

#include <cstdint>

struct NeuronExtraInfoHandle {
    CudaConfig::number_neurons_type num_neurons;
    const UpdateStatus* d_disable_flags;
};

struct FiredRecorderHandle {
    FiredStatus* d_fired;
    CudaConfig::counter_type** d_fired_recorder;
    int d_fire_recorder_size;
};

struct IzhikevichDeviceState {
    CudaConfig::membrane_potential_type* d_x; ///< Membrane potentials.
    CudaConfig::membrane_potential_type* d_u; ///< Recovery variables.
};

struct AeifDeviceState {
    CudaConfig::membrane_potential_type* d_x; ///< Membrane potentials.
    CudaConfig::membrane_potential_type* d_w; ///< Adaptation currents.
};

struct FitzHughNagumoDeviceState {
    CudaConfig::membrane_potential_type* d_x; ///< Membrane-potential-like state.
    CudaConfig::membrane_potential_type* d_w; ///< Recovery variables.
};

struct PoissonDeviceState {
    CudaConfig::membrane_potential_type* d_x; ///< Membrane-potential proxy.
    unsigned int* d_refractory_time;          ///< Remaining refractory periods per neuron.
};

struct IzhikevichKernelParams {
    CudaConfig::membrane_potential_type k1;
    CudaConfig::membrane_potential_type k2;
    CudaConfig::membrane_potential_type k3;
    CudaConfig::membrane_potential_type a;
    CudaConfig::membrane_potential_type b;
    CudaConfig::membrane_potential_type c;
    CudaConfig::membrane_potential_type d;
    CudaConfig::membrane_potential_type V_spike;
};

struct AeifKernelParams {
    CudaConfig::membrane_potential_type d_T;
    CudaConfig::membrane_potential_type tau_w;
    CudaConfig::membrane_potential_type C;
    CudaConfig::membrane_potential_type g_L;
    CudaConfig::membrane_potential_type E_L;
    CudaConfig::membrane_potential_type V_T;
    CudaConfig::membrane_potential_type a;
    CudaConfig::membrane_potential_type b;
    CudaConfig::membrane_potential_type V_spike;
};

struct FitzHughNagumoKernelParams {
    CudaConfig::membrane_potential_type a;
    CudaConfig::membrane_potential_type b;
    CudaConfig::membrane_potential_type phi;
};

struct PoissonKernelParams {
    CudaConfig::membrane_potential_type tau_x;
    CudaConfig::membrane_potential_type x_0;
    unsigned int refractory_period;
};

/**
 * @brief Advances all Izhikevich neurons by one simulation step.
 * @param extra_info  Neuron count and per-neuron disable flags.
 * @param h           Euler integration step size.
 * @param parameters  Izhikevich model parameters (a, b, c, d, v_threshold, ...).
 * @param d_input     Device array of accumulated synaptic input currents (read-only).
 * @param state       Per-neuron device arrays (d_x, d_u); updated in place.
 * @param recorder    Fire-status and per-rank fire-count recorders.
 */
void update_activity_izhikevich_entry(NeuronExtraInfoHandle extra_info, unsigned h,
                                      const models::izhikevich::Parameters<RelearnTypes::activity_type>& parameters,
                                      const CudaConfig::input_type* d_input, IzhikevichDeviceState state,
                                      FiredRecorderHandle recorder);

/**
 * @brief Advances all AdEx (AEIF) neurons by one simulation step.
 * @param extra_info  Neuron count and per-neuron disable flags.
 * @param h           Euler integration step size.
 * @param parameters  AEIF model parameters.
 * @param d_input     Device array of accumulated synaptic input currents (read-write; AEIF may clamp).
 * @param state       Per-neuron device arrays (d_x, d_w); updated in place.
 * @param recorder    Fire-status and per-rank fire-count recorders.
 */
void update_activity_aeif_entry(NeuronExtraInfoHandle extra_info, unsigned h,
                                const models::aeif::Parameters<RelearnTypes::activity_type>& parameters,
                                CudaConfig::input_type* d_input, AeifDeviceState state,
                                FiredRecorderHandle recorder);

/**
 * @brief Advances all FitzHugh-Nagumo neurons by one simulation step.
 * @param extra_info  Neuron count and per-neuron disable flags.
 * @param h           Euler integration step size.
 * @param parameters  FitzHugh-Nagumo model parameters.
 * @param d_input     Device array of accumulated synaptic input currents (read-only).
 * @param state       Per-neuron device arrays (d_x, d_w); updated in place.
 * @param recorder    Fire-status and per-rank fire-count recorders.
 */
void update_activity_fitzhughnagumo_entry(NeuronExtraInfoHandle extra_info, unsigned h,
                                          const models::fitzhughnagumo::Parameters<RelearnTypes::activity_type>& parameters,
                                          const CudaConfig::input_type* d_input, FitzHughNagumoDeviceState state,
                                          FiredRecorderHandle recorder);

/**
 * @brief Advances all Poisson neurons by one simulation step.
 * @param extra_info  Neuron count and per-neuron disable flags.
 * @param h           Euler integration step size.
 * @param parameters  Poisson model parameters (firing rate, ...).
 * @param d_input     Device array of accumulated synaptic input currents (read-only).
 * @param state       Per-neuron device arrays (d_x, d_refractory_time); updated in place.
 * @param recorder    Fire-status and per-rank fire-count recorders.
 * @param random_key  Key selecting the cuRAND stream for spike-rate sampling.
 */
void update_activity_poisson_entry(NeuronExtraInfoHandle extra_info, unsigned h,
                                   const models::poisson::Parameters<RelearnTypes::activity_type>& parameters,
                                   const CudaConfig::input_type* d_input, PoissonDeviceState state,
                                   FiredRecorderHandle recorder, std::uint32_t random_key);
