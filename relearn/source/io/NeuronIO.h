#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/SynapticElementType.h"
#include "sim/LoadedNeuron.h"
#include "sim/file/AdditionalPositionInformation.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

class LocalGroupTranslator;
class Partition;

/**
 * This class provides a static interface to load/store neurons and synapses from/to files,
 * as well as other linked information
 */
class NeuronIO {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using position_type = RelearnTypes::position_type;
    using space_type = RelearnTypes::space_type;

    using PlasticInSynapses = std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses>;
    using PlasticOutSynapses = std::tuple<PlasticLocalSynapses, PlasticDistantOutSynapses>;

    using StaticInSynapses = std::tuple<StaticLocalSynapses, StaticDistantInSynapses>;
    using StaticOutSynapses = std::tuple<StaticLocalSynapses, StaticDistantOutSynapses>;

    using InSynapses = std::tuple<StaticInSynapses, PlasticInSynapses>;
    using OutSynapses = std::tuple<StaticOutSynapses, PlasticOutSynapses>;

    [[nodiscard]] static bool is_valid_group_name(const RelearnTypes::group_name& group_name);

    [[nodiscard]] static bool is_valid_neuron_id_value(const std::string& neuron_id);

    /**
     * @brief Reads all comments from the beginning of the file and returns those.
     *      Comments start with '#'. It stops at the file end or the fist non-comment line
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if opening the file failed
     * @return Returns all comments at the beginning of the file
     */
    [[nodiscard]] static std::vector<std::string> read_comments(const std::filesystem::path& file_path);

    [[nodiscard]] static AdditionalPositionInformation parse_additional_position_information(const std::vector<std::string>& comments);

    /**
     * @brief Reads all neurons from the files and returns those.
     *       The first file must be ascendingly sorted wrt. to the neuron ids (starting at 1). All positions must be non-negative
     * @param paths The object containing the path to the positions and the path to the groups
     * @return Returns a tuple with (1) all loaded neurons, (2) a vector which assigns a neuron id to its group ids,
     *      (3) a vector which assigns a group id to its group name and (4) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<LoadedNeuron>, std::vector<RelearnTypes::group_ids>, RelearnTypes::group_names, LoadedNeuronsInfo, AdditionalPositionInformation> read_neurons_all_information(const NeuronFilePaths& paths);
    /**
     * @brief Reads all neurons from the files and returns the information in their components.
     *       The first file must be ascendingly sorted wrt. to the neuron ids (starting at 1). All positions must be non-negative
     * @param paths The object containing the path to the positions and the path to the groups
     * @return Returns a tuple with
     * (1) all neuron ids (which index (2)-(3) and (5)-(6)),
     * (2) their respective position types,
     * (3) their group ids
     * (4) a vector assigning group ids to group names
     * (5) the neurons' signal types
     * (6) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<RelearnTypes::group_ids>,
                                    RelearnTypes::group_names, std::vector<SignalType>, LoadedNeuronsInfo>
    read_neurons_all_information_componentwise(const NeuronFilePaths& paths);

    /**
     * @brief Reads all neurons and their positions and signal types from the file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1). All positions must be non-negative
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with (1) all loaded neurons and (2) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<LoadedNeuron>, LoadedNeuronsInfo, AdditionalPositionInformation> read_neuron_positions_and_signals(const std::filesystem::path& file_path);

    /**
     * @brief Reads all neurons and their positions and signal types from the file and returns those in their components.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with
     *      (1) The IDs (which index (2)-(4))
     *      (2) The positions
     *      (3) The signal types
     *      (4) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<SignalType>, LoadedNeuronsInfo>
    read_neuron_positions_and_signals_componentwise(const std::filesystem::path& file_path);

    /**
     * @brief Reads the groups of the neurons from the file and writes the groups in the loaded_neurons.
     *       The file does not need to follow a specific ordering and can also leave out neurons. In that case the neurons are assigned to the default group.
     *       However the neuron ids must be contained in the loaded neurons.
     * @param file_path The path to the file with the groups
     * @param number_neurons The total number of neurons whose groups could be read, even if some of them don't appear in the file
     * @exception Throws a RelearnException if: the file can't be found, the first token in a line is neither a group name nor a neuron id, a neuron id is out of bounds, a group name is not valid
     * @return A tuple with (1) a vector assigning a neuron id to its group ids (neuron_id -> group_ids) and (2) a vector assigning a group id to its group name (group_id <-> group_name)
     */
    [[nodiscard]] static std::tuple<std::vector<RelearnTypes::group_ids>, RelearnTypes::group_names> read_neuron_groups(const std::filesystem::path& file_path, RelearnTypes::number_neurons_type number_neurons);

    /**
     * @brief Writes all neurons to the files
     * @param neurons The neurons
     * @param paths The object containing the path to write the positions in and the path to write the groups in
     * @param local_group_translator Maps local group id to group map
     * @param partition Partition of the entire simulation
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const NeuronFilePaths& paths,
                              const std::shared_ptr<LocalGroupTranslator>& local_group_translator, const std::shared_ptr<Partition>& partition);

    /**
     * @brief Writes all neurons to the files
     * @param neurons The neurons
     * @param paths The object containing the path to write the positions in and the path to write the groups in
     * @param local_group_translator Maps local group id to group map
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const NeuronFilePaths& paths, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes all neurons to the file (does not include groups)
     * @param neurons The neurons
     * @param sstream StringStream in which the content is written
     * @param partition The partition of the simulation
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neuron_positions_and_signals(const std::vector<LoadedNeuron>& neurons, std::stringstream& sstream, const std::shared_ptr<Partition>& partition);

    /**
     * @brief Writes all neurons to the file (does not include groups)
     * @param neurons The neurons
     * @param file_path The file path to write in
     * @param partition The partition of the simulation
     */
    static void write_neuron_positions_and_signals(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<Partition>& partition);

    /**
     * @brief Writes the groups of neurons to the file
     * @param sstream The stringstream to write the content in
     * @param local_group_translator The local group translator
     */
    static void write_neuron_groups(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes the groups of neurons to the file
     * @param file_path The file path to write in
     * @param local_group_translator The local group translator
     */
    static void write_neuron_groups(const std::filesystem::path& file_path, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes the groups of specific neurons to the file
     * @param ids The neurons whose groups should be written out
     * @param sstream The stringstream to write the content in
     * @param local_group_translator The local group translator to map the neuron ids to their group ids
     */
    static void write_neuron_groups_of_specific_neurons(std::span<const NeuronID> ids, std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes the groups of specific neurons to the file
     * @param ids The neurons whose groups should be written out
     * @param file_path The file path to write in
     * @param local_group_translator The local group translator to map the neuron ids to their group ids
     */
    static void write_neuron_groups_of_specific_neurons(std::span<const NeuronID> ids, const std::filesystem::path& file_path, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes the assignment of group ids to group names
     * @param sstream Stringstream in which the group mapping is written
     * @param local_group_translator Maps local group id to group map
     */
    static void write_group_names(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes the group name to file name mapping
     * @param sstream Stringstream in which the group to file mapping is written
     * @param local_group_translator The local group translator
     */
    static void write_group_name_to_file_name(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator);

    /**
     * @brief Writes all positions and signal types of the neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param signal_types The signal types
     * @param sstream Stringstream to which is written
     * @param total_number_neurons Number of all neurons in the simulation
     * @param simulation_box Bounding box of the entire simulation
     * @param local_subdomain_boundaries List of bounding boxes (as pair of bounding box min and bounding box max) of the local subdomains
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neuron_positions_and_signals_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                                                 std::span<const SignalType> signal_types, std::stringstream& sstream,
                                                                 number_neurons_type total_number_neurons, RelearnTypes::bounding_box_type simulation_box, std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries);

    /**
     * @brief Writes all positions and signal types of the neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param signal_types The signal types
     * @param file_path The file path to write in
     * @param total_number_neurons Number of all neurons in the simulation
     * @param simulation_box Bounding box of the entire simulation
     * @param local_subdomain_boundaries List of bounding boxes (as pair of bounding box min and bounding box max) of the local subdomains
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neuron_positions_and_signals_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                                                 std::span<const SignalType> signal_types, const std::filesystem::path& file_path,
                                                                 number_neurons_type total_number_neurons, RelearnTypes::bounding_box_type simulation_box, std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries);

    /**
     * @brief Writes all positions and signal types of the neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param signal_types The signal types
     * @param file_path The file path to write in
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neuron_positions_and_signals_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                                                 std::span<const SignalType> signal_types, const std::filesystem::path& file_path);

    /**
     * @brief Writes all neurons to the files. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param local_group_translator Maps local group id to group map
     * @param signal_types The signal types
     * @param paths The object containing the path to write the positions in and the path to write the groups in
     * @param total_number_neurons Number of all neurons in the simulation
     * @param simulation_box Bounding box of the entire simulation
     * @param local_subdomain_boundaries List of bounding boxes (as pair of bounding box min and bounding box max) of the local subdomains
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                            const std::shared_ptr<LocalGroupTranslator>& local_group_translator, std::span<const SignalType> signal_types,
                                            const NeuronFilePaths& paths, number_neurons_type total_number_neurons, RelearnTypes::bounding_box_type simulation_box,
                                            std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries);

    /**
     * @brief Writes all neurons to the files. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
     * @param ids The IDs
     * @param positions The positions
     * @param local_group_translator Maps local group id to group map
     * @param signal_types The signal types
     * @param paths The object containing the path to write the positions in and the path to write the groups in
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                            const std::shared_ptr<LocalGroupTranslator>& local_group_translator, std::span<const SignalType> signal_types, const NeuronFilePaths& paths);

    /**
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<NeuronID>> read_neuron_ids(const std::filesystem::path& file_path);

    /**
     * @brief Reads all in-synapses from a file and returns those.
     *      Checks that no target id is larger or equal to number_local_neurons and that no source rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synapse is 0
     *      (3) a target rank is not my_rank
     *      (4) a source rank is not from [0, number_mpi_ranks)
     *      (5) or a target id is not from [0, number_local_neurons)
     * @return All in-synapses as a tuple: { { (1) Static synapses: (1.1) The local ones and (1.2) the distant ones }, { (2) Plastic synapses: (2.1) The local ones and (2.2) the distant ones } }
     */
    static InSynapses read_in_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, mpiPP::MPIRank my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all in-synapses to the specified file
     * @param local_in_synapses_static The local in-synapses that are static
     * @param distant_in_synapses_static The distant in-synapses that are static
     * @param local_in_synapses_plastic The local in-synapses that are plastic
     * @param distant_in_synapses_plastic The distant in-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param number_neurons Number of local neurons on the current mpi rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the source rank of a distant in-synapse is equal to my_rank
     */
    static void write_in_synapses(const StaticLocalSynapses& local_in_synapses_static, const StaticDistantInSynapses& distant_in_synapses_static, const PlasticLocalSynapses& local_in_synapses_plastic,
                                  const PlasticDistantInSynapses& distant_in_synapses_plastic, mpiPP::MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all in-synapses to the specified stream
     * @param local_in_edges_static The local in-synapses that are static
     * @param distant_in_edges_static The distant in-synapses that are static
     * @param local_in_edges_plastic The local in-synapses that are plastic
     * @param distant_in_edges_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param mpi_ranks Number of used mpi ranks
     * @param number_local_neurons Number of local neurons on the current mpi rank
     * @param number_total_neurons Number of neurons over all ranks
     * @param step The current step of the simulation
     * @param sstream StringStream to which the output is written
     */
    static void write_in_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_in_edges_static,
                                  const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_in_edges_static,
                                  const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_in_edges_plastic,
                                  const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_in_edges_plastic,
                                  mpiPP::MPIRank my_rank, int mpi_ranks, RelearnTypes::number_neurons_type number_local_neurons, RelearnTypes::number_neurons_type number_total_neurons, std::stringstream& sstream, RelearnTypes::step_type step);

    /**
     * @brief Reads all out-synapses from a file and returns those.
     *      Checks that no source id is larger or equal to number_local_neurons and that no target rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synapse is 0
     *      (3) a source rank is not my_rank
     *      (4) a target rank is not from [0, number_mpi_ranks)
     *      (5) or a source id is not from [0, number_local_neurons)
     * @return All out-synapses as a tuple: { { (1) Static synapses: (1.1) The local ones and (1.2) the distant ones }, { (2) Plastic synapses: (2.1) The local ones and (2.2) the distant ones } }
     */
    static OutSynapses read_out_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, mpiPP::MPIRank my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all out-synapses to the specified file
     * @param local_out_synapses_static The local out-synapses that are static
     * @param distant_out_synapses_static The distant out-synapses that are static
     * @param local_out_synapses_plastic The local out-synapses that are plastic
     * @param distant_out_synapses_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param number_neurons Number of local neurons on the current mpi rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the target rank of a distant out-synapse is equal to my_rank
     */
    static void write_out_synapses(const StaticLocalSynapses& local_out_synapses_static, const StaticDistantOutSynapses& distant_out_synapses_static,
                                   const PlasticLocalSynapses& local_out_synapses_plastic, const PlasticDistantOutSynapses& distant_out_synapses_plastic,
                                   mpiPP::MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all out-synapses to the specified stream
     * @param local_out_edges_static The local out-synapses that are static
     * @param distant_out_edges_static The distant out-synapses that are static
     * @param local_out_edges_plastic The local out-synapses that are plastic
     * @param distant_out_edges_plastic The distant out-synapses that are plastic
     * @param my_rank The current MPI rank
     * @param mpi_ranks Number of used mpi ranks
     * @param number_local_neurons Number of local neurons on the current mpi rank
     * @param number_total_neurons Number of neurons over all ranks
     * @param step The current step of the simulation
     * @param sstream StringStream to which the output is written
     */
    static void write_out_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_out_edges_static,
                                   const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_out_edges_static,
                                   const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_out_edges_plastic,
                                   const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_out_edges_plastic,
                                   mpiPP::MPIRank my_rank, int mpi_ranks, RelearnTypes::number_neurons_type number_local_neurons, RelearnTypes::number_neurons_type number_total_neurons, std::stringstream& sstream, RelearnTypes::step_type step);
};
