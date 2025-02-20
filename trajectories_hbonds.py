#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.1.0"

import argparse
import logging
import numpy as np
import os
import pickle
import re
import statistics
import sys
import yaml

from mpi4py import MPI
import pandas as pd
import pytraj as pt


def restricted_float(value_to_inspect):
    """Inspect if a float is between 0.0 and 100.0

    :param value_to_inspect: the value to inspect
    :type value_to_inspect: str
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the float value if float_to_inspect is between 0.0 and 100.0
    :rtype: float
    """
    x = float(value_to_inspect)
    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")
    return x


def restricted_positive(value_to_inspect):
    """Inspect if the value is positive.

    :param value_to_inspect: the value to inspect
    :type value_to_inspect: str
    :raises ArgumentTypeError: is not > 0.0
    :return: the float value if float_to_inspect is between 0.0 and 100.0
    :rtype: float
    """
    x = float(value_to_inspect)
    if x < 0.0:
        raise argparse.ArgumentTypeError(f"{x} not a positive value.")
    return x


def restricted_angle(value_to_inspect):
    """Inspect if an angle value is between 0 and 359.

    :param value_to_inspect: the value to inspect
    :type value_to_inspect: str
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the float value if float_to_inspect is between 0.0 and 100.0
    :rtype: float
    """
    x = int(value_to_inspect)
    if x < 0 or x > 359:
        raise argparse.ArgumentTypeError(f"{x} not a valid angle, it should be between 0 and 359.")
    return x


def create_log(path, level):
    """Create the log as a text file and as a stream.

    :param path: the path of the log.
    :type path: str
    :param level: the level of the log.
    :type level: str
    :return: the logging:
    :rtype: logging
    """

    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    if level is None:
        log_level = log_level_dict["INFO"]
    else:
        log_level = log_level_dict[level]

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def parse_frames(frames_selections, traj_files_paths):
    """
    Parse the frames' selection by trajectory file.

    :param frames_selections: the frames selection by trajectory file.
    :type frames_selections: str
    :param traj_files_paths: the trajectory files paths.
    :type traj_files_paths: list
    :return: the selected frames by trajectory file.
    :rtype: dict
    """
    frames_selection_data = {}
    # the pattern to get the beginning and the end of the frames selection
    pattern = re.compile("(.+):(\\d+|\\*)-(\\d+|\\*)")

    if frames_selections:
        start = 'the first frame'
        end = 'the last frame'
        traj_basenames = [os.path.basename(traj_files_path) for traj_files_path in traj_files_paths]
        for frames_sel in frames_selections.split(","):
            match = pattern.search(frames_sel)
            if match:
                current_traj = match.group(1)
                start = match.group(2)
                end = match.group(3)
                if current_traj not in traj_basenames:
                    raise argparse.ArgumentTypeError(f"The trajectory file {current_traj} in frame selection part "
                                                     f"'{frames_sel}' is not a file belonging to the inputs trajectory "
                                                     f"files: {','.join(traj_basenames)}")
                if start != "*":
                    frames_selection_data[current_traj] = {"begin": int(start)}
                if end != "*":
                    if current_traj not in frames_selection_data:
                        frames_selection_data[current_traj] = {"end": int(end)}
                    else:
                        frames_selection_data[current_traj]["end"] = int(end)
            else:
                raise argparse.ArgumentTypeError(f"The frame selection part '{frames_sel}' do not match the correct "
                                                 f"pattern {pattern.pattern}'")
        if comm.rank == 0:
            logging.info("Frames selection on input trajectory files:")
        for current_traj in frames_selection_data:
            logging.info(f"\t{current_traj}: frames selection from {start} to {end}.")
    return frames_selection_data


def resume_or_initialize_analysis(trajectory_files, topology_file, smp, distance_hbonds, angle_cutoff,
                                  proportion_hbonds, sim_time, resume_yaml, frames_sel):
    """
    Load the previous analysis data or create a new one if no previous analysis path was performed.

    :param trajectory_files: the current analysis trajectory files path.
    :type trajectory_files: list
    :param topology_file: the trajectories' topology file.
    :type topology_file: str
    :param smp: the sample name.
    :type smp: str
    :param distance_hbonds: the threshold atoms distance in Angstroms for hydrogen bonds.
    :type distance_hbonds: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param proportion_hbonds: the minimal percentage of hydrogen bonds for atoms' contacts of different residues in the
    selected frames.
    :type proportion_hbonds: float
    :param sim_time: the molecular dynamics simulation time in ns.
    :type sim_time: int
    :param resume_yaml: the path to the YAML file of previous analysis.
    :type resume_yaml: str
    :param frames_sel: the frames' selection for new trajectory files.
    :type frames_sel: dict
    :return: the initialized or resumed analysis data, the trajectory files' already analyzed.
    :rtype: dict, list
    """
    trajectory_files_to_skip = []
    if resume_yaml:
        with open(resume_yaml, "r") as file_handler:
            data = yaml.safe_load(file_handler.read())

        # check the processed analyzed trajectories files
        for t_file_path in trajectory_files:
            t_file = os.path.basename(t_file_path)
            if t_file in data["trajectory files processed"]:
                trajectory_files_to_skip.append(t_file)
                if comm.rank == 0:
                    logging.warning(f"\t{t_file} already processed in the previous analysis (check the YAML file, "
                                    f"section 'trajectory files processed'), this trajectory analysis is skipped.")

        discrepancies = []
        if data["sample"] != smp:
            discrepancies.append(f"discrepancy in --sample, current analysis is {smp}, previous analysis was "
                                 f"{data['sample']}")
        if data["topology file"] != os.path.basename(topology_file):
            discrepancies.append(f"discrepancy in --topology, current analysis is {topology_file}, previous analysis "
                                 f"was {data['topology file']}")
        if data["parameters"]["maximal atoms distance"] != distance_hbonds:
            discrepancies.append(f"discrepancy in --distance-hbonds, current analysis is {distance_hbonds}, "
                                 f"previous analysis was {data['parameters']['maximal atoms distance']}")
        if data["parameters"]["angle cutoff"] != angle_cutoff:
            discrepancies.append(f"discrepancy in --angle-cutoff, current analysis is {angle_cutoff}, previous "
                                 f"analysis was {data['parameters']['angle cutoff']}")
        if data["parameters"]["proportion hbonds"] != proportion_hbonds:
            discrepancies.append(f"discrepancy in --proportion-hbonds, current analysis is {proportion_hbonds}, "
                                 f"previous analysis was {data['parameters']['proportion hbonds']}")
        if discrepancies:
            discrepancies_txt = None
            for item in discrepancies:
                if discrepancies_txt:
                    discrepancies_txt = f"{discrepancies_txt}; {item}"
                else:
                    discrepancies_txt = item
                discrepancies_txt = f"{discrepancies_txt}. Check {resume_yaml}"
            raise KeyError(discrepancies_txt)

        # load the H bonds from the pickle file
        try:
            with open(data["pickle hydrogen bonds"], "rb") as file_handler:
                data["H bonds"] = pickle.load(file_handler)
        except FileNotFoundError as fnf_ex:
            logging.error(fnf_ex, exc_info=True)
            sys.exit(1)

        # add frames selection in new trajectory files
        if frames_sel:
            if "frames selections" in data["parameters"]:
                for traj_fn in frames_sel:
                    data["parameters"]["frames selections"][traj_fn] = frames_sel[traj_fn]
            else:
                data["parameters"]["frames selections"] = frames_sel
    else:
        data = {"sample": smp, "size Gb": 0, "frames": 0,
                "parameters": {"maximal atoms distance": distance_hbonds, "angle cutoff": angle_cutoff,
                               "proportion hbonds": proportion_hbonds},
                "topology file": os.path.basename(topology_file)}
    # set the simulation time
    data["parameters"]["time"] = f"{sim_time} ns"
    # add an H bonds section if necessary
    if "H bonds" not in data:
        data["H bonds"] = {}
    return data, trajectory_files_to_skip


def remove_processed_trajectories(all_traj, traj_to_skip, yaml_file):
    """
    Remove already processed trajectories.

    :param all_traj: the input trajectories.
    :type all_traj: list
    :param traj_to_skip: the trajectories already processed.
    :type traj_to_skip: list
    :param yaml_file: the path to the yaml file.
    :type yaml_file: str
    :return: the remaining trajectories.
    :rtype: list
    """
    traj_to_process = []
    for traj_path in all_traj:
        if os.path.basename(traj_path) not in traj_to_skip:
            traj_to_process.append(traj_path)
    if not traj_to_process:
        if comm.rank == 0:
            logging.warning(f"All trajectories ({','.join(all_traj)}) have already been processed, check the "
                            f"'trajectory files processed' section of: {yaml_file}")
    return traj_to_process


def load_trajectory(trajectory_file, topology_file, frames_sel):
    """
    Load the trajectory file.

    :param trajectory_file: the trajectory file path.
    :type trajectory_file: str
    :param topology_file: the topology file path.
    :type topology_file: str
    :param frames_sel: the frames' selection for new trajectory files.
    :type frames_sel: dict
    :return: the loaded trajectory.
    :rtype: pytraj.Trajectory
    """
    if comm.rank == 0:
        logging.info(f"\tLoading trajectory file, please be patient..")
    traj = None
    try:
        if trajectory_file in frames_sel:
            traj = pt.iterload(trajectory_file, top=topology_file,
                               frames_indices=range(frames_sel[trajectory_file]["begin"] - 1,
                                                    frames_sel[trajectory_file]["end"] - 1))
        else:
            traj = pt.iterload(trajectory_file, top=topology_file)
    except ValueError as ve_ex:
        logging.error(f"\tOne of the following files is missing: {trajectory_file} or {topology_file}")
        sys.exit(1)
    if comm.rank == 0:
        logging.info(f"\t\tMolecules:{traj.topology.n_mols:>20}")
        logging.info(f"\t\tResidues:{traj.topology.n_residues:>22}")
        logging.info(f"\t\tAtoms:{traj.topology.n_atoms:>27}")
        logging.info(f"\t\tTrajectory total frames:{traj.n_frames:>7}")
        logging.info(f"\t\tTrajectory memory size:{round(traj._estimated_GB, 6):>14} Gb")
    if os.path.basename(trajectory_file) in frames_sel:
        traj_bn = os.path.basename(trajectory_file)
        if frames_sel[traj_bn]["end"] > traj.n_frames:
            raise IndexError(f"Selected upper frame limit for {traj_bn} ({frames_sel[traj_bn]['end']}) from "
                             f"--frames argument is greater than the total frames number ({traj.n_frames}) of the MD "
                             f"trajectory.")
        frames_range = range(frames_sel[traj_bn]["begin"], frames_sel[traj_bn]["end"])
        if frames_sel[traj_bn]["begin"] == 1:
            frames_range[0] = 0
        traj = traj[frames_range]
        if comm.rank == 0:
            logging.info(f"\t\tSelected frames:{frames_sel[traj_bn]['begin']:>14} to {frames_sel[traj_bn]['end']}")
            logging.info(f"\t\tSelected frames memory size:{round(traj._estimated_GB, 6):>9} GB")
    else:
        txt = f"1 to {traj.n_frames}"
        if comm.rank == 0:
            logging.info(f"\t\tSelected frames:{txt:>20}")
    return traj


def check_trajectories_consistency(traj, path, data, frames_sel):
    """
    Check if the trajectory attributes match with the previous trajectories.

    :param traj: the current trajectory.
    :type traj: pytraj.Trajectory
    :param path: the current trajectory path.
    :type path: str
    :param data: the trajectories' data.
    :type data: dict
    :param frames_sel: the frames' selection on the trajectory files.
    :type frames_sel: dict
    :return: the updated trajectories' data.
    :rtype: dict
    """
    if "residues" not in data:
        data["residues"] = traj.topology.n_residues
        data["atoms"] = traj.topology.n_atoms
        data["molecules"] = traj.topology.n_mols
    else:
        if data["residues"] != traj.topology.n_residues:
            raise ValueError(f"the residues number ({traj.topology.n_residues}) is different from the residues number "
                             f"of the previous trajectories ({data['residues']}), check if "
                             f"{os.path.basename(path)} is from the same trajectory than the previous ones.")
        if data["atoms"] != traj.topology.n_atoms:
            raise ValueError(f"the atoms number ({traj.topology.n_atoms}) is different from the atoms number of "
                             f"the previous trajectories ({data['atoms']}), check if {os.path.basename(path)} is "
                             f"from the same trajectory than the previous ones.")
        if data["molecules"] != traj.topology.n_mols:
            raise ValueError(f"the molecules number ({traj.topology.n_mols}) is different from the molecules number of "
                             f"the previous trajectories ({data['molecules']}), check if {os.path.basename(path)} "
                             f"is from the same trajectory than the previous ones.")
    if frames_sel:
        if "frames selection" not in data:
            data["frames selection"] = {}
            for traj_name in frames_sel:
                data["frames selection"][traj_name] = frames_sel[traj_name]
    data["size Gb"] += traj._estimated_GB
    data["frames"] += traj.n_frames
    return data


def from_hbond_parallel_to_amber_mask(hb_parallelized):
    """
    Convert the keys of hb_parallelized dictionary to amber mask
    :param hb_parallelized: dictionary with the hydrogen bonds keys.
    :type hb_parallelized: dict
    :return: lists of tuples with the amber masks of the keys.
    :rtype: list
    """
    # get all the keys from hb_parallelized dictionary
    keys = list(hb_parallelized.keys())
    # remove the key 'total_solute_hbonds'
    keys.remove("total_solute_hbonds")
    # change the format of keys, i.e.: HIE4_O-LYS8_NZ-HZ2 to HIE_4@O-LYS_8@NZ-HZ2
    for i in range(len(keys)):
        keys[i] = keys[i].replace("_", " ").replace("-", " ").split()
        # slip the first element after 3 characters
        keys[i][0] = keys[i][0][:3] + '_' + keys[i][0][3:]
        keys[i][2] = keys[i][2][:3] + '_' + keys[i][2][3:]
        acceptor_mask = '@'.join((keys[i][0], keys[i][1]))
        donor_mask = '@'.join((keys[i][2], keys[i][3]))
        keys[i] = '-'.join((acceptor_mask, donor_mask, keys[i][4]))
    # Use function to_amber_mask to convert the keys to amber mask
    amber_masks = list(pt.hbond_analysis.to_amber_mask(keys))
    # split the list of tuples to two independent lists
    distances_mask, angles_mask = list(zip(*amber_masks))
    return distances_mask, angles_mask


def hydrogen_bonds(inspected_traj, data, atoms_dist, angle):
    """
    Extract the hydrogen bonds and add the distances' values.

    :param inspected_traj: the trajectory.
    :type inspected_traj: pytraj.Trajectory
    :param data: the trajectories' data.
    :type data: dict
    :param atoms_dist: the threshold atoms distance in Angstroms for an hydrogen bond.
    :type atoms_dist: float
    :param angle: the angle cutoff for the hydrogen bonds.
    :type angle: int
    :return: the updated trajectories' data.
    :rtype: dict
    """
    # search hydrogen bonds with distance < atoms' distance threshold and angle > angle cut-off.
    if comm.rank == 0:
        logging.info("\tSearch for hydrogen bonds, please be patient..")

    h_bonds = pt.pmap_mpi(pt.hbond, inspected_traj, distance=atoms_dist, angle=angle)

    # get the distances
    logging.info(f"\tGet the hydrogen bonds distances from process {comm.rank}, please be patient..")
    # MPI broadcast the hydrogen bonds data to all the MPI processes
    h_bonds = comm.bcast(h_bonds, root=0)
    # get all the keys from hb_parallelized dictionary
    donors_acceptors = list(h_bonds.keys())
    # remove the key 'total_solute_hbonds'
    donors_acceptors.remove("total_solute_hbonds")
    # convert the keys of the parallelized hbonds ordered dictionary to amber mask
    hbonds_distance_mask, _ = from_hbond_parallel_to_amber_mask(h_bonds)
    # get the distances of the hbonds
    distances = pt.pmap_mpi(pt.distance, inspected_traj, hbonds_distance_mask)
    # MPI broadcast the distances data to all the MPI processes
    distances = comm.bcast(distances, root=0)

    # filter the Hydrogen bonds
    # record the distances of all hydrogen bonds (donors-acceptors) detected in the chunk
    for idx in range(len(donors_acceptors)):
        # filter the whole frames distances for this contact on the atoms contact distance threshold
        key_distance = list(distances.keys())[idx]
        filtered_distances = distances[key_distance][distances[key_distance] <= atoms_dist]
        if donors_acceptors[idx] in data["H bonds"]:
            data["H bonds"][donors_acceptors[idx]] = np.concatenate((data["H bonds"][donors_acceptors[idx]],
                                                                     filtered_distances))
        else:
            data["H bonds"][donors_acceptors[idx]] = filtered_distances

    if comm.rank == 0:
        logging.info(f"\t\t{len(data['H bonds'])} hydrogen bonds found in the {inspected_traj.n_frames} frames of the "
                     f"trajectory.")
    return data


def record_analysis(data, out_dir, current_trajectory_file, smp):
    """
    Record the analysis to a YAML file and pickle the hydrogen bonds data in a binary file.

    :param data: the trajectory analysis data.
    :type data: dict
    :param out_dir: the path to the output directory.
    :type out_dir: str
    :param current_trajectory_file: the current trajectory file.
    :type current_trajectory_file: str
    :param smp: the sample name.
    :type smp: str
    :return: the trajectory analysis data.
    :rtype: dict
    """
    if "trajectory files processed" not in data:
        data["trajectory files processed"] = []
    data["trajectory files processed"].append(os.path.basename(current_trajectory_file))

    # extract the H bonds from the data and pickle them
    out_pickle = os.path.join(out_dir, f"{smp.replace(' ', '_')}_analysis.pkl")
    hydrogen_bond_analysis_data = data.pop("H bonds")
    with open(out_pickle, "wb") as file_handler:
        pickle.dump(hydrogen_bond_analysis_data, file_handler)
    if comm.rank == 0:
        logging.info(f"\t\tHydrogen bonds analysis saved: {os.path.abspath(out_pickle)}")

    # save the analysis parameters without the H bonds to a YAML file
    data["pickle hydrogen bonds"] = os.path.abspath(out_pickle)
    out_yaml = os.path.join(out_dir, f"{smp.replace(' ', '_')}_analysis.yaml")
    with open(out_yaml, "w") as file_handler:
        yaml.dump(data, file_handler)
    if comm.rank == 0:
        logging.info(f"\t\tAnalysis parameters saved: {os.path.abspath(out_yaml)}")

    # add the extracted H bonds to the data again
    data["H bonds"] = hydrogen_bond_analysis_data
    return data


def sort_hbonds(contact_names, pattern):
    """
    Get the order of the hbonds on the first residue then on the second one.

    :param contact_names: the hbonds' identifiers.
    :type contact_names: KeysView[Union[str, Any]]
    :param pattern: the pattern to extract the residues' positions of the atoms hbonds.
    :type pattern: re.pattern
    :return: the ordered list of hbonds.
    :rtype: list
    """
    tmp = {}
    ordered = []
    for contact_name in contact_names:
        match = pattern.search(contact_name)
        if match:
            if int(match.group(2)) in tmp:
                if int(match.group(5)) in tmp[int(match.group(2))]:
                    tmp[int(match.group(2))][int(match.group(5))].append(contact_name)
                else:
                    tmp[int(match.group(2))][int(match.group(5))] = [contact_name]
            else:
                tmp[int(match.group(2))] = {int(match.group(5)): [contact_name]}
        else:
            logging.error(f"no match for {pattern.pattern} in {contact_name}")
            sys.exit(1)

    for key1 in sorted(tmp):
        for key2 in sorted(tmp[key1]):
            for contact_name in sorted(tmp[key1][key2]):
                ordered.append(contact_name)

    return ordered


def filter_hbonds(analysis_data, pattern):
    """
    Filter out the hydrogen bonds' that belong to the same residue.

    :param analysis_data: the whole analysis data
    :type analysis_data: dict
    :param pattern: the donor/acceptor pattern.
    :type pattern: re.pattern
    :return: the filtered hydrogen bonds.
    :rtype: pandas.DataFrame
    """
    if comm.rank == 0:
        logging.info(f"Filtering {len(analysis_data['trajectory files processed'])} trajector"
                     f"{'ies' if len(analysis_data['trajectory files processed']) > 1 else 'y'} file"
                     f"{'s' if len(analysis_data['trajectory files processed']) > 1 else ''} with "
                     f"{len(analysis_data['H bonds'])} hydrogen bonds:")
    intra_residue_hbonds = 0
    inter_residue_hbonds_failed_thr = 0
    data = {}
    for donor_acceptor, distances in analysis_data["H bonds"].items():
        match = pattern.search(donor_acceptor)
        if match:
            if match.group(2) == match.group(5):
                intra_residue_hbonds += 1
                if comm.rank == 0:
                    logging.debug(f"\t[REJECTED INTRA] {donor_acceptor}: H bond between the atoms of the same residue.")
            else:
                # retrieve only the hbonds < to the max atoms threshold and having a percentage of
                # frames >= percentage threshold in the selected frames
                pct_hbonds = len(distances) / analysis_data["frames"] * 100
                if pct_hbonds >= analysis_data["parameters"]["proportion hbonds"]:
                    # get the median of all the distances for two atoms hbonds
                    data[donor_acceptor] = statistics.median(distances)
                    if comm.rank == 0:
                        logging.debug(f"\t[VALID {len(data)}] {donor_acceptor}: median atoms distance "
                                      f"{round(data[donor_acceptor], 2)} \u212B, proportion of valid frames "
                                      f"{pct_hbonds:.1f}% ({len(distances)}/{analysis_data['frames']} frames with H "
                                      f"bonds).")
                else:
                    inter_residue_hbonds_failed_thr += 1
                    if comm.rank == 0:
                        logging.debug(f"\t[REJECTED PERCENTAGE] {donor_acceptor}: frames with H bonds "
                                      f"{pct_hbonds:.1f}% < {analysis_data['parameters']['proportion hbonds']:.1f}%"
                                      f" threshold ({len(distances)}/{analysis_data['frames']} frames with H bonds).")
        else:
            raise Exception(f"no match for pattern '{pattern.pattern}' in donor/acceptor '{donor_acceptor}'")
    nb_used_hbonds = len(analysis_data["H bonds"]) - intra_residue_hbonds - inter_residue_hbonds_failed_thr
    if comm.rank == 0:
        logging.info(f"\t{intra_residue_hbonds}/{len(analysis_data['H bonds'])} hydrogen bonds with intra residues "
                     f"atoms hbonds discarded.")
        logging.info(f"\t{inter_residue_hbonds_failed_thr}/{len(analysis_data['H bonds'])} hydrogen bonds with inter "
                     f"residues atoms hydrogen bond discarded with hydrogen bond frames proportion under the threshold of "
                     f"{analysis_data['parameters']['proportion hbonds']:.1f}%.")
    if nb_used_hbonds == 0:
        if comm.rank == 0:
            logging.error(f"\t{nb_used_hbonds} inter residues atoms hydrogen bonds remaining, analysis stopped.")
            sys.exit(1)
    if comm.rank == 0:
        logging.info(f"\t{nb_used_hbonds} inter residues atoms hydrogen bonds used.")
    ordered_columns = sort_hbonds(data, pattern)
    tmp_df = pd.DataFrame(data, index=[0])
    tmp_df = tmp_df[ordered_columns]
    df = tmp_df.transpose()
    df = df.reset_index()
    df.columns = ["hydrogen bonds", "median distances"]
    return df


def hbonds_csv(df, out_dir, smp, pattern):
    """
    Get the median distances for the hydrogen bonds in the molecular dynamics.

    :param df: the hbonds dataframe.
    :type df: pd.DataFrame
    :param out_dir: the directory output path.
    :type out_dir: str
    :param smp: the sample name.
    :type smp: str
    :param pattern: the donor/acceptor pattern.
    :type pattern: re.pattern
    :return: the dataframe of the hbonds' statistics.
    :rtype: pandas.DataFrame
    """
    data = {"contact": [],
            "donor position": [],
            "donor residue": [],
            "acceptor position": [],
            "acceptor residue": [],
            "median distance": []}

    for _, contact in df.iterrows():
        match = pattern.search(contact["hydrogen bonds"])
        if match:
            data["contact"].append(contact["hydrogen bonds"])
            data["donor position"].append(int(match.group(2)))
            data["acceptor position"].append(int(match.group(5)))
            data["donor residue"].append(match.group(1))
            data["acceptor residue"].append(match.group(4))
        else:
            data["contact"].append(contact["hydrogen bonds"])
            data["donor position"].append(f"no match with {pattern.pattern}")
            data["donor residue"].append(f"no match with {pattern.pattern}")
            data["acceptor position"].append(f"no match with {pattern.pattern}")
            data["acceptor_residue"].append(f"no match with {pattern.pattern}")
        data["median distance"].append(contact["median distances"])
    hbonds_stat = pd.DataFrame(data)
    out_path = os.path.join(out_dir, f"hbonds_by_residue_{smp.replace(' ', '_')}.csv")
    hbonds_stat.to_csv(out_path, index=False)
    if comm.rank == 0:
        logging.info(f"Contacts by residue CSV file saved: {os.path.abspath(out_path)}")

    return hbonds_stat


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From molecular dynamics trajectories files (*.nc), the script performs a trajectory analysis to search contacts.
    It looks for the hydrogen bonds between the atoms of two different residues.

    An hydrogen bond is defined as A-HD, where A is the acceptor heavy atom, H is the hydrogen and D is the donor heavy
    atom. An hydrogen bond is formed when A to D distance < distance cutoff and A-H-D angle > angle cutoff.
    A contact is valid if the number of frames (defined by the user with --frames or on the whole data) where a contact
    is produced between 2 atoms is greater or equal to the proportion threshold of hydrogen bonds.

    The hydrogen bonds are represented as 2 CSV files:
        - the median of the frames hydrogen bonds.
        - the hydrogen bonds median distance by residue.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-s", "--sample", required=True, type=str, help="the sample ID used for the files name.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-n", "--nanoseconds", required=True, type=int,
                        help="the molecular dynamics simulation time in nano seconds.")
    parser.add_argument("-f", "--frames", required=False, type=str,
                        help="the frames selection by trajectory file. The arguments should be <TRAJ_FILE>:100-1000. "
                             "If the <TRAJ_FILE> contains 2000 frames, only the frames from 100-1000 will be selected."
                             "Multiple frames selections can be performed with comma separators, i.e: "
                             "'<TRAJ_FILE_1>:100-1000,<TRAJ_FILE_2>:1-500'. If a '*' is used as '<TRAJ_FILE_1>:100-*', "
                             "the used frames will be the 100th until the end frame.")
    parser.add_argument("-d", "--distance-hbonds", required=False, type=restricted_positive, default=3.0,
                        help="An hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. An hydrogen bond is formed when A to D distance < distance. Default is "
                             "3.0 Angstroms.")
    parser.add_argument("-a", "--angle-cutoff", required=False, type=restricted_angle, default=135,
                        help="Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. One condition to form an hydrogen bond is A-H-D angle > angle cut-off. "
                             "Default is 135 degrees.")
    parser.add_argument("-p", "--proportion-hbonds", required=False, type=restricted_float, default=20.0,
                        help="the minimal percentage of frames which make contact between 2 atoms of different "
                             "residues in the selected frame of the molecular dynamics simulation, default is 20%%.")
    parser.add_argument("-r", "--resume", required=False, type=str,
                        help="the YAML file path of the previous trajectory analysis. The analysis of the new "
                             "trajectory files of the same system will resume on the previous trajectory analysis. The "
                             "new trajectory file analysis will be added to the previous data.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("inputs", nargs="+", type=str, help="the paths to the molecular dynamics trajectory files.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # create output directory if necessary
        os.makedirs(args.out, exist_ok=True)

        # create the logger
        if args.log:
            log_path = args.log
        else:
            log_path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
        create_log(log_path, args.log_level)

        logging.info(f"version: {__version__}")
        logging.info(f"CMD: {' '.join(sys.argv)}")
        logging.info(f"Atoms maximal hydrogen bonds distance threshold: {args.distance_hbonds:>7} \u212B")
        logging.info(f"Angle minimal cut-off: {args.angle_cutoff:>27}°")
        logging.info(f"Minimal frames proportion with atoms hydrogen bonds: {args.proportion_hbonds:.1f}%")
        logging.info(f"Molecular Dynamics duration: {args.nanoseconds:>19} ns")

    frames_selection = None
    try:
        frames_selection = parse_frames(args.frames, args.inputs)
    except argparse.ArgumentTypeError as ex:
        logging.error(ex, exc_info=True)
        sys.exit(1)

    data_traj = None
    traj_files = None
    try:
        data_traj, skipped_traj = resume_or_initialize_analysis(args.inputs, args.topology, args.sample,
                                                                args.distance_hbonds, args.angle_cutoff,
                                                                args.proportion_hbonds, args.nanoseconds,
                                                                args.resume, frames_selection)
        traj_files = remove_processed_trajectories(args.inputs, skipped_traj, args.resume)
    except KeyError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    trajectory = None
    for traj_file in traj_files:
        # load the trajectory
        if comm.rank == 0:
            logging.info(f"Processing trajectory file: {traj_file}")
        try:
            trajectory = load_trajectory(traj_file, args.topology, frames_selection)
            data_traj = check_trajectories_consistency(trajectory, traj_file, data_traj, frames_selection)
        except RuntimeError as exc:
            logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({', '.join(args.inputs)}) "
                          f"files exists", exc_info=True)
            sys.exit(1)
        except ValueError as exc:
            logging.error(exc, exc_info=True)
            sys.exit(1)
        except IndexError as exc:
            logging.error(exc, exc_info=True)
            sys.exit(1)

        # find the Hydrogen bonds
        data_traj = hydrogen_bonds(trajectory, data_traj, args.distance_hbonds, args.angle_cutoff)

        # pickle the analysis
        record_analysis(data_traj, args.out, traj_file, args.sample)

    # filter the hydrogen bonds
    filtered_hydrogen_bonds = None
    pattern_donor_acceptor = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    try:
        filtered_hydrogen_bonds = filter_hbonds(data_traj, pattern_donor_acceptor)
    except Exception as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)
    # write the CSV for the hbonds
    stats = hbonds_csv(filtered_hydrogen_bonds, args.out, args.sample, pattern_donor_acceptor)

    if comm.rank == 0:
        logging.info(f"{len(data_traj['trajectory files processed'])} processed trajectory files: "
                     f"{', '.join(data_traj['trajectory files processed'])}")
        logging.info(f"Whole trajectories memory size: {round(data_traj['size Gb'], 6):>17} Gb")
        logging.info(f"Whole trajectories frames: {data_traj['frames']:>16}")
        logging.info(f"Whole trajectories hydrogen bonds found: {len(filtered_hydrogen_bonds)}")
