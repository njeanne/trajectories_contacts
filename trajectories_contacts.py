#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "3.0.0"

import argparse
import copy
import logging
import numpy as np
import os
import re
import statistics
import sys
import yaml

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
    :param level: the level og the log.
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
        log_level = log_level_dict[args.log_level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def parse_frames(frames_selections, traj_files_paths):
    """
    Parse the frames selection by trajectory file.

    :param frames_selections: the frames selection by trajectory file.
    :type frames_selections: str
    :param traj_files_paths: the trajectory files paths.
    :type traj_files_paths: list
    :return: the selected frames by trajectory file.
    :rtype: dict
    """
    data = {}
    pattern = re.compile("(.+):(\\d+)-(\\d+)")

    if frames_selections:
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
                data[current_traj] = {"begin": int(start), "end": int(end)}
            else:
                raise argparse.ArgumentTypeError(f"The frame selection part '{frames_sel}' do not match the correct "
                                                 f"pattern {pattern.pattern}'")
        logging.info("Frames selection on input trajectory files:")
        for current_traj in data:
            logging.info(f"\t{current_traj}: frames selection from {data[current_traj]['begin']} to "
                         f"{data[current_traj]['end']}.")
    return data


def resume_or_initialize_analysis(trajectory_files, topology_file, smp, distance_contacts, angle_cutoff,
                                  proportion_contacts, sim_time, resume_path, frames_sel):
    """
    Load the previous analysis data or create a new one if no previous analysis path was performed.

    :param trajectory_files: the current analysis trajectory files path.
    :type trajectory_files: list
    :param topology_file: the trajectories topology file.
    :type topology_file: str
    :param smp: the sample name.
    :type smp: str
    :param distance_contacts: the threshold atoms distance in Angstroms for contacts.
    :type distance_contacts: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param proportion_contacts: the minimal percentage of contacts for atoms contacts of different residues in the
    selected frames.
    :type proportion_contacts: float
    :param sim_time: the molecular dynamics simulation time in ns.
    :type sim_time: int
    :param resume_path: the path to the YAML file of previous analysis.
    :type resume_path: str
    :param frames_sel: the frames selection for new trajectory files.
    :type frames_sel: dict
    :return: the initialized or resumed analysis data.
    :rtype: dict
    """
    if resume_path:
        logging.info(f"resumed analysis from YAML file: {resume_path}")
        with open(resume_path, "r") as file_handler:
            data = yaml.safe_load(file_handler.read())

        discrepancies = []
        if data["sample"] != smp:
            discrepancies.append(f"discrepancy in --sample, current analysis is {smp}, previous analysis was "
                                 f"{data['sample']}")
        if data["topology file"] != os.path.basename(topology_file):
            discrepancies.append(f"discrepancy in --topology, current analysis is {topology_file}, previous analysis "
                                 f"was {data['topology file']}")
        if data["parameters"]["maximal atoms distance"] != distance_contacts:
            discrepancies.append(f"discrepancy in --distance-contacts, current analysis is {distance_contacts}, "
                                 f"previous analysis was {data['parameters']['maximal atoms distance']}")
        if data["parameters"]["angle cutoff"] != angle_cutoff:
            discrepancies.append(f"discrepancy in --angle-cutoff, current analysis is {angle_cutoff}, previous "
                                 f"analysis was {data['parameters']['angle cutoff']}")
        if data["parameters"]["proportion contacts"] != proportion_contacts:
            discrepancies.append(f"discrepancy in --proportion-contacts, current analysis is {proportion_contacts}, "
                                 f"previous analysis was {data['parameters']['proportion contacts']}")
        if discrepancies:
            discrepancies_txt = None
            for item in discrepancies:
                if discrepancies_txt:
                    discrepancies_txt = f"{discrepancies_txt}; {item}"
                else:
                    discrepancies_txt = item
                discrepancies_txt = f"{discrepancies_txt}. Check {resume_path}"
            raise KeyError(discrepancies_txt)

        # cast lists to numpy arrays
        for h_bond in data["H bonds"]:
            data["H bonds"][h_bond] = np.array(data["H bonds"][h_bond])
        # add the new files path
        data["trajectory files"] = data["trajectory files"] + sorted(
            [os.path.basename(t_file) for t_file in trajectory_files])
        # add frames selection in new trajectory files
        if frames_sel:
            if "frames selections" in data["parameters"]:
                for traj_fn in frames_sel:
                    data["parameters"]["frames selections"][traj_fn] = frames_sel[traj_fn]
            else:
                data["parameters"]["frames selections"] = frames_sel
    else:
        data = {"sample": smp, "size Gb": 0, "frames": 0,
                "parameters": {"maximal atoms distance": distance_contacts, "angle cutoff": angle_cutoff,
                               "proportion contacts": proportion_contacts},
                "trajectory files": sorted([os.path.basename(t_file) for t_file in trajectory_files]),
                "topology file": os.path.basename(topology_file), "H bonds": {}}
    # set the simulation time
    data["parameters"]["time"] = f"{sim_time} ns"
    return data


def load_trajectory(trajectory_file, topology_file, frames_sel):
    """
    Load the trajectory file.

    :param trajectory_file: the trajectory file path.
    :type trajectory_file: str
    :param topology_file: the topology file path.
    :type topology_file: str
    :param frames_sel: the frames selection for new trajectory files.
    :type frames_sel: dict
    :return: the loaded trajectory.
    :rtype: pytraj.Trajectory
    """
    logging.info(f"\tLoading trajectory file, please be patient..")
    traj = pt.iterload(trajectory_file, top=topology_file, )
    logging.info(f"\t\tMolecules:{traj.topology.n_mols:>20}")
    logging.info(f"\t\tResidues:{traj.topology.n_residues:>22}")
    logging.info(f"\t\tAtoms:{traj.topology.n_atoms:>27}")
    logging.info(f"\t\tTrajectory total frames:{traj.n_frames:>7}")
    logging.info(f"\t\tTrajectory memory size:{round(traj._estimated_GB, 6):>14} Gb")
    if os.path.basename(trajectory_file) in frames_selection:
        traj_bn = os.path.basename(trajectory_file)
        if frames_selection[traj_bn]["end"] > traj.n_frames:
            raise IndexError(f"Selected upper frame limit for {traj_bn} ({frames_selection[traj_bn]['end']}) from "
                             f"--frames argument is greater than the total frames number ({traj.n_frames}) of the MD "
                             f"trajectory.")
        frames_range = range(frames_selection[traj_bn]["begin"], frames_selection[traj_bn]["end"])
        if frames_selection[traj_bn]["begin"] == 1:
            frames_range[0] = 0
        traj = traj[frames_range]
        logging.info(f"\t\tSelected frames:{frames_selection[traj_bn]['begin']:>14} to "
                     f"{frames_selection[traj_bn]['end']}")
        logging.info(f"\t\tSelected frames memory size:{round(traj._estimated_GB, 6):>9} GB")
    else:
        txt = f"1 to {traj.n_frames}"
        logging.info(f"\t\tSelected frames:{txt:>20}")
    return traj


def check_trajectories_consistency(traj, path, data):
    """
    Check if the trajectory attributes match with the previous trajectories.

    :param traj: the current trajectory.
    :type traj: pytraj.Trajectory
    :param path: the current trajectory path.
    :type path: str
    :param data: the trajectories data.
    :type data: dict
    :return: the updated trajectories data.
    :rtype: dict
    """
    if "residues" not in data:
        data["residues"] = traj.topology.n_residues
        data["atoms"] = traj.topology.n_atoms
        data["molecules"] = traj.topology.n_mols
    else:
        if data_traj["residues"] != traj.topology.n_residues:
            logging.error(f"the residues number ({traj.topology.n_residues}) is different from the residues number of "
                          f"the previous trajectories ({data_traj['residues']}), check if {os.path.basename(path)} is "
                          f"from the same trajectory than the previous ones.")
            sys.exit(1)
        if data_traj["atoms"] != traj.topology.n_atoms:
            logging.error(f"the atoms number ({traj.topology.n_atoms}) is different from the atoms number of "
                          f"the previous trajectories ({data_traj['atoms']}), check if {os.path.basename(path)} is "
                          f"from the same trajectory than the previous ones.")
            sys.exit(1)
        if data_traj["molecules"] != traj.topology.n_mols:
            logging.error(f"the molecules number ({traj.topology.n_mols}) is different from the molecules number of "
                          f"the previous trajectories ({data_traj['molecules']}), check if {os.path.basename(path)} is "
                          f"from the same trajectory than the previous ones.")
            sys.exit(1)
    data["size Gb"] += traj._estimated_GB
    data["frames"] += traj.n_frames
    return data


def hydrogen_bonds(inspected_traj, data, atoms_dist, angle):
    """
    Extract the hydrogen bonds and add the distances values.

    :param inspected_traj: the trajectory.
    :type inspected_traj: pytraj.Trajectory
    :param data: the trajectories data.
    :type data: dict
    :param atoms_dist: the threshold atoms distance in Angstroms for contacts.
    :type atoms_dist: float
    :param angle: the angle cutoff for the hydrogen bonds.
    :type angle: int
    :return: the updated trajectories data.
    :rtype: dict
    """
    logging.info("\tsearch for hydrogen bonds:")
    # search hydrogen bonds with distance < atoms distance threshold and angle > angle cut-off.
    h_bonds = pt.search_hbonds(inspected_traj, distance=atoms_dist, angle=angle)
    # get the distances
    distances = pt.distance(inspected_traj, h_bonds.get_amber_mask()[0])
    # record the distances of all hydrogen bonds (donors-acceptors) detected in the chunk
    for idx in range(len(h_bonds.donor_acceptor)):
        donor_acceptor = h_bonds.donor_acceptor[idx]
        # filter the whole frames distances for this contact on the atoms contact distance threshold
        filtered_distances = distances[idx][distances[idx] <= atoms_dist]
        if donor_acceptor in data["H bonds"]:
            data["H bonds"][donor_acceptor] = np.concatenate((data["H bonds"][donor_acceptor], filtered_distances))
        else:
            data["H bonds"][donor_acceptor] = filtered_distances
    logging.info(f"\t\t{len(data['H bonds'])} hydrogen bonds found in the {inspected_traj.n_frames} frames of the "
                 f"trajectory.")
    return data


def sort_contacts(contact_names, pattern):
    """
    Get the order of the contacts on the first residue then on the second one.

    :param contact_names: the contacts identifiers.
    :type contact_names: KeysView[Union[str, Any]]
    :param pattern: the pattern to extract the residues positions of the atoms contacts.
    :type pattern: re.pattern
    :return: the ordered list of contacts.
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
    Filter out the hydrogen contacts that belongs to the same residue.

    :param analysis_data: the whole analysis data
    :type analysis_data: dict
    :param pattern: the donor/acceptor pattern.
    :type pattern: re.pattern
    :return: the filtered hydrogen bonds.
    :rtype: pandas.DataFrame
    """
    logging.info(f"Filtering {len(analysis_data['trajectory files'])} trajector"
                 f"{'ies' if len(analysis_data['trajectory files']) > 1 else 'y'} file"
                 f"{'s' if len(analysis_data['trajectory files']) > 1 else ''} with {len(analysis_data['H bonds'])} "
                 f"hydrogen bonds:")
    intra_residue_contacts = 0
    inter_residue_contacts_failed_thr = 0
    data = {}
    for donor_acceptor, distances in analysis_data["H bonds"].items():
        match = pattern.search(donor_acceptor)
        if match:
            if match.group(2) == match.group(5):
                intra_residue_contacts += 1
                logging.debug(f"\t[REJECTED INTRA] {donor_acceptor}: H bond between the atoms of the same residue.")
            else:
                # retrieve only the contacts < to the max atoms threshold and having a percentage of
                # frames >= percentage threshold in the selected frames
                pct_contacts = len(distances) / analysis_data["frames"] * 100
                if pct_contacts >= analysis_data["parameters"]["proportion contacts"]:
                    data[donor_acceptor] = statistics.median(distances)
                    logging.debug(f"\t[VALID {len(data)}] {donor_acceptor}: median atoms distance "
                                  f"{round(data[donor_acceptor], 2)} \u212B, proportion of valid frames "
                                  f"{pct_contacts:.1f}% ({len(distances)}/{analysis_data['frames']} frames with H "
                                  f"bonds).")
                else:
                    inter_residue_contacts_failed_thr += 1
                    logging.debug(f"\t[REJECTED PERCENTAGE] {donor_acceptor}: frames with H bonds "
                                  f"{pct_contacts:.1f}% < {analysis_data['parameters']['proportion contacts']:.1f}% "
                                  f"threshold ({len(distances)}/{analysis_data['frames']} frames with H bonds).")
        else:
            logging.error(f"no match for pattern '{pattern.pattern}' in donor/acceptor '{donor_acceptor}'")
            sys.exit(1)
    nb_used_contacts = len(analysis_data["H bonds"]) - intra_residue_contacts - inter_residue_contacts_failed_thr
    logging.info(f"\t{intra_residue_contacts}/{len(analysis_data['H bonds'])} hydrogen bonds with intra residues "
                 f"atoms contacts discarded.")
    logging.info(f"\t{inter_residue_contacts_failed_thr}/{len(analysis_data['H bonds'])} hydrogen bonds with inter "
                 f"residues atoms contacts discarded with contacts frames proportion under the threshold of "
                 f"{analysis_data['parameters']['proportion contacts']:.1f}%.")
    if nb_used_contacts == 0:
        logging.error(f"\t{nb_used_contacts} inter residues atoms contacts remaining, analysis stopped.")
        sys.exit(1)
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
    ordered_columns = sort_contacts(data, pattern)
    tmp_df = pd.DataFrame(data, index=[0])
    tmp_df = tmp_df[ordered_columns]
    df = tmp_df.transpose()
    df = df.reset_index()
    df.columns = ["hydrogen bonds", "median distances"]
    return df


def contacts_csv(df, out_dir, smp, pattern):
    """
    Get the median distances for the contacts in the molecular dynamics.

    :param df: the contacts dataframe.
    :type df: pd.DataFrame
    :param out_dir: the directory output path.
    :type out_dir: str
    :param smp: the sample name.
    :type smp: str
    :param pattern: the donor/acceptor pattern.
    :type pattern: re.pattern
    :return: the dataframe of the contacts statistics.
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
    contacts_stat = pd.DataFrame(data)
    out_path = os.path.join(out_dir, f"contacts_by_residue_{smp.replace(' ', '_')}.csv")
    contacts_stat.to_csv(out_path, index=False)
    logging.info(f"Contacts by residue CSV file saved: {os.path.abspath(out_path)}")

    return contacts_stat


def record_analysis_yaml(out_dir, smp, data):
    """
    Record the analysis in a YAML file.

    :param out_dir: the path to the output directory.
    :type out_dir: str
    :param smp: the sample name.
    :type smp: str
    :param data: the trajectory analysis data.
    :type data: dict
    """
    out = os.path.join(out_dir, f"{smp.replace(' ', '_')}_analysis.yaml")
    tmp = copy.copy(data)
    for h_bond in tmp["H bonds"]:
        tmp["H bonds"][h_bond] = tmp["H bonds"][h_bond].tolist()
    with open(out, "w") as file_handler:
        yaml.dump(tmp, file_handler)
    logging.info(f"Analysis parameter YAML file: {os.path.abspath(out)}")


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
    is produced between 2 atoms is greater or equal to the proportion threshold of contacts.

    The hydrogen bonds are represented as 2 CSV files:
        - the median of the frames contacts.
        - the contacts median distance by residue.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-s", "--sample", required=True, type=str,
                        help="the sample ID used for the files name.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-n", "--nanoseconds", required=True, type=int,
                        help="the molecular dynamics simulation time in nano seconds.")
    parser.add_argument("-f", "--frames", required=False, type=str,
                        help="the frames selection by trajectory file. The arguments should be <TRAJ_FILE>:100-1000. "
                             "If the <TRAJ_FILE> contains 1000 frames, only the frames from 100-1000 will be selected."
                             "Multiple frames selections can be performed with comma separators, i.e: "
                             "'<TRAJ_FILE_1>:100-1000,<TRAJ_FILE_2>:1-500'")
    parser.add_argument("-d", "--distance-contacts", required=False, type=restricted_positive, default=3.0,
                        help="An hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. An hydrogen bond is formed when A to D distance < distance. Default is "
                             "3.0 Angstroms.")
    parser.add_argument("-a", "--angle-cutoff", required=False, type=restricted_angle, default=135,
                        help="Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. One condition to form an hydrogen bond is A-H-D angle > angle cut-off. "
                             "Default is 135 degrees.")
    parser.add_argument("-p", "--proportion-contacts", required=False, type=restricted_float, default=20.0,
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
    logging.info(f"Atoms maximal contacts distance threshold: {args.distance_contacts:>7} \u212B")
    logging.info(f"Angle minimal cut-off: {args.angle_cutoff:>27}°")
    logging.info(f"Minimal frames proportion with atoms contacts: {args.proportion_contacts:.1f}%")

    try:
        frames_selection = parse_frames(args.frames, args.inputs)
    except argparse.ArgumentTypeError as ex:
        logging.error(ex, exc_info=True)
        sys.exit(1)

    try:
        data_traj = resume_or_initialize_analysis(args.inputs, args.topology, args.sample, args.distance_contacts,
                                                  args.angle_cutoff, args.proportion_contacts, args.nanoseconds,
                                                  args.resume, frames_selection)
    except KeyError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    for traj_file in args.inputs:
        # load the trajectory
        logging.info(f"Processing trajectory file: {os.path.splitext(os.path.basename(traj_file))[0]}")
        try:
            trajectory = load_trajectory(traj_file, args.topology, frames_selection)
            data_traj = check_trajectories_consistency(trajectory, traj_file, data_traj)
        except RuntimeError as exc:
            logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({', '.join(args.inputs)}) "
                          f"files exists", exc_info=True)
            sys.exit(1)
        except ValueError as exc:
            logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({', '.join(args.inputs)}) "
                          f"files exists.", exc_info=True)
            sys.exit(1)
        except IndexError as exc:
            logging.error(exc, exc_info=True)
            sys.exit(1)

        # find the Hydrogen bonds
        data_traj = hydrogen_bonds(trajectory, data_traj, args.distance_contacts, args.angle_cutoff)

    # filter the hydrogen bonds
    pattern_donor_acceptor = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    filtered_hydrogen_bonds = filter_hbonds(data_traj, pattern_donor_acceptor)
    # write the CSV for the contacts
    stats = contacts_csv(filtered_hydrogen_bonds, args.out, args.sample, pattern_donor_acceptor)

    logging.info(f"{len(data_traj['trajectory files'])} processed trajectory files: "
                 f"{', '.join(data_traj['trajectory files'])}")
    logging.info(f"Whole trajectories memory size: {round(data_traj['size Gb'], 6):>17} Gb")
    logging.info(f"Whole trajectories frames: {data_traj['frames']:>16}")
    logging.info(f"Whole trajectories hydrogen bonds found: {len(filtered_hydrogen_bonds)}")

    # record the analysis parameter in a yaml file
    record_analysis_yaml(args.out, args.sample, data_traj)
