#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "3.0.0"

import argparse
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
    """Inspect if an angle value is between 0 and 359.

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


def check_limits(frames):
    """Check if the selected frames are valid.

    :param frames: the limits of the frames to use.
    :type frames: str
    :raises ArgumentTypeError: values not in the fixed limits.
    :return: the frames limits.
    :rtype: dict
    """
    frames_lim = None
    pattern = re.compile("(\\d+)-(\\d+)")
    if frames:
        frames_lim = {}
        match = pattern.search(frames)
        if match:
            frames_lim["min"] = int(match.group(1))
            frames_lim["max"] = int(match.group(2))
        else:
            raise argparse.ArgumentTypeError(f"--frames {frames} is not a valid format, valid format should be: "
                                             f"--frames <DIGITS>-<DIGITS>")
    return frames_lim


def load_trajectories(trajectory_files, topology_file, frames=None):
    """
    Load the trajectory files and select frames if needed.

    :param trajectory_files: the trajectory file paths.
    :type trajectory_files: list
    :param topology_file: the topology file path.
    :type topology_file: str
    :param frames: the frames to use.
    :type frames: dict
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    logging.info("Loading trajectory files, please be patient..")
    traj = pt.iterload(trajectory_files, top=topology_file)
    logging.info(f"\tTotal trajectories frames:\t{traj.n_frames}")
    if frames:
        traj = traj[range(frames["min"], frames["max"])]
        logging.info(f"\tSelected frames:\t\t{frames['min']} to {frames['max']}")
    else:
        logging.info(f"\tSelected frames:\t\t1 to {traj.n_frames}")
    logging.info(f"\tUsed frames:\t\t\t{traj.n_frames}")
    logging.info(f"\tMolecules:\t\t\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t\t\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t\t\t{traj.topology.n_atoms}")
    return traj


def check_chunk_size(chunk, n_frames):
    """
    Modify the number of frames on which each step of the hydrogen bonds will be performed by each chunk in the case
    that the chunk size is greater than the number of frames of the trajectory.

    :param chunk: the chunk representing the number of frames selected by the user.
    :type chunk: int
    :param n_frames: the number of studied frames.
    :type n_frames: int
    :return: the chunk size modified if necessary
    :rtype: int
    """
    if chunk > n_frames:
        logging.warning(f"chunk size of {chunk} frames for hydrogen bonds search is greater than the {n_frames} frames "
                        f"of the trajectory , the chunk size is adjusted to {n_frames} frames.")
        chunk = n_frames
    return chunk


def record_analysis_parameters(out_dir, smp, traj, distance_contacts, angle_cutoff, proportion_contacts, frames_lim,
                               sim_time):
    """
    Record the analysis parameters in a YAML file.

    :param out_dir: the path to the output directory.
    :type out_dir: str
    :param smp: the sample name.
    :type smp: str
    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param distance_contacts: the threshold atoms distance in Angstroms for contacts.
    :type distance_contacts: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param proportion_contacts: the minimal percentage of contacts for atoms contacts of different residues in the
    selected frames.
    :type proportion_contacts: float
    :param frames_lim: the frames selected by the user.
    :type frames_lim: dict
    :param sim_time: the molecular dynamics simulation time.
    :type sim_time: str
    :type: str
    """
    parameters = {"maximal atoms distance": distance_contacts,
                  "angle cutoff": angle_cutoff,
                  "proportion contacts": proportion_contacts,
                  "frames": frames_lim,
                  "protein length": traj.topology.n_residues,
                  "sample": smp}
    if sim_time:
        parameters["MD duration"] = sim_time
    out = os.path.join(out_dir, f"{smp.replace(' ', '_')}_analysis_parameters.yaml")
    with open(out, "w") as file_handler:
        yaml.dump(parameters, file_handler)
    logging.info(f"Analysis parameters recorded: {out}")


def chunk_hbonds_distance(inspected_traj, atoms_dist, angle, steps):
    """
    Using chunks of the trajectory to save memory, extract the hydrogen bonds and add the distances values.

    :param inspected_traj: the trajectory.
    :type inspected_traj: pytraj.Trajectory
    :param atoms_dist: the threshold atoms distance in Angstroms for contacts.
    :type atoms_dist: float
    :param angle: the angle cutoff for the hydrogen bonds.
    :type angle: int
    :param steps: the step for each chunk.
    :type: int
    :return: the whole hydrogen bonds data.
    :rtype: dict
    """
    start_frame = 0
    data = {}
    # set the steps and check the case only one frame remains for the last step
    if steps == inspected_traj.n_frames:
        chunk_steps = [inspected_traj.n_frames]
    else:
        chunk_steps = list(range(steps - 1, inspected_traj.n_frames, steps))
        # add the last frame of the trajectory if the last range value do not correspond to the last frame, i.e:
        # 7 frames with a chunk of 2, the range produced is [2, 4, 6], we need to add 7 -> [2, 4, 6, 7]
        if chunk_steps[-1] < inspected_traj.n_frames - 1:
            chunk_steps.append(inspected_traj.n_frames - 1)
    # get all the hydrogen bonds
    for chunk_step in chunk_steps:
        logging.info(f"\tChunk on frames index {start_frame} to {chunk_step}.")
        frames_chunk = range(start_frame, chunk_step) if start_frame != chunk_step else [start_frame]
        # search hydrogen bonds with distance < atoms distance threshold and angle > angle cut-off.
        logging.debug(f"\t\tSearch for hydrogen bonds:")
        chunk_h_bonds = pt.search_hbonds(inspected_traj, distance=atoms_dist, angle=angle, frame_indices=frames_chunk)
        logging.debug(f"\t\t\t{len(chunk_h_bonds.donor_acceptor)} hydrogen bonds found.")
        # get the distances in the chunk
        logging.debug(f"\t\tGet the contacts distances in the chunk frames for the hydrogen bonds found.")
        chunk_distances = pt.distance(inspected_traj, chunk_h_bonds.get_amber_mask()[0], frame_indices=frames_chunk)
        # record the distances of all hydrogen bonds (donors-acceptors) detected in the chunk
        for idx in range(len(chunk_h_bonds.donor_acceptor)):
            donor_acceptor = chunk_h_bonds.donor_acceptor[idx]
            # filter the whole frames distances for this contact on the atoms contact distance threshold
            filtered_distances = chunk_distances[idx][chunk_distances[idx] <= atoms_dist]
            print(f"{donor_acceptor}:\t{len(chunk_distances[idx])} {len(filtered_distances)}")
            if donor_acceptor in data:
                data[donor_acceptor] = np.concatenate((data[donor_acceptor], filtered_distances))
            else:
                data[donor_acceptor] = filtered_distances
            np.set_printoptions(threshold=sys.maxsize)
        start_frame = chunk_step + 1

    logging.info(f"\t{len(data)} hydrogen bonds found.")
    return data


def filter_hbonds(raw_data, pattern, nb_selected_frames, percent_thr, txt_frames_selection):
    """
    Filter the hydrogen that not belongs to the same residue and which the number of frames where a hydrogen bond is
    formed (which residues atoms distance of contact is less than the distance threshold) is greater than the
    proportion threshold.

    :param raw_data: the whole hydrogen bonds data
    :type raw_data: dict
    :param pattern: the pattern to extract the residues positions of the atoms contacts.
    :type pattern: re.pattern
    :param nb_selected_frames: the number of selected frames for the analysis.
    :type nb_selected_frames: int
    :param percent_thr: the minimal percentage of contacts for atoms contacts of different residues in the selected
    frames.
    :type percent_thr: float
    :param txt_frames_selection: the description of the used frames.
    :type txt_frames_selection: str
    :return: the filtered hydrogen bonds.
    :rtype: dict
    """
    data = {}
    intra_residue_contacts = 0
    inter_residue_contacts_failed_thr = 0
    for donor_acceptor in raw_data:
        match = pattern.search(donor_acceptor)
        if match:
            if match.group(2) == match.group(5):
                intra_residue_contacts += 1
                logging.debug(f"\t[REJECTED INTRA] {donor_acceptor}: H bond between the atoms of the same residue.")
            else:
                # retrieve only the contacts < to the max atoms threshold and having a percentage of
                # frames >= percentage threshold in the selected frames
                pct_contacts = len(raw_data[donor_acceptor]) / nb_selected_frames * 100
                if pct_contacts >= percent_thr:
                    data[donor_acceptor] = statistics.median(raw_data[donor_acceptor])
                    logging.debug(f"\t[VALID {len(data)}] {donor_acceptor}: median atoms distance "
                                  f"{round(data[donor_acceptor], 2)} \u212B, proportion of valid frames "
                                  f"{pct_contacts:.1f}% ({len(raw_data[donor_acceptor])}/{nb_selected_frames} frames "
                                  f"with H bonds).")
                else:
                    inter_residue_contacts_failed_thr += 1
                    logging.debug(f"\t[REJECTED PERCENTAGE] {donor_acceptor}: frames with H bonds {pct_contacts:.1f}% "
                                  f"< {percent_thr:.1f}% threshold "
                                  f"({len(raw_data[donor_acceptor])}/{nb_selected_frames} frames with H bonds).")
    nb_used_contacts = len(raw_data) - intra_residue_contacts - inter_residue_contacts_failed_thr
    logging.info(f"\t{intra_residue_contacts}/{len(raw_data)} hydrogen bonds with intra residues atoms contacts "
                 f"discarded.")
    logging.info(f"\t{inter_residue_contacts_failed_thr}/{len(raw_data)} hydrogen bonds with inter residues atoms "
                 f"contacts discarded: contacts frames under the threshold of {percent_thr:.1f}% for "
                 f"{txt_frames_selection}.")
    if nb_used_contacts == 0:
        logging.error(f"\t{nb_used_contacts} inter residues atoms contacts remaining, analysis stopped.")
        sys.exit(1)
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
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


def search_hydrogen_bonds(traj, atoms_cutoff, angle_cutoff, out_dir, smp, pct_thr, pattern_hb, chunk, lim_frames=None):
    """
    Get the hydrogen bonds between the different atoms of the protein during the molecular dynamics simulation.
    Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is donor heavy atom. Hydrogen
    bond is formed when A to D distance < distance cutoff and A-HD angle > angle cutoff.

    :param traj: the trajectory.
    :type traj: pytraj.Trajectory
    :param atoms_cutoff: the threshold atoms distance in Angstroms for contacts.
    :type atoms_cutoff: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param out_dir: the output directory.
    :type out_dir: str
    :param smp: the sample name.
    :type smp: str
    :param pct_thr: the minimal percentage of contacts for atoms contacts of different residues in the selected frames.
    :type pct_thr: float
    :param pattern_hb: the pattern for the hydrogen bond name.
    :type pattern_hb: re.pattern
    :param lim_frames: the frames selected by the user.
    :type lim_frames: dict
    :return: the dataframe of the polar contacts.
    :rtype: pd.DataFrame
    """
    logging.info("Hydrogen bonds retrieval from the trajectories files, please be patient..")
    raw_hbonds = chunk_hbonds_distance(traj, atoms_cutoff, angle_cutoff, steps=chunk)
    logging.info("Filter the hydrogen bonds:")
    frames_txt = "the whole frames" if lim_frames is None else f"frames {lim_frames['min']} to {lim_frames['max']}"
    filtered_hbonds = filter_hbonds(raw_hbonds, pattern_hb, traj.n_frames, pct_thr, frames_txt)
    ordered_columns = sort_contacts(filtered_hbonds, pattern_hb)
    tmp_df = pd.DataFrame(filtered_hbonds, index=[0])
    tmp_df = tmp_df[ordered_columns]
    df = tmp_df.transpose()
    df = df.reset_index()
    df.columns = ["hydrogen bonds", "median distances"]
    out_path_bn = os.path.join(out_dir, f"median_h-bond_frames_{smp.replace(' ', '_')}")
    # write the CSV file
    path_csv = f"{out_path_bn}.csv"
    df.to_csv(f"{out_path_bn}.csv", index=False, header=["hydrogen bonds", "median distances (\u212B)"])
    logging.info(f"\tMedian contacts on {frames_txt if lim_frames else 'whole frames'} file saved: {path_csv}")

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
    :param pattern: the pattern for the contact.
    :type pattern: re.pattern
    :return: the dataframe of the contacts statistics.
    :rtype: pd.DataFrame
    """
    logging.info("Inter-residues atoms contacts:")
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
    logging.info(f"\tcontacts by residue CSV file saved: {out_path}")

    return contacts_stat


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectories files, the script performs a trajectory analysis to search contacts. The 
    script looks for the hydrogen bonds between the atoms of two different residues. 

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
    parser.add_argument("-d", "--distance-contacts", required=False, type=restricted_positive, default=3.0,
                        help="An hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. An hydrogen bond is formed when A to D distance < distance. "
                             "Default is 3.0 Angstroms.")
    parser.add_argument("-a", "--angle-cutoff", required=False, type=restricted_angle, default=135,
                        help="Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is "
                             "donor heavy atom. One condition to form an hydrogen bond is A-H-D angle > angle cutoff. "
                             "Default is 135 degrees.")
    parser.add_argument("-x", "--frames", required=False, type=str,
                        help="the frames to load from the trajectory, the format must be two integers separated by "
                             "an hyphen, i.e to load the trajectory from the frame 500 to 2000: --frames 500-2000")
    parser.add_argument("-c", "--chunk", required=False, type=int, default=1000,
                        help="to save memory, the chunk size of the trajectory frames on which a search for hydrogen "
                             "bonds will be performed. The chunk size must be less than the difference of max selected "
                             "frame and min of the selected frame if the argument --frame is used or the total of "
                             "frames if this argument is not used. Otherwise, the chunk will be reduced to the total of"
                             "frames of the trajectory and only one chunk will be produced.")
    parser.add_argument("-p", "--proportion-contacts", required=False, type=restricted_float, default=20.0,
                        help="the minimal percentage of frames which make contact between 2 atoms of different "
                             "residues in the selected frame of the molecular dynamics simulation, default is 20%%.")
    parser.add_argument("-m", "--md-time", required=False, type=str,
                        help="the molecular dynamics simulation time as free text.")
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
    logging.info(f"atoms maximal contacts distance threshold: {args.distance_contacts} \u212B")
    logging.info(f"minimal proportion of frames with atoms contacts between two different residues in the selected "
                 f"frames of the molecular dynamics: {args.proportion_contacts:.1f}%")
    try:
        frames_limits = check_limits(args.frames)
    except argparse.ArgumentTypeError as exc:
        logging.error(exc)
        sys.exit(1)

    # load the trajectory
    try:
        trajectory = load_trajectories(args.inputs, args.topology, frames_limits)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({', '.join(args.inputs)}) files "
                      f"exists", exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({', '.join(args.inputs)}) files "
                      f"exists.", exc_info=True)
        sys.exit(1)

    # check if the chunk size is less than the number of frames of the trajectory
    chunk_size = check_chunk_size(args.chunk, trajectory.n_frames)

    # record the analysis parameter in a yaml file
    record_analysis_parameters(args.out, args.sample, trajectory, args.distance_contacts, args.angle_cutoff,
                               args.proportion_contacts, frames_limits, args.md_time)

    # find the Hydrogen bonds
    pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    hydrogen_bonds = search_hydrogen_bonds(trajectory, args.distance_contacts, args.angle_cutoff, args.out, args.sample,
                                           args.proportion_contacts, pattern_contact, chunk_size, frames_limits)

    # write the CSV for the contacts
    stats = contacts_csv(hydrogen_bonds, args.out, args.sample, pattern_contact)
