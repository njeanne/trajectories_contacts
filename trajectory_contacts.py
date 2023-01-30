#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "3.0.0"

import argparse
import gzip
import logging
import os
import re
import shutil
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


def load_trajectory(trajectory_file, topology_file, frames=None):
    """
    Load a trajectory and apply a mask if mask argument is set.

    :param trajectory_file: the trajectory file path.
    :type trajectory_file: str
    :param topology_file: the topology file path.
    :type topology_file: str
    :param frames: the frames to use.
    :type frames: str
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    logging.info("Loading trajectory file, please be patient..")
    frame_indices = None
    if frames:
        frames_split = frames.split("-")
        frame_indices = range(int(frames_split[0]), int(frames_split[1]))

    traj = pt.load(trajectory_file, top=topology_file, frame_indices=frame_indices)

    if frames:
        logging.info(f"\tFrames used:\t{frame_indices[0]} to {frame_indices[-1] + 1}")
    else:
        logging.info(f"\tFrames used:\t1 to {traj.n_frames}")
    logging.info(f"\tFrames total:\t{traj.n_frames}")
    logging.info(f"\tMolecules:\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t{traj.topology.n_atoms}")
    return traj


def record_analysis_parameters(out_dir, bn, traj, distance_contacts, angle_cutoff, proportion_contacts, frames_lim,
                               smp, sim_time):
    """
    Record the analysis parameters in a YAML file.

    :param out_dir: the path to the output directory.
    :type out_dir: str
    :param bn: the basename of the file.
    :type bn: str
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
    :param smp: the sample name.
    :type smp: str
    :param sim_time: the molecular dynamics simulation time.
    :type sim_time: str
    :type: str
    """
    parameters = {"maximal atoms distance": distance_contacts,
                  "angle cutoff": angle_cutoff,
                  "proportion contacts": proportion_contacts,
                  "frames": frames_lim,
                  "protein length": traj.topology.n_residues}
    if smp:
        parameters["sample"] = smp
    else:
        parameters["sample"] = bn.replace("_", " ")
    if sim_time:
        parameters["MD duration"] = sim_time
    out = os.path.join(out_dir, f"{bn}_analysis_parameters.yaml")
    with open(out, "w") as file_handler:
        yaml.dump(parameters, file_handler)
    logging.info(f"Analysis parameters recorded: {out}")


def sort_contacts(contact_names, pattern):
    """
    Get the order of the contacts on the first residue then on the second one.

    :param contact_names: the contacts identifiers.
    :type contact_names: KeysView[Union[str, Any]]
    :param pattern: the regex pattern to extract the residues positions of the atoms contacts.
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


def hydrogen_bonds(traj, atoms_dist_thr, angle_cutoff, pct_thr, pattern_hb, out_dir, out_basename, lim_frames=None):
    """
    Get the hydrogen bonds between the different atoms of the protein during the molecular dynamics simulation.
    Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is donor heavy atom. Hydrogen
    bond is formed when A to D distance < distance cutoff and A-HD angle > angle cutoff.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param atoms_dist_thr: the threshold atoms distance in Angstroms for contacts.
    :type atoms_dist_thr: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param pct_thr: the minimal percentage of contacts for atoms contacts of different residues in the
    selected frames.
    :type pct_thr: float
    :param pattern_hb: the pattern for the hydrogen bond name.
    :type pattern_hb: re.pattern
    :param out_dir: the output directory.
    :type out_dir: str
    :param out_basename: the basename for the output CSV file.
    :type out_basename: str
    :param lim_frames: the frames selected by the user.
    :type lim_frames: dict
    :return: the dataframe of the polar contacts.
    :rtype: pd.DataFrame
    """
    logging.info("Hydrogen bonds retrieval from the trajectory file, please be patient..")
    # search hydrogen bonds with distance < atoms distance threshold and angle > angle cut-off. Return the H bonds by
    # donor-acceptor with a dataset of frames, 0 when the contacts do not pass a threshold, 1 when they pass all the
    # thresholds, in ex: [0 0 1 0 1 ... 1 0 0 1]
    h_bonds = pt.hbond(traj, distance=atoms_dist_thr, angle=angle_cutoff)
    nb_total_contacts = len(h_bonds.data) - 1
    # from the get the distances of of all hydrogen bonds (donors-acceptors) detected
    distances_hbonds = pt.distance(traj, h_bonds.get_amber_mask()[0])
    frames_txt = "the whole frames" if lim_frames is None else f"frames {lim_frames['min']} to {lim_frames['max']}"
    logging.info(f"Search for inter-residues polar contacts in {nb_total_contacts} total polar contacts:")

    nb_intra_residue_contacts = 0
    nb_frames_contacts_selected_interval = 0
    data_hydrogen_bonds = {}
    idx = 0
    for h_bond in h_bonds.data:
        if h_bond.key != "total_solute_hbonds":
            match = pattern_hb.search(h_bond.key)
            if match:
                if match.group(2) == match.group(5):
                    nb_intra_residue_contacts += 1
                    logging.debug(f"\t {h_bond.key}: atoms contact from same residue, contact skipped")
                else:
                    # retrieve only the contacts < to the max atoms threshold and having
                    # a percentage of frames >= percentage threshold in the selected frames
                    pct_contacts = len(distances_hbonds[idx][distances_hbonds[idx] <= atoms_dist_thr]) / len(
                        distances_hbonds[idx]) * 100
                    if pct_contacts >= pct_thr:
                        data_hydrogen_bonds[h_bond.key] = statistics.median(distances_hbonds[idx])
                    else:
                        nb_frames_contacts_selected_interval += 1
                        logging.debug(f"\t {h_bond.key}: {pct_contacts:.1f}% of the frames with contacts under the "
                                      f"threshold of {pct_thr:.1f}% in {frames_txt}, contact skipped.")
            idx += 1
    nb_used_contacts = nb_total_contacts - nb_intra_residue_contacts - nb_frames_contacts_selected_interval
    logging.info(f"\t{nb_intra_residue_contacts}/{nb_total_contacts} intra residues atoms contacts discarded.")
    logging.info(f"\t{nb_frames_contacts_selected_interval}/{nb_total_contacts} inter residues atoms contacts "
                 f"discarded with number of contacts frames under the threshold of {pct_thr:.1f}% for {frames_txt}.")
    if nb_used_contacts == 0:
        logging.error(f"\t{nb_used_contacts} inter residues atoms contacts remaining, analysis stopped.")
        sys.exit(1)
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
    ordered_columns = sort_contacts(data_hydrogen_bonds.keys(), pattern_hb)
    tmp_df = pd.DataFrame(data_hydrogen_bonds, index=[0])
    tmp_df = tmp_df[ordered_columns]
    df = tmp_df.transpose()
    df = df.reset_index()
    df.columns = ["hydrogen bonds", "median distances"]
    out_path_bn = os.path.join(out_dir, f"median_h-bond_frames_{out_basename}")
    # write the CSV file
    path_csv = f"{out_path_bn}.csv"
    df.to_csv(f"{out_path_bn}.csv", index=False, header=["hydrogen bonds", "median distances (\u212B)"])
    logging.info(f"\tMedian contacts on {frames_txt if lim_frames else 'whole frames'} file saved: {path_csv}")

    return df


def contacts_csv(df, out_dir, out_basename, pattern):
    """
    Get the median distances for the contacts in the molecular dynamics.

    :param df: the contacts dataframe.
    :type df: pd.DataFrame
    :param out_dir: the directory output path.
    :type out_dir: str
    :param out_basename: the basename.
    :type out_basename: str
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
    out_path = os.path.join(out_dir, f"contacts_by_residue_{out_basename}.csv")
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

    From a molecular dynamics trajectory file, the script performs a trajectory analysis to search contacts. The script 
    looks for the hydrogen bonds between the atoms of two different residues. 

    An hydrogen bond is defined as A-HD, where A is the acceptor heavy atom, H is the hydrogen and D is the donor heavy 
    atom. An hydrogen bond is formed when A to D distance < distance cutoff and A-H-D angle > angle cutoff.
    A contact is valid if the number of frames (defined by the user with --frames or on the whole data) where a contact 
    is produced between 2 atoms is greater or equal to the proportion threshold of contacts.

    The hydrogen bonds are represented as 2 CSV files:
        - the contacts by frame (compressed file).
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
    parser.add_argument("input", type=str, help="the path to the molecular dynamics trajectory file.")
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
        trajectory = load_trajectory(args.input, args.topology, args.frames)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists",
                      exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists.",
                      exc_info=True)
        sys.exit(1)

    # record the analysis parameter in a yaml file
    basename = os.path.splitext(os.path.basename(args.input))[0]
    record_analysis_parameters(args.out, basename, trajectory, args.distance_contacts, args.angle_cutoff,
                               args.proportion_contacts, frames_limits, args.sample.replace(' ', '_'), args.md_time)

    # find the Hydrogen bonds
    pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    data_h_bonds = hydrogen_bonds(trajectory, args.distance_contacts, args.angle_cutoff, args.proportion_contacts,
                                  pattern_contact, args.out, args.sample.replace(' ', '_'), frames_limits)

    # write the CSV for the contacts
    stats = contacts_csv(data_h_bonds, args.out, args.sample.replace(' ', '_'), pattern_contact)