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
        if contact_name == "frames":
            ordered.append(contact_name)
        else:
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


def hydrogen_bonds(traj, dist_thr, angle_cutoff, thr, pattern_hb, out_dir, out_basename, lim_frames=None):
    """
    Get the hydrogen bonds between the different atoms of the protein during the molecular dynamics simulation.
    Hydrogen bond is defined as A-HD, where A is acceptor heavy atom, H is hydrogen, D is donor heavy atom. Hydrogen
    bond is formed when A to D distance < distance cutoff and A-H-D angle > angle cutoff.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param dist_thr: the threshold distance in Angstroms for contacts.
    :type dist_thr: float
    :param angle_cutoff: the angle cutoff for the hydrogen bonds.
    :type angle_cutoff: int
    :param thr: the minimal percentage of contacts for atoms contacts of different residues in the
    second half of the simulation.
    :type thr: float
    :param pattern_hb: the pattern for the hydrogen bond name.
    :type pattern_hb: re.pattern
    :param out_dir: the output directory.
    :type out_dir: str
    :param out_basename: the basename for the output CSV file.
    :type out_basename: str
    :param lim_frames: the frames used in the trajectory.
    :type lim_frames: dict
    :return: the dataframe of the polar contacts.
    :rtype: pd.DataFrame
    """
    logging.info("Hydrogen bonds retrieval from the trajectory file, please be patient..")
    h_bonds = pt.hbond(traj, distance=dist_thr, angle=angle_cutoff)
    nb_total_contacts = len(h_bonds.data) - 1
    distances_hbonds = pt.distance(traj, h_bonds.get_amber_mask()[0])
    frames_txt = "the whole frames" if lim_frames is None else f"frames {lim_frames['min']} to {lim_frames['max']}"
    logging.info(f"Search for inter-residues polar contacts in {nb_total_contacts} total polar contacts:")

    nb_intra_residue_contacts = 0
    nb_frames_contacts_2nd_half_thr = 0
    data_hydrogen_bonds = {"frames": range(traj.n_frames)}
    idx = 0
    for h_bond in h_bonds.data:
        if h_bond.key != "total_solute_hbonds":
            match = pattern_hb.search(h_bond.key)
            if match:
                if match.group(2) == match.group(5):
                    nb_intra_residue_contacts += 1
                    logging.debug(f"\t {h_bond.key}: atoms contact from same residue, contact skipped")
                else:
                    # retrieve only the contacts >= percentage threshold of frames in the selected frames
                    pct_contacts = len(distances_hbonds[idx][distances_hbonds[idx] <= dist_thr]) / len(
                        distances_hbonds[idx]) * 100
                    if pct_contacts >= thr:
                        data_hydrogen_bonds[h_bond.key] = distances_hbonds[idx]
                    else:
                        nb_frames_contacts_2nd_half_thr += 1
                        logging.debug(f"\t {h_bond.key}: {pct_contacts:.1f}% of the frames with contacts under the "
                                      f"threshold of {thr:.1f}% in {frames_txt}, contact skipped.")
            idx += 1
    nb_used_contacts = nb_total_contacts - nb_intra_residue_contacts - nb_frames_contacts_2nd_half_thr
    logging.info(f"\t{nb_intra_residue_contacts}/{nb_total_contacts} intra residues atoms contacts discarded.")
    logging.info(f"\t{nb_frames_contacts_2nd_half_thr}/{nb_total_contacts} inter residues atoms contacts discarded "
                 f"with number of contacts frames under the threshold of {thr:.1f}% for {frames_txt}.")
    if nb_used_contacts == 0:
        logging.error(f"\t{nb_used_contacts} inter residues atoms contacts remaining, analysis stopped.")
        sys.exit(0)
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
    ordered_columns = sort_contacts(data_hydrogen_bonds.keys(), pattern_hb)
    df = pd.DataFrame(data_hydrogen_bonds)
    df = df[ordered_columns]
    if lim_frames:
        df["frames"] = [x + lim_frames["min"] for x in list(df["frames"])]
    out_path_bn = os.path.join(out_dir, f"contacts_by_frame_{out_basename}")
    # write the CSV file
    path_csv = f"{out_path_bn}.csv"
    df.to_csv(f"{out_path_bn}.csv", index=False)
    # compress the CSV file
    path_compressed = f"{out_path_bn}.gz"
    with open(path_csv, 'rb') as file_handler_in:
        with gzip.open(path_compressed, 'wb') as file_handler_out:
            shutil.copyfileobj(file_handler_in, file_handler_out)
    os.remove(path_csv)
    logging.info(f"\tContacts by frame file saved: {path_compressed}")

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

    for contact_id in df.columns[1:]:
        match = pattern.search(contact_id)
        if match:
            data["contact"].append(contact_id)
            data["donor position"].append(int(match.group(2)))
            data["acceptor position"].append(int(match.group(5)))
            data["donor residue"].append(match.group(1))
            data["acceptor residue"].append(match.group(4))
        else:
            data["contact"].append(contact_id)
            data["donor position"].append(f"no match with {pattern.pattern}")
            data["donor residue"].append(f"no match with {pattern.pattern}")
            data["acceptor position"].append(f"no match with {pattern.pattern}")
            data["acceptor_residue"].append(f"no match with {pattern.pattern}")
        data["median distance"].append(round(statistics.median(df.loc[:, contact_id]), 2))
    contacts_stat = pd.DataFrame(data)
    out_path = os.path.join(out_dir, f"contacts_by_residue_{out_basename}.csv")
    contacts_stat.to_csv(out_path, index=False)
    logging.info(f"\tcontacts by residue CSV file saved: {out_path}")

    return contacts_stat


def reduce_contacts_dataframe(raw_df, dist_col, thr, roi_lim):
    """
    The dataframe are filtered on the values under the distance threshold, then when multiple rows of the same
    combination of donor and acceptor residues, keep the one with the minimal contact distance (between the atoms of
    this 2 residues) and create a column with the number of contacts between this 2 residues.

    :param raw_df: the contact residues dataframe.
    :type raw_df: pd.Dataframe
    :param dist_col: the name of the distances column.
    :type dist_col: str
    :param thr: the distance threshold.
    :type thr: float
    :param roi_lim: the region of interest limits for the heatmap.
    :type roi_lim: dict
    :return: the reduced dataframe with the minimal distance value of all the couples of donors-acceptors and the
    column with the number of contacts.
    :rtype: pd.Dataframe
    """
    # convert the donor and acceptor positions columns to int
    raw_df["donor position"] = pd.to_numeric(raw_df["donor position"])
    raw_df["acceptor position"] = pd.to_numeric(raw_df["acceptor position"])
    # get only the rows with a contact distance less or equal to the threshold
    df = raw_df[raw_df[dist_col].between(0.0, thr)]
    # select rows of the dataframe if limits for the heatmap were set
    if roi_lim:
        df = df[df["donor position"].between(roi_lim["min"], roi_lim["max"])]
    # donors_acceptors is used to register the combination of donor and acceptor and select only the value with the
    # minimal contact distance and also the number of contacts
    donors_acceptors = []
    idx_to_remove = []
    donor_acceptor_nb_contacts = []
    for _, row in df.iterrows():
        donor = f"{row['donor position']}{row['donor residue']}"
        acceptor = f"{row['acceptor position']}{row['acceptor residue']}"
        if f"{donor}_{acceptor}" not in donors_acceptors:
            donors_acceptors.append(f"{donor}_{acceptor}")
            tmp_df = df[
                (df["donor position"] == row["donor position"]) & (df["acceptor position"] == row["acceptor position"])]
            # get the index of the minimal distance
            idx_min = tmp_df[[dist_col]].idxmin()
            # record the index to remove of the other rows of the same donor - acceptor positions
            tmp_index_to_remove = list(tmp_df.index.drop(idx_min))
            if tmp_index_to_remove:
                idx_to_remove = idx_to_remove + tmp_index_to_remove
            donor_acceptor_nb_contacts.append(len(tmp_df.index))
    df = df.drop(idx_to_remove)
    df["number contacts"] = donor_acceptor_nb_contacts
    return df


def get_df_distances_nb_contacts(df, dist_col):
    """
    Create a distances and a number of contacts dataframes for the couples donors and acceptors.

    :param df: the initial dataframe
    :type df: pd.Dataframe
    :param dist_col: the name of the distances column.
    :type dist_col: str
    :return: the dataframe of distances and the dataframe of the number of contacts.
    :rtype: pd.Dataframe, pd.Dataframe
    """

    # create the dictionaries of distances and number of contacts
    distances = {}
    nb_contacts = {}
    donors = []
    acceptors = []
    unique_donor_positions = sorted(list(set(df["donor position"])))
    unique_acceptor_positions = sorted(list(set(df["acceptor position"])))
    for donor_position in unique_donor_positions:
        donor = f"{donor_position}{df.loc[(df['donor position'] == donor_position), 'donor residue'].values[0]}"
        if donor not in donors:
            donors.append(donor)
        for acceptor_position in unique_acceptor_positions:
            acceptor = f"{acceptor_position}" \
                       f"{df.loc[(df['acceptor position'] == acceptor_position), 'acceptor residue'].values[0]}"
            if acceptor not in acceptors:
                acceptors.append(acceptor)
            # get the distance
            if acceptor_position not in distances:
                distances[acceptor_position] = []
            dist = df.loc[(df["donor position"] == donor_position) & (df["acceptor position"] == acceptor_position),
                          dist_col]
            if not dist.empty:
                distances[acceptor_position].append(dist.values[0])
            else:
                distances[acceptor_position].append(None)
            # get the number of contacts
            if acceptor_position not in nb_contacts:
                nb_contacts[acceptor_position] = []
            contacts = df.loc[(df["donor position"] == donor_position) & (df["acceptor position"] == acceptor_position),
                              "number contacts"]
            if not contacts.empty:
                nb_contacts[acceptor_position].append(contacts.values[0])
            else:
                nb_contacts[acceptor_position].append(None)
    source_distances = pd.DataFrame(distances, index=donors)
    source_distances.columns = acceptors
    source_nb_contacts = pd.DataFrame(nb_contacts, index=donors)
    source_nb_contacts.columns = acceptors
    return source_distances, source_nb_contacts


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

    # find the Hydrogen bonds
    basename = os.path.splitext(os.path.basename(args.input))[0]
    pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    data_h_bonds = hydrogen_bonds(trajectory, args.distance_contacts, args.angle_cutoff, args.proportion_contacts,
                                  pattern_contact, args.out, basename, frames_limits)

    # write the CSV for the contacts
    stats = contacts_csv(data_h_bonds, args.out, basename, pattern_contact)
