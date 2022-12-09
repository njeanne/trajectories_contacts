#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.2.0"

import argparse
import logging
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytraj as pt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "references"))
import polarPairs


def restricted_float(float_to_inspect):
    """Inspect if a float is between 0.0 and 100.0

    :param float_to_inspect: the float to inspect
    :type float_to_inspect: float
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the float value if float_to_inspect is between 0.0 and 100.0
    :rtype: float
    """
    x = float(float_to_inspect)
    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")
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


def check_limits(frames, roi):
    """Check if the selected frames and the region of interest (roi) are valid .

    :param frames: the limits of the frames to use.
    :type frames: str
    :param roi: the region of interest limits.
    :type roi: str
    :raises ArgumentTypeError: values not in the fixed limits.
    :return: the frames and region of interest limits.
    :rtype: dict, dict
    """
    frames_lim = {}
    roi_lim = {}
    pattern = re.compile("(\\d+)-(\\d+)")
    if frames:
        match = pattern.search(frames)
        if match:
            frames_lim["min"] = int(match.group(1))
            frames_lim["max"] = int(match.group(2))
        else:
            raise argparse.ArgumentTypeError(f"--frames {frames} is not a valid format, valid format should be: "
                                             f"--frames <DIGITS>-<DIGITS>")
    if roi:
        match = pattern.search(roi)
        if match:
            roi_lim["min"] = int(match.group(1))
            roi_lim["max"] = int(match.group(2))
        else:
            raise argparse.ArgumentTypeError(f"--roi-hm {roi} is not a valid format, valid format should be: "
                                             f"--roi-hm <DIGITS>-<DIGITS>")
    return frames_lim, roi_lim


def load_trajectory(trajectory_file, topology_file, out_dir, frames=None, save=False):
    """
    Load a trajectory and apply a mask if mask argument is set.

    :param trajectory_file: the trajectory file path.
    :type trajectory_file: str
    :param topology_file: the topology file path.
    :type topology_file: str
    :param out_dir: the output directory.
    :type out_dir: str
    :param frames: the frames to use.
    :type frames: str
    :param save: save the trajectory file with the selected frames, only if frames is not None.
    :type save: bool
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    logging.info("Loading trajectory file:")
    frame_indices = None
    if frames:
        frames_split = frames.split("-")
        frame_indices = range(int(frames_split[0]), int(frames_split[1]))
        logging.info(f"\tUsing frames:\t{frame_indices[0]} to {frame_indices[-1] + 1}")

    logging.info("\tComputing trajectory, please be patient..")
    traj = pt.load(trajectory_file, top=topology_file, frame_indices=frame_indices)

    if frames and save:
        path_traj_out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(trajectory_file))[0]}_"
                                              f"frames-{frame_indices[0]}-{frame_indices[-1] + 1}.nc")
        pt.save(path_traj_out, traj, overwrite=True)
        logging.info(f"\tTrajectory file with frames selection saved: {path_traj_out}")
    logging.info(f"\tFrames:\t\t{traj.n_frames}")
    logging.info(f"\tMolecules:\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t{traj.topology.n_atoms}")
    return traj


def rmsd(traj, out_dir, out_basename, format_output, frames_lim=None, mask=None):
    """
    Compute the Root Mean Square Deviation and create the plot.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param out_dir: the output directory path
    :type out_dir: str
    :param out_basename: the plot and CSV basename.
    :type out_basename: str
    :param format_output: the output format for the plots.
    :type format_output: str
    :param frames_lim: the frames limits.
    :type frames_lim: dict
    :param mask: the applied mask.
    :type mask: str
    """
    logging.info("RMSD computation:")
    path_basename = os.path.join(out_dir, f"RMSD_{out_basename}")
    rmsd_traj = pt.rmsd(traj, mask=mask, ref=0)
    if frames_lim:
        range_frames = [x + frames_lim["min"] for x in range(traj.n_frames)]
    else:
        range_frames = [x + 1 for x in range(traj.n_frames)]
    source = pd.DataFrame({"frames": range_frames, "RMSD": rmsd_traj})
    path_csv = f"{path_basename}.csv"
    source.to_csv(path_csv, index=False)
    logging.info(f"\tdata saved: {path_csv}")
    rmsd_ax = sns.lineplot(data=source, x="frames", y="RMSD")
    plot = rmsd_ax.get_figure()
    title = f"Root Mean Square Deviation: {out_basename}"
    plt.suptitle(title, fontsize="large", fontweight="bold")
    if mask and frames_lim:
        plt.title(f"Applied mask: {mask}   Frames used: {frames_lim['min']}-{frames_lim['max']}")
    elif mask:
        plt.title(f"Applied mask: {mask}")
    elif frames_lim:
        plt.title(f"Frames used: {frames_lim['min']}-{frames_lim['max']}")
    plt.xlabel("Frame", fontweight="bold")
    plt.ylabel("RMSD (\u212B)", fontweight="bold")
    path_plot = f"{path_basename}.{format_output}"
    plot.savefig(path_plot)
    # clear the plot for the next use of the function
    plt.clf()
    logging.info(f"\tplot saved: {path_plot}")


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


def hydrogen_bonds(traj, dist_thr, thr_2nd_half_frames, pattern_hb, out_dir, out_basename, lim_frames=None):
    """
    Get the polar bonds (hydrogen) between the different atoms of the protein during the molecular dynamics simulation.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param dist_thr: the threshold distance in Angstroms for contacts.
    :type dist_thr: float
    :param thr_2nd_half_frames: the minimal percentage of contacts for atoms contacts of different residues in the
    second half of the simulation.
    :type thr_2nd_half_frames: float
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
    h_bonds = pt.hbond(traj, distance=dist_thr)
    nb_total_contacts = len(h_bonds.data) - 1
    dist = pt.distance(traj, h_bonds.get_amber_mask()[0])
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
                    # get the second half of the simulation
                    second_half = dist[idx][int(len(dist[idx])/2):]
                    # retrieve only the contacts >= percentage threshold of frames in the second half of the simulation
                    pct_contacts = len(second_half[second_half <= dist_thr]) / len(second_half) * 100
                    if pct_contacts >= thr_2nd_half_frames:
                        data_hydrogen_bonds[h_bond.key] = dist[idx]
                    else:
                        nb_frames_contacts_2nd_half_thr += 1
                        logging.debug(f"\t {h_bond.key}: {pct_contacts:.1f}% of the frames with contacts under the "
                                      f"threshold of {thr_2nd_half_frames:.1f}% for the second half of the "
                                      f"simulation, contact skipped")
            idx += 1

    nb_used_contacts = nb_total_contacts - nb_intra_residue_contacts - nb_frames_contacts_2nd_half_thr
    logging.info(f"\t{nb_intra_residue_contacts}/{nb_total_contacts} intra residues atoms contacts discarded.")
    logging.info(f"\t{nb_frames_contacts_2nd_half_thr}/{nb_total_contacts} inter residues atoms contacts discarded "
                 f"with number of contacts frames under the threshold of {thr_2nd_half_frames:.1f}% for the "
                 f"second half of the simulation.")
    if nb_used_contacts == 0:
        logging.error(f"\t{nb_used_contacts} inter residues atoms contacts remaining, analysis stopped. Check the "
                      f"applied mask selection value if used.")
        sys.exit(0)
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
    ordered_columns = sort_contacts(data_hydrogen_bonds.keys(), pattern_hb)
    df = pd.DataFrame(data_hydrogen_bonds)
    df = df[ordered_columns]
    if lim_frames:
        df["frames"] = [x + lim_frames["min"] for x in list(df["frames"])]
    out_path = os.path.join(out_dir, f"h_bonds_by_frame_{out_basename}.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"\tHydrogen bonds file saved: {out_path}")

    return df


def plot_individual_contacts(df, out_dir, out_basename, dist_thr, format_output, frames_lim=None):
    """
    Plot individual inter residues polar contacts.

    :param df: the inter residues polar contacts.
    :type df: pd.Dataframe
    :param out_dir: the output directory path.
    :type out_dir: str
    :param out_basename: the plot basename.
    :type out_basename: str
    :param dist_thr: the threshold distance in Angstroms for contacts.
    :type dist_thr: float
    :param format_output: the output format for the plots.
    :type format_output: str
    :param frames_lim: the frames limits.
    :type frames_lim: dict
    """
    contacts_plots_dir = os.path.join(out_dir, "individual_contacts")
    os.makedirs(contacts_plots_dir, exist_ok=True)

    # plot all the contacts
    logging.info("Creating individual plots for inter-residues atoms contacts.")
    nb_plots = 0
    for contact_id in df.columns[1:]:
        source = df[["frames", contact_id]]
        scattered_plot = sns.scatterplot(data=source, x="frames", y=contact_id)
        plot = scattered_plot.get_figure()
        title = f"Contact: {contact_id}"
        plt.suptitle(title, fontsize="large", fontweight="bold")
        plt.xlabel("Frames", fontweight="bold")
        plt.ylabel("distance (\u212B)", fontweight="bold")
        # add the threshold horizontal line
        scattered_plot.axhline(dist_thr, color="red")
        # add the 2nd part of MD vertical line
        if frames_lim:
            thr_2nd_half = frames_lim["min"] + int((frames_lim["max"] - frames_lim["min"]) / 2)
        else:
            thr_2nd_half = int(len(df["frames"]) / 2)
        scattered_plot.axvline(thr_2nd_half, color="red")
        out_path = os.path.join(contacts_plots_dir, f"{out_basename}_{contact_id}.{format_output}")
        plot.savefig(out_path)
        nb_plots += 1
        # clear the plot for the next use of the function
        plt.clf()
    logging.info(f"\t{nb_plots}/{len(df.columns[1:])} inter residues atoms contacts plot saved in {contacts_plots_dir}")


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
            "median distance whole MD": [],
            "median distance 2nd half MD": []}

    df_second_half = df.iloc[int(len(df.index) / 2):]
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
        data["median distance whole MD"].append(round(statistics.median(df.loc[:, contact_id]), 2))
        data["median distance 2nd half MD"].append(round(statistics.median(df_second_half.loc[:, contact_id]), 2))
    contacts_stat = pd.DataFrame(data)
    out_path = os.path.join(out_dir, f"contacts_{out_basename}.csv")
    contacts_stat.to_csv(out_path, index=False)
    logging.info(f"\tcontacts CSV saved: {out_path}")

    return contacts_stat


def reduce_contacts_dataframe(raw_df, dist_col, thr, roi_lim):
    """
    The dataframe are filtered on the values under the distance threshold, then when multiple rows of the same
    combination of donor and acceptor residues keep the one with the minimal contact distance (between the atoms of
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


def heatmap_contacts(df_residues, distances_col, threshold_contact, out_basename, out_dir, output_fmt, lim_roi,
                     lim_frames):
    """
    Create the heatmap of contacts between residues.

    :param df_residues: the statistics dataframe.
    :type df_residues: pd.DataFrame
    :param distances_col: the column in the dataframe to get the distances.
    :type distances_col: str
    :param threshold_contact: the maximal contact distance in Angstroms.
    :type threshold_contact: float
    :param out_basename: the basename.
    :type out_basename: str
    :param out_dir: the output directory.
    :type out_dir: str
    :param output_fmt: the output format for the heatmap.
    :type output_fmt: str
    :param lim_roi: the region of interest limits for the heatmap.
    :type lim_roi: dict
    :param lim_frames: the frames used in the trajectory.
    :type lim_frames: dict
    """
    # keep only the minimal distance between 2 residues and add the number of contacts
    df_residues = reduce_contacts_dataframe(df_residues, distances_col, threshold_contact, lim_roi)

    # create the distances and number of contacts dataframes to produce the heatmap
    source_distances, source_nb_contacts = get_df_distances_nb_contacts(df_residues, distances_col)

    # increase the size of the heatmap if too much entries
    factor = int(len(source_distances) / 40) if len(source_distances) / 40 >= 1 else 1
    logging.debug(f"{len(source_distances)} entries, the size of the figure is multiplied by a factor {factor}.")
    rcParams["figure.figsize"] = 15 * factor, 12 * factor
    # create the heatmap
    heatmap = sns.heatmap(source_distances, annot=source_nb_contacts, cbar_kws={"label": "Distance (\u212B)"},
                          linewidths=0.5, xticklabels=True, yticklabels=True)
    heatmap.figure.axes[-1].yaxis.label.set_size(15)
    plot = heatmap.get_figure()
    title = f"Contact residues {distances_col}: {out_basename}"
    plt.suptitle(title, fontsize="large", fontweight="bold")
    subtitle = "Number of residues atoms in contact are displayed in the squares."
    if lim_roi:
        subtitle = f"{subtitle}\nHeatmap focus on donor residues {lim_roi['min']} to {lim_roi['max']}"
    if lim_frames:
        subtitle = f"{subtitle}\nMolecular dynamics frames used: {lim_frames['min']} to {lim_frames['max']}"
    plt.title(subtitle)
    plt.xlabel("Acceptors", fontweight="bold")
    plt.ylabel("Donors", fontweight="bold")
    out_path = os.path.join(out_dir, f"heatmap_{distances_col.replace(' ', '-')}_{out_basename}.{output_fmt}")
    plot.savefig(out_path)
    # clear the plot for the next use of the function
    plt.clf()
    logging.info(f"\tmedian distance heatmap saved: {out_path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectory file perform trajectory analysis. The script computes the Root Mean Square 
    Deviation (RMSD) and the hydrogen contacts between atoms of two different residues represented as a CSV file and 
    heatmaps.
    WARNING: the mask selection is only used to compute the RMSD plot not for loading the trajectory because if the 
    mask defined the backbone, no hydrogen bond will be find.
    See: https://amber-md.github.io/pytraj/latest/atom_mask_selection.html#examples-atom-mask-selection-for-trajectory
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-x", "--frames", required=False, type=str,
                        help="the frames to load from the trajectory, the format must be two integers separated by "
                             "an hyphen, i.e to load the trajectory from the frame 500 to 2000: --frames 500-2000")
    parser.add_argument("--save", required=False, action="store_true",
                        help="if a frame selection is done, specify if the new trajectory file should be saved.")
    parser.add_argument("-m", "--mask", required=False, type=str,
                        help="Only used for the RMSD computation, the residues mask selection format is defined in "
                             "amber documentation.")
    parser.add_argument("-r", "--roi-hm", required=False, type=str,
                        help="the boundaries of the region to display in the heatmap within the mask selection if any. "
                             "In example if the mask '682-850' is applied and the region of interest for the "
                             "heatmap is '--roi 35-39', only positions 717 to 721 will be displayed in the heatmap.")
    parser.add_argument("-f", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("-d", "--distance-contacts", required=False, type=float, default=3.0,
                        help="the contacts distances threshold, default is 3.0 Angstroms.")
    parser.add_argument("-s", "--second-half-percent", required=False, type=restricted_float, default=20.0,
                        help="the minimal percentage of frames which make contact between 2 atoms of different "
                             "residues in the second half of the molecular dynamics simulation, default is 20%%.")
    parser.add_argument("-i", "--individual-plots", required=False, action="store_true",
                        help="plot the individual contacts.")
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
    logging.info(f"minimal proportion of frames with atoms contacts between two different residues in the second "
                 f"half of the molecular dynamics: {args.second_half_percent:.1f}%")
    try:
        frames_limits, roi_limits = check_limits(args.frames, args.roi_hm)
    except argparse.ArgumentTypeError as exc:
        logging.error(exc)
        sys.exit(1)

    # set the seaborn plots theme and size
    sns.set_theme()
    rcParams["figure.figsize"] = 15, 12

    # load the trajectory
    try:
        trajectory = load_trajectory(args.input, args.topology, args.out, args.frames, args.save)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists",
                      exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists.",
                      exc_info=True)
        sys.exit(1)

    # compute RMSD and create the plot
    basename = os.path.splitext(os.path.basename(args.input))[0]
    rmsd(trajectory, args.out, basename, args.format, frames_limits, args.mask)

    # find the Hydrogen bonds
    pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
    data_h_bonds = hydrogen_bonds(trajectory, args.distance_contacts, args.second_half_percent, pattern_contact,
                                  args.out, basename, frames_limits)

    if args.individual_plots:
        # plot individual contacts
        plot_individual_contacts(data_h_bonds, args.out, basename, args.distance_contacts, args.format, frames_limits)

    # write the CSV for the contacts
    stats = contacts_csv(data_h_bonds, args.out, basename, pattern_contact)

    # get the heatmaps of validated contacts by residues for each column of the statistics dataframe
    for distances_column_id in stats.columns[5:]:
        heatmap_contacts(stats, distances_column_id, args.distance_contacts, basename, args.out, args.format,
                         roi_limits, frames_limits)
