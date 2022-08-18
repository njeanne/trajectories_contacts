#!/usr/bin/env python3

"""
Created on 17 Mar. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.0.0"

import argparse
import logging
import os
import re
import statistics
import sys

import altair as alt
import pandas as pd
import pytraj as pt

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
        raise argparse.ArgumentTypeError("{} not in range [0.0, 100.0]".format(x))
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


def load_trajectory(trajectory_file, topology_file, mask):
    """
    Load a trajectory and apply a mask if mask argument is set.

    :param trajectory_file: the trajectory file path.
    :type trajectory_file: str
    :param topology_file: the topology file path.
    :type topology_file: str
    :param mask: the mask.
    :type mask: str
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    traj = pt.iterload(trajectory_file, topology_file)
    logging.info("Load trajectory file:")
    if mask is not None:
        logging.info(f"\tapplied mask:\t{mask}")
        traj = traj[mask]
    logging.info(f"\tFrames:\t\t{traj.n_frames}")
    logging.info(f"\tMolecules:\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t{traj.topology.n_atoms}")

    return traj


def rmsd(traj, out_dir, out_basename, mask, format_output):
    """
    Compute the Root Mean Square Deviation and create the plot.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param out_dir: the output directory path
    :type out_dir: str
    :param out_basename: the plot basename.
    :type out_basename: str
    :param mask: the selection mask.
    :type mask: str or None
    :param format_output: the output format for the plots.
    :type format_output: str
    """
    rmsd_traj = pt.rmsd(traj, ref=0)
    source = pd.DataFrame({"frames": range(traj.n_frames), "RMSD": rmsd_traj})
    rmsd_plot = alt.Chart(data=source).mark_line().encode(
        x=alt.X("frames", title="Frame"),
        y=alt.Y("RMSD", title="RMSD (\u212B)")
    ).properties(
        title={
            "text": f"Root Mean Square Deviation: {out_basename}",
            "subtitle": [f"Mask:\t{mask}" if mask else ""],
            "subtitleColor": "gray"
        },
        width=600,
        height=400
    )
    basename_plot_path = os.path.join(out_dir, f"RMSD_{out_basename}_{mask}" if mask else f"RMSD_{out_basename}")
    out_path = f"{basename_plot_path}.{format_output}"
    rmsd_plot.save(out_path)
    logging.info(f"RMSD plot saved: {out_path}")


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
                    if int(match.group(4)) in tmp[int(match.group(2))]:
                        tmp[int(match.group(2))][int(match.group(4))].append(contact_name)
                    else:
                        tmp[int(match.group(2))][int(match.group(4))] = [contact_name]
                else:
                    tmp[int(match.group(2))] = {int(match.group(4)): [contact_name]}
            else:
                logging.error(f"no match for {pattern.pattern} in {contact_name}")
                sys.exit(1)

    for key1 in sorted(tmp):
        for key2 in sorted(tmp[key1]):
            for contact_name in sorted(tmp[key1][key2]):
                ordered.append(contact_name)

    return ordered


def hydrogen_bonds(traj, dist_thr, contacts_frame_thr_2nd_half, pattern_hb):
    """
    Get the polar bonds (hydrogen) between the different atoms of the protein during the molecular dynamics simulation.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param dist_thr: the threshold distance in Angstroms for contacts.
    :type dist_thr: float
    :param contacts_frame_thr_2nd_half: the minimal percentage of contacts for atoms contacts of different residues in
    the second half of the simulation.
    :type contacts_frame_thr_2nd_half: float
    :param pattern_hb: the pattern for the hydrogen bond name.
    :type pattern_hb: re.pattern
    :return: the dataframe of the polar contacts.
    :rtype: pd.DataFrame
    """
    logging.info("Hydrogen bonds retrieval from the trajectory file, please be patient..")
    h_bonds = pt.hbond(traj, distance=dist_thr)
    nb_total_contacts = len(h_bonds.data) - 1
    dist = pt.distance(traj, h_bonds.get_amber_mask()[0])
    logging.info(f"Search for inter residues polar contacts in {nb_total_contacts} total polar contacts:")

    nb_intra_residue_contacts = 0
    nb_frames_contacts_2nd_half_thr = 0
    data_hydrogen_bonds = {"frames": range(traj.n_frames)}
    idx = 0
    for h_bond in h_bonds.data:
        if h_bond.key != "total_solute_hbonds":
            match = pattern_hb.search(h_bond.key)
            if match:
                if match.group(2) == match.group(4):
                    nb_intra_residue_contacts += 1
                    logging.debug(f"\t {h_bond.key}: atoms contact from same residue, contact skipped")
                else:
                    # get the second half of the simulation
                    second_half = dist[idx][int(len(dist[idx])/2):]
                    # retrieve only the contacts >= percentage threshold of frames in the second half of the simulation
                    pct_contacts = len(second_half[second_half <= dist_thr]) / len(second_half) * 100
                    if pct_contacts >= contacts_frame_thr_2nd_half:
                        data_hydrogen_bonds[h_bond.key] = dist[idx]
                    else:
                        nb_frames_contacts_2nd_half_thr += 1
                        logging.debug(f"\t {h_bond.key}: {pct_contacts:.1f}% of the frames with contacts under the "
                                      f"threshold of {contacts_frame_thr_2nd_half:.1f}% for the second half of the "
                                      f"simulation, contact skipped")
            idx += 1

    nb_used_contacts = nb_total_contacts - nb_intra_residue_contacts - nb_frames_contacts_2nd_half_thr
    logging.info(f"\t{nb_intra_residue_contacts}/{nb_total_contacts} intra residues atoms contacts discarded.")
    logging.info(f"\t{nb_frames_contacts_2nd_half_thr}/{nb_total_contacts} inter residues atoms contacts discarded "
                 f"with number of contacts frames under the threshold of {contacts_frame_thr_2nd_half:.1f}% for the "
                 f"second half of the simulation.")
    logging.info(f"\t{nb_used_contacts} inter residues atoms contacts used.")
    ordered_columns = sort_contacts(data_hydrogen_bonds.keys(), pattern_hb)
    df = pd.DataFrame(data_hydrogen_bonds)
    df = df[ordered_columns]

    return df


def plot_individual_contacts(df, out_dir, out_basename, dist_thr, mask, format_output):
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
    :param mask: the selection mask.
    :type mask: str
    :param format_output: the output format for the plots.
    :type format_output: str
    """
    contacts_plots_solo_dir = os.path.join(out_dir, "contacts_individual", "solo")
    os.makedirs(contacts_plots_solo_dir, exist_ok=True)
    contacts_plots_merged_dir = os.path.join(out_dir, "contacts_individual", "merged")
    os.makedirs(contacts_plots_merged_dir, exist_ok=True)

    # plot all the contacts
    nb_used_contacts = len(df.columns[1:])
    plots = []
    nb_plots = 0
    nb_merged_plots = 1
    basename_merged = f"contacts_{out_basename}_{mask}" if mask else f"contacts_{out_basename}"
    for contact_id in df.columns[1:]:
        source = df[["frames", contact_id]]
        contact_plot = alt.Chart(data=source).mark_circle().encode(
            x=alt.X("frames", title="Frame"),
            y=alt.Y(contact_id, title="distance (\u212B)")
        ).properties(
            title={"text": f"Contact: {contact_id}"},
            width=400,
            height=260
        )
        # add a distance threshold line
        h_line = alt.Chart().mark_rule(color="red").encode(y=alt.datum(dist_thr))
        contact_plot = contact_plot + h_line
        # save the plot of the contact
        path_solo = os.path.join(contacts_plots_solo_dir,
                                 f"{out_basename}_{contact_id}_{'_'+mask if mask else ''}.{format_output}")
        contact_plot.save(path_solo)
        # add the plot to the list to create a whole picture
        nb_plots += 1
        plots.append(contact_plot)

        # create the merged plot if 15 plots are present in the list
        if nb_plots % 15 == 0:
            # merge the plots by row of 3
            contacts_plots = alt.concat(*plots, columns=3)
            # save the plot
            out_path_plot = os.path.join(contacts_plots_merged_dir,
                                         f"{basename_merged}_{nb_merged_plots}.{format_output}")
            contacts_plots.save(out_path_plot)
            logging.info(f"\t{nb_plots}/{nb_used_contacts} inter residues atoms contacts plot saved: "
                         f"{out_path_plot}")
            plots = []
            nb_merged_plots += 1

    # merge the remaining plots by row of 3 and save them
    if plots:
        contacts_plots = alt.concat(*plots, columns=3)
        # save the plot
        out_path_plot = os.path.join(contacts_plots_merged_dir,
                                     f"{basename_merged}_{nb_merged_plots}.{format_output}")
        contacts_plots.save(out_path_plot)
        logging.info(f"\t{nb_plots}/{nb_used_contacts} inter residues atoms contacts plot saved: {out_path_plot}")


def contacts_csv(df, out_path, pattern):
    """
    Get the mean and median distances for the contacts in the whole molecular dynamics simulation and in the second
    half of the simulation.

    :param df: the contacts dataframe.
    :type df: pd.DataFrame
    :param out_path: the CSV output path.
    :type out_path: str
    :param pattern: the pattern for the contact.
    :type pattern: re.pattern
    :return: the dataframe of the contacts statistics.
    :rtype: pd.DataFrame
    """
    data = {"contact": [],
            "donor position": [],
            "donor residue": [],
            "acceptor position": [],
            "acceptor residue": [],
            "mean_distance_2nd_half": [],
            "median_distance_2nd_half": [],
            "mean_distance_whole": [],
            "median_distance_whole": []}

    df_half = df.iloc[int(len(df.index)/2):]
    for contact_id in df.columns[1:]:
        data["contact"].append(contact_id)
        match = pattern.search(contact_id)
        if match:
            data["donor position"].append(match.group(2))
            data["donor residue"].append(match.group(1))
            data["acceptor position"].append(match.group(4))
            data["acceptor residue"].append(match.group(3))
        else:
            data["donor position"].append("not found")
            data["donor residue"].append("not found")
            data["acceptor position"].append("not found")
            data["acceptor_residue"].append("not found")
        data["mean_distance_2nd_half"].append(round(statistics.mean(df_half.loc[:, contact_id]), 2))
        data["median_distance_2nd_half"].append(round(statistics.median(df_half.loc[:, contact_id]), 2))
        data["mean_distance_whole"].append(round(statistics.mean(df.loc[:, contact_id]), 2))
        data["median_distance_whole"].append(round(statistics.median(df.loc[:, contact_id]), 2))
    contacts_stat = pd.DataFrame(data)
    contacts_stat.to_csv(out_path, index=False)
    logging.info(f"Inter residues atoms contacts CSV saved: {out_path}")

    return contacts_stat


def heat_map_contacts(df, stat_col, out_basename, mask, out_dir, output_fmt):
    """
    Create the heat map of contacts between residues.

    :param df: the statistics dataframe.
    :type df: pd.DataFrame
    :param stat_col: the column in the dataframe to get the distances.
    :type stat_col: str
    :param out_basename: the basename.
    :type out_basename: str
    :param mask: the selection mask
    :type mask: str
    :param out_dir: the output directory.
    :type out_dir: str
    :param output_fmt: the output format for the heat map.
    :type output_fmt: str
    """
    donor_acceptor = {}
    for _, row in df.iterrows():
        donor = f"{row['donor position']}{row['donor residue']}"
        acceptor = f"{row['acceptor position']}{row['acceptor residue']}"
        if donor in donor_acceptor:
            if acceptor in donor_acceptor[donor]:
                donor_acceptor[donor][acceptor].append(row[stat_col])
            else:
                donor_acceptor[donor][acceptor] = [row[stat_col]]
        else:
            donor_acceptor[donor] = {acceptor: [row[stat_col]]}
    donors = []
    acceptors = []
    nb_contacts = []
    min_distances = []
    for donor in donor_acceptor:
        donors = donors + [donor] * len(donor_acceptor[donor])
        for acceptor in donor_acceptor[donor]:
            acceptors.append(acceptor)
            nb_contacts.append(len(donor_acceptor[donor][acceptor]))
            min_distances.append(min(donor_acceptor[donor][acceptor]))

    # get the heat map
    source = pd.DataFrame({"acceptors": acceptors, "donors": donors, "number of contacts": nb_contacts,
                           stat_col: min_distances})

    heatmap = alt.Chart(data=source).mark_rect().encode(
        x=alt.X("acceptors", type="nominal", sort=None),
        y=alt.Y("donors", type="nominal", sort=None),
        color=alt.Color(stat_col, type="quantitative", title="Distance (\u212B)", sort="descending",
                        scale=alt.Scale(scheme="yelloworangered"))
    ).properties(
        title={
            "text": f"Contact residues {stat_col.replace('_', ' ')}: {out_basename}",
            "subtitle": ["Number of contacts displayed in the squares", f"Mask:\t{mask}" if mask else ""],
            "subtitleColor": "gray"
        },
        width=600,
        height=400
    )
    # Configure the text with the number of contacts
    switch_color = min(source[stat_col]) + (max(source[stat_col]) - min(source[stat_col])) / 2
    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("number of contacts"),
        color=alt.condition(
            f"datum.{stat_col} > {switch_color}",
            alt.value("black"),
            alt.value("white")
        )
    )
    plot = heatmap + text
    mask_str = f"_{mask}" if mask else ""
    out_path = os.path.join(out_dir, f"heatmap_{stat_col.replace(' ', '-')}_{out_basename}{mask_str}.{output_fmt}")
    plot.save(out_path)
    # heatmap.save(out_path)
    logging.info(f"\t{stat_col} heat map saved: {out_path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectory file and eventually a mask selection
    (https://amber-md.github.io/pytraj/latest/atom_mask_selection.html#examples-atom-mask-selection-for-trajectory),
    perform trajectory analysis. The script computes the Root Mean Square Deviation (RMSD) and the hydrogen contacts
    between atoms of two different residues.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-m", "--mask", required=False, type=str, help="the mask selection.")
    parser.add_argument("-f", "--output-format", required=False, choices=["svg", "png", "html", "pdf"], default="html",
                        help="the output plots format, if not used the default is HTML.")
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
    logging.info(f"maximal threshold for atoms contacts distance: {args.distance_contacts} \u212B")
    logging.info(f"minimal threshold for number of frames atoms contacts between two different residues in the second "
                 f"half of the simulation: {args.second_half_percent:.1f}%")

    # load the trajectory
    try:
        trajectory = load_trajectory(args.input, args.topology, args.mask)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists. \
        {exc}")
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists. \
        {exc}")
        sys.exit(1)

    # compute RMSD and create the plot
    basename = os.path.splitext(os.path.basename(args.input))[0]
    # data_traj = rmsd(trajectory, args.out, basename, args.mask, args.output_format)
    rmsd(trajectory, args.out, basename, args.mask, args.output_format)

    # find Hydrogen bonds
    pattern_contact = re.compile("(\\D{3})(\\d+).+-(\\D{3})(\\d+)")
    data_h_bonds = hydrogen_bonds(trajectory, args.distance_contacts, args.second_half_percent, pattern_contact)
    if args.individual_plots:
        # plot individual contacts
        plot_individual_contacts(data_h_bonds, args.out, basename, args.distance_contacts, args.mask,
                                 args.output_format)

    # write the CSV for the contacts
    stats = contacts_csv(data_h_bonds, os.path.join(args.out, f"contacts_{basename}.csv"), pattern_contact)

    # get the heat maps of validated contacts by residues for each column of the statistics dataframe
    logging.info("Heat maps contacts:")
    for stat_column_id in stats.columns[5:]:
        heat_map_contacts(stats, stat_column_id, basename, args.mask, args.out, args.output_format)

    # clean the geckodriver log file
    path_geckodriver_log = os.path.join(args.out, "geckodriver.log")
    if os.path.exists(path_geckodriver_log):
        os.remove(path_geckodriver_log)
