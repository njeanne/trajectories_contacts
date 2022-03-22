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
import sys

import altair as alt
import pandas as pd
import pytraj as pt
from Bio import PDB
from pymol import cmd

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
    :param out_dir: the output directory path/
    :type out_dir: str
    :param out_basename: the plot basename.
    :type out_basename: str
    :param mask: the selection mask.
    :type mask: str
    :param format_output: the output format for the plots.
    :type format_output: bool
    :return: the trajectory data.
    :rtype: pd.DataFrame
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

    return source


def hydrogen_bonds(traj, out_dir, out_basename, mask, dist_thr, second_half_pct_thr, format_output):
    """
    Get the polar bonds (hydrogen) between the different atoms of the protein during the molecular dynamics simulation.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param out_dir: the output directory path/
    :type out_dir: str
    :param out_basename: the plot basename.
    :type out_basename: str
    :param mask: the selection mask.
    :type mask: str
    :param dist_thr: the threshold distance in Angstroms for contacts.
    :type dist_thr: float
    :param second_half_pct_thr: the minimal percentage of contacts for atoms contacts of different residues in the
    second half of the simulation.
    :type second_half_pct_thr: float
    :param format_output: the output format for the plots.
    :type format_output: bool
    :return: the dataframe of the polar contacts.
    :rtype: pd.DataFrame
    """
    logging.info("Search for polar contacts:")
    h_bonds = pt.hbond(traj, distance=dist_thr)
    dist = pt.distance(traj, h_bonds.get_amber_mask()[0])
    pattern_hb = re.compile("\\D{3}(\\d+).+-\\D{3}(\\d+)")
    idx = 0
    nb_intra_residue_contacts = 0
    nb_contacts_2nd_part_simu_under_pct_thr = 0
    h_bonds_data = {"frames": range(traj.n_frames)}
    for h_bond in h_bonds.data:
        if h_bond.key != "total_solute_hbonds":
            match = pattern_hb.search(h_bond.key)
            if match:
                if match.group(1) == match.group(2):
                    nb_intra_residue_contacts += 1
                    logging.debug(f"\t atom contact from same residue ({h_bond.key}), contacts skipped")
                else:
                    # get the second part of the simulation
                    second_half = dist[idx][int(len(dist[idx])/2):]
                    # retrieve only the contacts >= percentage threshold in the second half of the simulation
                    if (len(second_half[second_half <= dist_thr]) / len(second_half)) * 100 >= second_half_pct_thr:
                        h_bonds_data[h_bond.key] = dist[idx]
                    else:
                        nb_contacts_2nd_part_simu_under_pct_thr += 1
                        logging.debug(f"\t atom contact under {second_half_pct_thr} second half threshold "
                                      f"({h_bond.key}), contacts skipped")
            idx += 1
    logging.info(f"\t{nb_intra_residue_contacts} intra residues atoms contacts discarded.")
    logging.info(f"\t{nb_contacts_2nd_part_simu_under_pct_thr} inter residues atoms contacts discarded with number of "
                 f"contacts under the percentage threshold of {second_half_pct_thr:.1f}% for the second half of the "
                 f"simulation.")
    df = pd.DataFrame(h_bonds_data)
    # plot all the contacts
    plots = []
    nb_plots = 0
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
        nb_plots += 1
        plots.append(contact_plot)

    contacts_plots = alt.concat(*plots, columns=3)
    basename_plot_path = os.path.join(out_dir,
                                      f"contacts_{out_basename}_{mask}" if mask else f"contacts_{out_basename}")
    out_path = f"{basename_plot_path}.{format_output}"
    contacts_plots.save(out_path)
    logging.info(f"\t{nb_plots} inter residues atoms contacts saved to plot: {out_path}")

    return df


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
    logging.info(f"minimal threshold for number of atoms contacts between two different residues: "
                 f"{args.second_half_percent:.1f}%")

    # load the trajectory
    trajectory = load_trajectory(args.input, args.topology, args.mask)
    # compute RMSD and create the plot
    basename = os.path.splitext(os.path.basename(args.input))[0]
    data_traj = rmsd(trajectory, args.out, basename, args.mask, args.output_format)

    # find Hydrogen bonds
    data_h_bonds = hydrogen_bonds(trajectory, args.out, basename, args.mask, args.distance_contacts,
                                  args.second_half_percent, args.output_format)
