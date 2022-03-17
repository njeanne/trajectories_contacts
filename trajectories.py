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


def rmsd(traj, out_dir, out_basename, mask):
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
    :return: the trajectory data.
    :rtype: pd.DataFrame
    """
    rmsd_traj = pt.rmsd(traj, ref=0)
    source = pd.DataFrame({"frames": range(traj.n_frames), "RMSD": rmsd_traj})
    rmsd_plot = alt.Chart(data=source).mark_line().encode(
        x=alt.X("frames", title="Frame"),
        y=alt.Y("RMSD", title="RMSD")
    ).properties(
        title={
            "text": out_basename,
            "subtitle": [f"Mask {mask}" if mask else ""],
            "subtitleColor": "gray"
        },
        width=600,
        height=400
    )
    out_path = os.path.join(out_dir, f"RMSD_{out_basename}_{mask}.html" if mask else f"RMSD_{out_basename}.html")
    rmsd_plot.save(out_path)
    logging.info(f"RMSD plot saved: {out_path}")

    return source


def hydrogen_bonds(traj, out_dir, out_basename):

    h_bonds = pt.hbond(traj)
    dist = pt.distance(traj, h_bonds.get_amber_mask()[0])
    print(h_bonds.data)  # 1: have hbond; 0: does not have hbond
    pattern_hb = re.compile("\\D{3}(\\d+).+-\\D{3}(\\d+)")
    idx = 0
    h_bonds_data = {"frames": range(traj.n_frames)}
    for h_bond in h_bonds.data:
        if h_bond.key != "total_solute_hbonds":
            match = pattern_hb.search(h_bond.key)
            if match:
                if match.group(1) == match.group(2):
                    logging.debug(f"atom contact from same residue ({h_bond.key}), contacts skipped")
                else:
                    h_bonds_data[h_bond.key] = dist[idx]
            idx += 1
    df = pd.DataFrame(h_bonds_data)
    # plot all the contacts
    # todo: code angstrom
    angstrom_str = r"($\AA$)"
    for contact_id in df.columns[1:]:
        source = df[["frames", contact_id]]
        contact_plot = alt.Chart(data=source).mark_circle().encode(
            x=alt.X("frames", title="Frame"),
            y=alt.Y(contact_id, title=f"contact distance {contact_id} {angstrom_str}")
        ).properties(
            title={
                "text": f"{contact_id} distance during molecular dynamics simulation"
            },
            width=600,
            height=400
        )
        out_path = os.path.join(out_dir, f"contact_{out_basename}_{contact_id}.html")
        contact_plot.save(out_path)
    logging.info(f"Contacts plots saved in: {out_dir}")

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
    perform trajectory analysis.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-m", "--mask", required=False, type=str, help="the mask selection.")
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

    # load the trajectory
    trajectory = load_trajectory(args.input, args.topology, args.mask)
    # compute RMSD and create the plot
    basename = os.path.splitext(os.path.basename(args.input))[0]
    data_traj = rmsd(trajectory, args.out, basename, args.mask)

    # find Hydrogen bonds
    data_h_bonds = hydrogen_bonds(trajectory, args.out, basename)
    print(data_h_bonds)


    # pd.DataFrame.to_excel(hb.data.to_dataframe, os.path.join(args.out, "data.xlsx"))

    # # compute distance between donor-acceptor for ALL frames (also include frames that do not form hbond)


    #
    # angle = pt.angle(trajectory, hb.get_amber_mask()[1])
    # print(f"angles: {angle}")

    # distances_plot = alt.Chart(data=source).mark_rect().encode(
    #     x=alt.X(chain1.replace(".", "_"), title=chain1),
    #     y=alt.Y(chain2.replace(".", "_"), title=chain2, sort=None),
    #     color=alt.Color("minimal_contact_distance:Q", title="Distance (Angstroms)", sort="descending",
    #                     scale=alt.Scale(scheme="yelloworangered"))
    # )
