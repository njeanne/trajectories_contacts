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


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectory file, select the residues to follow.
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

    traj = pt.iterload(args.input, args.topology)
    if args.mask:
        traj = traj[args.mask]

    print(traj)
    rmsd = pt.rmsd(traj, ref=0)
    print(dir(traj))
    print(traj.n_frames)
    source = pd.DataFrame({"frames": range(traj.n_frames), "RMSD": rmsd})
    print(source)
    rmsd_plot = alt.Chart(data=source).mark_line().encode(
        x=alt.X("frames", title="Frame"),
        y=alt.Y("RMSD", title="RMSD")
    )
    out_path_plot = os.path.join(args.out, f"RMSD_{os.path.splitext(os.path.basename(args.input))[0]}.html")
    rmsd_plot.save(out_path_plot)

    # # find Hydrogen bonds
    # hb = pt.hbond(traj)
    # distance_masks = hb.get_amber_mask()[0]
    # print(f"hbond distance mask: {distance_masks} \n")
    # i = 0
    # for dist_mask in sorted(distance_masks):
    #     i += 1
    #     print(f"{i}:\t{dist_mask}")
    # angle_mask = hb.get_amber_mask()[1]
    # print(f'hbond angle mask: {angle_mask} \n')
    #
    # print("hbond data\t1: have hbond; 0: does not have hbond")
    # print(hb.data)  # 1: have hbond; 0: does not have hbond
    #
    # # compute distance between donor-acceptor for ALL frames (also include frames that do not form hbond)
    # dist = pt.distance(traj, hb.get_amber_mask()[0])
    # print('all hbond distances: ', dist)
    #
    # angle = pt.angle(traj, hb.get_amber_mask()[1])
    # print(f"angles: {angle}")

    # distances_plot = alt.Chart(data=source).mark_rect().encode(
    #     x=alt.X(chain1.replace(".", "_"), title=chain1),
    #     y=alt.Y(chain2.replace(".", "_"), title=chain2, sort=None),
    #     color=alt.Color("minimal_contact_distance:Q", title="Distance (Angstroms)", sort="descending",
    #                     scale=alt.Scale(scheme="yelloworangered"))
    # )
