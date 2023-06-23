#!/usr/bin/env python3

import argparse
import pytraj as pt
import logging
import os
from datetime import datetime


def str_elapsed_time(time_of_start):
    """
    compute elapsed time.
    :param datetime time_of_start: time at the start of the analysis
    :return: time at the end of the analysis
    :rtype: str
    """
    delta = (datetime.now() - time_of_start).total_seconds()
    elapsed_time = "{:02}h:{:02}m:{:02}s".format(int(delta // 3600),
                                                 int(delta % 3600 // 60),
                                                 int(delta % 60))
    return elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test distance computation with MPI",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-t", "--top", required=True, type=str, help="the path to the topology file.")
    parser.add_argument("input", type=str, help="the trajectory file path.")
    args = parser.parse_args()

    sample = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level="INFO",
                        handlers=[logging.FileHandler(os.path.join(args.out, f"SERIAL_distances_{sample}.log")),
                                  logging.StreamHandler()])

    time_start = datetime.now()
    logging.info(f"SERIAL PROCESSING")

    # load trajectory to each core. Use iterload to save memory
    logging.info(f"Trajectory file: {args.input}")
    traj = pt.iterload(args.input, args.top)
    logging.info(f"\tMolecules:{traj.topology.n_mols:>20}")
    logging.info(f"\tResidues:{traj.topology.n_residues:>22}")
    logging.info(f"\tAtoms:{traj.topology.n_atoms:>27}")
    logging.info(f"\tTrajectory total frames:{traj.n_frames:>7}")
    logging.info(f"\tTrajectory memory size:{round(traj._estimated_GB, 6):>14} Gb")

    hb = pt.hbond(traj)
    logging.info(f"serial: {hb.get_amber_mask()[0]}")

    # compute distances
    data = pt.distance(traj, hb.get_amber_mask()[0])
    pt.to_pickle(data, os.path.join(args.out, f"SERIAL_{sample}.pk"))
    logging.info(f"Analysis time: {str_elapsed_time(time_start)}")
    logging.info(f"length data: {len(data)}")


