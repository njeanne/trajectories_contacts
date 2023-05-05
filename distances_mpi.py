import pytraj as pt
import logging
import os
from datetime import datetime
# import mpi4py to get rank of each core
from mpi4py import MPI


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


out_dir = "results/parallel/distances_mpi"
os.makedirs(out_dir, exist_ok=True)
logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                    datefmt="%Y/%m/%d %H:%M:%S",
                    level="INFO",
                    handlers=[logging.FileHandler(os.path.join(out_dir, "distances.log")), logging.StreamHandler()])

time_start = datetime.now()

comm = MPI.COMM_WORLD
# load trajectory to each core. Use iterload to save memory
traj = pt.iterload("data/HEPAC-6_RNF19A_ORF1_2000-frames.nc",
                   "data/HEPAC-6_RNF19A_ORF1_0.parm")
logging.info(f"trajectory on process {comm.rank}: {traj}")
hb = pt.hbond(traj, distance=3.0, angle=135)
if comm.rank == 0:
    logging.info(f"process {comm.rank}: {hb.get_amber_mask()[0]}")

# compute distances by sending this method to pt.pmap_mpi function, data is sent to the first core
data = pt.pmap_mpi(pt.distance, traj, hb.get_amber_mask()[0])

# data is sent to first core (rank=0)
if comm.rank == 0:
    # save data
    pt.to_pickle(data, os.path.join(out_dir, "MPI_HEPAC-6_RNF19A_ORF1_2000-frame.pk"))
    logging.info(f"Analysis time (process {comm.rank}): {str_elapsed_time(time_start)}")
    logging.info(f"process {comm.rank}, length data: {len(data)}:\n{data}\n")

