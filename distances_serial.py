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


out_dir = "results/serial/distances_serial"
os.makedirs(out_dir, exist_ok=True)
logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                    datefmt="%Y/%m/%d %H:%M:%S",
                    level="INFO",
                    handlers=[logging.FileHandler(os.path.join(out_dir, "distances.log")), logging.StreamHandler()])

time_start = datetime.now()
logging.info(f"launched on process")

# load trajectory to each core. Use iterload to save memory
traj = pt.iterload("data/HEPAC-6_RNF19A_ORF1_2000-frames.nc",
                   "data/HEPAC-6_RNF19A_ORF1_0.parm")
hb = pt.hbond(traj)
logging.info(f"process: {hb.get_amber_mask()[0]}")

# compute distances
data = pt.distance(traj, hb.get_amber_mask()[0])
pt.to_pickle(data, os.path.join(out_dir, "SERIAL_HEPAC-6_RNF19A_ORF1_2000-frame.pk"))
out_path = os.path.join(out_dir, "MPI_HEPAC-6_RNF19A_ORF1_2000-frame.pk")
logging.info(f"SERIAL, length data: {len(data)}, saved: {out_path}")
logging.info(f"Analysis time (SERIAL): {str_elapsed_time(time_start)}")


