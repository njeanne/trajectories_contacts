#!/usr/bin/env python3

__author__ = "Nicolas Jeanne"
__license__ = "GNU General Public License"
__version__ = "3.0.0"
__email__ = "jeanne.n@chu-toulouse.fr"

import argparse
import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid

from trajectories import check_limits, load_trajectory, rmsd, hydrogen_bonds, contacts_csv

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join("data", "test_files")
BIN_DIR = os.path.dirname(TEST_DIR)
sys.path.append(BIN_DIR)


def format_csv(path):
    with open(path, "r") as csv_file:
        return "".join(csv_file.readlines())


class TestTrajectories(unittest.TestCase):

    def setUp(self):
        system_tmp_dir = tempfile.gettempdir()
        self.tmp_dir = os.path.join(system_tmp_dir, "tmp_tests_trajectories")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.chunk = 500
        self.dist_thr = 3.0
        self.contacts_frame_thr_2nd_half = 50.0
        self.pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
        self.frames = check_limits("500-2000")
        self.traj = load_trajectory(os.path.join(TEST_FILES_DIR, "JQ679014_hinge_WT_ranked_0_20-frames.nc"),
                                    os.path.join(TEST_FILES_DIR, "JQ679014_hinge_WT_ranked_0.parm"),
                                    self.tmp_dir, ":25-45")
        self.rmsd = format_csv(os.path.join(TEST_FILES_DIR,
                                            "RMSD_JQ679014_hinge_WT_ranked_0_20-frames_mask-25-45.csv"))
        self.h_bonds = format_csv(os.path.join(TEST_FILES_DIR,
                                               "h_bonds_JQ679014_hinge_WT_ranked_0_20-frames_mask-25-45.csv"))
        self.contacts = format_csv(os.path.join(TEST_FILES_DIR,
                                                "contacts_JQ679014_hinge_WT_ranked_0_20-frames_mask-25-45.csv"))

    def tearDown(self):
        # Clean temporary files
        shutil.rmtree(self.tmp_dir)

    def test_limits(self):
        self.assertEqual(check_limits("500-2000"), self.frames)
        self.assertNotEqual(check_limits("1-2000"), self.frames)
        self.assertRaises(argparse.ArgumentTypeError, check_limits, ":25to45", "7-15")
        self.assertRaises(argparse.ArgumentTypeError, check_limits, ":25-45", "7to15")
        self.assertRaises(argparse.ArgumentTypeError, check_limits, ":25-45", "7-55")
        self.assertRaises(argparse.ArgumentTypeError, check_limits, ":25-45", "20-35")

    def test_rmsd(self):
        unique_id = str(uuid.uuid1())
        rmsd(self.traj, self.tmp_dir, unique_id, self.limits, self.format_output)
        path_observed = os.path.join(self.tmp_dir, f"RMSD_{unique_id}.csv")
        with open(path_observed, "r") as observed_file:
            observed_rmsd_csv = "".join(observed_file.readlines())
        self.assertEqual(observed_rmsd_csv, self.rmsd)

    def test_hydrogen_bonds(self):
        unique_id = str(uuid.uuid1())
        h_bonds = hydrogen_bonds(self.traj, self.dist_thr, self.contacts_frame_thr_2nd_half, self.pattern_contact,
                                 self.tmp_dir, unique_id)
        path_observed = os.path.join(self.tmp_dir, unique_id)
        h_bonds.to_csv(path_observed, index=False)
        with open(path_observed, "r") as observed_file:
            observed = "".join(observed_file.readlines())
        self.assertEqual(observed, self.h_bonds)

    def test_contacts_csv(self):
        unique_id = str(uuid.uuid1())
        h_bonds = hydrogen_bonds(self.traj, self.dist_thr, self.contacts_frame_thr_2nd_half, self.pattern_contact,
                                 self.tmp_dir, unique_id)
        _ = contacts_csv(h_bonds, self.tmp_dir, unique_id, self.pattern_contact, self.limits)
        with open(os.path.join(self.tmp_dir, f"contacts_{unique_id}.csv"), "r") as observed_file:
            observed_contacts = "".join(observed_file.readlines())
        self.assertEqual(observed_contacts, self.contacts)


if __name__ == "__main__":
    unittest.main()
