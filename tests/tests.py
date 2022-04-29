#!/usr/bin/env python3

__author__ = 'Nicolas Jeanne'
__copyright__ = 'Copyright (C) 2022 CHU-Toulouse'
__license__ = 'GNU General Public License'
__version__ = '1.0.0'
__email__ = 'jeanne.n@chu-toulouse.fr'

import os
import shutil
import sys
import tempfile
import unittest
import uuid

import pandas as pd
import pytraj as pt

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.dirname(TEST_DIR)
sys.path.append(BIN_DIR)

from trajectories import rmsd, hydrogen_bonds, contacts_csv


class TestTrajectories(unittest.TestCase):

    def setUp(self):
        system_tmp_dir = tempfile.gettempdir()
        self.tmp_dir = os.path.join(system_tmp_dir, "tmp_tests_trajectories")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.mask = None
        self.format_output = "png"
        self.dist_thr = 3.0
        self.contacts_frame_thr_2nd_half = 20.0
        self.traj = pt.iterload(os.path.join(TEST_DIR, "test_files", "JQ679014_hinge_WT_ranked_0_MD-1M.nc"),
                                os.path.join(TEST_DIR, "test_files", "JQ679014_hinge_WT_ranked_0.parm"))
        with open(os.path.join(TEST_DIR, "test_files", "rmsd.csv"), "r") as rmsd_csv_file:
            self.rmsd_csv = "".join(rmsd_csv_file.readlines())
        with open(os.path.join(TEST_DIR, "test_files", "h_bonds.csv"), "r") as h_bonds_csv_file:
            self.h_bonds_csv = "".join(h_bonds_csv_file.readlines())
        with open(os.path.join(TEST_DIR, "test_files", "contacts.csv"), "r") as contacts_csv_file:
            self.contacts_csv = "".join(contacts_csv_file.readlines())

    def tearDown(self):
        # Clean temporary files
        shutil.rmtree(self.tmp_dir)

    def test_rmsd(self):
        unique_id = str(uuid.uuid1())
        observed = rmsd(self.traj, self.tmp_dir, unique_id, self.mask, self.format_output)
        path_observed = os.path.join(self.tmp_dir, unique_id)
        observed.to_csv(path_observed)
        with open(path_observed, "r") as observed_file:
            observed_rmsd_csv = "".join(observed_file.readlines())
        self.assertEqual(observed_rmsd_csv, self.rmsd_csv)

    def test_hydrogen_bonds(self):
        unique_id = str(uuid.uuid1())
        observed = hydrogen_bonds(self.traj, self.tmp_dir, unique_id, self.mask, self.dist_thr,
                                  self.contacts_frame_thr_2nd_half, self.format_output)
        path_observed = os.path.join(self.tmp_dir, unique_id)
        observed.to_csv(path_observed)
        with open(path_observed, "r") as observed_file:
            observed_h_bonds = "".join(observed_file.readlines())
        self.assertEqual(observed_h_bonds, self.h_bonds_csv)

    # def test_contacts_csv(self):
    #     unique_id = str(uuid.uuid1())
    #     path_contacts_csv = os.path.join(self.tmp_dir, unique_id)
    #     _ = contacts_csv(pd.read_csv(self.h_bonds_csv, index=False), path_contacts_csv)
    #     with open(path_contacts_csv, "r") as observed_file:
    #         observed_contacts = "".join(observed_file.readlines())
    #     self.assertEqual(observed_contacts, self.contacts_csv)


if __name__ == "__main__":
    unittest.main()
