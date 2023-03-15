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

from trajectories_contacts import *

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.dirname(TEST_DIR)
sys.path.append(BIN_DIR)
TEST_DIR_EXPECTED = os.path.join(TEST_DIR, "expected")
TEST_DIR_INPUTS = os.path.join(TEST_DIR, "test_files")


# def format_csv(path):
#     with open(path, "r") as csv_file:
#         return "".join(csv_file.readlines())


class TestTrajectories(unittest.TestCase):

    def setUp(self):
        system_tmp_dir = tempfile.gettempdir()
        self.tmp_dir = os.path.join(system_tmp_dir, "tmp_tests_trajectories_contacts")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.dist_cutoff = 3.0
        self.angle_cutoff = 135
        self.pct_cutoff = 50.0
        self.sample = "test"
        self.nanoseconds = 1
        self.frames = "test_data_20-frames.nc:5-20"
        self.parsed_frames = {'test_data_20-frames.nc': {'begin': 5, 'end': 20}}
        self.pattern_contact = re.compile("(\\D{3})(\\d+)_(.+)-(\\D{3})(\\d+)_(.+)")
        self.inputs = [os.path.join(TEST_DIR_INPUTS, "test_data_20-frames.nc"),
                       os.path.join(TEST_DIR_INPUTS, "test_data_20-frames_2.nc")]
        self.topology = os.path.join(TEST_DIR_INPUTS, "test_data.parm")
        # self.traj1 = load_trajectory([self.inputs[0]], os.path.join(TEST_DIR_INPUTS, "test_data.parm"))
        with open(os.path.join(TEST_DIR_EXPECTED, "test_analysis.yaml"), "r") as file_handler:
            self.analysis_yaml = yaml.safe_load(file_handler.read())
        # self.contacts = format_csv(os.path.join(TEST_FILES_DIR,
        #                                         "contacts_JQ679014_hinge_WT_ranked_0_20-frames_mask-25-45.csv"))

    def tearDown(self):
        # Clean temporary files
        shutil.rmtree(self.tmp_dir)

    def test_restricted_float(self):
        self.assertEqual(50.0, restricted_float("50.0"))
        self.assertEqual(50.0, restricted_float("50"))
        self.assertRaises(argparse.ArgumentTypeError, restricted_float, "-10.0")
        self.assertRaises(argparse.ArgumentTypeError, restricted_float, "105.0")

    def test_restricted_positive(self):
        self.assertEqual(50.0, restricted_positive("50.0"))
        self.assertRaises(argparse.ArgumentTypeError, restricted_positive, "-10.0")

    def test_restricted_angle(self):
        self.assertEqual(135.0, restricted_angle("135"))
        self.assertRaises(argparse.ArgumentTypeError, restricted_angle, "-10")
        self.assertRaises(argparse.ArgumentTypeError, restricted_angle, "361")

    def test_parse_frames(self):
        current = parse_frames(self.frames, self.inputs)
        self.assertDictEqual(current, self.parsed_frames)

    def test_resume_or_initialize_analysis(self):
        expected = copy.copy(self.analysis_yaml)
        expected["H bonds"] = {}
        expected["size Gb"] = 0
        expected["frames"] = 0
        del expected["molecules"]
        del expected["residues"]
        del expected["atoms"]
        no_resume = resume_or_initialize_analysis(self.inputs, self.topology, self.sample, self.dist_cutoff,
                                                  self.angle_cutoff, self.pct_cutoff, self.nanoseconds, None,
                                                  self.parsed_frames)
        self.assertDictEqual(no_resume, expected)
        with self.assertLogs() as cm_logs:
            resumed = resume_or_initialize_analysis(self.inputs, self.topology, self.sample, self.dist_cutoff,
                                                    self.angle_cutoff, self.pct_cutoff, self.nanoseconds,
                                                    os.path.join(TEST_DIR_EXPECTED, "test_analysis.yaml"),
                                                    self.parsed_frames)
            with open(os.path.join(TEST_DIR_EXPECTED, "test_analysis_resumed.yaml"), "r") as file_handler:
                expected_resumed = yaml.safe_load(file_handler.read())
            self.assertDictEqual(resumed, expected_resumed)
        self.assertEqual(cm_logs.records[0].getMessage(), f"resumed analysis from YAML file: {self.analysis_yaml}")



    def test_load_trajectory(self):
        traj = load_trajectory([os.path.join(TEST_DIR, "test_files", "test_data_20-frames.nc")],
                               os.path.join(TEST_DIR, "test_files", "test_data.parm"))
        self.assertEqual(traj.n_frames, 10)


    # def test_record_analysis_parameters(self):
    #     observed_path = os.path.join(self.tmp_dir, os.path.join(f"{self.sample}_analysis_parameters.yaml"))
    #     record_analysis_yaml(self.tmp_dir, self.sample, self.traj, self.dist_cutoff, self.angle_cutoff,
    #                          self.pct_cutoff, self.frames, self.md_time)
    #     with open(observed_path, "r") as file_handler:
    #         observed = yaml.safe_load(file_handler.read())
    #         self.assertEqual(observed, self.parameters)



    # def test_hydrogen_bonds(self):
    #     unique_id = str(uuid.uuid1())
    #     h_bonds = hydrogen_bonds(self.traj, self.dist_thr, self.contacts_frame_thr_2nd_half, self.pattern_contact,
    #                              self.tmp_dir, unique_id)
    #     path_observed = os.path.join(self.tmp_dir, unique_id)
    #     h_bonds.to_csv(path_observed, index=False)
    #     with open(path_observed, "r") as observed_file:
    #         observed = "".join(observed_file.readlines())
    #     self.assertEqual(observed, self.h_bonds)
    #
    # def test_contacts_csv(self):
    #     unique_id = str(uuid.uuid1())
    #     h_bonds = hydrogen_bonds(self.traj, self.dist_thr, self.contacts_frame_thr_2nd_half, self.pattern_contact,
    #                              self.tmp_dir, unique_id)
    #     _ = contacts_csv(h_bonds, self.tmp_dir, unique_id, self.pattern_contact, self.limits)
    #     with open(os.path.join(self.tmp_dir, f"contacts_{unique_id}.csv"), "r") as observed_file:
    #         observed_contacts = "".join(observed_file.readlines())
    #     self.assertEqual(observed_contacts, self.contacts)


if __name__ == "__main__":
    unittest.main()
