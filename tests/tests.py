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
        self.traj1 = load_trajectory(self.inputs[0], self.topology, self.parsed_frames)
        self.traj2 = load_trajectory(self.inputs[1], self.topology, self.parsed_frames)
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

    def test_initialize_resume_analysis(self):
        # test creation
        actual = resume_or_initialize_analysis(self.inputs, self.topology, self.sample, self.dist_cutoff,
                                               self.angle_cutoff, self.pct_cutoff, self.nanoseconds, None,
                                               self.parsed_frames)
        with open(os.path.join(TEST_DIR_EXPECTED, "test_analysis_virgin.yaml"), "r") as file_handler:
            expected = yaml.safe_load(file_handler.read())
        self.assertDictEqual(actual, expected)
        # test resume
        with self.assertLogs() as cm_logs:
            actual = resume_or_initialize_analysis(self.inputs, self.topology, self.sample, self.dist_cutoff,
                                                   self.angle_cutoff, self.pct_cutoff, self.nanoseconds,
                                                   os.path.join(TEST_DIR_EXPECTED, "test_analysis.yaml"),
                                                   self.parsed_frames)
            with open(os.path.join(TEST_DIR_EXPECTED, "test_analysis_resumed.yaml"), "r") as file_handler:
                expected = yaml.safe_load(file_handler.read())
            for key in actual["H bonds"]:
                np.testing.assert_array_equal(actual["H bonds"][key], expected["H bonds"][key])
            self.assertEqual(actual["atoms"], expected["atoms"])
            self.assertEqual(actual["frames"], expected["frames"])
            self.assertEqual(actual["molecules"], expected["molecules"])
            self.assertDictEqual(actual["parameters"], expected["parameters"])
            self.assertEqual(actual["residues"], expected["residues"])
            self.assertEqual(actual["sample"], expected["sample"])
            self.assertEqual(actual["size Gb"], expected["size Gb"])
            self.assertEqual(actual["topology file"], expected["topology file"])
            self.assertListEqual(actual["trajectory files"], expected["trajectory files"])
        self.assertEqual(cm_logs.records[0].getMessage(), f"resumed analysis from YAML file: "
                                                          f"{os.path.join(TEST_DIR_EXPECTED, 'test_analysis.yaml')}")
        # test resume with discrepancies
        self.assertRaises(KeyError, resume_or_initialize_analysis, self.inputs,
                          os.path.join(TEST_DIR_EXPECTED, "test_analysis.yaml"), "John Doe", 5.0, 200, 90.0,
                          self.nanoseconds, os.path.join(TEST_DIR_EXPECTED,
                                                         "test_analysis.yaml"),
                          self.parsed_frames)

    def test_load_trajectory(self):
        self.assertEqual(self.traj1.n_frames, 15)
        self.assertRaises(IndexError, load_trajectory, self.inputs[0], os.path.join(TEST_DIR_INPUTS, "test_data.parm"),
                          {"test_data_20-frames.nc": {"begin": 5, "end": 100}})

    def test_check_trajectories_consistency(self):
        with open(os.path.join(TEST_DIR_EXPECTED, "test_analysis.yaml"), "r") as file_handler:
            expected = yaml.safe_load(file_handler.read())
            del expected["H bonds"]
            del expected["parameters"]
            del expected["sample"]
            del expected["topology file"]
            del expected["trajectory files"]
        with open(os.path.join(TEST_DIR_EXPECTED, "test_check_consistency.yaml"), "r") as file_handler:
            actual_data1 = yaml.safe_load(file_handler.read())
        actual1 = check_trajectories_consistency(self.traj2, self.inputs[1], actual_data1)
        self.assertDictEqual(actual1, expected)
        with open(os.path.join(TEST_DIR_EXPECTED, "test_check_consistency.yaml"), "r") as file_handler:
            actual_data2 = yaml.safe_load(file_handler.read())
        del actual_data2["residues"]
        actual2 = check_trajectories_consistency(self.traj2, self.inputs[1], copy.copy(actual_data2))
        self.assertDictEqual(actual2, expected)
        actual_data2["residues"] = 10
        self.assertRaises(ValueError, check_trajectories_consistency, self.traj2, self.inputs[1], actual_data2)
        actual_data2["residues"] = 73
        actual_data2["atoms"] = 10000
        self.assertRaises(ValueError, check_trajectories_consistency, self.traj2, self.inputs[1], actual_data2)
        actual_data2["atoms"] = 1041
        actual_data2["molecules"] = 1000
        self.assertRaises(ValueError, check_trajectories_consistency, self.traj2, self.inputs[1], actual_data2)

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
