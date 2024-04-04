from typing import Dict
import numpy as np
import copy

class ExperimentDescription():
    def __init__(self):
        pass

    def setup_description(self, s: Dict):
        # cali_files should be n x 3, n is the num of antennas. Pay antennation
        # that the idx 0, 1, 2 may not correpsond the the actual label of each
        # antenna.
        self.cali_files = s["cali_files"]

        # config_file should be a string. 
        self.config_file = s["config_file"]

        # expr_name should be a string.
        self.expr_name = s["expr_name"]

        # cir file should be 1x3.
        self.cir_files = s["cir_files"]

        # If use oncable_delay. (No calibration will be made if using this method)
        self.oncable_delay = s['oncable_delay']

        # Initial position in this experiment.
        self.init_position = s['init_position']

        # Dirty debug file in this experiment.
        if 'dirty_debug_file' in s.keys():
            self.dirty_debug_file = s['dirty_debug_file']
        else:
            self.dirty_debug_file = ""

        # Ground truth file in this experiment.
        if 'groundtruth_file' in s.keys():
            self.groundtruth_file = s["groundtruth_file"]
        else:
            self.groundtruth_file = None

        # Aligned indices for estimated trajectory and ground-truths
        if 'align_indices' in s.keys():
            self.align_indices = s["align_indices"]
        else:
            self.align_indices = [0, -1, 0, -1]