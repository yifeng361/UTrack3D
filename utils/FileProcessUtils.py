import numpy as np
import copy
import collections
import json
import sys
sys.path.append("data_structure")
from EnvironmentConfiguration import *
from ExperimentDescription import *

class FileProcessUtils():
    cir_data_elements_perline = 209
    n_dist_info = 8
    n_cir_elements = 200

    def __init__(self):
        self.print_detailed_info = True
    
    def get_cir_and_dist(self, filepath: str, start_cir_index: int, 
                         n_taps: int):
        """ Given filepath, start_cir_index and n_taps, return the distinfo and
            cirs included in this file.

            start_cir_index and n_taps are used to filter invalid packets where
            detected first peaks are not in range [start_cir_index, start_cir_index + n_taps]
        """
        n_dist_info = self.n_dist_info
        n_cir_elements = self.n_cir_elements
        n_cir_taps = n_cir_elements // 2
        cir_data_elements_perline = self.cir_data_elements_perline
        f = open(filepath)
        data = f.readlines()

        all_dists, all_cirs = np.zeros((50000, n_dist_info)), \
            np.zeros((50000, n_cir_taps), dtype=np.complex64)
        
        error_code = collections.defaultdict(int)
        n_error = 0
        i_data = 0
        prev_seq = -1
        for line in data:
            linedata_list = line.split(',')
            n = len(linedata_list)
            if n != cir_data_elements_perline:
                x = 1
                if "Error in packet " in line:
                    error_info_list = line.split()
                    error_code[error_info_list[-1]] += 1
                    n_error += 1
                continue
            distinfo = linedata_list[0:n_dist_info]
            cir_elements = linedata_list[n_dist_info:n_dist_info + n_cir_elements]
            distinfo = np.array([float(x) for x in distinfo])
            cir_elements = np.array([float(x) for x in cir_elements])
            seq = int(distinfo[3])

            fp_index = distinfo[6] + distinfo[7] / 64
            # if fp_index >= start_cir_index + n_taps or fp_index <= start_cir_index:
            #     continue
            cir_taps = cir_elements[0:n_cir_elements:2]+1j*cir_elements[1:n_cir_elements:2]

            all_dists[i_data] = copy.deepcopy(distinfo)
            all_cirs[i_data] = copy.deepcopy(cir_taps)
            i_data += 1

            if seq == 31856:
                x = 1
            
            if prev_seq != -1 and prev_seq + 1 != seq:
                x = 1
            prev_seq = seq

        all_dists = all_dists[0:i_data]
        all_cirs = all_cirs[0: i_data]

        # Statistics
        timestamps = all_dists[:, 5] / 1e6
        n_wraps = 0
        global_timestamps = np.zeros(timestamps.shape)
        for i in range(global_timestamps.shape[0]):
            if i >= 1 and timestamps[i] < timestamps[i-1]:
                n_wraps += 1
            global_timestamps[i] = timestamps[i] + n_wraps * 2**40 / 499.2e6 / 128

        global_timestamps -= global_timestamps[0]
        t = global_timestamps[-1] - global_timestamps[0]
        seqs = all_dists[:, 3]
        n_send = seqs[-1] - seqs[0] + 1 if seqs[-1] > seqs[0] else seqs[-1] + 65536 - seqs[0] + 1
        n_recvd = i_data
        packet_rate = n_recvd / t
        if self.print_detailed_info:
            print(f"filepath={filepath} statistics: send: {n_send}, received: {n_recvd}, "
                f"reception ratio={n_recvd / n_send}, error detected={n_error}, "
                f"packet rate={packet_rate}Hz")
            print(f"error: {error_code}")
        return all_dists, all_cirs



    def decode_linedata(self, linedata: str):
        """ This function reads one line of data and outputs distinfo, raw cirs,
        and if this is a valid line.
        """
        linedata_list = linedata.split(',')
        n = len(linedata_list)
        if n != self.cir_data_elements_perline:
            return [], [], False
        distinfo = linedata_list[0:self.n_dist_info]
        cir_elements = linedata_list[self.n_dist_info:self.n_dist_info + self.n_cir_elements]
        distinfo = np.array([float(x) for x in distinfo])
        cir_elements = np.array([float(x) for x in cir_elements])
        cir_taps = cir_elements[0:self.n_cir_elements:2]+1j*\
            cir_elements[1:self.n_cir_elements:2]
        return distinfo, cir_taps, True
    
    def load_groundtruth_data(self, filepath, marker_count=3, idx0=6): 
        """ This function reads the csv file given by MoCap to extract 
        ground-truth positions.
        Args:
            filepath: The path of the ground-truth file.
            marker_count: The number of markers in your rigid body. Check MoCap
                          settings to get this value.
            idx0: The starting column index indicating the postions of the first
                  marker. Check csv file to specify this value.
        Returns:
            frames: frame seq.
            time: timnstamps.
            positions: n x (3k) array where k is the number of markers.
            masks: n x k indicating if there is a measurement for idx=i and 
                   marker id = j.
        """
        f = open(filepath, 'r')
        data = f.readlines()
        data_start = False
        n_lines = 200000
        i_frame = 0
        frames = np.zeros((n_lines, 1))
        time = np.zeros((n_lines, 1))
        positions = np.zeros((n_lines, 3*marker_count))
        masks = np.zeros((n_lines, marker_count))
        idx0 = 6
        for linedata in data:
            if "Frame,Time," in linedata:
                data_start = True
                continue
            if data_start:
                strlist = linedata.split(',')
                frames[i_frame] = int(strlist[0])
                time[i_frame] = float(strlist[1])
                for j in range(marker_count):
                    tmp = [x for x in strlist[idx0+4*j: idx0+4*j+3]]
                    if len(tmp[0]) == 0 or len(tmp[1]) == 0 or len(tmp[2]) == 0:
                        masks[i_frame, j] = 0
                    else:
                        masks[i_frame, j] = 1
                        positions[i_frame, 3*j:3*j+3] = np.array([float(x) for x in tmp])

                i_frame += 1

        frames = frames[:i_frame]
        time = time[:i_frame]
        positions = positions[:i_frame]
        masks = masks[:i_frame]
        return frames, time, positions, masks

    def parse_config_file(self, filepath) -> EnvironmentConfiguration:
        """ This function reads a configuration file and returns the config
        object.
        """
        f = open(filepath, 'r')
        data = json.load(f)
        config = EnvironmentConfiguration()
        config.setup_config(data)
        return config

    def parse_expr_description_file(self, filepath) -> ExperimentDescription:
        """ This function reads a description file while describes cir files,
        calibration files, expr name, etc.
        """
        f = open(filepath, 'r')
        data = json.load(f)
        expr_descp = ExperimentDescription()
        expr_descp.setup_description(data)
        return expr_descp