from typing import Dict
import numpy as np
import copy

class EnvironmentConfiguration():
    def __init__(self):
        pass

    def setup_config(self, s: Dict):
        self.n_receivers = s['n_receivers']
        self.n_cali_points = s['n_cali_points']
        self.receiver_positions = np.zeros((self.n_receivers, 2, 3))
        self.cali_points = np.zeros((self.n_cali_points, 3))

        receivers = s['receivers']
        for i in range(self.n_receivers):
            id = str(i + 1)
            receiver_name = "receiver_" + id
            self.receiver_positions[i][0] = copy.deepcopy(receivers[receiver_name]['at1_pos'])
            self.receiver_positions[i][1] = copy.deepcopy(receivers[receiver_name]['at2_pos'])

        cali_points = s['cali_points']
        for i in range(self.n_cali_points):
            id = str(i + 1)
            cali_point_name = "cp" + id
            self.cali_points[i] = copy.deepcopy(cali_points[cali_point_name])

        pass