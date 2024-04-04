from typing import Dict
import numpy as np
import copy

class TrackingConfiguration():
    
    def __init__(self):
        # For signals

        # For peak selector
        self.minimum_gap = 800
        self.search_region_length = 1600
        self.peak_threshold = 40
        self.minimum_peak_threshold = 35
        self.invisible_peak_threshold = 40
        self.max_sidepeak_distance = 320
        self.invisible_peak_der_threshold = 0.8
        self.topn = 2

        self.speed_of_light = 299792458
        self.freq = 3.993e9


        