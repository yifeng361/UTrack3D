import sys
sys.path.append("utils")
sys.path.append("data_structure")

import os

from FileProcessUtils import *
from SignalProcessUtils import *
from TrackingConfiguration import *
from ParticleFilterConfiguration import *

from Tracker import *


# Setup. 
file_dir = "./raw_data/test3/"
expr_description_file = "expr_description.json"
n_taps = 100
up_factor = 64
start_cir_index = 30

target_start_index = 20
infinitesimal = 1e-32

# Global utils and variables.
expr_description_path = os.path.join(file_dir, expr_description_file)
fpu = FileProcessUtils()
spu = SignalProcessUtils()
expr_descp = fpu.parse_expr_description_file(expr_description_path)
dd_vars = None
groundtruth_file = os.path.join("raw_data", expr_descp.groundtruth_file) if \
    (expr_descp.groundtruth_file is not None and len(expr_descp.groundtruth_file) > 0) else None

config = fpu.parse_config_file(os.path.join(file_dir, expr_descp.config_file))
do_calibration = (len(expr_descp.oncable_delay) == 0)
oncable_delay = copy.deepcopy(expr_descp.oncable_delay) if not do_calibration else []

tracking_config = TrackingConfiguration()

mTracker = Tracker(env_config=config,
                   expr_descp = expr_descp,
                   tracking_config = tracking_config,
                   do_calibration = do_calibration,
                   calibration_files=expr_descp.cali_files, 
                   cir_files=expr_descp.cir_files,
                   file_dir=file_dir,
                   n_taps=n_taps,
                   up_factor=up_factor,
                   start_cir_index=start_cir_index,
                   target_start_index = target_start_index,
                   groundtruth_file = groundtruth_file
                   )
mTracker.preprocess()

pf_config = ParticleFilterConfiguration() 
mTracker.test_peak_selection_fused_with_PF(pf_config, debug=True)