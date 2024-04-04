from typing import List
import os
import sys
sys.path.append("utils")
sys.path.append("data_structure")

from FileProcessUtils import *
from SignalProcessUtils import *
from PeakSelector import *
from MathUtils import *
import numpy as np
from EnvironmentConfiguration import *
from TrackingConfiguration import *
from ParticleFilterConfiguration import *
import scipy.io as sio
import scipy.stats as sstats
import time
import copy

class Tracker():
    def __init__(self, tracking_config: TrackingConfiguration,
                 env_config: EnvironmentConfiguration, 
                 expr_descp: ExperimentDescription,
                 do_calibration: bool,
                 calibration_files: List[List[str]], cir_files: List[str],
                 file_dir: str, n_taps: int, up_factor: int, start_cir_index: int,
                 target_start_index: int, groundtruth_file=None):
        self.speed_of_light = tracking_config.speed_of_light
        self.freq = tracking_config.freq
        self.wavelen = self.speed_of_light / self.freq

        # Fixed variables. DON'T change it unless Trek1000 code is modified (Even
        # if Trek1000 code is modified, it is suggested that these variables are
        # not changed).
        self.SEQ_IDX = 3
        self.BOARD_ID_IDX = 1
        self.FP_INT_IDX = 6
        self.FP_FRAC_IDX = 7

        # Copy the external parameters.
        self.cali_files = [[fn for fn in line] for line in calibration_files]
        self.cir_files = [fn for fn in cir_files]
        self.cali_cirs = []
        self.cali_upsample_cirs = []
        self.do_calibration = do_calibration
        self.n_receivers = env_config.n_receivers
        self.antenna_pos = copy.deepcopy(env_config.receiver_positions)
        self.n_cali_points = env_config.n_cali_points
        self.cali_points = copy.deepcopy(env_config.cali_points)
        self.oncable_delay = copy.deepcopy(expr_descp.oncable_delay)
        self.init_position = copy.deepcopy(expr_descp.init_position)
        self.align_indices = copy.deepcopy(expr_descp.align_indices)

        # Ground-truth variables.
        self.groundtruth_file = groundtruth_file
        self.gt_frames = None
        self.gt_time = None
        self.gt_marker_positions = None # n x (3k), k is the number of markers.
        self.gt_masks = None


        # Parameters for tracking.
        self.tracking_config = tracking_config

        # Variables to maintain.
        self.dist_infos = []
        self.cirs = []
        self.upsample_cirs = []
        self.file_dir = file_dir
        self.fpu = FileProcessUtils()
        self.spu = SignalProcessUtils()
        self.mpu = MathUtils()
        self.n_taps = n_taps
        self.up_factor = up_factor
        self.start_cir_index = start_cir_index
        self.target_start_index = target_start_index
        self.start_up_cir_index = target_start_index * up_factor
        self.env_config = env_config
        
        self.received_seqs = None # sequences from min_seq to max_seq

        # Received packets at each seq. Fill None if there is no received cir 
        # of this board at this seq #. E.g.,
        #-------------------------------------------------
        #          board 1      board 2     board 3
        #   seq17    cir          cir         cir
        #   seq18    cir          None        cir
        #    ...
        self.cirs_by_at = None
        self.cirs_all_available_masks = None # False if the cir of seq # is missing. 

    def preprocess(self):
        """ This function reads CIRs from files and store them.
        """
        file_dir = self.file_dir
        n_taps, up_factor, start_cir_index = self.n_taps, self.up_factor, \
            self.start_cir_index
        target_up_fp_idx = self.start_up_cir_index
        fpu = self.fpu
        spu = self.spu

        # Load groundtruth.
        if self.groundtruth_file is not None:
            self.gt_frames, self.gt_time, self.gt_marker_positions, self.gt_masks=\
                self.fpu.load_groundtruth_data(self.groundtruth_file)
            print(f"Ground-truth loaded. Frame count={self.gt_frames[-1]}, time={self.gt_time[-1]}")

        # Read cirs.
        iCir = 0
        t1 = time.time()
        for cir_file in self.cir_files:
            dists, cirs = fpu.get_cir_and_dist(
                    os.path.join(file_dir, cir_file), start_cir_index, n_taps)
            fp_indices = start_cir_index + dists[:, self.FP_FRAC_IDX] / 64
            up_fp_indices = np.round(fp_indices*up_factor).astype(np.int32)
            upsamp_cirs = np.zeros((len(cirs), up_factor*n_taps), dtype=np.complex64) # upsampled cirs

            for i in range(len(cirs)):
                up_fp_idx = up_fp_indices[i]
                tmp = spu.upsample(cirs[i], factor=64)
                vq = spu.freq_filter(tmp, up_factor*n_taps, n_taps//2)
                vq_aligned = spu.align_cir(vq, up_fp_idx, target_up_fp_idx)
                upsamp_cirs[i] = copy.deepcopy(vq_aligned)
                iCir += 1
            self.cirs.append(cirs)
            self.upsample_cirs.append(upsamp_cirs)
            self.dist_infos.append(dists)

        print(f"Receive {iCir} CIRs, take time = {time.time() - t1}s")

        # Sync cir data, reshape CIRs.
        n_boards = len(self.dist_infos)
        min_seq, max_seq = int(1e8), -1
        for i in range(len(self.dist_infos)):
            min_seq = min(min_seq, min(self.dist_infos[i][:, self.SEQ_IDX]))
            max_seq = max(max_seq, max(self.dist_infos[i][:, self.SEQ_IDX]))
        min_seq, max_seq = int(min_seq), int(max_seq)
        self.received_seqs = [i for i in range(min_seq, max_seq+1)]
        self.cirs_by_at = [[np.array([]) for j in range(n_boards)] for i in range(min_seq, max_seq + 1)]
        self.cirs_all_available_masks = [[True for j in range(n_boards)] for i in range(min_seq, max_seq + 1)]

        for i in range(len(self.cirs)):
            for j in range(len(self.cirs[i])):
                seq = int(self.dist_infos[i][j][self.SEQ_IDX])
                seq_idx = seq - min_seq
                board_id_idx = int(self.dist_infos[i][j][self.BOARD_ID_IDX] - 1) # board_id_idx labels from 1, need to minus 1
                self.cirs_by_at[seq_idx][board_id_idx] = copy.deepcopy(self.upsample_cirs[i][j])

        # The first seq no where all the three boards receive signals.
        first_goodidx = 0
        for i in range(len(self.cirs_by_at)):
            all_good = True
            for j in range(len(self.cirs_by_at[i])):
                if self.cirs_by_at[i][j].shape[0] == 0:
                    all_good = False
                    break
            if all_good:
                first_goodidx = i
                break

        for i in range(len(self.cirs_by_at)):
            for j in range(len(self.cirs_by_at[i])):
                if self.cirs_by_at[i][j].shape[0] == 0:
                    self.cirs_all_available_masks[i][j] = False

        self.skipstart = first_goodidx
        self.skipend = 1
        self.cirs_by_at = self.cirs_by_at[first_goodidx:-self.skipend]
        self.cirs_all_available_masks = self.cirs_all_available_masks[first_goodidx:-self.skipend]
        self.received_seqs = self.received_seqs[first_goodidx:-self.skipend]

    def test_calibrate_oncable_delay(self, debug=False):
        """ Compute on-cable delay for each board.

        Formula:
            cir_t2 - cir_t1 = air_t2 - air_t1 + on_cable_delay
            Use CIR channel to estimate phase wrapping integer (Could be wrong).

        Returns:
            A triple (t1, t2, t3)
        """
        start_idx = self.start_cir_index
        minimum_gap = 800
        search_region_length = 400
        peak_threshold = 20

        # compute air_time_diff for each cali point and each board using 
        # ground truth.
        n_cali_points = self.n_cali_points
        n_receivers = self.n_receivers
        n_at_per_receiver = 2

        # dist(p, antenna2) - dist(p, antenna1)
        peak_selector = PeakSelector(self.tracking_config, n_at_per_receiver)
        mpu = self.mpu
        alalg = self.alalg
        air_time_diff = np.zeros((n_cali_points, n_receivers)) 
        for i in range(n_cali_points):
            for j in range(n_receivers):
                d1 = np.linalg.norm(self.cali_points[i] - self.antenna_pos[j, 0])
                d2 = np.linalg.norm(self.cali_points[i] - self.antenna_pos[j, 1])
                air_time_diff[i][j] = d2 - d1
        
        debug_dict = {}
        peak_indices = [[np.zeros((self.cali_cirs[i][j].shape[0], 2)) 
                         for i in range(n_receivers)] for j in range(n_cali_points)]
        peak_phases =  [[np.zeros((self.cali_cirs[i][j].shape[0], 2)) 
                         for i in range(n_receivers)] for j in range(n_cali_points)]
        peak_timediffs = [[np.zeros((self.cali_cirs[i][j].shape[0])) 
                         for i in range(n_receivers)] for j in range(n_cali_points)]
        peak_phasediffs = [[np.zeros((self.cali_cirs[i][j].shape[0])) 
                         for i in range(n_receivers)] for j in range(n_cali_points)]
        
        # Obtain first peak and second peak for each sequence.
        for i in range(n_cali_points):
            for j in range(n_receivers):
                n = len(self.cali_cirs[i][j])
                peak_indices = np.zeros((n, n_at_per_receiver), dtype=np.int64)
                for k in range(n):
                    fp, sp = peak_selector.primary_search_notimediff(
                        self.cali_cirs[i][j], thre=peak_threshold, 
                        start_idx=start_idx, minimum_gap=minimum_gap, 
                        search_region_length=search_region_length)
                    if fp == 0 or sp == 0:
                        fp, sp, fp_phase, sp_phase = 0, 0, 0, 0
                    else:
                        fp, sp = 0, 0
                        fp_phase, sp_phase = np.angle(self.cali_cirs[i][j][k][fp]),\
                            np.angle(self.cali_cirs[i][j][k][sp])
                    peak_indices[i][j][k][0] = fp
                    peak_indices[i][j][k][1] = sp
                    peak_phases[i][j][k][0] = fp_phase
                    peak_phases[i][j][k][1] = sp_phase
                    peak_timediffs[i][j][k] = sp - fp
                    peak_phasediffs[i][j][k] = mpu.wrap_phase(sp_phase - fp_phase)
                
                if debug:
                    debug_dict[f"cali_{i+1}_rx_{j+1}_peak_indices"] = peak_indices[i][j]
                    debug_dict[f"cali_{i+1}_rx_{j+1}_peak_phases"] = peak_phases[i][j]

        # Compute phase diff.
        
        if debug:
            sio.savemat("./output/debug/debug.mat", debug_dict)
        pass

    def test_peak_selection_fused_with_PF(self, pf_config: ParticleFilterConfiguration,
                                          debug=False):
        OPTIMZIATION_ONLY, PF_ONLY = 0, 1

        n_particles = pf_config.n_particles
        xyz_sigma = pf_config.xyz_sigma
        xyz_beta = pf_config.xyz_beta
        n_at_per_receiver = 2
        topn=self.tracking_config.topn

        mpu = self.mpu

        peak_selector = PeakSelector(self.tracking_config, n_at_per_receiver)
        start_idx = self.start_up_cir_index
        n_seqs, n_receivers = len(self.cirs_by_at), self.n_receivers
        if debug:
            debug_dict = {}


        # Extract two peaks from each CIR.
        candidate_peaks = np.zeros((n_seqs, n_receivers, 2, 5), dtype=int)
        candidate_peak_features = [[0 for j in range(n_receivers)] for i in range(n_seqs)]
        
        # A list with shape=n_cirs x n_receivers. Each entry (i,j) is a list of k_ij x 2 array.
        peak_indices = [[np.zeros((1,2), dtype=int) for j in range(n_receivers)] for i in range(n_seqs)] 
        peak_phases = [[np.zeros((1,2)) for j in range(n_receivers)] for i in range(n_seqs)] 
        # A list with the length=n_cirs x n_receivers. Each entry (i,j) is a list of k_ij x 1 array.
        peak_phasediffs = [[np.zeros((1,1), dtype=int) for j in range(n_receivers)] for i in range(n_seqs)]
        peak_timediffs = [[np.zeros((1,1)) for j in range(n_receivers)] for i in range(n_seqs)] 

        # Track: global states.
        localization_method = np.zeros(n_seqs, dtype=int)
        est_positions = np.zeros((n_seqs, 3))
        est_velocities = np.zeros((n_seqs, 3))
        phasediffs = np.zeros((n_seqs, n_receivers))
        timediffs = np.zeros((n_seqs, n_receivers), dtype=int)
        accum_phasediff_array = np.zeros((n_seqs, n_receivers))

        # Track: previous states.
        prev_accum_phasediff = np.zeros(n_receivers)
        prev_position = copy.deepcopy(self.init_position)
        prev_phasediff = np.zeros(n_receivers)
        prev_timediff = np.zeros(n_receivers, dtype=int)


        manual_set_start_position = self.init_position
        start_time = time.time()

        peak_proposing_time = np.zeros(n_seqs)
        localization_time = np.zeros(n_seqs)

        for i in range(n_seqs):
            if i % 100 == 0:
                print(f"i_seq={i}, time elapsed={time.time() - start_time}s")

            if i == 0:
                if not np.all(self.cirs_all_available_masks[i]):
                    raise ValueError(f"A broken CIR is detected at the 0th "
                                         "packet. Please reset the start packet.")
                for j in range(n_receivers):
                    fp, sp = peak_selector.primary_search_notimediff(
                        self.cirs_by_at[i][j], start_idx=start_idx)
                    if fp == 0 or sp == 0:
                        raise ValueError(f"Good peaks are not available in the 0th "
                                         "packet. Please reset the start packet.")
                    fp_phase, sp_phase = np.angle(self.cirs_by_at[i][j][fp]), \
                                             np.angle(self.cirs_by_at[i][j][sp])
                    
                    peak_indices[i][j] = np.array((fp, sp))
                    peak_phases[i][j] = np.array((fp_phase, sp_phase))
                    peak_phasediffs[i][j] =  np.array([mpu.wrap_phase(sp_phase - fp_phase)])
                    peak_timediffs[i][j] = np.array([sp - fp])
                    prev_phasediff[j] = mpu.wrap_phase(sp_phase - fp_phase)
                    prev_timediff[j] = sp - fp
                    prev_accum_phasediff[j] = 0
                    accum_phasediff_array[i][j] = 0
                    phasediffs[i][j] = prev_phasediff[j]
                    timediffs[i][j] = prev_timediff[j]

                localization_method[i] = OPTIMZIATION_ONLY
                prev_position = np.array(manual_set_start_position)
                est_positions[i] = np.array(manual_set_start_position)
                est_velocities[i] = np.array([0,0,0])
           

            else:
                accum_phasediff = np.zeros(n_receivers)
                
                t1 = time.time()
                for j in range(n_receivers):
                    if not self.cirs_all_available_masks[i][j]:
                        accum_phasediff[j] = 2 * accum_phasediff_array[i-1][j] - \
                            accum_phasediff_array[i-2][j]
                        prev_phasediff[j] = prev_phasediff[j] # Do not change
                        prev_timediff[j] = prev_timediff[j] # Do not change
                        prev_accum_phasediff[j] = accum_phasediff[j]
                        timediffs[i][j] = prev_timediff[j]
                        phasediffs[i][j] =  prev_phasediff[j]

                    else:
                        if False:
                            raise NotImplementedError("PF is not implemented")
                        else:
                            # Propose candidates.
                            
                            peaks, features = peak_selector.multi_candidate_search(
                                    self.cirs_by_at[i][j], start_idx=start_idx, 
                                    prev_timediff=prev_timediff[j])
                            if peaks is None:
                                accum_phasediff[j] = 2 * accum_phasediff_array[i-1][j] - \
                                    accum_phasediff_array[i-2][j]
                                prev_phasediff[j] = prev_phasediff[j] # Do not change
                                prev_timediff[j] = prev_timediff[j] # Do not change
                                prev_accum_phasediff[j] = accum_phasediff[j]
                                timediffs[i][j] = prev_timediff[j]
                                phasediffs[i][j] =  prev_phasediff[j]
                                continue
                            else:
                                candidate_peaks[i][j] = copy.deepcopy(peaks)
                                candidate_peak_features[i][j] = copy.deepcopy(features)

                            # Filter and rank candidates
                            tmp_pairs, tmp_w = peak_selector.rank_peaks_particle_filter(
                                peaks, features, prev_timediff[j],
                                mpu)
                            if tmp_pairs.shape[0] == 0:
                                accum_phasediff[j] = 2 * accum_phasediff_array[i-1][j] - \
                                    accum_phasediff_array[i-2][j]
                                prev_phasediff[j] = prev_phasediff[j] # Do not change
                                prev_timediff[j] = prev_timediff[j] # Do not change
                                prev_accum_phasediff[j] = accum_phasediff[j]
                                timediffs[i][j] = prev_timediff[j]
                                phasediffs[i][j] =  prev_phasediff[j]
                            else:
                                fp, sp = tmp_pairs[0, 0], tmp_pairs[0, 1]
                                fp_phase = np.angle(self.cirs_by_at[i][j][fp])
                                sp_phase = np.angle(self.cirs_by_at[i][j][sp])
                                tmp_pd = mpu.wrap_phase(sp_phase - fp_phase)
                            
                                peak_indices[i][j] = np.array([fp, sp])
                                peak_phases[i][j] = np.array([fp_phase, sp_phase])
                                peak_phasediffs[i][j] = np.array([tmp_pd])
                                peak_timediffs[i][j] = np.array([sp-fp])
                                # Update states.

                                phasediffs[i][j] = tmp_pd
                                timediffs[i][j] = sp - fp
                        
                                pdc = mpu.wrap_phase(phasediffs[i][j] - phasediffs[i-1][j])
                                accum_phasediff[j] = prev_accum_phasediff[j] + pdc

                                prev_timediff[j] = timediffs[i][j]
                                prev_phasediff[j] = phasediffs[i][j]
                                prev_accum_phasediff[j] = accum_phasediff[j]
                peak_proposing_time[i] = time.time() - t1
                
                t2 = time.time()
                this_range = [(prev_position[0] - 0.2, prev_position[0] + 0.2), 
                          (prev_position[1] - 0.2, prev_position[1] + 0.2), 
                          (prev_position[2] - 0.2, prev_position[2] + 0.2)]
                est_position = mpu.localization_solver_multiboards(
                self.antenna_pos, manual_set_start_position, (-1) * accum_phasediff, 
                this_range, prev_position)
                localization_time[i] = time.time() - t2
                # Update variables.
                accum_phasediff_array[i] = accum_phasediff
                est_positions[i] = est_position
                est_velocities[i] = est_positions[i] - est_positions[i-1]
                prev_position = est_position


        if debug:
            debug_dict['candidate_peaks'] = candidate_peaks
            debug_dict['peak_indices'] = peak_indices
            debug_dict['peak_phases'] = peak_phases
            debug_dict['accum_phasediff'] = accum_phasediff_array
            debug_dict['peak_phasediffs'] = peak_phasediffs
            debug_dict['est_positions'] = est_positions
            debug_dict["peak_timediffs"] = peak_timediffs
            debug_dict["antenna_positions"] = self.antenna_pos
            debug_dict["start_position"] = self.init_position
            debug_dict["timediffs"] = timediffs
            debug_dict["align_indices"] = self.align_indices


            if self.groundtruth_file is not None:
                debug_dict['gt_frames'] = self.gt_frames
                debug_dict['gt_time'] = self.gt_time
                debug_dict['gt_marker_positions'] = self.gt_marker_positions
                debug_dict['gt_masks'] = self.gt_masks
                
            debug_dict["peak_proposing_time"] = peak_proposing_time
            debug_dict["localization_time"] = localization_time

            # for i in range(n_receivers):
            #     debug_dict[f'seq_cirs'] = self.cirs_by_at
            sio.savemat("./output/tracking_results.mat", debug_dict)

    def track_selective_approach(self, debug=True):
        raise NotImplementedError("not implemented")
        