import sys
sys.path.append('data_structure')
sys.path.append('utils')

from TrackingConfiguration import *
from PeakFeature import *

from MathUtils import *

import numpy as np
import scipy
import scipy.signal
import bisect
import copy


class PeakSelector():
    ''' This class will try different peak selecting algorithms. The output
    is a list of k arrays. k = number of antennas of each board.
    '''
    prev_diff_sigma = 9.3859 # Obtained with data


    def __init__(self, tracking_config: TrackingConfiguration,
                 n_antennas_per_board):
        self.n_antennas_per_board = n_antennas_per_board
        self.minimum_gap = tracking_config.minimum_gap
        self.search_region_length = tracking_config.search_region_length
        self.peak_threshold = tracking_config.peak_threshold
        self.minimum_peak_threshold = tracking_config.minimum_peak_threshold
        self.invisible_peak_threshold =  tracking_config.invisible_peak_threshold
        self.max_sidepeak_distance = tracking_config.max_sidepeak_distance
        self.invisible_peak_der_threshold = tracking_config.invisible_peak_der_threshold
        
        self.mpu = MathUtils()
        self.tdc_sigma = 9.3859 * 3
        
        self.sp_power_L = 1
        self.sp_power_K = 0.1
        self.sp_power_x0 = 30


        self.tdc_cut_threshold = 128
        self.prominent_power_threshold = 80

        pass

    def primary_search_notimediff(self, cir: np.ndarray, start_idx):
        ''' This is the primary searching algorithm. This algorithm does not
        take timediff into consideration. This algorithm avoids the accumulated 
        tracking error, but has a higher risk of selecting unwanted peaks.
        It searches the first prominent peak in [l0, l0 + search_region_length] 
        as the first peak, and the first prominent peak in 
        [r0, r0 + search_region_length] as the second peak.

        Args:
            threshold: The minimum acceptable amplitude to recognize a peak as the
                  received signal. If there is no strong_enough peak, return 
                  None for this antenna.
            start_idx: cir_start_index (after upsampling)
            minimum_gap: minimum gap between the first peak, and the second peak.
            search_region_length: length of the region in which we search for the
                                  peak.
        '''
        minimum_gap = self.minimum_gap
        search_region_length = self.search_region_length
        peak_threshold = self.prominent_power_threshold
        peak_threshold = 50

        l, r = start_idx, start_idx + search_region_length
        all_peakidxs = scipy.signal.find_peaks(abs(cir))[0]
        tmp_idxs = list(np.where(abs(cir[all_peakidxs]) > peak_threshold)[0])
        goodpeak_idxs = all_peakidxs[tmp_idxs]
        
        first_found, second_found, fpidx, spidx = False, False, 0, 0
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        if tmpidx < goodpeak_idxs.shape[0]:
            fpidx = goodpeak_idxs[tmpidx]
        
        if fpidx <= r:
            first_found = True
        
        l, r = fpidx + minimum_gap, fpidx + minimum_gap + search_region_length
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        if tmpidx < goodpeak_idxs.shape[0]:
            spidx = goodpeak_idxs[tmpidx]

        if spidx <= r:
            second_found = True        
        peak_indices = (fpidx if first_found else 0, 
                        spidx if second_found else 0)
        return peak_indices
    
    def primary_search_withtimediff(self, cir: np.ndarray, start_idx, prev_timediff):
        ''' This is the primary searching algorithm. This algorithm considers
        timediff. This algorithm may have accumulated error, but is less likely 
        to choose the unwanted peaks. It searches the first prominent peak in 
        [l0, l0 + search_region_length]  as the first peak. Then using first peak
        as reference, it searches a prominent peak which is closest to 
        fp + prev_timediff.

        Args:
            threshold: The minimum acceptable amplitude to recognize a peak as the
                  received signal. If there is no strong_enough peak, return 
                  None for this antenna.
            start_idx: cir_start_index (after upsampling)
            minimum_gap: minimum gap between the first peak, and the second peak.
            search_region_length: length of the region in which we search for the
                                  peak.
            prev_timediff: Time difference of the first and the second peak in
                           the previous packet.
        '''
        minimum_gap = self.minimum_gap
        search_region_length = self.search_region_length
        peak_threshold = self.peak_threshold

        l, r = start_idx, start_idx + search_region_length
        all_peakidxs = scipy.signal.find_peaks(abs(cir))[0]
        tmp_idxs = list(np.where(abs(cir[all_peakidxs]) > peak_threshold)[0])
        goodpeak_idxs = all_peakidxs[tmp_idxs]
        
        first_found, second_found, fpidx, spidx = False, False, 0, 0
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        if tmpidx < goodpeak_idxs.shape[0]:
            fpidx = goodpeak_idxs[tmpidx]
        
        if fpidx <= r:
            first_found = True
        
        half_length = search_region_length // 2
        mid = fpidx + prev_timediff
        l, r = fpidx + prev_timediff - half_length, fpidx + prev_timediff + half_length
        iL = bisect.bisect_left(goodpeak_idxs, mid) - 1
        lidx = goodpeak_idxs[iL] if iL < goodpeak_idxs.shape[0] else 0
        iR = bisect.bisect_left(goodpeak_idxs, mid)
        ridx = goodpeak_idxs[iR] if iR < goodpeak_idxs.shape[0] else 0
        spidx = ridx if abs(mid - lidx) > abs(ridx - mid) else lidx

        if spidx <= r:
            second_found = True        
        peak_indices = (fpidx if first_found else 0, 
                        spidx if second_found else 0)
        return peak_indices

    def primary_search_with_neighbor_peaks(self, cir: np.ndarray, start_idx,
                                           use_timediff=False, prev_timediff=0):
        ''' The algorithm is the same with primary search, but apart from the 
        anchor peaks, this algorithm will also return the left and right peak
        which satisfying the result.

        Args (read from config):
            minimum_threshold: All returned peaks must have an amplitude larger 
            than minimum_threshold. Otherwise fill with 0.
            max_sidepeak_distance: The left and the right peak should have a 
            distance less or equal to max_sidepeak_distance.
        Returns:
            result_peaks: n_antennas x 3 array
            neighbor_peaks: n_antennas x 2 (the exact left and right peak of the
                anchors, regardless of the amplitude)
        '''
        minimum_gap = self.minimum_gap
        search_region_length = self.search_region_length
        peak_threshold = self.peak_threshold
        minimum_peak_threshold = self.minimum_peak_threshold
        max_sidepeak_distance = self.max_sidepeak_distance

        
        result_peaks = np.zeros((2, 3), dtype=int)
        neighbor_peaks = np.zeros((2, 2), dtype=int)

        # l, r = start_idx, start_idx + search_region_length
        l, r = start_idx, start_idx + 300
        all_peakidxs = scipy.signal.find_peaks(abs(cir))[0]
        all_peakvals = abs(cir[all_peakidxs])
        
        tmp_idxs = list(np.where(abs(cir[all_peakidxs]) > peak_threshold)[0])
        goodpeak_idxs = all_peakidxs[tmp_idxs]
        
        fpidx, spidx = 0, 0
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        fpidx =  goodpeak_idxs[tmpidx] if tmpidx < goodpeak_idxs.shape[0] else 0
        if fpidx == 0:
            return result_peaks, neighbor_peaks

        result_peaks[0, 1] = fpidx if fpidx <= r else 0

        # j is the index of left peak of the first peak index.
        # k is the index of right peak of the first peak index.
        j = bisect.bisect_left(all_peakidxs, fpidx) - 1
        k = bisect.bisect_left(all_peakidxs, fpidx) + 1
        if j >= 0:
            neighbor_peaks[0, 0] = all_peakidxs[j]
        if k < all_peakidxs.shape[0]:
            neighbor_peaks[0, 1] = all_peakidxs[k]
        
        while j >= 0:
            if abs(all_peakidxs[j] - fpidx) >= max_sidepeak_distance:
                break
            if all_peakvals[j] > minimum_peak_threshold:
                result_peaks[0, 0] = all_peakidxs[j]
                break
            j -= 1

        while k < all_peakidxs.shape[0]:
            if abs(all_peakidxs[k] - fpidx) >= max_sidepeak_distance:
                break
            if all_peakvals[k] > minimum_peak_threshold:
                result_peaks[0, 2] = all_peakidxs[k]
                break
            k += 1
            
        # Search for second peak.
        if use_timediff:
            half_length = search_region_length // 2
            mid = fpidx + prev_timediff
            l, r = fpidx + prev_timediff - half_length, fpidx + prev_timediff + half_length
            iL = bisect.bisect_left(goodpeak_idxs, mid) - 1 # Not include mid if mid is in
            lidx = goodpeak_idxs[iL] if iL >= 0 and iL < goodpeak_idxs.shape[0] else 0
            iR = bisect.bisect_left(goodpeak_idxs, mid)
            ridx = goodpeak_idxs[iR] if iR >= 0 and iR < goodpeak_idxs.shape[0] else 0
            spidx = ridx if abs(mid - lidx) > abs(ridx - mid) else lidx
        else:
            l, r = fpidx + minimum_gap, fpidx + minimum_gap + search_region_length
            tmpidx = bisect.bisect_left(goodpeak_idxs, l)
            spidx = goodpeak_idxs[tmpidx] if tmpidx < goodpeak_idxs.shape[0] else 0

        if spidx == 0:
            return result_peaks, neighbor_peaks

        result_peaks[1, 1] = spidx if spidx <= r else 0

        # j is the index of left peak of the first peak index.
        # k is the index of right peak of the first peak index.
        j = bisect.bisect_left(all_peakidxs, spidx) - 1
        k = bisect.bisect_left(all_peakidxs, spidx) + 1
        if j >= 0:
            neighbor_peaks[1, 0] = all_peakidxs[j]
        if k < all_peakidxs.shape[0]:
            neighbor_peaks[1, 1] = all_peakidxs[k]

        while j >= 0:
            if abs(all_peakidxs[j] - spidx) >= max_sidepeak_distance:
                break
            if all_peakvals[j] > minimum_peak_threshold:
                result_peaks[1, 0] = all_peakidxs[j]
                break
            j -= 1
        while k < all_peakidxs.shape[0]:
            if abs(all_peakidxs[k] - spidx) >= max_sidepeak_distance:
                break
            if all_peakvals[k] > minimum_peak_threshold:
                result_peaks[1, 2] = all_peakidxs[k]
                break
            k += 1
        
        return result_peaks, neighbor_peaks
    
    def multi_candidate_search(self, cir: np.ndarray, start_idx, prev_timediff):
        '''
        First locate the "anchor peak" for each antenna.
        Then search around the anchor peak to propose multiple peaks.
        For each antenna, there are at most 5 candidates, we should locate at 
        most 1 left peak, and 1 right peak, and an "invisible peak" on the left,
         "an invisible peak" on the right.

        This method cannot find low-power peaks, so we need some other scheme 
        to find low-power peaks.

        Args:
            threshold: The minimum acceptable amplitude to recognize a peak as the
                       received signal. If there is no strong_enough peak, return 
                       None for this antenna.
            start_idx: cir_start_index (after upsampling)
            minimum_gap: minimum gap between the first peak, and the second peak.
            search_region_length: length of the region in which we search for the
                                  peak.
            minimum_threshold: All returned peaks must have an amplitude larger 
                               than minimum_threshold. Otherwise fill with 0.
            max_sidepeak_distance: The left and the right peak should have a 
                                   distance less or equal to max_sidepeak_distance.
            invisible_peak_threshold: The minimum amplitude of the invisible (merged)
                                      peak.
        
        Returns:
            2 x 5 array representing the proposed peaks for each antenna.
            For each antenna, the format is
            [left peak of the anchor peak, left invisible peak of the anchor peak,
            anchor peak, right invisible peak of the anchor peak,
            right peak of the anchor peak]
        '''
        minimum_gap = self.minimum_gap
        search_region_length = self.search_region_length
        peak_threshold = self.peak_threshold
        minimum_peak_threshold = self.minimum_peak_threshold
        max_sidepeak_distance = self.max_sidepeak_distance
        invisible_peak_threshold = self.invisible_peak_threshold
        invisible_peak_der_threshold = self.invisible_peak_der_threshold

        # Step 1: Find anchor peaks, and its one left and one right peak.
        peak_search_error = False
        fp, sp = 0, 0
        thre = peak_threshold
        while fp == 0 or sp == 0: 
            if thre < minimum_peak_threshold:
                peak_search_error = True
                break
            result_peaks, neighbor_peaks = self.primary_search_with_neighbor_peaks(
                cir, start_idx, use_timediff=True, prev_timediff=prev_timediff)
            fp, sp = result_peaks[0, 1], result_peaks[1, 1]
            thre /= 2

        if peak_search_error:
            return None, None

        # Step 2: Find invisible peaks between anchor and its direct left/right 
        # peak.
        y = copy.deepcopy(abs(cir))
        n = cir.shape[0]
        der_y, der_der_y, zc = np.zeros(n), np.zeros(n), np.zeros(n)
        der_y[0:n-1] = y[1: n] - y[0: n-1]
        der_der_y[0: n-1] = der_y[1: n] - der_y[0: n-1]
        der_der_y[-2] = 0
        zc[0:n-1] = np.multiply(der_der_y[0:n-1], der_der_y[1:n]) 

        # Condition: zc < 0,  y large enough, with the smallest der_y.
        invisible_peaks = np.zeros((2, 2))
        for i in range(neighbor_peaks.shape[0]):
            for j in range(2):
                if j == 0:
                    l, r = neighbor_peaks[i, 0], result_peaks[i, 1]
                else:
                    l, r = result_peaks[i, 1], neighbor_peaks[i, 1]
                idxs = np.where((zc[l:r] <= 0) & (y[l:r] > invisible_peak_threshold))[0]
                if idxs.shape[0] == 0:
                    continue
                idxs += l
                abs_der_y_of_idxs = abs(der_y[idxs])
                minidx = np.argmin(abs_der_y_of_idxs)
                if abs_der_y_of_idxs[minidx] < invisible_peak_der_threshold:
                    mink = idxs[minidx]
                    if abs(cir[mink]) < minimum_peak_threshold:
                        for k in range(max(mink-64, l), min(mink+64, r)):
                            if abs(cir[k]) >= minimum_peak_threshold:
                                invisible_peaks[i][j] = mink
                                break
                    else:
                        invisible_peaks[i][j] = mink
                else:
                    invisible_peaks[i][j] = 0
                    
        proposed_peaks = np.zeros((2, 5), dtype=int)
        for i in range(proposed_peaks.shape[0]):
            proposed_peaks[i, 0] = result_peaks[i, 0]
            proposed_peaks[i, 1] = invisible_peaks[i, 0]
            proposed_peaks[i, 2] = result_peaks[i, 1]
            proposed_peaks[i, 3] = invisible_peaks[i, 1]
            proposed_peaks[i, 4] = result_peaks[i, 2]

        # Peak features.
        peak_features = [[PeakFeature(cir, proposed_peaks[i,j]) 
                          for j in range(5)] for i in range(2)]

        return proposed_peaks, peak_features
    
    def rank_peaks_particle_filter(self, proposed_peaks, peak_features, 
                                   prev_tdc, mpu: MathUtils):
        """ Given a few peak candidates for the first, and the second peak,
        Output the high-condident peaks. Rank them. 

        Algorithm:
            The algorithm is two step:
            Step 1: Check if there is a proliminent peak that definitely is the 
            peak we want. The way to determine this is to check if it is temorally
            early and has high power.
            Step 2: Given all the candidates in step 1, rank them according to
            timediff error. Compute the difference between the errors of best and
            the worst. If they have a big gap, remove the worst one. Iteratively 
            doing this until all of the condidates have approximately similar errors. 

        Returns:
             peak_pairs: kx2 (k depends on the algorithm, could be zero)
             w_prior: the prior confidence which is calculated from timediff only.
        """
        if prev_tdc == 0:
            raise ValueError("Prev_tdc is zero. This function should not be called. "
                             "Please use peak selection algorithms for reseting. ")
        if np.all(proposed_peaks[0] == 0) and np.all(proposed_peaks[1] == 0):
            return np.empty((0, 2), dtype=int), np.empty((0, 2))
        
        # Parameters:
        td_threshold = 128
        w_prior_sigma = 64
        w_prior_beta = 5

        # Step 1:
        ppthre = self.prominent_power_threshold
        fp_candidates, sp_candidates = proposed_peaks[0], proposed_peaks[1]
        find_fp, find_sp = False, False

        # Find prominent fp.
        if fp_candidates[0] > 0 and peak_features[0][0].amplitude > ppthre:
            find_fp = True
            fp = fp_candidates[0]
        elif fp_candidates[0] == 0 and fp_candidates[1] > 0 \
            and peak_features[0][1].amplitude > ppthre:
            find_fp = True
            fp = fp_candidates[1]
        elif fp_candidates[0] == 0 and fp_candidates[1] == 0 and \
            fp_candidates[2] > 0 and peak_features[0][2].amplitude > ppthre:
            find_fp = True
            fp = fp_candidates[2]

        # Find prominent sp.
        if sp_candidates[0] > 0 and peak_features[1][0].amplitude > ppthre \
            and abs(sp_candidates[0] - sp_candidates[2] < 128):
            find_sp = True
            sp = sp_candidates[0]
        elif sp_candidates[0] == 0 and sp_candidates[1] > 0 and\
            peak_features[1][1].amplitude > ppthre and abs(sp_candidates[1] - sp_candidates[2]) < 64:
            find_sp = True
            sp = sp_candidates[1]
        elif  sp_candidates[0] == 0 and sp_candidates[1] == 0 and\
            sp_candidates[2] > 0 and peak_features[0][2].amplitude > ppthre:
            find_sp = True
            sp = sp_candidates[2]

        # Step 2: Get all candidates
        if find_fp and find_sp:
            return np.array([[fp, sp]], dtype=int), np.ones(1)
        elif find_fp and (not find_sp):
            tmp = []
            for i in range(sp_candidates.shape[0]):
                if sp_candidates[i] != 0:
                    tmp.append([fp, sp_candidates[i]])    
                indices_ij = np.array(tmp, dtype=int)
        elif (not find_fp) and find_sp:
            tmp = []
            for i in range(fp_candidates.shape[0]):
                if fp_candidates[i] != 0:
                    tmp.append([fp_candidates[i], sp])    
                indices_ij = np.array(tmp, dtype=int)
        else:
            tmp = []
            for i in range(fp_candidates.shape[0]):
                if fp_candidates[i] == 0:
                    continue
                for j in range(sp_candidates.shape[0]):
                    if sp_candidates[j] == 0:
                        continue
                    tmp.append([fp_candidates[i], sp_candidates[j]])
                indices_ij = np.array(tmp, dtype=int) 

        # Step 2.
        # Rank algorithm:
        # 1. Two pair of peaks: if their timediff error difference is less than 64,
        # the earlier pair ranks before the later pair;
        #


        td = indices_ij[:, 1] - indices_ij[:, 0]
        td_error = abs(td - prev_tdc)
        idxs = np.argsort(td_error)
        indices_ij = indices_ij[idxs]
        td_error = td_error[idxs]

        k = indices_ij.shape[0] - 1
        while abs(td_error[0] - td_error[k]) > td_threshold:
            k -= 1


        indices_ij = indices_ij[0: k+1]
        td_error = td_error[0: k+1]
        # Small sum means it is earlier, and it should have more weight.
        indices_sum = np.sum(indices_ij, axis=1)
        idxsum_idxs = np.argsort(np.argsort(indices_sum))
        k = indices_sum.shape[0]
        w_time = np.ones(k)
        for i in range(w_time.shape[0]):
            j = idxsum_idxs[i]
            w_time[i] = max(1 - j * 0.1, 0.5)
        
        peak_pairs = indices_ij[0: k+1]
        errors = td_error[0: k+1]
        w_prior = mpu.generalized_gaussian(errors, 0, w_prior_sigma, w_prior_beta)
        w_prior = w_prior / np.sum(w_prior)

        w = w_time * w_prior
        idxs = np.argsort(w)
        peak_pairs = peak_pairs[idxs[::-1]]
        w = w[idxs[::-1]]

        return peak_pairs, w



    def select_peaks_deterministic(self, proposed_peaks, peak_features, 
                                   prev_tdc, topn = 1):
        """ Given a few peak candidates for the first, and the second peak,
        Output n x 2 pairs. 
        
        Args:
            proposed_peaks: 2x5 array representing the proposed peak indices
                            of the first and the second antenna.
            peak_features:  2x5 array of PeakFeature storing the features of each
                            proposed peak.
            prev_tdc:       The time difference of the two peaks in the previous
                            CIR (assuming up_sampling factor is 64).
            topn:           Return topn candidates for each first peak, and topn
                            first peaks.
        
        Returns:
            pairs: k x 2
            confidence: (n, ) array
        """
        if prev_tdc == 0:
            fp, sp = proposed_peaks[0, 2], proposed_peaks[1, 2]
            w1, w2 = np.zeros(5), np.zeros((5, 5))
            return np.array([[fp, sp]], dtype=int), w1, w2

        indices_ij = np.zeros((topn*topn, 2), dtype=int)
        w1 = np.zeros(5)
        w2 = np.ones((5, 5))
        fp_candidates, sp_candidates = proposed_peaks[0], proposed_peaks[1]
                
        # Algorithm 1: If the first peak before fp, or the first invisible peak
        #              before fp is prominent (e.g., >100), directly set it as
        #              the first peak. Then using timediff to find second peak.
        find_fp, find_sp = False, False
        if fp_candidates[0] > 0 or fp_candidates[1] > 0:
            if peak_features[0][0].amplitude > self.prominent_power_threshold:
                find_fp = True
                fp = fp_candidates[0]
            elif peak_features[0][1].amplitude > self.prominent_power_threshold:
                find_fp = True
                fp = fp_candidates[1]
        else:   
            if peak_features[0][2].amplitude > self.prominent_power_threshold:
                find_fp = True
                fp = fp_candidates[2]

        if find_fp: # In this case, we assume fp must be right.
            # Pick 3 candidates which are closest to pred_sp. If earlier peaks
            # are prominent, choose it.
            pred_sp = fp + prev_tdc
            offset = abs(sp_candidates - pred_sp)
            sort_idxs = np.argsort(offset)
            sp = 6400
            for i in range(3):
                idx = sort_idxs[i]
                if peak_features[1][idx].amplitude > self.prominent_power_threshold and \
                    offset[idx] < 64 and sp_candidates[idx] < sp:
                    sp = sp_candidates[idx]
                    find_sp = True

        if find_fp and find_sp:
            indices_ij[0][0], indices_ij[0][1] = fp, sp
        elif find_fp and (not find_sp):
            w = np.ones(5)
            for i in range(sp_candidates.shape[0]):
                idx_i = sp_candidates[i]
                if sp_candidates[i] == 0:
                    w[i] = 0
                    continue
                
                # Probability incurred by time difference change.
                tdc = abs(idx_i - fp) - prev_tdc
                w[i] *= self.mpu.gaussian(
                    x=tdc, mu=0, sigma=self.tdc_sigma)
                
                # Probability incurred by second peak power
                ampl = peak_features[1][i].amplitude
                w[i] *= self.mpu.logistic_function(
                    x=ampl, L=self.sp_power_L, K=self.sp_power_K, 
                    x0=self.sp_power_x0)
        
            sort_idxs = np.argsort(w)
            for k in range(topn):
                best_i = sort_idxs[-1 - k]
                idx_i = sp_candidates[best_i]
                indices_ij[k][0] = fp
                indices_ij[k][1] = idx_i
        
        else:
            # For each first peak, find the best matched second peak.
            # Give top n candidates for each first peak
            candidate_pairs = np.zeros((5, topn, 3), dtype=int)
            for i in range(fp_candidates.shape[0]):
                idx_i = fp_candidates[i]
                if idx_i == 0:
                    w2[i] = copy.deepcopy([0 for i in range(w2.shape[0])])
                    continue

                for j in range(sp_candidates.shape[0]):
                    idx_j = sp_candidates[j]
                    if sp_candidates[j] == 0:
                        w2[i][j] = 0
                        continue
                    
                    # Probability incurred by time difference change.
                    tdc = idx_j - idx_i - prev_tdc
                    w2[i][j] *= self.mpu.gaussian(
                        x=tdc, mu=0, sigma=self.tdc_sigma)
                    
                    # Probability incurred by second peak power
                    ampl = peak_features[1][j].amplitude
                    w2[i][j] *= self.mpu.logistic_function(
                        x=ampl, L=self.sp_power_L, K=self.sp_power_K, 
                        x0=self.sp_power_x0)
            
                sort_idxs = np.argsort(w2[i])
                for k in range(topn):
                    best_j = sort_idxs[-1 - k]
                    best_idx_j = sp_candidates[best_j]
                    candidate_pairs[i][k][0] = i
                    candidate_pairs[i][k][1] = idx_i
                    candidate_pairs[i][k][2] = best_idx_j

            # Then choose the best first peak.
            
            w1[3], w1[4] = 0, 0 # We dislike later peaks.

            w_tdc = np.zeros(5)
            w_ampl = np.zeros(5)
            w_base = np.array([4,2,1,0.01,0.01])

            for i, idx_i, idx_j in candidate_pairs[:,0,:]: 
                tdc = idx_j - idx_i - prev_tdc
                w_tdc[i] = self.mpu.gaussian(
                        x=tdc, mu=0, sigma=self.tdc_sigma)
                ampl = peak_features[0][i].amplitude
                w_ampl[i] = self.mpu.logistic_function(
                        x=ampl, L=self.sp_power_L, K=self.sp_power_K, 
                        x0=self.sp_power_x0)
                w1[i] = w_tdc[i] * w_ampl[i] * w_base[i]
        

            w_tdc = w_tdc / np.sum(w_tdc)
            w_ampl = w_ampl/ np.sum(w_ampl)
            
            sort_idxs = np.argsort(w1)
            
            k = 0
            for i in range(topn):
                best_i = sort_idxs[-1 - i]
                for j in range(topn):
                    indices_ij[k] = copy.deepcopy(candidate_pairs[best_i][j][1:])
                    k += 1

        return indices_ij, w1, w2