import numpy as np
import copy
import bisect
import scipy.signal

class SignalProcessUtils():

    def upsample(self, data: np.ndarray, factor: int):
        nlen = data.shape[0]
        N = factor
        aUpSound = np.zeros((N*nlen, ), dtype=np.complex64)
        aUpSound[0:N*nlen:N] = copy.deepcopy(data)
        return aUpSound

    def get_freqs(self, nFFT, fs):
        freqRes = fs / nFFT
        if nFFT % 2 == 0:
            aFreq = freqRes * (np.arange(nFFT) - nFFT // 2)
        else:
            aFreq = freqRes * (np.arange(nFFT) - (nFFT-1) // 2)
        return aFreq

    def freq_filter(self, data, fs, filtfreq):
        nlen = data.shape[0]
        afreq = self.get_freqs(nlen, fs)
        aSigf = np.fft.fftshift(np.fft.fft(data))
        iFreq = np.where(abs(afreq) < filtfreq)[0].astype("int")
        aSigf2 = np.zeros(aSigf.shape, dtype=np.complex64)
        aSigf2[iFreq] = aSigf[iFreq]
        aSig2 = np.fft.ifft(np.fft.ifftshift(aSigf2))
        return aSig2
    
    def align_cir(self, vq, original_up_first_index, aligned_up_first_index):
        """ This function aligns a CIR temporally (the original up first index
            will be moved to aligned_up_first_index).
        """
        a, b = original_up_first_index, aligned_up_first_index
        n = vq.shape[0]
        vq_aligned = np.zeros((n, ), dtype=np.complex64)
        if a > b:
            dloc = a - b
            vq_aligned[0:n-dloc] = copy.deepcopy(vq[dloc:n])
        else:
            dloc = b - a
            vq_aligned[dloc:n] = copy.deepcopy(vq[0:n-dloc])
        return vq_aligned

    def normalize_phase_ampl(self, data, ref_idx, norm_phase=True, norm_ampl=True):
        """ This function normalizes a CIR to zero-phase, unit-amplitude at the ref_idx.
        """
        val = data[ref_idx]
        A0, phi0 = abs(val), np.angle(val)
        norm_data = copy.deepcopy(data)
        if norm_phase:
            norm_data = copy.deepcopy(data * np.exp(1j*(-phi0)))
        if norm_ampl:
            if A0 < 1e-4:
                norm_data = np.zeros(data.shape)
            else:
                norm_data = norm_data / A0
        return norm_data
    
    def search_peaks(self, cir: np.ndarray, thre, start_idx, minimum_gap, 
                     search_region_length):
        """ This function searches for the first peak and the second peak
        corresponding to the received signals of two antennas
        Args:
            thre: The minimum acceptable amplitude to recognize a peak as the
                  received signal. If there is no strong_enough peak, return 
                  None for this antenna.
            start_idx: cir_start_index (after upsampling)
            minimum_gap: minimum gap between the first peak, and the second peak.
            search_region_length: length of the region in which we search for the
                                  peak.
        """
        
        l, r = start_idx, start_idx + search_region_length
        all_peakidxs = scipy.signal.find_peaks(abs(cir))[0]
        tmp_idxs = list(np.where(abs(cir[all_peakidxs]) > thre)[0])
        goodpeak_idxs = all_peakidxs[tmp_idxs]
        

        first_found, second_found, fpidx, spidx = False, False, 0, 0
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        if tmpidx < goodpeak_idxs.shape[0]:
            fpidx = goodpeak_idxs[tmpidx]
        
        if fpidx <= r:
            first_found = True
        
        l, r = fpidx + minimum_gap, fpidx + minimum_gap + search_region_length
        # goodpeak_idxs = list(np.where(abs(cir) > thre)[0])
        tmpidx = bisect.bisect_left(goodpeak_idxs, l)
        if tmpidx < goodpeak_idxs.shape[0]:
            spidx = goodpeak_idxs[tmpidx]

        if spidx <= r:
            second_found = True        
        peak_indices = (fpidx if first_found else 0, 
                        spidx if second_found else 0)
        return peak_indices

    def find_prominent_peak_intervals(cir, ref_idx, thre):
        """ This function tries to find prominent peak intervals of a CIR.
        """
        pass