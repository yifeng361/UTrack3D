import numpy as np
import copy
import math
import scipy.optimize
import scipy.special
from numpy.random import Generator, PCG64
import bisect

class MathUtils():
    def __init__(self, freq=3.993e9):
        self.cf = freq
        self.wavelen = 299792458 / freq

    def logistic_function(self, x, L, K, x0):
        return L / (1 + np.exp(-K * (x - x0)))

    def gaussian(self, x, mu, sigma):
        return (
            1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2)
        )

    def generalized_gaussian(self, x, mu, sigma, beta):
        return beta / (2 * sigma * scipy.special.gamma(1 / beta)) *\
              (np.exp(-(abs(x - mu)/ sigma) ** beta))

    def compute_phase_from_physical_positions(self, current_position, antennas):
        """ This function computes the phase given the current positions.
        Args:
            current_position: ndarray (3, )
            antennas: ndarray(k1, k2, ... , 3)
        """
        n_dim = len(antennas.shape)
        wavelen = self.wavelen
        dist_to_antennas = np.linalg.norm(antennas - current_position, axis=n_dim-1)
        phases = dist_to_antennas / wavelen * (2*math.pi)
        wrapped_phases = self.wrap_phase(phases)
        return wrapped_phases
        

    def compute_nonwrapphase_from_physical_positions(self, current_position, antennas):
        """ This function computes the phase given the current positions.
        Args:
            current_position: ndarray (3, )
            antennas: ndarray(4, 3)
        """
        wavelen = self.wavelen
        dist_to_antennas = np.linalg.norm(antennas - current_position, axis=1)
        phases = dist_to_antennas / wavelen * (2*math.pi)
        return phases

    def compute_phasediff_from_physical_positions(self, current_position, antennas):
        """ This function computes the phasediff w.r.t. the first antenna given 
        the current positions.
        Args:
            current_position: ndarray (n, 3)
            antennas: ndarray(k, 2, 3)
        """
        if len(current_position.shape) == 1:
            wavelen = self.wavelen
            dist_to_antennas1 = np.linalg.norm(antennas[:,0] - current_position, axis=1)
            dist_to_antennas2 = np.linalg.norm(antennas[:,1] - current_position, axis=1)
            phases = (dist_to_antennas2 -  dist_to_antennas1) / wavelen * (2*math.pi)
            wrapped_phasediffs = self.wrap_phase(phases - phases[0])
        else:
            n_recievers = antennas.shape[0]
            pos = np.tile(current_position[:,None,:], reps=(1,n_recievers,1)) # n_pos x n_r x 3
            at_pos1 = antennas[:, 0][None, :, :] # 1 x n_r x 3
            at_pos2 = antennas[:, 1][None, :, :] # 1 x n_r x 3
            dist_to_at1 = np.linalg.norm(pos - at_pos1, axis=2) # n_pos x n_r
            dist_to_at2 = np.linalg.norm(pos - at_pos2, axis=2) # n_pos x n_r
            phasediffs = (dist_to_at2 -  dist_to_at1) / wavelen * (2*math.pi)
            wrapped_phasediffs = self.wrap_phase(phasediffs)
        return wrapped_phasediffs
    
    def compute_airpath_difference(self, current_position, antennas):
        """
        Args:
            current_position: ndarray (3, ) or (n, 3)
            antennas: ndarray(k, 2, 3)
        Returns:
            k x 3
        """
        
        if len(current_position.shape) == 1:
            dist_to_antennas1 = np.linalg.norm(antennas[:,0] - current_position, axis=1)
            dist_to_antennas2 = np.linalg.norm(antennas[:,1] - current_position, axis=1)
            apd = dist_to_antennas2 - dist_to_antennas1
        else:
            n_recievers = antennas.shape[0]
            pos = np.tile(current_position[:,None,:], reps=(1,n_recievers,1)) # n_pos x n_r x 3
            at_pos1 = antennas[:, 0][None, :, :] # 1 x n_r x 3
            at_pos2 = antennas[:, 1][None, :, :] # 1 x n_r x 3
            dist_to_at1 = np.linalg.norm(pos - at_pos1, axis=2) # n_pos x n_r
            dist_to_at2 = np.linalg.norm(pos - at_pos2, axis=2) # n_pos x n_r
            apd = dist_to_at2 - dist_to_at1
        return apd
        
    def wrap_phase(self, raw_phases):
        """ Transform phase info [-pi, pi).
        Args:
            raw_phases: can be a float, or a ndarray.
        """
        return (raw_phases + math.pi) % (2 * math.pi ) - math.pi

    def rand_sample_discrete_distribution(self, rshape, w):
        """ 
        Args:
            r_shape: givens the shape of randomly generated numbers
            w: A list of array. Each array gives the probability of each variable.
            All w should sum up to 1
        Returns:
            indices of rshape
        """
        assert len(rshape) == 2, "Shape of rshape must equal to 2"
        assert len(w) == rshape[1]
        for i in range(len(w)):
            assert abs(np.sum(w[i]) - 1) < 1e-6, "discrete distribution error, sum(w) != 1"
        
        n_sample, n_dim = rshape[0], rshape[1]
        rng = Generator(PCG64())
        values = rng.random(rshape)
        indices = np.zeros(rshape, dtype=int)
        
        for i in range(n_dim):
            # n = w[j].shape[0]
            bins = np.cumsum(w[i])
            indices[:,i] = np.digitize(np.random.random_sample(n_sample), bins)

        return indices
        
        

    def localization_solver(self, antennas, start_pos, accum_phasediffchange, 
                            feasible_range, x0):
        """ Given antennas, initial position, accumulated phase change, range. 
        Compute current position.

        Args:
            antennas: ndarray (4, 3)
            start_pos: ndarray (3, )
            accum_phasediffchange: ndarray (4, ). The accumulated phase diff 
                                change from t=0 to current time.
            range: a list of 3 tuples. [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        """
        def error_function(est_pos, antennas, start_pos, accum_phasediffchange):
            # Given est_pos, compute distance diff
            est_dist = np.linalg.norm(antennas - est_pos, axis=1)
            est_distdiff = est_dist - est_dist[0]

            # Phase diff change gives the dist diff change.
            init_dist = np.linalg.norm(antennas - start_pos, axis=1)
            init_distdiff = init_dist - init_dist[0]
            distdiffchange = accum_phasediffchange / (math.pi * 2) * self.wavelen
            est_distdiff2 = init_distdiff + distdiffchange
            error = np.mean((est_distdiff - est_distdiff2) ** 2)
            return error
            
        res = scipy.optimize.minimize(error_function, x0, 
                                args=(antennas, start_pos, accum_phasediffchange),
                                bounds=feasible_range)
        return res.x
    
    def localization_solver_multiboards(self, antennas, start_pos, 
                                        accum_phasediffchange, feasible_range, x0):
        """ Given antennas, initial position, accumulated phase change, range. 
        Compute current position.

        Args:
            antennas: ndarray (n_receivers, 2, 3)
            start_pos: ndarray (3, )
            accum_phasediffchange: ndarray (3, ). The accumulated phase diff 
                                change from t=0 to current time.
            range: a list of 3 tuples. [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        """
        def error_function(est_pos, antennas, start_pos, accum_phasediffchange):
            # Given est_pos, compute distance diff
            n_receivers = antennas.shape[0]
            est_distdiff = np.zeros((n_receivers, ))
            init_distdiff = np.zeros((n_receivers, ))
            for i in range(n_receivers):
                # est_distdiff[i] = np.linalg.norm(antennas[i, 1, :] - est_pos, axis=1) - \
                #     np.linalg.norm(antennas[i, 0, :] - est_pos, axis=1)
                est_distdiff[i] = np.linalg.norm(antennas[i, 1, :] - est_pos) - \
                    np.linalg.norm(antennas[i, 0, :] - est_pos)
                init_distdiff[i] = np.linalg.norm(antennas[i, 1, :] - start_pos) - \
                    np.linalg.norm(antennas[i, 0, :] - start_pos)

            # Phase diff change gives the dist diff change.
            # init_dist = np.linalg.norm(antennas - start_pos, axis=1)
            # init_distdiff = init_dist - init_dist[0]
            distdiffchange = accum_phasediffchange / (math.pi * 2) * self.wavelen
            est_distdiff2 = init_distdiff + distdiffchange
            error = np.mean((est_distdiff - est_distdiff2) ** 2)
            return error
            
        res = scipy.optimize.minimize(error_function, x0, 
                                args=(antennas, start_pos, accum_phasediffchange),
                                bounds=feasible_range)
        return res.x
    

    def absolute_localization_solver_with_timediff(self, antennas, 
                                        observed_distdiff, feasible_range, x0):
        def error_function(est_pos, antennas, observed_distdiff):
            # Given est_pos, compute distance diff
            n_receivers = antennas.shape[0]
            est_distdiff = np.zeros(n_receivers)
            for i in range(n_receivers):
                est_distdiff[i] = np.linalg.norm(antennas[i, 1, :] - est_pos) - \
                    np.linalg.norm(antennas[i, 0, :] - est_pos)
            error = np.mean((observed_distdiff - est_distdiff) ** 2)
            return error
        
        res = scipy.optimize.minimize(error_function, x0, 
                                args=(antennas, observed_distdiff),
                                bounds=feasible_range)
        return res.x
    
    