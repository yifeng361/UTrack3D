import numpy as np
import math

class PeakFeature():
    def __init__(self, cir, index):
        self.phase_dev = 0
        self.amplitude = 0
        self.index = 0
        if index == 0:
            return
        
        l = int(max(index-64, 0))
        r = int(min(index+64, cir.shape[0]))
        step = 8
        phases = np.angle(cir[l:r:step])
        unwrapped_phases = np.zeros(phases.shape[0])
        unwrapped_phases[0] = phases[0]
        for i in range(1, phases.shape[0]):
            diff = phases[i] - phases[i-1]
            if diff < -math.pi:
                diff += 2*math.pi
            elif diff >= math.pi:
                diff -= 2*math.pi
            unwrapped_phases[i] = unwrapped_phases[i-1] + diff

        self.phase_dev = np.std(unwrapped_phases)
        self.index = index
        self.amplitude = abs(cir[index])