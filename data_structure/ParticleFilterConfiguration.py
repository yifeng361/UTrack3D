class ParticleFilterConfiguration():
    def __init__(self) -> None:
        self.n_particles = 10000
        self.xyz_mu = 0
        self.xyz_sigma = 0.02
        self.xyz_beta = 10