from numpy.random import rand
import numpy as np


class SSA:
    def __init__(
        self, k0_production_rate_const, k1_degradation_rate_const, stochiometry=[1, -1]
    ):
        self.k0_production_rate_const = k0_production_rate_const
        self.k1_degradation_rate_const = k1_degradation_rate_const
        self.stoichiometry = stochiometry

    def propensities(self, mRNA_level):
        return [
            self.k0_production_rate_const,
            self.k1_degradation_rate_const * mRNA_level,
        ]

    def reaction_times(self, x) -> np.ndarray:
        a = self.propensities(x)
        aInv = [1 / s if s > 0 else np.inf for s in a]
        noise = -np.log(rand(2))
        return noise * aInv

    def compute_step(self, mRNA_in, tIn, tOut):
        # take the state x at time tIn and return the state at time tOut
        t = tIn

        curr_mRNA = mRNA_in
        while t < tOut:
            rt = self.reaction_times(curr_mRNA)
            idx = np.argmin(rt)
            tau = np.min(rt)
            # tau = rt[idx]

            curr_mRNA += self.stoichiometry[idx]
            t += tau
        return curr_mRNA

    def run_simulation(
        self, timesteps: int, initial_mRNA_level=0.0, start_time=0.0, timestep_sec=0.5
    ):
        mrna = [initial_mRNA_level]
        time = [start_time]
        t = start_time
        curr_mRNA = initial_mRNA_level
        for i in range(timesteps):
            curr_mRNA = self.compute_step(curr_mRNA, t, t + timestep_sec)
            t += timestep_sec
            mrna.append(curr_mRNA)
            time.append(t)

        return (time, mrna)
