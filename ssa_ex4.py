from numpy.random import rand
import numpy as np


class SSA_w_protein:
    def __init__(
        self,
        k0_mRNA_production_rate,
        k1_mRNA_degradation_rate,
        k2_protein_production_rate,
        k3_protein_degradation_rate,
        stochiometry=[1, -1],
    ):
        self.k0_mRNA_production_rate = k0_mRNA_production_rate
        self.k1_mRNA_degradation_rate = k1_mRNA_degradation_rate
        self.k2_protein_production_rate = k2_protein_production_rate
        self.k3_protein_degradation_rate = k3_protein_degradation_rate
        self.stoichiometry = stochiometry

    def mrna_propensities(self, mRNA_level):
        return [
            self.k0_mRNA_production_rate,
            self.k1_mRNA_degradation_rate * mRNA_level,
        ]

    def protein_propensities(self, mRNA_level, protein_level):
        return [
            mRNA_level * self.k2_protein_production_rate,
            self.k3_protein_degradation_rate * protein_level,
        ]

    def reaction_times_mrna(self, x) -> np.ndarray:
        a = self.mrna_propensities(x)
        aInv = [1 / s if s > 0 else np.inf for s in a]
        noise = -np.log(rand(2))
        return noise * aInv

    def reaction_times_protein(self, protein_level, mRNA_level) -> np.ndarray:
        a = self.protein_propensities(
            protein_level=protein_level, mRNA_level=mRNA_level
        )
        aInv = [1 / s if s > 0 else np.inf for s in a]
        noise = -np.log(rand(2))
        return noise * aInv

    def compute_mRNA_step(self, mRNA_in, t_in, t_out):
        # take the state x at time tIn and return the state at time tOut
        t = t_in

        curr_mRNA = mRNA_in

        while t < t_out:
            rt = (self.reaction_times_mrna(curr_mRNA),)
            idx = np.argmin(rt)
            tau = np.min(rt)

            curr_mRNA += self.stoichiometry[idx]
            t += tau
        return curr_mRNA

    def compute_protein_step(self, protein_in, mRNA_in, t_in, t_out):
        # take the state x at time tIn and return the state at time tOut
        t = t_in

        curr_protein = protein_in
        while t < t_out:
            rt = self.reaction_times_protein(
                protein_level=curr_protein,
                mRNA_level=mRNA_in,  # we assume constant mRNA level for the duration of the timestep
            )
            idx = np.argmin(rt)
            tau = np.min(rt)
            # tau = rt[idx]

            curr_protein += self.stoichiometry[idx]
            t += tau
        return curr_protein

    def run_simulation(
        self,
        timesteps: int,
        initial_mRNA_level=0.0,
        intial_protein_level=0.0,
        start_time=0.0,
        timestep_sec=0.5,
    ):
        mrna = [initial_mRNA_level]
        protein = [intial_protein_level]

        time = [start_time]
        t = start_time

        curr_mRNA = initial_mRNA_level
        curr_protein = intial_protein_level

        for i in range(timesteps):
            next_mRNA = self.compute_mRNA_step(
                mRNA_in=curr_mRNA, t_in=t, t_out=t + timestep_sec
            )
            curr_protein = self.compute_protein_step(
                protein_in=curr_protein,
                mRNA_in=curr_mRNA,
                t_in=t,
                t_out=t + timestep_sec,
            )

            curr_mRNA = next_mRNA

            t += timestep_sec

            mrna.append(curr_mRNA)
            protein.append(curr_protein)
            time.append(t)

        return {"time": time, "mRNA": mrna, "protein": protein}
