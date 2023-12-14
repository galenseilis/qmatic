import numpy as np
from scipy.special import iv
from scipy.stats import rv_discrete

class mm1_len(rv_discrete):

    "Distribution of number of customers in a MM1 queue."

    def _pmf(self, k: int, lam: float, mu: float, i: int, t: float) -> float:
        """
        Compute the probability p_k(t) based on the provided formula.

        Parameters:
        - k (int): Number of customers in queue.
        - i (int): Initial number of customers in queue.
        - t (float): Time parameter.
        - lam (float): Arrival rate parameter.
        - mu (float): Service rate parameter.

        Returns:
        float: The computed probability p_k(t).
        """

        rho = lam / mu
        a = 2 * np.sqrt(lam * mu)
        at = a * t
        
        exp_decay_factor = np.exp(-(lam + mu) * t)
        term1 = np.power(rho, (k - i) / 2) * iv(k - i, at)
        term2 = np.power(rho, (k - i - 1) / 2) * iv(k + i + 1, at)

        expansion_factor = (1 - rho) * rho
        
        j = k + i + 2
        expansion = 0
        while True:
            new_term = np.power(rho, -j / 2) * iv(j, at)
            expansion += new_term
            if np.all(np.abs(new_term)) < 1e-15:
                break
            j += 1
        
        return exp_decay_factor * (term1 + term2 + expansion_factor * expansion)

def sample_mm1_len_seq(t, k0, taus, lambdas, mus):

    dist = mm1_len()

    assert t > np.max(taus) # Only predict in the last interval
    assert np.all(np.diff(taus) > 0) # Ensure time is strictly monotonic
    assert len({len(taus), len(lambdas), len(mus)}) == 1 # Size/len must match

    end_points = [k0]
    for i, (tau,l,m) in enumerate(zip(taus, lambdas, mus)):
        print(l, m, end_points[-1], tau)
        end_points.append(dist.rvs(l, m, end_points[-1], tau))

    return dist.rvs(l, m, end_points[-1], t) 
            
        
if __name__ == "__main__":
    np.random.seed(1)
    sample_mm1_len_seq(10, 5, list(range(1,9)), [10] * 8, [10] * 8)
