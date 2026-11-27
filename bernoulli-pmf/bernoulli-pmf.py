
import numpy as np

def bernoulli_pmf_and_moments(x, p):

    x = np.array(x)

    pmf = np.where(x == 0, 1 - p, p)

    mean = float(p)

    variance = float(p * (1 - p))

    return {
        "pmf": pmf,
        "mean": mean,
        "variance": variance
    }
