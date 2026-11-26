
import math

def binomial_pmf_cdf(n, p, k):

    # PMF: Exactly k successes
    pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # CDF: At most k successes
    cdf = 0.0

    for i in range(k + 1):
        cdf += (
            math.comb(n, i)
            * (p ** i)
            * ((1 - p) ** (n - i))
        )

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }
