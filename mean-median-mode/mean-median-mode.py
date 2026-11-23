
import numpy as np
from collections import Counter

def mean_median_mode(x):

    # Mean
    mean = np.mean(x)

    # Median
    median = np.median(x)

    # Mode
    frequency = Counter(x)

    max_freq = max(frequency.values())

    mode = min(
        num for num, count in frequency.items()
        if count == max_freq
    )

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode)
    }
