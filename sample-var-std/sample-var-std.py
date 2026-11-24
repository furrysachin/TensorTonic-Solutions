import numpy as np

def sample_var_std(x):

    arr = np.array(x)

    mean = np.mean(arr)

    variance = np.sum((arr - mean) ** 2) / (len(arr) - 1)

    standard_deviation = np.sqrt(variance)

    return {
        "variance": float(variance),
        "standard_deviation": float(standard_deviation)
    }