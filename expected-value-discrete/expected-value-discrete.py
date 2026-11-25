
import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x)
    p = np.array(p)

    result = np.sum(x * p)

    return float(result)
