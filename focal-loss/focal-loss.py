
import numpy as np

def focal_loss(p, y, gamma):
    p = np.array(p)
    y = np.array(y)

    # Clip probabilities for numerical stability
    p = np.clip(p, 1e-7, 1 - 1e-7)

    # Calculate focal loss elementwise
    loss = -(
        (1 - p)**gamma * y * np.log(p)
        +
        p**gamma * (1 - y) * np.log(1 - p)
    )

    # Return mean loss as Python float
    return float(np.mean(loss))
