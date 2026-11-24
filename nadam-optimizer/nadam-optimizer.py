import numpy as np

def nadam_step(w, m, v, grad, lr, beta1, beta2, eps):

    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Step 1: First moment
    new_m = beta1 * m + (1 - beta1) * grad

    # Step 2: Second moment
    new_v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Nesterov adjustment
    m_hat = beta1 * new_m + (1 - beta1) * grad

    # Step 4: Update weights
    new_w = w - lr * m_hat / (np.sqrt(new_v) + eps)

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }