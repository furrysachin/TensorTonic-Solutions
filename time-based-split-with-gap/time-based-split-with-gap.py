
import numpy as np

def time_based_split(timestamps, train_end, gap, val_end):

    timestamps = np.array(timestamps)

    train_idx = np.where(timestamps < train_end)[0]

    val_idx = np.where(
        (timestamps >= train_end + gap) &
        (timestamps < val_end)
    )[0]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx
    }
