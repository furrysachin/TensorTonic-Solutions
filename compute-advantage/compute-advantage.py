
import numpy as np

def compute_advantage(states, rewards, V, gamma):

    G = 0.0
    returns = []

    # Calculate returns backwards
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    # Reverse to restore original order
    returns.reverse()

    # Calculate advantages
    advantages = np.array(returns) - np.array(V)[states]

    return np.round(advantages, 4)
