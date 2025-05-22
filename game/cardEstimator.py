import numpy as np

def estimate_opponent_cards(current_hand, played_cards, remaining_counts, history):
    """
    Placeholder: Generate random uniform probabilities for each of the 3 opponents.
    """
    dummy = np.random.rand(3, 4, 13).astype(np.float32)
    # Normalize each opponent's card probabilities to sum to their remaining count
    for i in range(3):
        total = np.sum(dummy[i])
        if total > 0:
            dummy[i] *= remaining_counts[i] / total
    return dummy
