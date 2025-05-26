from game.cards import PlayType
from itertools import combinations, product, permutations
import numpy as np

AllPlays: dict[PlayType, np.ndarray] = {}

AllPlays[PlayType.SINGLE] = np.arange(52).reshape(52, 1)

AllPlays[PlayType.PAIR] = np.stack(
    [
        np.array([v * 4 + s1, v * 4 + s2])
        for v in range(13)
        for s1, s2 in combinations(range(4), 2)
    ]
)

# AllPlays[PlayType.THREE_OF_A_KIND] = np.stack(
#     [
#         np.array([v * 4 + s1, v * 4 + s2, v * 4 + s3])
#         for v in range(13)
#         for s1, s2, s3 in combinations(range(4), 3)
#     ]
# )

AllPlays[PlayType.STRAIGHT] = np.stack(
    [
        np.array([v * 4 + s1, (v + 1) * 4 + s2, (v + 2) * 4 + s3, (v + 3) * 4 + s4, (v + 4) * 4 + s5])
        for v in range(9)
        for s1, s2, s3, s4, s5 in product(range(4), repeat=5)
    ]
)

# AllPlays[PlayType.FLUSH] = np.stack(
#     [
#         np.array([v1 * 4 + s, v2 * 4 + s, v3 * 4 + s, v4 * 4 + s, v5 * 4 + s])
#         for s in range(4)
#         for v1, v2, v3, v4, v5 in combinations(range(13), 5)
#     ]
# )

AllPlays[PlayType.FULL_HOUSE] = np.stack(
    [
        np.array([v1 * 4 + s1, v1 * 4 + s2, v1 * 4 + s3, v2 * 4 + s4, v2 * 4 + s5])
        for v1, v2 in permutations(range(13), 2)
        for s1, s2, s3 in combinations(range(4), 3)
        for s4, s5 in combinations(range(4), 2)
    ]
)

AllPlays[PlayType.FOUR_OF_A_KIND] = np.stack(
    [
        np.array([v1 * 4, v1 * 4 + 1, v1 * 4 + 2, v1 * 4 + 3, v2 * 4 + s1])
        for v1, v2 in permutations(range(13), 2)
        for s1 in range(4)
    ]
)

# AllPlays[PlayType.STRAIGHT_FLUSH] = np.stack(
#     [
#         np.array([v * 4 + s, (v + 1) * 4 + s, (v + 2) * 4 + s, (v + 3) * 4 + s, (v + 4) * 4 + s])
#         for v in range(9)
#         for s in range(4)
#     ]
# )