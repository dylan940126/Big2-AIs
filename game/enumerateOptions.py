from .cards import Card, CardPlay, PlayType
from itertools import combinations, permutations, product

cards = [[Card.fromValue(v, s) for v in range(13)] for s in range(4)]  # 4 * 13

# dictionary to store all possible plays
AllPlays: dict[PlayType, list[CardPlay]] = {}

AllPlays[PlayType.SINGLE] = [
    CardPlay([cards[s][v]]) for s in range(4) for v in range(13)
]

AllPlays[PlayType.PAIR] = [
    CardPlay([cards[s1][v], cards[s2][v]])
    for s1, s2 in combinations(range(4), 2)
    for v in range(13)
]

# AllPlays[PlayType.THREE_OF_A_KIND] = [
#     CardPlay([cards[s1][v], cards[s2][v], cards[s3][v]])
#     for s1, s2, s3 in combinations(range(4), 3)
#     for v in range(13)
# ]

AllPlays[PlayType.STRAIGHT] = [
    CardPlay(
        [
            cards[s1][v],
            cards[s2][v + 1],
            cards[s3][v + 2],
            cards[s4][v + 3],
            cards[s5][v + 4],
        ]
    )
    for s1, s2, s3, s4, s5 in product(range(4), repeat=5)
    if not (s1 == s2 == s3 == s4 == s5)
    for v in range(9)
]

# AllPlays[PlayType.FLUSH] = [
#     CardPlay([cards[s][v1], cards[s][v2], cards[s][v3], cards[s][v4], cards[s][v5]])
#     for s in range(4)
#     for v1, v2, v3, v4, v5 in combinations(range(13), 5)
#     if not (v2 == v1 + 1 and v3 == v2 + 1 and v4 == v3 + 1 and v5 == v4 + 1)
# ]

AllPlays[PlayType.FULL_HOUSE] = [
    CardPlay(
        [cards[s1][v1], cards[s2][v1], cards[s3][v1], cards[s4][v2], cards[s5][v2]]
    )
    for s1, s2, s3 in combinations(range(4), 3)
    for s4, s5 in combinations(range(4), 2)
    for v1, v2 in permutations(range(13), 2)
]

AllPlays[PlayType.FOUR_OF_A_KIND] = [
    CardPlay([cards[0][v1], cards[1][v1], cards[2][v1], cards[3][v1], cards[s5][v2]])
    for s5 in range(4)
    for v1, v2 in permutations(range(13), 2)
]

# AllPlays[PlayType.STRAIGHT_FLUSH] = [
#     CardPlay(
#         [
#             cards[s][v],
#             cards[s][v + 1],
#             cards[s][v + 2],
#             cards[s][v + 3],
#             cards[s][v + 4],
#         ]
#     )
#     for s in range(4)
#     for v in range(9)
# ]

AllPlays[PlayType.INVALID] = []
