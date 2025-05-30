from typing import List
from enum import Enum
import numpy as np


class PlayType(Enum):
    UNKNOWN = 0  # 無效牌型
    SINGLE = 1  # 單牌
    PAIR = 2  # 對子，兩張同點數
    THREE_OF_A_KIND = 3  # 三條，三張同點數
    STRAIGHT = 4  # 順子，五張連續點數
    FLUSH = 5  # 同花，五張同花色
    FULL_HOUSE = 6  # 葫蘆，三條加一對
    FOUR_OF_A_KIND = 7  # 四條，四張同點數和一張其他牌
    STRAIGHT_FLUSH = 8  # 同花順，五張連續點數且同花色
    PASS = 9  # 不出牌


class CardPlay:
    cards: np.ndarray
    size: int
    _combination: PlayType | None

    def __init__(self, cards: List[int]):
        self.cards = np.array(cards)
        self.cards.sort()
        self.cards.flags.writeable = False
        self._combination = None

    def get_cards_index(self) -> List[int]:
        return self.cards.tolist()

    def _check_card_uniqueness(self) -> bool:
        # Check if all cards are unique
        return np.unique(self.cards).size == self.cards.size

    def _is_straight(self):
        # Check if the cards form a straight (consecutive values)
        values = self.cards // 4
        return np.all(np.diff(values) == 1)

    def _is_flush(self):
        # Check if the cards are of the same suit
        suits = self.cards % 4
        return np.all(suits == suits[0])

    def get_type(self) -> PlayType:
        # read the cache
        if self._combination is not None:
            return self._combination

        if self.cards.size == 0:
            self._combination = PlayType.PASS
        elif self._check_card_uniqueness():
            value_count = np.bincount(self.cards // 4, minlength=13)
            value_count = value_count[value_count > 0]
            value_count = np.sort(value_count)

            if np.array_equal(value_count, [1]):
                self._combination = PlayType.SINGLE
            elif np.array_equal(value_count, [2]):
                self._combination = PlayType.PAIR
            # elif np.array_equal(value_count, [3]):
            #     self._combination = PlayType.THREE_OF_A_KIND
            elif np.array_equal(value_count, [1, 4]):
                self._combination = PlayType.FOUR_OF_A_KIND
            elif np.array_equal(value_count, [2, 3]):
                self._combination = PlayType.FULL_HOUSE
            # elif self._is_flush() and self._is_straight():
            #     self._combination = PlayType.STRAIGHT_FLUSH
            # elif self._is_flush():
            #     self._combination = PlayType.FLUSH
            elif self._is_straight():
                self._combination = PlayType.STRAIGHT
            else:
                self._combination = PlayType.UNKNOWN
        else:
            self._combination = PlayType.UNKNOWN

        return self._combination

    def __len__(self):
        return self.cards.size

    def __eq__(self, value):
        if not isinstance(value, CardPlay):
            return NotImplemented
        return np.array_equal(self.cards, value.cards)

    def __lt__(self, value):
        if not isinstance(value, CardPlay):
            return NotImplemented
        if self.get_type() != value.get_type():
            raise ValueError("Cannot compare different play types")
        if self.get_type() == PlayType.FULL_HOUSE:
            # Compare by the rank of the three of a kind
            self_three = np.max(self.cards[self.cards // 4 == np.bincount(self.cards // 4).argmax()])
            value_three = np.max(value.cards[value.cards // 4 == np.bincount(value.cards // 4).argmax()])
            return self_three < value_three
        return self.cards[-1] < value.cards[-1]

    def __repr__(self) -> str:
        str_ = ""
        match self.get_type():
            case PlayType.SINGLE:
                str_ = "Single"
            case PlayType.PAIR:
                str_ = "Pair"
            case PlayType.THREE_OF_A_KIND:
                str_ = "Three of a Kind"
            case PlayType.STRAIGHT:
                str_ = "Straight"
            case PlayType.FLUSH:
                str_ = "Flush"
            case PlayType.FULL_HOUSE:
                str_ = "Full House"
            case PlayType.FOUR_OF_A_KIND:
                str_ = "Four of a Kind"
            case PlayType.STRAIGHT_FLUSH:
                str_ = "Straight Flush"
            case PlayType.PASS:
                str_ = "Pass"
            case _:
                str_ = "Unknown"

        return f"{str_}: {convert_index_to_str(self.cards)}"


def convert_index_to_str(cards: np.ndarray) -> str:
    """
    Convert a list of card indices to a string representation.
    """
    value_str = np.array(
        ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
    )
    suit_str = np.array(["♧ ", "♢ ", "♡ ", "♤ "])
    cards_str = suit_str[cards % 4] + value_str[cards // 4] if cards.size > 0 else ""
    return f"[{', '.join(cards_str)}]"
