from typing import List
from enum import Enum


class Card:
    def __init__(self, index: int) -> None:
        if index < 0 or index >= 52:
            raise ValueError("Card index must be between 0 and 51")
        self.index = index
        self.value = index // 4
        self.suit = index % 4

    @classmethod
    def fromIndex(cls, index: int) -> "Card":
        return cls(index)

    @classmethod
    def fromValue(cls, value: int, suit: int) -> "Card":
        return cls(value * 4 + suit)

    def __repr__(self) -> str:
        if self.suit == 0:
            suit_str = "♧ "
        elif self.suit == 1:
            suit_str = "♢ "
        elif self.suit == 2:
            suit_str = "♡ "
        else:
            suit_str = "♤ "

        if self.value < 7:
            value_str = str(self.value + 3)
            value_str = value_str[0]
        elif self.value == 7:
            value_str = "10"
        elif self.value == 8:
            value_str = "J"
        elif self.value == 9:
            value_str = "Q"
        elif self.value == 10:
            value_str = "K"
        elif self.value == 11:
            value_str = "A"
        elif self.value == 12:
            value_str = "2"
        return suit_str + value_str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.index == other.index

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.index < other.index


class PlayType(Enum):
    INVALID = 0  # 無效牌型
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
    cards: List[Card]
    size: int
    _combination: PlayType | None

    def __init__(self, cards: List[int | Card]):
        self.cards = [i if isinstance(i, Card) else Card.fromIndex(i) for i in cards]
        self.size = len(self.cards)
        # sort
        self.cards.sort()
        self._combination = None

    def get_cards_index(self) -> List[int]:
        return [card.index for card in self.cards]

    def _check_card_uniqueness(self) -> bool:
        # Check if all cards are different
        for i in range(1, self.size):
            if self.cards[i] == self.cards[i - 1]:
                return False
        return True

    def _get_value_count(self, sort_and_remove_zeros: bool = False) -> List[int]:
        # Count the number of cards of each suit
        value_count = [0] * 13
        for card in self.cards:
            value_count[card.value] += 1
        if sort_and_remove_zeros:
            value_count.sort()
            for i in range(13):
                if value_count[i] == 0:
                    continue
                value_count = value_count[i:]
                break
        return value_count

    def _is_straight(self):
        if self.size != 5:
            return False
        values = [card.value for card in self.cards]
        values.sort()
        for i in range(1, 5):
            if values[i] - values[i - 1] != 1:
                return False
        return True

    def get_type(self) -> PlayType:
        # read the cache
        if self._combination is not None:
            return self._combination

        if self.size == 0:
            self._combination = PlayType.PASS
        elif self._check_card_uniqueness():
            suits_count = self._get_value_count(sort_and_remove_zeros=True)

            if suits_count == [1]:
                self._combination = PlayType.SINGLE
            elif suits_count == [2]:
                self._combination = PlayType.PAIR
            # elif suits_count == [3]:
            #     self._combination = PlayType.THREE_OF_A_KIND
            elif suits_count == [1, 4]:
                self._combination = PlayType.FOUR_OF_A_KIND
            elif suits_count == [2, 3]:
                self._combination = PlayType.FULL_HOUSE
            # elif suits_count == [5] and self._is_straight():
            #     self._combination = PlayType.STRAIGHT_FLUSH
            # elif suits_count == [5]:
            #     self._combination = PlayType.FLUSH
            elif self._is_straight():
                self._combination = PlayType.STRAIGHT
            else:
                self._combination = PlayType.INVALID
        else:
            self._combination = PlayType.INVALID

        return self._combination

    def __eq__(self, value):
        if not isinstance(value, CardPlay):
            return NotImplemented
        return self.cards == value.cards

    def __lt__(self, value):
        if not isinstance(value, CardPlay):
            return NotImplemented
        if self.get_type() != value.get_type():
            raise ValueError("Cannot compare different play types")
        return self.cards[-1] < value.cards[-1]

    def __repr__(self) -> str:
        match self.get_type():
            case PlayType.INVALID:
                return f"Invalid: {self.cards}"
            case PlayType.SINGLE:
                return f"Single: {self.cards}"
            case PlayType.PAIR:
                return f"Pair: {self.cards}"
            case PlayType.THREE_OF_A_KIND:
                return f"Three of a kind: {self.cards}"
            case PlayType.STRAIGHT:
                return f"Straight: {self.cards}"
            case PlayType.FLUSH:
                return f"Flush: {self.cards}"
            case PlayType.FULL_HOUSE:
                return f"Full house: {self.cards}"
            case PlayType.FOUR_OF_A_KIND:
                return f"Four of a kind: {self.cards}"
            case PlayType.STRAIGHT_FLUSH:
                return f"Straight flush: {self.cards}"
            case PlayType.PASS:
                return "Pass: []"
