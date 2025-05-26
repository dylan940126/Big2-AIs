from typing import List
from game.cards import CardPlay, PlayType, convert_index_to_str
import numpy as np
from game.enumerateOptions import AllPlays


class PlayerHandCard:
    handcards: np.ndarray
    player_available_plays: dict[PlayType, np.ndarray]

    def __init__(self, cards: List[int]):
        self.handcards = np.array(cards)
        self.handcards.sort()

        # 從所有可能的出牌組中，篩選出玩家手牌中有的出牌組
        self.player_available_plays = {}
        for type, plays_in_type in AllPlays.items():
            if type == PlayType.PASS:
                continue
            self.player_available_plays[type] = plays_in_type[
                np.isin(plays_in_type, self.handcards).all(axis=1)
            ]

    def get_cards_index(self) -> List[int]:
        return self.handcards.tolist()

    def has_play(self, play: CardPlay) -> bool:
        """
        Check if the player has the cards in the play.
        """
        return np.all(np.isin(play.cards, self.handcards))

    def test_play(self, play: CardPlay, playOnTop: CardPlay, control: bool) -> bool:
        return play in self.get_available_plays(playOnTop, control)

    def remove_played_cards(self, current_play: CardPlay) -> None:
        # 更新手牌
        self.handcards = np.setdiff1d(self.handcards, current_play.cards)

        # 更新可出牌組
        for type, plays_in_type in self.player_available_plays.items():
            self.player_available_plays[type] = plays_in_type[
                np.isin(plays_in_type, self.handcards).all(axis=1)
            ]

    def get_available_plays(self, lastPlayOnTop: CardPlay, control: bool) -> List[CardPlay]:
        # 篩選可出牌型
        available_plays = []
        # 是否有梅花三，要第一個出牌
        if self.handcards.size > 0 and self.handcards[0] == 0:
            for type, plays_in_type in self.player_available_plays.items():
                available_plays.extend(
                    plays_in_type[np.isin(plays_in_type, [0]).any(axis=1)]
                )
            available_plays = [CardPlay(i) for i in available_plays]
        # 是否有控制權，可出所有類型
        elif control:
            for type, plays_in_type in self.player_available_plays.items():
                available_plays.extend([CardPlay(i) for i in plays_in_type])
        # 沒有控制權，篩選出比上家出的牌大的牌
        else:
            for play in self.player_available_plays[lastPlayOnTop.get_type()]:
                play = CardPlay(play)
                if play > lastPlayOnTop:
                    available_plays.append(play)

            # 可以過牌
            available_plays.append(CardPlay([]))

        return available_plays

    def __repr__(self) -> str:
        return f"PlayerHandCard: {convert_index_to_str(self.handcards)}"

    def __len__(self) -> int:
        return self.handcards.size
