from typing import List
from .cards import Card, CardPlay, PlayType
from .enumerateOptions import AllPlays


class PlayerHandCard:
    handcards: List[Card]
    player_available_plays: dict[PlayType, List[CardPlay]]

    def __init__(self, cards: List[int]):
        self.handcards = [Card.fromIndex(card) for card in cards]
        self.handcards.sort()

        self.player_available_plays = {}
        for play_type, all_plays_in_type in AllPlays.items():
            self.player_available_plays[play_type] = list(
                filter(self.has_play, all_plays_in_type)
            )

    def get_cards_index(self) -> List[int]:
        return [card.index for card in self.handcards]

    def has_play(self, play: CardPlay) -> bool:
        for card in play.cards:
            if card not in self.handcards:
                return False
        return True

    def test_play(self, play: CardPlay, playOnTop: CardPlay, control: bool) -> bool:
        return play in self.get_available_plays(playOnTop, control)

    def remove_played_cards(self, play: CardPlay) -> None:
        for card in play.cards:
            self.handcards.remove(card)
        for play_type, available_plays_in_type in self.player_available_plays.items():
            self.player_available_plays[play_type] = list(
                filter(self.has_play, available_plays_in_type)
            )

    def get_available_plays(self, playOnTop: CardPlay, control: bool) -> List[CardPlay]:
        # 只保留可出牌
        available_plays = []
        if Card(0) in self.handcards:
            for (
                play_type,
                available_plays_in_type,
            ) in self.player_available_plays.items():
                if play_type == PlayType.PASS:
                    continue
                available_plays.extend(
                    list(
                        filter(
                            lambda play: Card(0) in play.cards,
                            available_plays_in_type,
                        )
                    )
                )
        elif control:
            for (
                play_type,
                available_plays_in_type,
            ) in self.player_available_plays.items():
                if play_type == PlayType.PASS:
                    continue
                available_plays.extend(available_plays_in_type)
        else:
            available_plays += list(
                filter(
                    lambda play: play > playOnTop,
                    self.player_available_plays[playOnTop.get_type()],
                )
            )
            available_plays.append(CardPlay([]))

        return available_plays

    def __repr__(self) -> str:
        return f"PlayerHandCard: {self.handcards}"

    def __len__(self) -> int:
        return len(self.handcards)
