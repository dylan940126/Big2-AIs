# big 2 class
from typing import Any, List
from game.gameLogic import CardPlay, PlayerHandCard, PlayType

import numpy as np
import random


class big2Game:
    playHistory: List[CardPlay]

    def __init__(self, cards: List[int] | None = None):
        self.reset(cards)

    def reset(self, cards: List[int] | None = None):
        self.rewards = np.zeros(4)

        if cards is None:
            cards = list(range(52))
            random.shuffle(cards)

        self.PlayersHand = [
            PlayerHandCard(cards[0:13]),
            PlayerHandCard(cards[13:26]),
            PlayerHandCard(cards[26:39]),
            PlayerHandCard(cards[39:52]),
        ]
        for i in range(4):
            if self.PlayersHand[i].handcards[0] == 0:
                self.firstPlayer = i
                self.playersGo = i
                break

        self.playHistory = []

    def getRewards(self):
        cardsLeft = np.array([len(self.PlayersHand[i]) for i in range(4)])
        winner = cardsLeft.argmin()
        for i in range(4):
            if i == winner:
                continue
            self.rewards[i] -= cardsLeft[i]
            self.rewards[winner] += cardsLeft[i]
        return self.rewards

    def step(self, play: CardPlay):
        currentPlayerHand = self.PlayersHand[self.playersGo]
        if not currentPlayerHand.test_play(
            play, self.getPlayOnTop(), self.playerHasControl()
        ):
            raise ValueError("Invalid play")
        currentPlayerHand.remove_played_cards(play)
        self.playHistory.append(play)
        if self.isGameOver():
            self.getRewards()
            return
        self.playersGo = (self.playersGo + 1) % 4

    def getFirstPlayer(self) -> int:
        return self.firstPlayer

    def getHistory(self, length: int | None = None, no_pass: bool = False) -> list[CardPlay] | list:
        """
        Get the last n plays in the play history.
        If length is None, return the entire play history.
        If history is shorter than length, pad with empty plays.
        """
        if no_pass:
            history = [i for i in self.playHistory if i.get_type() != PlayType.PASS]
        else:
            history = self.playHistory

        if length is None:
            return history
        if length < 1:
            raise ValueError("length must be greater than 0")
        if len(history) < length:
            res = []
            for _ in range(length - len(history)):
                res.append(CardPlay([]))
            res.extend(history)
            return res
        else:
            return history[-length:]

    def isGameOver(self) -> bool:
        return any(len(self.PlayersHand[i]) == 0 for i in range(4))

    def playerHasControl(self) -> bool:
        return all(i.get_type() == PlayType.PASS for i in self.getHistory(length=3))

    def getPlayOnTop(self) -> CardPlay | None:
        """
        Get the last play that was not a pass. If this is the first play, return None.
        """
        for prev_play in self.playHistory[::-1]:
            if prev_play.get_type() != PlayType.PASS:
                return prev_play
        return None

    def getCurrentState(self) -> tuple[int, int, List[CardPlay], PlayerHandCard, List[CardPlay]]:
        return (
            self.playersGo,
            self.firstPlayer,
            self.getHistory(),
            self.PlayersHand[self.playersGo],
            self.PlayersHand[self.playersGo].get_available_plays(
                self.getPlayOnTop(), self.playerHasControl()
            ),
        )

    def getInfoForDrawing(self) -> dict[str, Any]:
        info = {
            "type": "updateGame",
            "playersHand": self.PlayersHand[0].get_cards_index(),
            "playersGo": self.playersGo + 1,  # 1~4
            "control": self.playerHasControl(),
            "nCards": [
                len(self.PlayersHand[1]),
                len(self.PlayersHand[2]),
                len(self.PlayersHand[3]),
            ],
            "previousHands": [
                i.get_cards_index() for i in self.getHistory(length=3, no_pass=True)
            ],
            "gameOver": self.isGameOver(),
            "rewards": self.rewards.tolist(),
        }
        return info
