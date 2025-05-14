# big 2 class
from .gameLogic import CardPlay, PlayerHandCard, PlayType

import numpy as np
import random


class big2Game:
    playHistory: list[CardPlay]

    def __init__(self):
        self.reset()

    def reset(self):
        self.rewards = np.zeros(4)

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
                self.playersGo = i
                break

        self.playHistory = []

    def assignRewards(self):
        cardsLeft = np.array([len(self.PlayersHand[i]) for i in range(4)])
        winner = cardsLeft.argmin()
        for i in range(4):
            if i == winner:
                continue
            self.rewards[i] -= cardsLeft[i]
            self.rewards[winner] += cardsLeft[i]

    def step(self, play: CardPlay):
        currentPlayerHand = self.PlayersHand[self.playersGo]
        if not currentPlayerHand.test_play(
            play, self.getPlayOnTop(), self.playerHasControl()
        ):
            raise ValueError("Invalid play")
        currentPlayerHand.remove_played_cards(play)
        self.playHistory.append(play)
        if self.isGameOver():
            self.assignRewards()
            return
        self.playersGo = (self.playersGo + 1) % 4

    def getHistory(self, length: int | None = None, no_pass: bool = False):
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

    def isGameOver(self):
        return any(len(self.PlayersHand[i]) == 0 for i in range(4))

    def playerHasControl(self):
        return all(i.get_type() == PlayType.PASS for i in self.getHistory(length=3))

    def getPlayOnTop(self):
        for prev_play in self.playHistory[::-1]:
            if prev_play.get_type() != PlayType.PASS:
                return prev_play

    def getCurrentState(self):
        return (
            self.playersGo,
            self.getHistory(),
            self.PlayersHand[self.playersGo].get_available_plays(
                self.getPlayOnTop(), self.playerHasControl()
            ),
        )

    def getInfoForDrawing(self):
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
