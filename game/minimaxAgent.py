from channels.generic.websocket import AsyncWebsocketConsumer
import json
from big2Game import big2Game
import numpy as np
import random
from gameLogic import CardPlay, PlayType, PlayerHandCard
from math import comb
import copy

class big2MinimaxSimulation(big2Game):
    def reset(self, hand_card: PlayerHandCard, agent_index: int, OpponentsCardCount: int[3]):
        self.rewards = np.zeros(4)
        cards = list(range(52))
        self.hand_cards = PlayerHandCard(),
        self.stochastic_cards = PlayerHandCard()

        self.playHistory = []
        self.hand_cards = copy.deepcopy(hand_card)
        self.agent_index = agent_index
        self.OpponentsCardCount = OpponentsCardCount
    
    def getOpponentCardCount(self):
        return [count for count in self.OpponentsCardCount]
    
    def possibilityWeight(self, action: CardPlay):
        """
        Calculate the possibility of bigger play with the number of each opponent's card and 
        """
        if not self.stochastic_cards
        return []

    def step(self, play: CardPlay):
        if not self.hand_cards.test_play(
            play, self.getPlayOnTop(), self.playerHasControl()
        ):
            raise ValueError("Invalid play")
        self.hand_cards.remove_played_cards(play)
        self.playHistory.append(play)
        if self.isGameOver():
            self.assignRewards()
            return
        self.playersGo = (self.playersGo + 1) % 4

    def getHistory(self, length: int | None = None, no_pass: bool = False):
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

    # done
    def getCardCount(self):
        # return a list of card counts
        return self.OpponentsCardCount
    
    # done
    def getStochasticCards(self):
        # return a hand of stochastic cards, 'excluded' is compulsory
        return self.stochastic_cards
    
    # done
    def isGameOver(self):
        return len(self.hands_cards) == 0 | any(self.OpponentsCardCount[i] == 0 for i in range(4))

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
    
class MinimaxAgent:
    def __init__(self, hand_card, agent_index, depth = 2):
        self.hand_card = hand_card
        self.agent_index = agent_index
        self.depth = depth
        
    def step(self, history, available_actions: list[big2Game.CardPlay], card_counts: list[3]):
        """
        Called minimax agent to step
        """
        if len(available_actions) == 0:
            raise ValueError("No available actions")
        self.minimax
        action = random.choice(best_action)
        return action
    
    def minimax(self, simulated_game: big2Game, depth, maximizingPlayer):
        """
        Goal:   Implement recursive Minimax search for Big2.
        Return: (boardValue, {setOfCandidateMoves})
            - boardValue is the evaluated utility of the board state
            - {setOfCandidateMoves} is a set of actions that achieve this boardValue
        """
        if depth == 0 or simulated_game.isGameOver():
            return 
    
        val = []
        if maximizingPlayer:
            result = -2e10
            for c in grid.valid:
                val.append(minimax(game.step(grid, c), depth - 1, False)[0])
            result = max(val)
        else:
            result = 2e10
            for c in grid.valid:
                val.append(minimax(game.drop_piece(grid, c), depth - 1, True)[0])
            result = min(val)
    
        return 
        
    def ActionRating(self, history, availAcs: list[big2Game.CardPlay], card_counts: list[3]):
        """
        Rate the avaible actions, and return the ratings
        """
        availAcs[0]._combination
        self.StaticGuess(card_counts)
        return []
    
