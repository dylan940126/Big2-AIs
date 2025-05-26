import random
from typing import List
from game.cards import CardPlay
from game.gameLogic import PlayerHandCard
from agents.agent import Agent

class RandomAgent(Agent):
    def step(
        self,
        first_player: int,
        history: List[CardPlay],
        hand: PlayerHandCard,
        avail_actions: List[CardPlay],
    ) -> CardPlay:
        """
        Randomly selects an action from the available actions.
        """
        if len(avail_actions) == 0:
            raise ValueError("No available actions")
        return random.choice(avail_actions)