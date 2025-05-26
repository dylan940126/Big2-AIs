from typing import List
from game.cards import CardPlay
from game.gameLogic import PlayerHandCard
from agents.agent import Agent

class HumanAgent(Agent):
    def __init__(self):
        self.current_action = None

    def step(
        self,
        first_player: int,
        history: List[CardPlay],
        hand: PlayerHandCard,
        avail_actions: List[CardPlay],
    ) -> CardPlay:
        if len(avail_actions) == 0:
            raise ValueError("No available actions")
        # For human agent, we assume the action is provided externally
        if self.current_action is None or self.current_action not in avail_actions:
            raise ValueError("Current action is not valid or not set")
        return self.current_action

    def update_human_action(self, action: CardPlay):
        """
        Update the human agent's action based on external input.
        This method should be called when the human player selects an action.
        """
        self.current_action = action

    def reset(self):
        self.current_action = None