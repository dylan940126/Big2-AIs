from abc import ABC, abstractmethod
from typing import List
from game.cards import CardPlay
from game.gameLogic import PlayerHandCard


class Agent(ABC):
    """
    Abstract base class for agents in the Big2 game.
    Agents are responsible for selecting actions based on the game state.
    """

    @abstractmethod
    def step(
        self,
        first_player: int,
        history: List[CardPlay],
        hand: PlayerHandCard,
        avail_actions: List[CardPlay],
    ) -> CardPlay:
        """
        Selects the next action based on the current game state.
        Args:
            first_player (int): The index of the first player in the current round.
            history (List[CardPlay]): The history of card plays in the current game.
            hand (PlayerHandCard): The player's current hand of cards.
            avail_actions (List[CardPlay]): The list of available actions for the player.
        Returns:
            CardPlay: The selected action to play.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def reset(self):
        """
        Resets the agent's internal state if necessary.
        This method can be overridden by subclasses if they maintain state.
        """
        pass

    def set_final_reward(self, reward: float):
        """
        Sets the reward for the agent.
        This method can be overridden by subclasses if they need to track rewards.
        Args:
            reward (float): The final reward for the agent after the game ends.
        """
        pass