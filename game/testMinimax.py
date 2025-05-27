import numpy as np
import random
from big2Game import big2Game
from minimaxAgent import MinimaxAgent
from gameLogic import CardPlay, PlayType

NUM_GAMES = 100

minimax_wins = 0

for game_idx in range(NUM_GAMES):
    game = big2Game()
    minimax_player = random.randint(0, 3)
    minimax_agent = MinimaxAgent(game, minimax_player)

    while not game.isGameOver():
        player_id, history, available_actions = game.getCurrentState()

        if len(available_actions) == 0:
            game.step(CardPlay([]))
            continue

        if player_id == minimax_player:
            action = minimax_agent.step(
                history,
                available_actions,
                game.getCardCount(minimax_player)
                )
            game.step(action)

        else:
            # --- Random agent ---
            action = random.choice(available_actions)
            game.step(action)
    
    game.assignRewards()
    reward = game.rewards[minimax_player]
    if reward > 0: minimax_wins += 1

minimax_winrate = minimax_wins * 100 / NUM_GAMES