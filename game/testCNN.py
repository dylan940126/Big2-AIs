import torch
import numpy as np
import random
from cnnAgent import Big2CNN, CNNBot, play_to_matrix, hand_to_matrix
from big2Game import big2Game
from gameLogic import CardPlay, PlayType
from cardEstimator import estimate_opponent_cards  # dummy placeholder

# ======== Config ========
MODEL_PATH = "cnn_model.pt"
NUM_GAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load CNN Agent ========
model = Big2CNN().to(DEVICE)

# Only load model once
cnn_agent = CNNBot(model_path=None, device=DEVICE)
cnn_agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cnn_agent.model.eval()  # Make sure eval mode is set


# ======== Evaluation Counter ========
cnn_wins = 0

for game_idx in range(NUM_GAMES):
    game = big2Game()
    cnn_player = random.randint(0, 3)

    while not game.isGameOver():
        player_id, history, available_actions = game.getCurrentState()

        if len(available_actions) == 0:
            # No valid actions, pass
            game.step(CardPlay([]))
            continue

        if player_id == cnn_player:
            # --- Generate features ---
            hand_matrix = hand_to_matrix(game.PlayersHand[player_id])

            played_cards = [p for p in game.playHistory if p.get_type() != PlayType.PASS]
            remaining_counts = [len(game.PlayersHand[i]) for i in range(4) if i != cnn_player]

            predicted_3x4x13 = estimate_opponent_cards(
                current_hand=hand_matrix,
                played_cards=played_cards,
                remaining_counts=remaining_counts,
                history=game.playHistory
            )

            # Average across 3 opponents → 4x13
            predicted_matrix = np.mean(predicted_3x4x13, axis=0)

            # --- Get CNN-selected action ---
            action = cnn_agent.step(predicted_matrix, hand_matrix, available_actions)
            game.step(action)

        else:
            # --- Random agent ---
            action = random.choice(available_actions)
            game.step(action)

    # After game over
    game.assignRewards()
    reward = game.rewards[cnn_player]
    if reward > 0:
        cnn_wins += 1

    if (game_idx + 1) % 10 == 0:
        print(f"[Game {game_idx + 1}/{NUM_GAMES}] CNN win rate so far: {cnn_wins / (game_idx + 1):.2%}")

# ======== Final Accuracy ========
final_accuracy = cnn_wins / NUM_GAMES
print(f"\nFinal CNN Win Rate over {NUM_GAMES} games: {final_accuracy:.2%}")
