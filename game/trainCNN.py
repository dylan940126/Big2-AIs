import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from cnnAgent import Big2CNN
from gameLogic import CardPlay, PlayerHandCard, PlayType
from big2Game import big2Game
from cardEstimator import estimate_opponent_cards
import datetime

# ========== Hyperparameters ==========
NUM_EPISODES = 1500  # Increase training episodes
BATCH_SIZE = 64       # Increase batch size for better gradient stability
LEARNING_RATE = 5e-4  # Slightly lower learning rate for stability
SAVE_PATH = "cnn_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model Setup ==========
model = Big2CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.MSELoss()

# ========== Helper Functions ==========
def hand_to_matrix(hand: PlayerHandCard) -> np.ndarray:
    matrix = np.zeros((4, 13), dtype=np.float32)
    for card in hand.handcards:
        suit = card // 13
        rank = card % 13
        matrix[suit][rank] = 1
    return matrix

def play_to_matrix(play: CardPlay) -> np.ndarray:
    matrix = np.zeros((4, 13), dtype=np.float32)
    for card in play.cards:
        suit = card // 13
        rank = card % 13
        matrix[suit][rank] = 1
    return matrix

def create_input_tensor(predicted: np.ndarray, hand: np.ndarray, play: np.ndarray) -> torch.Tensor:
    extra1 = np.zeros((4, 13), dtype=np.float32)
    extra2 = np.zeros((4, 13), dtype=np.float32)
    stacked = np.stack([predicted, hand, play, extra1, extra2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ========== Training Data Storage ==========
training_data = []

# ========== Training Loop ==========
for episode in range(NUM_EPISODES):
    game = big2Game()
    cnn_player = random.randint(0, 3)
    move_history = []

    while not game.isGameOver():
        player_id, history, available_actions = game.getCurrentState()

        if player_id == cnn_player:
            hand_matrix = hand_to_matrix(game.PlayersHand[cnn_player])
            played_cards = [p for p in game.playHistory if p.get_type() != PlayType.PASS]
            remaining_counts = [len(game.PlayersHand[i]) for i in range(4) if i != cnn_player]

            predicted_3x4x13 = estimate_opponent_cards(
                current_hand=hand_matrix,
                played_cards=played_cards,
                remaining_counts=remaining_counts,
                history=game.playHistory
            )
            predicted = np.mean(predicted_3x4x13, axis=0)

            action_tensors = []
            for action in available_actions:
                play_matrix = play_to_matrix(action)
                input_tensor = create_input_tensor(predicted, hand_matrix, play_matrix)
                with torch.no_grad():
                    score = model(input_tensor).item()
                action_tensors.append((score, input_tensor, action))

            action_tensors.sort(key=lambda x: x[0], reverse=True)
            best_score, best_input, best_action = action_tensors[0]

            game.step(best_action)

            for score, input_tensor, action in action_tensors:
                is_best = (action == best_action)
                move_history.append((input_tensor, is_best))

        else:
            game.step(random.choice(available_actions))

    game.assignRewards()
    reward = game.rewards[cnn_player]
    cards_used = 52 - len(game.PlayersHand[cnn_player])
    win_rate = 0.6 + 0.1 * (cards_used / 52.0) if reward > 0 else 0.05 + 0.05 * (cards_used / 52.0)

    for input_tensor, is_best in move_history:
        label = win_rate if is_best else win_rate * 0.7  # sharper penalty
        training_data.append((input_tensor, torch.tensor([[label]], dtype=torch.float32).to(DEVICE)))

    if len(training_data) >= BATCH_SIZE:
        batch = random.sample(training_data, BATCH_SIZE)
        inputs = torch.cat([b[0] for b in batch])
        targets = torch.cat([b[1] for b in batch])
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 20 == 0:
            print(f"[DEBUG] Ep {episode} | Loss: {loss.item():.4f} | Sample: {outputs[:3].squeeze().detach().cpu().numpy()}| Win_rate: {win_rate}")

# ========== Save the model ==========
torch.save(model.state_dict(), SAVE_PATH)
print(f"CNN model saved to {SAVE_PATH}")
