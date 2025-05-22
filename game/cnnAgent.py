from channels.generic.websocket import AsyncWebsocketConsumer
import json
import big2Game as big2Game
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gameLogic import CardPlay, PlayerHandCard
import os


# ---------------------------
# CNN Model Architecture
# ---------------------------
class Big2CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 13, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (B, 32, 4, 13)
        x = F.relu(self.conv2(x))      # (B, 64, 4, 13)
        x = x.view(x.size(0), -1)      # Flatten
        x = F.relu(self.fc1(x))        # (B, 128)
        x = self.fc2(x)
        return torch.sigmoid(x)


# ---------------------------
# CNN Agent for Big 2
# ---------------------------
class CNNBot:
    def __init__(self, model_path: str, device="cpu"):
        self.device = torch.device(device)
        self.model = Big2CNN().to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.training_data = []

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[CNNBot] Loaded pretrained model from {model_path}")
        else:
            print(f"[CNNBot] No pretrained model found at '{model_path}'. Starting fresh.")

    def encode_state(
        self,
        predicted_hands: np.ndarray,     # (4,13)
        current_hand: np.ndarray,        # (4,13)
        candidate_play: np.ndarray       # (4,13)
    ) -> torch.Tensor:
        """
        Combines input features into a 5x4x13 tensor.
        Additional 2 channels (zeros) can be used for history/control/mask/etc.
        """
        # Placeholder channels
        empty = np.zeros((4, 13), dtype=np.float32)
        stacked = np.stack([
            predicted_hands,
            current_hand,
            candidate_play,
            empty,
            empty
        ], axis=0)  # (5, 4, 13)
        return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(self.device)

    def step(
        self,
        predicted_hands: np.ndarray,
        current_hand: np.ndarray,
        available_actions: list[CardPlay]
    ) -> CardPlay:
        """
        Evaluate all available actions and return the one with highest win probability.
        """
        best_score = -float("inf")
        best_action = None

        for action in available_actions:
            candidate_play = self.play_to_matrix(action)
            input_tensor = self.encode_state(predicted_hands, current_hand, candidate_play)

            with torch.no_grad():
                score = self.model(input_tensor).item()

            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            raise ValueError("CNNBot failed to select an action.")
        return best_action

    def play_to_matrix(self, play: CardPlay) -> np.ndarray:
        """
        Converts a CardPlay object into a 4x13 matrix.
        1 if the card is in the play, else 0.
        """
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card in play.cards:
            suit = card // 13
            rank = card % 13
            matrix[suit][rank] = 1.0
        return matrix

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