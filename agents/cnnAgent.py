import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from game.gameLogic import CardPlay, PlayerHandCard
from agents import Agent
from typing import List, Tuple


# ---------------------------
# CNN Model Architecture
# ---------------------------
class Big2CNN_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=(4, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.fc1 = nn.Linear(64 * 1 * 13, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x) -> Tensor:
        if x.shape == (5, 4, 13):
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))  # (B, 32, 1, 13)
        x = F.relu(self.conv2(x))  # (B, 64, 1, 13)
        x = x.flatten(start_dim=1)  # Flatten to (B, 64 * 1 * 13)
        x = F.relu(self.fc1(x))  # (B, 128)
        x = self.fc2(x)  # (B, 1)
        return x
    
class Big2CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, out_channels=32, kernel_size=(4, 1), padding=0)
        self.conv2 = nn.Conv2d(5, 32, kernel_size=(1, 5), padding=0)
        self.fc1 = nn.Linear(32 * 1 * 13 + 32 * 4 * 9 + 5 * 4 * 13, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x) -> Tensor:
        if x.shape == (5, 4, 13):
            x = x.unsqueeze(0)
        x1 = F.relu(self.conv1(x))  # (B, 32, 1, 13)
        x1 = x1.flatten(start_dim=1) # Flatten to (B, 32 * 1 * 13)
        x2 = F.relu(self.conv2(x))  # (B, 32, 4, 9)
        x2 = x2.flatten(start_dim=1) # Flatten to (B, 32 * 4 * 9)
        x = x.flatten(start_dim=1)  # Flatten to (B, 5 * 4 * 13)
        x = torch.cat((x, x1, x2), dim=1) # (B, 5 * 4 * 13 + 32 * 1 * 13 + 32 * 4 * 9)
        x = F.relu(self.fc1(x))  # (B, 128)
        x = self.fc2(x)  # (B, 1)
        return x


# ---------------------------
# CNN Agent for Big 2
# ---------------------------
class CNNAgent(Agent):
    history_data: List[Tuple[torch.Tensor, int, float]]

    def __init__(
        self,
        model: Big2CNN | str | None,
        device: str = "Auto",
        train: bool = False,
        gamma: float = 0.99,
        epsilon: float = 0.1,  # for epsilon-greedy exploration
        epsilon_min: float = 0.001,  # minimum epsilon value
        epsilon_decay: float = 0.995,  # epsilon decay rate
        lr: float = 1e-3,  # initial learning rate
        lr_min: float = 1e-4,  # minimum learning rate
        lr_decay: float = 0.999,  # learning rate decay rate
    ):
        if device == "Auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if isinstance(model, str):
            self.model = Big2CNN()
            self.model.load_state_dict(torch.load(model, map_location=self.device))
        elif isinstance(model, Big2CNN):
            self.model = model
        else:
            self.model = Big2CNN()

        self.model = self.model.to(self.device)
        self.model.train(train)
        self.history_data = []
        self.train = train

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Learning rate parameters
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        if train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=1e-5
            )
            self.criterion = nn.MSELoss()
            self.gamma = gamma

    def encode_state(
        self,
        predicted_hands: torch.Tensor,  # (3, 4, 13)
        current_hand: torch.Tensor,  # (4, 13)
        candidate_play: torch.Tensor,  # (batch_size, 4, 13)
    ) -> torch.Tensor:
        """
        Combines input features into a 5x4x13 tensor.
        Args:
            predicted_hands (torch.Tensor): Predicted hands of the opponents, shape (3, 4, 13).
            current_hand (torch.Tensor): Current player's hand, shape (4, 13).
            candidate_play (torch.Tensor): Candidate plays, shape (batch_size, 4, 13).
        Returns:
            torch.Tensor: Encoded state tensor of shape (batch_size, 5, 4, 13).
        """
        batch_size = candidate_play.size(0)

        player_hand = torch.cat(
            [current_hand.unsqueeze(0), predicted_hands], dim=0
        )  # (4, 4, 13)

        candidate_play = candidate_play.unsqueeze(1)  # (batch_size, 1, 4, 13)

        input_tensor = torch.cat(
            [player_hand.unsqueeze(0).repeat(batch_size, 1, 1, 1), candidate_play],
            dim=1,
        )  # (batch_size, 5, 4, 13)
        return input_tensor

    def assign_opponent_hands(self, history: list[CardPlay]) -> torch.Tensor:
        predicted_hands = torch.zeros(
            (3, 4, 13), dtype=torch.float32, device=self.device
        )
        for i in range(3):
            player_history = history[-1 - i :: -4]
            cards = [card for play in player_history for card in play.cards]
            if cards:
                cards_tensor = torch.tensor(
                    cards, dtype=torch.int64, device=self.device
                )
                suit = cards_tensor % 4
                rank = cards_tensor // 4
                predicted_hands[i, suit, rank] = 1

        return predicted_hands

    def step(
        self,
        first_player: int,
        history: list[CardPlay],
        handcards: PlayerHandCard,
        available_actions: list[CardPlay],
    ) -> CardPlay:
        """
        Selects the best action based on the current game state.
        """
        if not available_actions:
            raise ValueError("No available actions")

        # Prepare input tensors for all available actions
        with torch.no_grad():
            hand_matrix = self.hand_to_matrix(handcards)  # (4, 13)
            predicted_opponent_hands = self.assign_opponent_hands(history)  # (3, 4, 13)
            play_matrices = torch.stack(
                [self.play_to_matrix(play) for play in available_actions]
            )  # (batch_size, 4, 13)

            # Create input tensor for the CNN
            input_tensor = self.encode_state(
                predicted_opponent_hands, hand_matrix, play_matrices
            )
            q_values = self.model.forward(input_tensor).squeeze(-1)

            # Epsilon-greedy exploration
            if self.train and torch.rand(1).item() < self.epsilon:
                best_action_index = torch.randint(
                    0, len(available_actions), (1,)
                ).item()
            else:
                # Select the action with the highest Q-value
                best_action_index = torch.argmax(q_values).item()

        # Store experience for training (without immediate training)
        # Use a more meaningful reward signal based on game strategy
        card_count_reward = (
            -len(available_actions[best_action_index]) * 0.1
        )  # prefer fewer cards
        self.history_data.append(
            (input_tensor[best_action_index].clone(), card_count_reward)
        )

        return available_actions[best_action_index]

    def reset(self):
        """
        Resets the agent's internal state if necessary.
        This method can be overridden by subclasses if they maintain state.
        """
        self.history_data = []

    def update_epsilon(self):
        """
        Decay epsilon value for epsilon-greedy exploration.
        Should be called after each game/episode.
        """
        if self.train and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_learning_rate(self):
        """
        Decay learning rate for the optimizer.
        Should be called after each game/episode.
        """
        if self.train and self.lr > self.lr_min:
            self.lr = max(self.lr_min, self.lr * self.lr_decay)
            # Update the optimizer's learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    def update_hyperparameters(self):
        """
        Update both epsilon and learning rate.
        Convenience method to be called after each game/episode.
        """
        self.update_epsilon()
        self.update_learning_rate()

    def get_hyperparameters(self):
        """
        Get current hyperparameter values for monitoring.
        Returns:
            dict: Dictionary containing current epsilon and learning rate
        """
        return {
            "epsilon": self.epsilon,
            "learning_rate": self.lr,
            "epsilon_min": self.epsilon_min,
            "lr_min": self.lr_min,
        }

    def set_final_reward(self, reward: float):
        """
        Update the model using experiences from the entire game.
        This is called at the end of each game.
        """
        if not self.train or len(self.history_data) == 0:
            return

        # Calculate discounted rewards
        target_q_values = torch.tensor(
            reward + self.history_data[-1][1], dtype=torch.float32, device=self.device
        )
        losses = []

        # Train on experiences in reverse order (temporal difference learning)
        for i in range(len(self.history_data) - 1, -1, -1):
            best_input_tensor, local_reward = self.history_data[i]

            # Calculate Q-values for this state
            q_values = self.model.forward(best_input_tensor).squeeze()
            # Calculate loss
            loss = self.criterion(q_values, target_q_values)
            losses.append(loss)

            # Update discounted reward for next iteration
            target_q_values = self.gamma * target_q_values + local_reward

        # Batch update
        if losses:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update hyperparameters after training
            self.update_hyperparameters()

    def set_local_reward(self, reward: torch.Tensor):
        # This method is now deprecated - we do batch training in set_final_reward
        pass

    def hand_to_matrix(self, hand: PlayerHandCard) -> torch.Tensor:
        matrix = torch.zeros((4, 13), dtype=torch.float32, device=self.device)
        suit = hand.handcards % 4
        rank = hand.handcards // 4
        matrix[suit, rank] = 1
        return matrix

    def play_to_matrix(self, play: CardPlay) -> torch.Tensor:
        matrix = torch.zeros((4, 13), dtype=torch.float32, device=self.device)
        suit = play.cards % 4
        rank = play.cards // 4
        matrix[suit, rank] = 1
        return matrix
