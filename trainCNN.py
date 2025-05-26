import numpy as np
from tqdm import tqdm
from game.big2Game import big2Game
from agents import RandomAgent, CNNAgent, Agent, Big2CNN
from typing import List
import torch
import multiprocessing
from itertools import repeat
import cProfile
import pstats
import io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def run_game(model_state_dict) -> List[float]:
    """進行一場遊戲，一個CNN代理對抗三個隨機代理"""
    game = big2Game()

    # Create CNN model and load state dict
    cnn_model = Big2CNN()
    cnn_model.load_state_dict(model_state_dict)

    agents: List[Agent] = [
        CNNAgent(model=cnn_model, train=False),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    while not game.isGameOver():
        player_go, first_player, history, hand, avail_actions = game.getCurrentState()
        action = agents[player_go].step(first_player, history, hand, avail_actions)
        game.step(action)

    rewards = game.getRewards()

    # 遊戲結束，返回獎勵
    return rewards


def train_single_game(model):
    """訓練單場遊戲"""
    game = big2Game()
    agents: List[Agent] = [
        CNNAgent(model=model, train=True),
        CNNAgent(model=model, train=True),
        CNNAgent(model=model, train=True),
        CNNAgent(model=model, train=True),
    ]

    while not game.isGameOver():
        player_go, first_player, history, hand, avail_actions = game.getCurrentState()
        action = agents[player_go].step(first_player, history, hand, avail_actions)
        game.step(action)

    rewards = game.getRewards()
    for i, agent in enumerate(agents):
        agent.set_final_reward(rewards[i])

    return rewards


def evaluate_model(model, num_games=100):
    """評估模型性能"""
    total_rewards = np.zeros(4, dtype=float)
    wins = np.zeros(4, dtype=int)

    model_state_dict = model.state_dict()
    cpu_count = 4

    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = list(
            tqdm(
                pool.imap(run_game, repeat(model_state_dict, num_games)),
                total=num_games,
                desc="評估中",
            )
        )

    for rewards in results:
        total_rewards += rewards
        winner = np.argmax(rewards)
        wins[winner] += 1

    avg_rewards = total_rewards / num_games
    win_rates = wins / num_games * 100
    return avg_rewards, win_rates, wins


if __name__ == "__main__":
    num_games = 100
    training_games = 100  # Number of games to train before evaluation

    cnn_model = Big2CNN()
    cnn_model.load_state_dict(torch.load("cnn_agent_best.pt"))

    multiprocessing.set_start_method("spawn")

    print("開始訓練和評估循環...")

    best_win_rate = 0.0
    episode = 0
    while True:
        episode += 1
        print(f"\n========== Episode {episode} ==========")

        # Training phase
        print(f"訓練階段：進行 {training_games} 場遊戲...")
        training_rewards = []
        for i in tqdm(range(training_games), desc="訓練中"):
            rewards = train_single_game(cnn_model)
            training_rewards.append(rewards)

        # Calculate training statistics
        training_rewards = np.array(training_rewards)
        avg_training_rewards = np.mean(training_rewards, axis=0)
        training_wins = np.sum(
            training_rewards == np.max(training_rewards, axis=1, keepdims=True), axis=0
        )
        training_win_rates = training_wins / training_games * 100

        print(
            f"訓練結果 - CNN獲勝率: {training_win_rates[0]:.1f}%, 平均獎勵: {avg_training_rewards[0]:.2f}"
        )

        # Evaluation phase
        print(f"評估階段：進行 {num_games} 場遊戲...")
        avg_rewards, win_rates, wins = evaluate_model(cnn_model, num_games)

        # 輸出結果
        print("\n----------- 評估結果統計 -----------")
        print(f"總場數: {num_games}場")
        print("\n玩家獲勝次數:")
        print(f"玩家 1 (CNN): {int(wins[0])}場 ({win_rates[0]:.2f}%)")
        for i in range(1, 4):
            print(f"玩家 {i + 1} (隨機): {int(wins[i])}場 ({win_rates[i]:.2f}%)")

        print("\n平均獎勵:")
        print(f"玩家 1 (CNN): {avg_rewards[0]:.2f}")
        for i in range(1, 4):
            print(f"玩家 {i + 1} (隨機): {avg_rewards[i]:.2f}")

        # Save best model if CNN win rate is improving
        if win_rates[0] > best_win_rate:  # If CNN wins more than previous best
            best_win_rate = win_rates[0]
            torch.save(cnn_model.state_dict(), "cnn_agent_best.pt")
            print("🎉 新的最佳模型已保存！")
