#!/usr/bin/env python3
"""
訓練腳本：使用 epsilon 和學習率退火的 CNN Agent 訓練
"""

import torch
import numpy as np
from tqdm import tqdm
from game.big2Game import big2Game
from agents import RandomAgent, CNNAgent, Big2CNN
import matplotlib.pyplot as plt
import json
from datetime import datetime


def train_with_annealing():
    """使用超參數退火進行訓練"""
    
    # 訓練參數
    NUM_EPISODES = 1000
    EVAL_INTERVAL = 50  # 每50個episode評估一次
    NUM_EVAL_GAMES = 100  # 評估時進行的遊戲數量
    SAVE_INTERVAL = 100  # 每100個episode保存一次模型
    
    # CNN Agent 參數 (使用新的退火功能)
    cnn_params = {
        'train': True,
        'epsilon': 0.3,          # 初始探索率較高
        'epsilon_min': 0.01,     # 最小探索率
        'epsilon_decay': 0.995,  # 探索率衰減
        'lr': 1e-3,              # 初始學習率
        'lr_min': 1e-6,          # 最小學習率
        'lr_decay': 0.999,       # 學習率衰減
        'gamma': 0.95            # 折扣因子
    }
    
    # 創建 CNN Agent
    cnn_agent = CNNAgent(model=None, **cnn_params)
    
    # 訓練記錄
    training_log = {
        'episodes': [],
        'win_rates': [],
        'avg_rewards': [],
        'epsilon_values': [],
        'lr_values': [],
        'losses': []
    }
    
    print(f"開始訓練，共 {NUM_EPISODES} 個 episodes")
    print(f"初始參數: epsilon={cnn_params['epsilon']}, lr={cnn_params['lr']}")
    
    best_win_rate = 0.0
    
    for episode in tqdm(range(NUM_EPISODES), desc="訓練進度"):
        # 訓練一局遊戲
        game = big2Game()
        agents = [cnn_agent, RandomAgent(), RandomAgent(), RandomAgent()]
        
        # 重置 agents
        for agent in agents:
            agent.reset()
        
        # 進行遊戲
        while not game.isGameOver():
            player_go, first_player, history, hand, avail_actions = game.getCurrentState()
            action = agents[player_go].step(first_player, history, hand, avail_actions)
            game.step(action)
        
        # 分配獎勵
        game.assignRewards()
        rewards = game.rewards
        
        # 設置最終獎勵（只有訓練中的 agent 需要）
        if hasattr(agents[0], 'set_final_reward'):
            agents[0].set_final_reward(rewards[0])
        
        # 記錄當前超參數
        hyperparams = cnn_agent.get_hyperparameters()
        
        # 每隔一定 episodes 進行評估
        if (episode + 1) % EVAL_INTERVAL == 0:
            win_rate, avg_reward = evaluate_agent(cnn_agent, NUM_EVAL_GAMES)
            
            # 記錄訓練數據
            training_log['episodes'].append(episode + 1)
            training_log['win_rates'].append(win_rate)
            training_log['avg_rewards'].append(avg_reward)
            training_log['epsilon_values'].append(hyperparams['epsilon'])
            training_log['lr_values'].append(hyperparams['learning_rate'])
            
            print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
            print(f"勝率: {win_rate:.2%}")
            print(f"平均獎勵: {avg_reward:.3f}")
            print(f"Epsilon: {hyperparams['epsilon']:.4f}")
            print(f"學習率: {hyperparams['learning_rate']:.2e}")
            
            # 保存最佳模型
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(cnn_agent.model.state_dict(), "cnn_agent_best_annealing.pt")
                print(f"🎉 新的最佳模型已保存！勝率: {win_rate:.2%}")
        
        # 定期保存檢查點
        if (episode + 1) % SAVE_INTERVAL == 0:
            torch.save(cnn_agent.model.state_dict(), f"cnn_agent_episode_{episode + 1}_annealing.pt")
    
    # 保存最終模型和訓練記錄
    torch.save(cnn_agent.model.state_dict(), "cnn_agent_final_annealing.pt")
    
    # 保存訓練記錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"training_log_{timestamp}.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # 繪製訓練曲線
    plot_training_curves(training_log, timestamp)
    
    print(f"\n訓練完成！最佳勝率: {best_win_rate:.2%}")
    print(f"最終 epsilon: {cnn_agent.epsilon:.4f}")
    print(f"最終學習率: {cnn_agent.lr:.2e}")


def evaluate_agent(cnn_agent, num_games=100):
    """評估 agent 的表現"""
    # 暫時設置為評估模式（不探索，不訓練）
    original_train = cnn_agent.train
    original_epsilon = cnn_agent.epsilon
    
    cnn_agent.model.eval()
    cnn_agent.train = False
    cnn_agent.epsilon = 0.0  # 純 exploitation
    
    wins = 0
    total_reward = 0
    
    for _ in range(num_games):
        game = big2Game()
        agents = [cnn_agent, RandomAgent(), RandomAgent(), RandomAgent()]
        
        # 重置 agents
        for agent in agents:
            agent.reset()
        
        # 進行遊戲
        while not game.isGameOver():
            player_go, first_player, history, hand, avail_actions = game.getCurrentState()
            action = agents[player_go].step(first_player, history, hand, avail_actions)
            game.step(action)
        
        # 統計結果
        game.assignRewards()
        if game.rewards[0] > 0:  # CNN agent 獲勝
            wins += 1
        total_reward += game.rewards[0]
    
    # 恢復原始設置
    cnn_agent.model.train()
    cnn_agent.train = original_train
    cnn_agent.epsilon = original_epsilon
    
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    
    return win_rate, avg_reward


def plot_training_curves(training_log, timestamp):
    """繪製訓練曲線"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = training_log['episodes']
    
    # 勝率曲線
    ax1.plot(episodes, training_log['win_rates'], 'b-', linewidth=2)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('CNN Agent Win Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 平均獎勵曲線
    ax2.plot(episodes, training_log['avg_rewards'], 'g-', linewidth=2)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Reward Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Epsilon 衰減曲線
    ax3.plot(episodes, training_log['epsilon_values'], 'r-', linewidth=2)
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon Decay Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(training_log['epsilon_values']) * 1.1)
    
    # 學習率衰減曲線
    ax4.semilogy(episodes, training_log['lr_values'], 'm-', linewidth=2)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Learning Rate (log scale)')
    ax4.set_title('Learning Rate Decay Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"訓練曲線已保存為 training_curves_{timestamp}.png")


if __name__ == "__main__":
    # 檢查是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")
    
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        train_with_annealing()
    except KeyboardInterrupt:
        print("\n訓練被用戶中斷")
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
