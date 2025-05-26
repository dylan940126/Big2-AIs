#!/usr/bin/env python3
"""
簡單的示例：展示 epsilon 和學習率退火的效果
"""

import torch
from agents import CNNAgent, Big2CNN

# 創建一個帶有退火功能的 CNN Agent
def demo_annealing():
    """展示超參數退火的效果"""
    
    print("=== CNN Agent 超參數退火示例 ===\n")
    
    # 創建 CNN Agent 並設置退火參數
    agent = CNNAgent(
        model=None,
        train=True,
        epsilon=0.5,          # 初始探索率
        epsilon_min=0.01,     # 最小探索率
        epsilon_decay=0.95,   # 探索率衰減（較快衰減以便演示）
        lr=1e-3,              # 初始學習率
        lr_min=1e-6,          # 最小學習率
        lr_decay=0.9          # 學習率衰減（較快衰減以便演示）
    )
    
    print("初始超參數:")
    print_hyperparams(agent)
    
    print("\n模擬 10 個 episodes 的訓練...")
    print("=" * 50)
    
    # 模擬訓練過程中的超參數變化
    for episode in range(1, 11):
        print(f"\nEpisode {episode}:")
        
        # 模擬訓練後更新超參數
        agent.update_hyperparameters()
        
        print_hyperparams(agent)
    
    print("\n" + "=" * 50)
    print("訓練完成！")
    
    final_params = agent.get_hyperparameters()
    print(f"\n最終參數:")
    print(f"Epsilon: {final_params['epsilon']:.6f}")
    print(f"學習率: {final_params['learning_rate']:.2e}")


def print_hyperparams(agent):
    """打印當前超參數"""
    params = agent.get_hyperparameters()
    print(f"  Epsilon: {params['epsilon']:.6f}")
    print(f"  學習率: {params['learning_rate']:.2e}")


if __name__ == "__main__":
    demo_annealing()
