import numpy as np
from tqdm import tqdm
from game.big2Game import big2Game
import multiprocessing
from agents import RandomAgent

def run_game(_):
    """進行一場遊戲，所有玩家都是隨機代理"""
    game = big2Game()
    random_agents = [RandomAgent() for _ in range(4)]
    
    while not game.isGameOver():
        # 獲取當前玩家和可行動作
        player_go, first_player, history, hand, avail_actions = game.getCurrentState()
        # 讓隨機代理做出決策
        action = random_agents[player_go].step(first_player, history, hand, avail_actions)
        # 執行決策
        game.step(action)
    
    # 遊戲結束，返回獎勵
    return game.rewards

def run_experiments(num_games=1000):
    total_rewards = np.zeros(4, dtype=float)
    wins = np.zeros(4, dtype=int)
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = list(tqdm(pool.imap_unordered(run_game, range(num_games)), total=num_games))
    for rewards in results:
        total_rewards += rewards
        winner = np.argmax(rewards)
        wins[winner] += 1
    avg_rewards = total_rewards / num_games
    win_rates = wins / num_games * 100
    return avg_rewards, win_rates, wins

if __name__ == "__main__":
    print("開始運行1000場隨機代理遊戲...")
    
    # # 設置隨機種子以確保結果可重現
    # random.seed(42)
    # np.random.seed(42)
    
    # 運行實驗
    avg_rewards, win_rates, wins = run_experiments(1000)
    
    # 輸出結果
    print("\n----------- 結果統計 -----------")
    print("總場數: 1000場")
    print("\n玩家獲勝次數:")
    for i in range(4):
        print(f"玩家 {i+1}: {int(wins[i])}場 ({win_rates[i]:.2f}%)")
    
    print("\n平均獎勵:")
    for i in range(4):
        print(f"玩家 {i+1}: {avg_rewards[i]:.2f}")
