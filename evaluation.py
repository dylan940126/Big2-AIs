#!/usr/bin/env python3
"""
Big2 AI Agent Cross-Evaluation System
=====================================

This script evaluates the performance of different AI agents (Random, CNN, MLP) 
in a simplified 1 vs 3 tournament setting.

Evaluation scenarios:
1. Each agent type vs 3 others of same type (baseline)
2. Each agent type vs 3 Random agents 
3. Each agent type vs 3 CNN agents
4. Each agent type vs 3 MLP agents
"""

import numpy as np
import pandas as pd
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm

# Import game and agent modules
from game.big2Game import big2Game
from game.gameLogic import CardPlay
from agents.randomAgent import RandomAgent
from agents.cnnAgent import CNNAgent
from agents.mlpAgent import MLPAgent


class AgentFactory:
    """Factory class to create different types of agents"""
    
    @staticmethod
    def create_agent(agent_type: str, agent_id: int = 0, **kwargs) -> Any:
        """
        Create an agent of specified type
        
        Args:
            agent_type: 'random', 'cnn', or 'mlp'
            agent_id: Unique identifier for the agent
            **kwargs: Additional arguments for agent creation
        """
        if agent_type.lower() == 'random':
            return RandomAgent()
        
        elif agent_type.lower() == 'cnn':
            model_path = kwargs.get('model_path', 'CNN_agent_best.pt')
            device = kwargs.get('device', 'Auto')
            return CNNAgent(
                model=model_path,
                device=device,
                train=False  # Evaluation mode
            )
        
        elif agent_type.lower() == 'mlp':
            model_path = kwargs.get('model_path', 'MLP_agent_best.pt')
            device = kwargs.get('device', 'Auto')
            return MLPAgent(
                model=model_path,
                device=device,
                train=False  # Evaluation mode
            )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


def run_single_game_worker(args):
    """
    Worker function for multiprocessing - runs a single game
    
    Args:
        args: Tuple of (target_agent_type, opponent_agents, device, game_id, scenario_name)
        
    Returns:
        Dictionary with game results
    """
    target_agent_type, opponent_agents, device, game_id, scenario_name = args
    
    try:
        # Create agents for this game
        agents = []
        
        # Target agent (position 0)
        target_agent = AgentFactory.create_agent(
            target_agent_type, 
            agent_id=0,
            device=device
        )
        agents.append(target_agent)
        
        # Opponent agents (positions 1, 2, 3)
        for i, opponent_type in enumerate(opponent_agents):
            opponent_agent = AgentFactory.create_agent(
                opponent_type,
                agent_id=i+1,
                device=device
            )
            agents.append(opponent_agent)
        
        # Run the game
        game = big2Game()
        moves_count = 0
        
        # Reset all agents
        for agent in agents:
            if hasattr(agent, 'reset'):
                agent.reset()

        while not game.isGameOver():
            player_go, first_player, history, hand, avail_actions = game.getCurrentState()
            
            if len(avail_actions) == 0:
                action = CardPlay([])
            else:
                action = agents[player_go].step(first_player, history, hand, avail_actions)
            
            game.step(action)
            moves_count += 1
            
            # Safety check to prevent infinite loops
            if moves_count > 1000:
                break

        # Get final rewards and determine winner
        final_rewards = game.getRewards()
        winner = np.argmax(final_rewards)
        
        # Set final rewards for training agents
        for i, agent in enumerate(agents):
            if hasattr(agent, 'set_final_reward'):
                agent.set_final_reward(final_rewards[i])
        
        # Calculate game statistics
        cards_left = [len(game.PlayersHand[i]) for i in range(4)]
        game_length = len(game.playHistory)
        
        return {
            'scenario': scenario_name,
            'game_id': game_id,
            'winner': int(winner),
            'final_scores': final_rewards.tolist(),
            'cards_left': cards_left,
            'game_length': game_length,
            'total_moves': moves_count,
            'agent_types': [type(agent).__name__ for agent in agents]
        }
        
    except Exception as e:
        return {
            'scenario': scenario_name,
            'game_id': game_id,
            'winner': -1,  # Error indicator
            'final_scores': [0, 0, 0, 0],
            'cards_left': [13, 13, 13, 13],
            'game_length': 0,
            'total_moves': 0,
            'agent_types': [],
            'error': str(e)
        }


class GameEvaluator:
    """Main evaluation class for Big2 agent performance"""
    
    def __init__(self, 
                 games_per_scenario: int = 1000,
                 device: str = "Auto",
                 save_detailed_results: bool = True,
                 results_dir: str = "evaluation_results",
                 num_processes: int = None):
        """
        Initialize the evaluator
        
        Args:
            games_per_scenario: Number of games to play for each scenario
            device: Device for neural network agents
            save_detailed_results: Whether to save detailed game logs
            results_dir: Directory to save results
            num_processes: Number of processes to use (None for auto-detection)
        """
        self.games_per_scenario = games_per_scenario
        self.device = device
        self.save_detailed_results = save_detailed_results
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set number of processes
        if num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
        else:
            self.num_processes = max(1, num_processes)
        
        print(f"🚀 Using {self.num_processes} processes for parallel evaluation")
        
        # Results storage
        self.results = defaultdict(list)
        self.detailed_logs = []
        
        # Agent types available
        self.agent_types = ['random', 'cnn', 'mlp']
        
    def run_single_game(self, agents: List[Any], scenario_name: str) -> Dict[str, Any]:
        """
        Run a single game and return results
        
        Args:
            agents: List of 4 agents
            scenario_name: Name of the scenario for logging
            
        Returns:
            Dictionary with game results
        """
        try:
            game = big2Game()
            moves_count = 0
            
            # Reset all agents
            for agent in agents:
                if hasattr(agent, 'reset'):
                    agent.reset()

            while not game.isGameOver():
                player_go, first_player, history, hand, avail_actions = game.getCurrentState()
                
                if len(avail_actions) == 0:
                    # Force pass if no actions available
                    action = CardPlay([])
                else:
                    action = agents[player_go].step(first_player, history, hand, avail_actions)
                
                game.step(action)
                moves_count += 1
                
                # Safety check to prevent infinite loops
                if moves_count > 1000:  # Reasonable upper limit
                    print(f"Warning: Game {scenario_name} exceeded move limit")
                    break

            # Get final rewards and determine winner
            final_rewards = game.getRewards()
            winner = np.argmax(final_rewards)
            
            # Set final rewards for training agents
            for i, agent in enumerate(agents):
                if hasattr(agent, 'set_final_reward'):
                    agent.set_final_reward(final_rewards[i])
            
            # Calculate game statistics
            cards_left = [len(game.PlayersHand[i]) for i in range(4)]
            game_length = len(game.playHistory)
            
            game_result = {
                'scenario': scenario_name,
                'winner': int(winner),
                'final_scores': final_rewards.tolist(),
                'cards_left': cards_left,
                'game_length': game_length,
                'total_moves': moves_count,
                'agent_types': [type(agent).__name__ for agent in agents]
            }
            
            return game_result
            
        except Exception as e:
            print(f"Error in game {scenario_name}: {e}")
            return {
                'scenario': scenario_name,
                'winner': -1,  # Error indicator
                'final_scores': [0, 0, 0, 0],
                'cards_left': [13, 13, 13, 13],
                'game_length': 0,
                'total_moves': 0,
                'agent_types': [type(agent).__name__ for agent in agents],
                'error': str(e)
            }
    
    def evaluate_scenario(self, 
                         target_agent_type: str, 
                         opponent_agents: List[str],
                         scenario_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific scenario (1 target agent vs 3 opponents)
        
        Args:
            target_agent_type: Type of the agent being evaluated
            opponent_agents: List of 3 opponent agent types
            scenario_name: Name for this scenario
            
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n🎮 {scenario_name}")
        print(f"   {target_agent_type} vs {opponent_agents}")
        
        wins = 0
        total_games = 0
        game_results = []
        target_scores = []  # Collect target agent scores
        
        # Use tqdm for progress bar
        with tqdm(total=self.games_per_scenario, desc=f"  {scenario_name}", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for game_num in range(self.games_per_scenario):
                # Create agents for this game
                agents = []
                
                # Target agent (position 0)
                target_agent = AgentFactory.create_agent(
                    target_agent_type, 
                    agent_id=0,
                    device=self.device
                )
                agents.append(target_agent)
                
                # Opponent agents (positions 1, 2, 3)
                for i, opponent_type in enumerate(opponent_agents):
                    opponent_agent = AgentFactory.create_agent(
                        opponent_type,
                        agent_id=i+1,
                        device=self.device
                    )
                    agents.append(opponent_agent)
                
                # Run the game
                game_result = self.run_single_game(agents, scenario_name)
                game_results.append(game_result)
                
                # Check if target agent won and collect score
                if game_result['winner'] == 0:  # Target agent is at position 0
                    wins += 1
                
                # Collect target agent's score (position 0)
                if game_result['winner'] != -1:  # Valid game
                    target_scores.append(game_result['final_scores'][0])
                
                total_games += 1
                
                # Update progress bar with current win rate
                current_winrate = wins / (game_num + 1) * 100
                pbar.set_postfix({"Win Rate": f"{current_winrate:.1f}%"})
                pbar.update(1)
        
        # Calculate statistics
        win_rate = wins / total_games * 100 if total_games > 0 else 0
        avg_score = np.mean(target_scores) if target_scores else 0
        
        # Calculate additional statistics
        valid_games = [g for g in game_results if g['winner'] != -1]
        avg_game_length = np.mean([g['game_length'] for g in valid_games]) if valid_games else 0
        avg_total_moves = np.mean([g['total_moves'] for g in valid_games]) if valid_games else 0
        
        # Position statistics (where did target agent finish)
        positions = []
        for game in valid_games:
            # Determine final position based on scores
            scores = game['final_scores']
            target_score = scores[0]
            position = 1 + sum(1 for score in scores[1:] if score < target_score)
            positions.append(position)
        
        avg_position = np.mean(positions) if positions else 4
        position_distribution = {i: positions.count(i) for i in range(1, 5)}
        
        scenario_results = {
            'scenario_name': scenario_name,
            'target_agent': target_agent_type,
            'opponents': opponent_agents,
            'total_games': total_games,
            'wins': wins,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_position': avg_position,
            'position_distribution': position_distribution,
            'avg_game_length': avg_game_length,
            'avg_total_moves': avg_total_moves,
            'valid_games': len(valid_games),
            'error_games': total_games - len(valid_games)
        }
        
        print(f"   ✅ Results: {wins}/{total_games} wins ({win_rate:.1f}%), "
              f"Avg score: {avg_score:.2f}, Avg position: {avg_position:.2f}")
        
        # Store detailed results if requested
        if self.save_detailed_results:
            self.detailed_logs.extend(game_results)
        
        return scenario_results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive cross-evaluation of all agent types
        
        Returns:
            Dictionary with all evaluation results
        """
        print("🚀 Starting Comprehensive Big2 Agent Evaluation")
        print(f"📊 Configuration: {self.games_per_scenario} games per scenario")
        print(f"🎯 Agent types: {self.agent_types}")
        print(f"💾 Results will be saved to: {self.results_dir}")
        
        start_time = time.time()
        
        # 1. Baseline scenarios: Each agent vs 3 of same type
        print("\n" + "="*60)
        print("📋 PHASE 1: Baseline Scenarios (Same vs Same)")
        print("="*60)
        
        for agent_type in self.agent_types:
            scenario_name = f"{agent_type}_vs_3x{agent_type}"
            opponents = [agent_type] * 3
            
            results = self.evaluate_scenario(agent_type, opponents, scenario_name)
            self.results['baseline'].append(results)
        
        # 2. Cross-type scenarios: Each agent vs 3 different types
        print("\n" + "="*60)
        print("📋 PHASE 2: Cross-Type Scenarios (A vs 3xB)")
        print("="*60)
        
        for target_type in self.agent_types:
            for opponent_type in self.agent_types:
                if target_type != opponent_type:  # Skip same-type (already done)
                    scenario_name = f"{target_type}_vs_3x{opponent_type}"
                    opponents = [opponent_type] * 3
                    
                    results = self.evaluate_scenario(target_type, opponents, scenario_name)
                    self.results['cross_type'].append(results)
        
        # Calculate overall statistics
        total_time = time.time() - start_time
        total_games = sum(len(scenarios) for scenarios in self.results.values()) * self.games_per_scenario
        
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETE!")
        print("="*60)
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🎮 Total games played: {total_games}")
        print(f"📊 Scenarios evaluated: {sum(len(scenarios) for scenarios in self.results.values())}")
        
        return {
            'evaluation_summary': {
                'total_time': total_time,
                'total_games': total_games,
                'games_per_scenario': self.games_per_scenario,
                'agent_types': self.agent_types,
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': dict(self.results),
            'detailed_logs': self.detailed_logs if self.save_detailed_results else []
        }
    
    def save_results(self, evaluation_results: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save summary results as JSON
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create and save CSV summary
        csv_data = []
        for category, scenarios in evaluation_results['results'].items():
            for scenario in scenarios:
                csv_data.append({
                    'category': category,
                    'scenario': scenario['scenario_name'],
                    'target_agent': scenario['target_agent'],
                    'opponents': '_'.join(scenario['opponents']),
                    'win_rate': scenario['win_rate'],
                    'avg_score': scenario['avg_score'],
                    'avg_position': scenario['avg_position'],
                    'total_games': scenario['total_games'],
                    'wins': scenario['wins']
                })
        
        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / f"evaluation_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save detailed logs if available
        if self.save_detailed_results and evaluation_results['detailed_logs']:
            detailed_file = self.results_dir / f"detailed_logs_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(evaluation_results['detailed_logs'], f, indent=2)
        
        print("\n💾 Results saved to:")
        print(f"   📄 Summary: {summary_file}")
        print(f"   📊 CSV: {csv_file}")
        if self.save_detailed_results:
            print(f"   📋 Detailed logs: {detailed_file}")
    
    def print_summary_report(self, evaluation_results: Dict[str, Any]):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("📊 BIG2 AGENT EVALUATION SUMMARY REPORT")
        print("="*80)
        
        results = evaluation_results['results']
        
        # Overall win rates for each agent type
        print("\n🏆 OVERALL WIN RATES BY AGENT TYPE:")
        print("-" * 50)
        
        agent_stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'scenarios': 0, 'total_score': 0})
        
        for category, scenarios in results.items():
            for scenario in scenarios:
                agent = scenario['target_agent']
                agent_stats[agent]['wins'] += scenario['wins']
                agent_stats[agent]['games'] += scenario['total_games']
                agent_stats[agent]['scenarios'] += 1
                agent_stats[agent]['total_score'] += scenario['avg_score'] * scenario['total_games']
        
        for agent, stats in agent_stats.items():
            win_rate = stats['wins'] / stats['games'] * 100 if stats['games'] > 0 else 0
            avg_score = stats['total_score'] / stats['games'] if stats['games'] > 0 else 0
            print(f"  {agent.upper():8}: {stats['wins']:4}/{stats['games']:4} "
                  f"({win_rate:5.1f}%) | Avg Score: {avg_score:6.2f} | {stats['scenarios']:2} scenarios")
        
        # Overall average scores for each agent type
        print("\n📊 OVERALL AVERAGE SCORES BY AGENT TYPE:")
        print("-" * 50)
        
        for agent, stats in agent_stats.items():
            avg_score = stats['total_score'] / stats['games'] if stats['games'] > 0 else 0
            print(f"  {agent.upper():8}: {avg_score:6.2f}")
        
        # Best and worst scenarios for each agent
        print("\n🎯 BEST & WORST SCENARIOS:")
        print("-" * 50)
        
        for agent_type in self.agent_types:
            agent_scenarios = []
            for category, scenarios in results.items():
                for scenario in scenarios:
                    if scenario['target_agent'] == agent_type:
                        agent_scenarios.append(scenario)
            
            if agent_scenarios:
                best = max(agent_scenarios, key=lambda x: x['win_rate'])
                worst = min(agent_scenarios, key=lambda x: x['win_rate'])
                
                print(f"\n  {agent_type.upper()}:")
                print(f"    🏆 Best:  {best['scenario_name']:30} "
                      f"({best['win_rate']:5.1f}%)")
                print(f"    😓 Worst: {worst['scenario_name']:30} "
                      f"({worst['win_rate']:5.1f}%)")
        
        # Cross-comparison matrix
        print("\n🔄 CROSS-COMPARISON MATRIX (Win Rates %):")
        print("-" * 50)
        print("         vs 3xRandom  vs 3xCNN  vs 3xMLP")
        
        matrix = {}
        score_matrix = {}
        for category, scenarios in results.items():
            if category == 'cross_type' or category == 'baseline':
                for scenario in scenarios:
                    target = scenario['target_agent']
                    opponent = scenario['opponents'][0]  # All same type
                    if target not in matrix:
                        matrix[target] = {}
                        score_matrix[target] = {}
                    matrix[target][opponent] = scenario['win_rate']
                    score_matrix[target][opponent] = scenario['avg_score']
        
        for agent in self.agent_types:
            if agent in matrix:
                random_rate = matrix[agent].get('random', 0)
                cnn_rate = matrix[agent].get('cnn', 0)
                mlp_rate = matrix[agent].get('mlp', 0)
                print(f"{agent.upper():8}     {random_rate:6.1f}%    "
                      f"{cnn_rate:6.1f}%    {mlp_rate:6.1f}%")
        
        # Cross-comparison matrix for scores
        print("\n📈 CROSS-COMPARISON MATRIX (Average Scores):")
        print("-" * 50)
        print("         vs 3xRandom  vs 3xCNN  vs 3xMLP")
        
        for agent in self.agent_types:
            if agent in score_matrix:
                random_score = score_matrix[agent].get('random', 0)
                cnn_score = score_matrix[agent].get('cnn', 0)
                mlp_score = score_matrix[agent].get('mlp', 0)
                print(f"{agent.upper():8}     {random_score:8.2f}  {cnn_score:8.2f}  {mlp_score:8.2f}")


def main():
    """Main function to run the evaluation"""
    print("🎮 Big2 Agent Cross-Evaluation System")
    print("=" * 50)
    
    # Configuration
    games_per_scenario = 1000  # 1000 games per scenario
    device = "Auto"  # Will use CUDA if available
    
    # Create evaluator
    evaluator = GameEvaluator(
        games_per_scenario=games_per_scenario,
        device=device,
        save_detailed_results=True,
        results_dir="evaluation_results"
    )
    
    try:
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Save results
        evaluator.save_results(results)
        
        # Print summary report
        evaluator.print_summary_report(results)
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
