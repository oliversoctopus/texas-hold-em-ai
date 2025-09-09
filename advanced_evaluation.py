import os
import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from player import Player

class AdvancedEvaluator:
    """Advanced evaluation system that tests against strong, trained AI models"""
    
    def __init__(self, benchmark_models_paths: List[str] = None):
        """
        Initialize evaluator with paths to strong benchmark models
        
        Args:
            benchmark_models_paths: List of paths to .pth files of strong models
        """
        self.benchmark_models = []
        self.benchmark_names = []
        
        # Default strong models to load if not specified
        if benchmark_models_paths is None:
            # Try to find these strong models in the current directory
            potential_models = [
                'tuned_ai_v2.pth',      # Complex model with good strategy
                'tuned_ai_v4.pth',      # Another strong version
                'poker_ai_tuned.pth',   # Earlier strong model
                'standard_ai_v3.pth',   # Decent baseline
                'opponent_eval.pth'     # For comparison with all-in strategy
            ]
            benchmark_models_paths = [p for p in potential_models if os.path.exists(p)]
        
        # Load benchmark models
        for path in benchmark_models_paths:
            try:
                model = PokerAI()
                model.load(path)
                model.epsilon = 0  # No exploration during evaluation
                self.benchmark_models.append(model)
                self.benchmark_names.append(os.path.basename(path).replace('.pth', ''))
                print(f"Loaded benchmark model: {os.path.basename(path)}")
            except Exception as e:
                print(f"Could not load {path}: {e}")
        
        if not self.benchmark_models:
            print("Warning: No benchmark models loaded. Will use varied random strategies.")
            self._create_fallback_benchmarks()
    
    def _create_fallback_benchmarks(self):
        """Create varied AI strategies if no trained models are available"""
        # Aggressive AI (high raise frequency)
        aggressive = PokerAI(config={
            'epsilon': 0.1, 'learning_rate': 0,
            'gamma': 0.95, 'hidden_sizes': [256, 128],
            'dropout_rate': 0.2, 'batch_size': 1,
            'update_target_every': 1000000,
            'min_epsilon': 0.1, 'epsilon_decay': 1.0
        })
        self.benchmark_models.append(aggressive)
        self.benchmark_names.append("Aggressive_Random")
        
        # Conservative AI (high fold frequency)
        conservative = PokerAI(config={
            'epsilon': 0.8, 'learning_rate': 0,
            'gamma': 0.95, 'hidden_sizes': [128],
            'dropout_rate': 0.1, 'batch_size': 1,
            'update_target_every': 1000000,
            'min_epsilon': 0.8, 'epsilon_decay': 1.0
        })
        self.benchmark_models.append(conservative)
        self.benchmark_names.append("Conservative_Random")
    
    def evaluate_against_strong_opponents(self, 
                                         test_model: PokerAI,
                                         num_games: int = 100,
                                         num_players: int = 6,
                                         verbose: bool = True) -> Dict:
        """
        Evaluate a model against strong, trained opponents
        
        Returns dict with detailed performance metrics
        """
        original_epsilon = test_model.epsilon
        test_model.epsilon = 0  # No exploration during evaluation
        
        results = {
            'overall_win_rate': 0,
            'overall_earnings': 0,
            'survival_rate': 0,
            'vs_each_opponent': {},
            'action_distribution': {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all_in': 0},
            'showdown_win_rate': 0,
            'bluff_success_rate': 0,
            'position_performance': {},
            'hand_strength_performance': {}
        }
        
        total_wins = 0
        total_earnings = 0
        total_survivals = 0
        total_showdowns = 0
        showdowns_won = 0
        total_actions = 0
        action_counts = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all_in': 0}
        
        # Track performance against each benchmark
        for benchmark_idx, benchmark_model in enumerate(self.benchmark_models):
            benchmark_name = self.benchmark_names[benchmark_idx]
            if verbose:
                print(f"\nEvaluating against {benchmark_name}...")
            
            benchmark_wins = 0
            benchmark_earnings = 0
            
            for game_num in range(num_games // len(self.benchmark_models)):
                # Create game with mixed opponents
                training_game = TexasHoldEmTraining(num_players=num_players)
                
                # Build opponent pool - mix of benchmark and other models
                ai_models = [test_model]  # Test model always in position 0
                
                # Add the current benchmark model
                ai_models.append(benchmark_model)
                
                # Fill remaining slots with random selection from all benchmarks
                remaining_slots = num_players - 2
                for _ in range(remaining_slots):
                    # Mix between benchmark models and some random play
                    if random.random() < 0.7 and len(self.benchmark_models) > 1:
                        # Use another benchmark model
                        other_benchmark = random.choice(self.benchmark_models)
                        ai_models.append(other_benchmark)
                    else:
                        # Use a random player for diversity
                        random_ai = PokerAI(config={
                            'epsilon': random.uniform(0.5, 1.0),
                            'learning_rate': 0, 'gamma': 0,
                            'hidden_sizes': [64], 'dropout_rate': 0,
                            'batch_size': 1, 'update_target_every': 1000000,
                            'min_epsilon': 1.0, 'epsilon_decay': 1.0
                        })
                        ai_models.append(random_ai)
                
                # Play multiple hands
                hand_wins = 0
                hand_earnings = 0
                initial_chips = 1000
                
                for hand in range(20):  # 20 hands per game
                    training_game.reset_game()
                    
                    # Track actions for analysis
                    pre_hand_actions = len(training_game.action_history)
                    
                    winners = training_game.simulate_hand(ai_models)
                    
                    # Analyze actions taken
                    for action in training_game.action_history[pre_hand_actions:]:
                        if action.name.lower() in action_counts:
                            action_counts[action.name.lower()] += 1
                            total_actions += 1
                    
                    # Check if test model won
                    test_player = training_game.players[0]  # Test model is always first
                    if test_player in winners:
                        hand_wins += 1
                        benchmark_wins += 1
                    
                    # Track showdowns
                    if len([p for p in training_game.players if not p.folded]) > 1:
                        total_showdowns += 1
                        if test_player in winners:
                            showdowns_won += 1
                
                # Calculate earnings (simplified)
                if hand_wins > 0:
                    hand_earnings = (hand_wins * 100) - (20 - hand_wins) * 25
                    benchmark_earnings += hand_earnings
                    total_earnings += hand_earnings
                
                if hand_wins > 0:
                    total_survivals += 1
                
                if hand_wins >= 10:  # Won majority of hands
                    total_wins += 1
            
            # Store results vs this benchmark
            games_vs_benchmark = num_games // len(self.benchmark_models)
            results['vs_each_opponent'][benchmark_name] = {
                'win_rate': (benchmark_wins / (games_vs_benchmark * 20)) * 100,
                'earnings': benchmark_earnings / games_vs_benchmark
            }
        
        # Calculate overall statistics
        results['overall_win_rate'] = (total_wins / num_games) * 100
        results['overall_earnings'] = total_earnings / num_games
        results['survival_rate'] = (total_survivals / num_games) * 100
        
        # Calculate action distribution
        if total_actions > 0:
            for action, count in action_counts.items():
                results['action_distribution'][action] = (count / total_actions) * 100
        
        # Calculate showdown win rate
        if total_showdowns > 0:
            results['showdown_win_rate'] = (showdowns_won / total_showdowns) * 100
        
        # Restore original epsilon
        test_model.epsilon = original_epsilon
        
        return results
    
    def evaluate_head_to_head(self, 
                             model1: PokerAI, 
                             model2: PokerAI,
                             num_games: int = 100,
                             verbose: bool = True) -> Dict:
        """
        Direct head-to-head comparison between two models
        """
        model1.epsilon = 0
        model2.epsilon = 0
        
        model1_wins = 0
        model2_wins = 0
        model1_earnings = 0
        model2_earnings = 0
        
        training_game = TexasHoldEmTraining(num_players=2)
        
        for game in range(num_games):
            # Alternate positions for fairness
            if game % 2 == 0:
                ai_models = [model1, model2]
                model1_idx = 0
            else:
                ai_models = [model2, model1]
                model1_idx = 1
            
            # Play multiple hands
            for hand in range(10):
                training_game.reset_game()
                winners = training_game.simulate_hand(ai_models)
                
                if training_game.players[model1_idx] in winners:
                    model1_wins += 1
                else:
                    model2_wins += 1
        
        total_hands = num_games * 10
        
        return {
            'model1_win_rate': (model1_wins / total_hands) * 100,
            'model2_win_rate': (model2_wins / total_hands) * 100,
            'model1_expected_value': model1_wins - model2_wins,
            'model2_expected_value': model2_wins - model1_wins
        }
    
    def comprehensive_evaluation(self,
                                test_model: PokerAI,
                                model_name: str = "Test Model",
                                num_games: int = 200,
                                num_players: int = 6) -> None:
        """
        Run comprehensive evaluation and print detailed report
        """
        print("=" * 80)
        print(f"COMPREHENSIVE EVALUATION: {model_name}")
        print("=" * 80)
        
        # Test against strong opponents
        print("\n1. TESTING AGAINST STRONG OPPONENTS")
        print("-" * 40)
        results = self.evaluate_against_strong_opponents(
            test_model, num_games=num_games, num_players=num_players
        )
        
        print(f"\nOverall Performance:")
        print(f"  Win Rate: {results['overall_win_rate']:.1f}%")
        print(f"  Average Earnings: ${results['overall_earnings']:+.0f}")
        print(f"  Survival Rate: {results['survival_rate']:.1f}%")
        print(f"  Showdown Win Rate: {results['showdown_win_rate']:.1f}%")
        
        print(f"\nAction Distribution:")
        for action, pct in results['action_distribution'].items():
            print(f"  {action.capitalize()}: {pct:.1f}%")
        
        # Flag potential issues
        if results['action_distribution']['all_in'] > 50:
            print("\n⚠️ WARNING: Model shows excessive all-in behavior (>50%)")
            print("   This strategy may be exploitable by competent human players.")
        
        if results['action_distribution']['fold'] > 70:
            print("\n⚠️ WARNING: Model is too passive (>70% folds)")
        
        if results['showdown_win_rate'] < 40:
            print("\n⚠️ WARNING: Low showdown win rate suggests poor hand selection")
        
        print(f"\nPerformance vs Each Benchmark:")
        for opponent, stats in results['vs_each_opponent'].items():
            print(f"  vs {opponent}:")
            print(f"    Win Rate: {stats['win_rate']:.1f}%")
            print(f"    Earnings: ${stats['earnings']:+.0f}")
        
        # Test head-to-head against best benchmark if available
        if self.benchmark_models and len(self.benchmark_models) > 0:
            print("\n2. HEAD-TO-HEAD COMPARISON")
            print("-" * 40)
            
            # Find the strongest benchmark (assuming first is strongest)
            strongest_benchmark = self.benchmark_models[0]
            strongest_name = self.benchmark_names[0]
            
            print(f"\nTesting head-to-head against {strongest_name}...")
            h2h_results = self.evaluate_head_to_head(
                test_model, strongest_benchmark, num_games=50
            )
            
            print(f"  {model_name} Win Rate: {h2h_results['model1_win_rate']:.1f}%")
            print(f"  {strongest_name} Win Rate: {h2h_results['model2_win_rate']:.1f}%")
            print(f"  Expected Value Difference: {h2h_results['model1_expected_value']:+.0f}")
        
        # Overall assessment
        print("\n3. OVERALL ASSESSMENT")
        print("-" * 40)
        
        # Calculate overall score
        strategy_diversity_score = 100 - abs(25 - results['action_distribution']['raise'])
        strategy_diversity_score -= max(0, results['action_distribution']['all_in'] - 30) * 2
        
        overall_score = (
            results['overall_win_rate'] * 0.3 +
            results['showdown_win_rate'] * 0.2 +
            strategy_diversity_score * 0.3 +
            min(100, results['survival_rate']) * 0.2
        )
        
        print(f"Overall Score: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print("Rating: EXCELLENT - Tournament ready")
        elif overall_score >= 65:
            print("Rating: GOOD - Competitive against strong players")
        elif overall_score >= 50:
            print("Rating: MODERATE - Needs refinement")
        else:
            print("Rating: POOR - Significant improvements needed")
        
        print("\n" + "=" * 80)


def evaluate_all_models(model_paths: List[str], 
                        benchmark_models: List[str] = None,
                        num_games: int = 100) -> None:
    """
    Evaluate multiple models and rank them
    """
    evaluator = AdvancedEvaluator(benchmark_models)
    
    all_results = []
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Skipping {path} - file not found")
            continue
        
        try:
            model = PokerAI()
            model.load(path)
            model.epsilon = 0
            
            model_name = os.path.basename(path).replace('.pth', '')
            print(f"\nEvaluating {model_name}...")
            
            results = evaluator.evaluate_against_strong_opponents(
                model, num_games=num_games, verbose=False
            )
            
            # Calculate composite score
            strategy_penalty = max(0, results['action_distribution']['all_in'] - 30) * 2
            composite_score = (
                results['overall_win_rate'] * 0.4 +
                results['showdown_win_rate'] * 0.3 +
                (100 - strategy_penalty) * 0.3
            )
            
            all_results.append({
                'name': model_name,
                'win_rate': results['overall_win_rate'],
                'showdown_wr': results['showdown_win_rate'],
                'all_in_pct': results['action_distribution']['all_in'],
                'composite_score': composite_score
            })
            
        except Exception as e:
            print(f"Error evaluating {path}: {e}")
    
    # Rank and display results
    print("\n" + "=" * 80)
    print("MODEL RANKINGS (Against Strong Opponents)")
    print("=" * 80)
    
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<25} {'Win%':<8} {'Show%':<8} {'AllIn%':<8} {'Score':<8}")
    print("-" * 70)
    
    for i, result in enumerate(all_results, 1):
        print(f"{i:<5} {result['name']:<25} "
              f"{result['win_rate']:<8.1f} {result['showdown_wr']:<8.1f} "
              f"{result['all_in_pct']:<8.1f} {result['composite_score']:<8.1f}")
    
    print("\nLegend:")
    print("  Win%: Overall win rate against strong opponents")
    print("  Show%: Win rate when reaching showdown")
    print("  AllIn%: Percentage of all-in actions (lower is often better)")
    print("  Score: Composite score considering strategy diversity")


# Example usage function
def run_advanced_evaluation():
    """
    Example of how to use the advanced evaluation system
    """
    # List your strong benchmark models here
    benchmark_models = [
        'tuned_ai_v2.pth',      # Complex model with good strategy
        'tuned_ai_v4.pth',      # Another strong version
        'poker_ai_tuned.pth',   # Earlier strong model
    ]
    
    # Create evaluator with strong models as benchmarks
    evaluator = AdvancedEvaluator(benchmark_models)
    
    # Test a specific model
    test_model_path = 'best_final_2.pth'  # Change to your model
    
    if os.path.exists(test_model_path):
        test_model = PokerAI()
        test_model.load(test_model_path)
        
        # Run comprehensive evaluation
        evaluator.comprehensive_evaluation(
            test_model,
            model_name=test_model_path.replace('.pth', ''),
            num_games=200,
            num_players=6
        )
    
    # Or evaluate all models at once
    all_models = [
        'standard_poker_ai.pth',
        'enhanced_poker_ai.pth',
        'tuned_ai_v2.pth',
        'opponent_eval.pth',
        'best_checkpoint_2.pth',
        'best_final_2.pth',
        # Add more models as needed
    ]
    
    print("\n\nEvaluating all models against strong opponents...")
    evaluate_all_models(all_models, benchmark_models, num_games=100)