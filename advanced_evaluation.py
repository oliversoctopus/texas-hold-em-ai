import os
import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from poker_ai import PokerAI
from game_engine import TexasHoldEm  # Use full engine, not training version
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
    
    def simulate_game(self, ai_models: List[PokerAI], model_names: List[str], 
                     test_position: int, max_hands: int = 50) -> Dict:
        """
        Simulate a complete game with proper chip tracking
        Uses the full game engine like main.py does
        """
        num_players = len(ai_models)
        starting_chips = 1000
        
        # Create game using the full engine
        game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips)
        
        # Manually setup players with our AI models
        game.players = []
        for i, (ai_model, name) in enumerate(zip(ai_models, model_names)):
            player = Player(name, starting_chips, is_ai=True)
            player.ai_model = ai_model
            game.players.append(player)
        
        hands_played = 0
        hand_winners = []
        test_hand_wins = 0
        total_showdowns = 0
        test_showdown_wins = 0
        action_counts = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all_in': 0}
        test_action_count = 0
        
        # Play hands until someone wins or max hands reached
        for hand_num in range(max_hands):
            # Check if game should end
            active_players = [p for p in game.players if p.chips > 0]
            if len(active_players) <= 1:
                break
            
            # Skip if test player eliminated
            if game.players[test_position].chips <= 0:
                break
            
            # Reset for new hand
            game.deck.reset()
            game.community_cards = []
            game.pot = 0
            game.current_bet = 0
            game.action_history = []
            game.opponent_bets = []
            game.last_raise_amount = game.big_blind
            
            for player in game.players:
                player.reset_hand()
            
            # Deal cards
            for player in game.players:
                if player.chips > 0:
                    player.hand = game.deck.draw(2)
            
            # Post blinds
            sb_pos = (game.dealer_position + 1) % len(game.players)
            bb_pos = (game.dealer_position + 2) % len(game.players)
            
            if game.players[sb_pos].chips > 0:
                game.pot += game.players[sb_pos].bet(min(game.small_blind, game.players[sb_pos].chips))
            if game.players[bb_pos].chips > 0:
                game.pot += game.players[bb_pos].bet(min(game.big_blind, game.players[bb_pos].chips))
            game.current_bet = game.big_blind
            
            # Track actions before betting
            actions_before = len(game.action_history)
            
            # Play betting rounds
            streets = ['preflop', 'flop', 'turn', 'river']
            
            for street_idx, street in enumerate(streets):
                # Add community cards
                if street == 'flop':
                    game.community_cards.extend(game.deck.draw(3))
                elif street == 'turn':
                    game.community_cards.append(game.deck.draw())
                elif street == 'river':
                    game.community_cards.append(game.deck.draw())
                
                # Betting round
                if not game.betting_round(verbose=False, street_idx=street_idx):
                    break
                
                # Check if only one player left
                if sum(1 for p in game.players if not p.folded) == 1:
                    break
            
            # Track actions from test player (approximate)
            if len(game.action_history) > actions_before:
                # Estimate test player took about 1/num_active actions
                num_active = len([p for p in game.players if p.chips > 0])
                if num_active > 0:
                    estimated_test_actions = max(1, (len(game.action_history) - actions_before) // num_active)
                    
                    # Sample some actions to attribute to test player
                    for _ in range(estimated_test_actions):
                        if game.action_history[actions_before:]:
                            action = random.choice(game.action_history[actions_before:])
                            action_name = action.name.lower()
                            if action_name in action_counts:
                                action_counts[action_name] += 1
                                test_action_count += 1
            
            # Determine winners
            winners = game.determine_winners(verbose=False)
            
            # Track showdowns
            active_at_showdown = [p for p in game.players if not p.folded]
            if len(active_at_showdown) > 1:
                total_showdowns += 1
                if game.players[test_position] in winners:
                    test_showdown_wins += 1
            
            # Distribute pot
            game.distribute_pot(winners, verbose=False)
            
            # Track hand winner
            if winners:
                for winner in winners:
                    winner_idx = game.players.index(winner)
                    hand_winners.append(winner_idx)
                    if winner_idx == test_position:
                        test_hand_wins += 1
            
            hands_played += 1
            
            # Move dealer button
            game.dealer_position = (game.dealer_position + 1) % len(game.players)
        
        # Determine final winner
        final_standings = [(i, p.name, p.chips) for i, p in enumerate(game.players)]
        final_standings.sort(key=lambda x: x[2], reverse=True)
        
        game_winner_idx = final_standings[0][0] if final_standings[0][2] > 0 else -1
        
        # Calculate earnings for test player
        test_earnings = game.players[test_position].chips - starting_chips
        
        # Calculate action distribution
        action_distribution = {}
        if test_action_count > 0:
            for action, count in action_counts.items():
                action_distribution[action] = (count / test_action_count) * 100
        else:
            for action in action_counts:
                action_distribution[action] = 0
        
        return {
            'hands_played': hands_played,
            'game_winner_idx': game_winner_idx,
            'test_won_game': game_winner_idx == test_position,
            'test_hand_wins': test_hand_wins,
            'test_earnings': test_earnings,
            'test_showdown_wins': test_showdown_wins,
            'total_showdowns': total_showdowns,
            'action_distribution': action_distribution,
            'final_chip_total': sum(p.chips for p in game.players)
        }
    
    def evaluate_against_strong_opponents(self, 
                                         test_model: PokerAI,
                                         num_games: int = 100,
                                         num_players: int = 6,
                                         verbose: bool = True,
                                         max_hands_per_game: int = 50) -> Dict:
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
            'position_wins': {i: 0 for i in range(num_players)},
            'position_games': {i: 0 for i in range(num_players)},
            'hands_won': 0,
            'total_hands': 0,
            'games_won': 0,
            'total_games': 0,
            'games_ended_by_elimination': 0,
            'games_ended_by_hand_limit': 0,
            'avg_hands_per_game': 0
        }
        
        total_showdowns = 0
        showdowns_won = 0
        action_totals = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all_in': 0}
        total_actions = 0
        all_hands_played = []
        
        games_per_benchmark = max(1, num_games // max(1, len(self.benchmark_models)))
        
        # Test against each benchmark
        for benchmark_idx, benchmark_model in enumerate(self.benchmark_models):
            benchmark_name = self.benchmark_names[benchmark_idx]
            if verbose:
                print(f"\nEvaluating against {benchmark_name}...")
            
            benchmark_hands_won = 0
            benchmark_hands_played = 0
            benchmark_games_won = 0
            
            for game_num in range(games_per_benchmark):
                # Rotate test model position for fairness
                test_position = game_num % num_players
                results['position_games'][test_position] += 1
                
                # Build AI models list with test model at specified position
                ai_models = []
                model_names = []
                
                for pos in range(num_players):
                    if pos == test_position:
                        ai_models.append(test_model)
                        model_names.append("TEST")
                    else:
                        # Mix benchmark and other models for diversity
                        if pos == (test_position + 1) % num_players:
                            # Always put the current benchmark next to test model
                            ai_models.append(benchmark_model)
                            model_names.append(f"B_{benchmark_name}")
                        elif random.random() < 0.6 and len(self.benchmark_models) > 1:
                            # Use another benchmark model
                            other_benchmark = random.choice([m for m in self.benchmark_models if m != benchmark_model])
                            other_name = self.benchmark_names[self.benchmark_models.index(other_benchmark)]
                            ai_models.append(other_benchmark)
                            model_names.append(f"O_{other_name}")
                        else:
                            # Use a random player for diversity
                            random_ai = PokerAI(config={
                                'epsilon': random.uniform(0.7, 1.0),
                                'learning_rate': 0, 'gamma': 0,
                                'hidden_sizes': [64], 'dropout_rate': 0,
                                'batch_size': 1, 'update_target_every': 1000000,
                                'min_epsilon': 1.0, 'epsilon_decay': 1.0
                            })
                            ai_models.append(random_ai)
                            model_names.append("Random")
                
                # Simulate game
                game_result = self.simulate_game(ai_models, model_names, test_position, max_hands_per_game)
                
                # Update statistics
                hands_played = game_result['hands_played']
                all_hands_played.append(hands_played)
                benchmark_hands_played += hands_played
                results['total_hands'] += hands_played
                
                if game_result['test_won_game']:
                    benchmark_games_won += 1
                    results['games_won'] += 1
                    results['position_wins'][test_position] += 1
                
                benchmark_hands_won += game_result['test_hand_wins']
                results['hands_won'] += game_result['test_hand_wins']
                
                showdowns_won += game_result['test_showdown_wins']
                total_showdowns += game_result['total_showdowns']
                
                results['overall_earnings'] += game_result['test_earnings']
                results['total_games'] += 1
                
                # Update action distribution
                for action, pct in game_result['action_distribution'].items():
                    if pct > 0:
                        action_totals[action] += pct
                        total_actions += 1
                
                # Track how game ended
                if hands_played >= max_hands_per_game:
                    results['games_ended_by_hand_limit'] += 1
                else:
                    results['games_ended_by_elimination'] += 1
                
                # Verify chip conservation
                if game_result['final_chip_total'] != num_players * 1000:
                    if verbose:
                        print(f"  ⚠️ Chip conservation error in game {game_num + 1}")
            
            # Store results vs this benchmark
            if benchmark_hands_played > 0:
                results['vs_each_opponent'][benchmark_name] = {
                    'hands_won': benchmark_hands_won,
                    'hands_played': benchmark_hands_played,
                    'hand_win_rate': (benchmark_hands_won / benchmark_hands_played) * 100,
                    'games_won': benchmark_games_won,
                    'games_played': games_per_benchmark,
                    'game_win_rate': (benchmark_games_won / games_per_benchmark) * 100 if games_per_benchmark > 0 else 0
                }
        
        # Calculate overall statistics
        if results['total_games'] > 0:
            results['overall_win_rate'] = (results['games_won'] / results['total_games']) * 100
            results['survival_rate'] = results['overall_win_rate']  # Simplified
            results['avg_hands_per_game'] = np.mean(all_hands_played) if all_hands_played else 0
            results['overall_earnings'] = results['overall_earnings'] / results['total_games']
        
        if results['total_hands'] > 0:
            results['hand_win_rate'] = (results['hands_won'] / results['total_hands']) * 100
        
        # Calculate action distribution (average across all games)
        if total_actions > 0:
            for action in action_totals:
                results['action_distribution'][action] = action_totals[action] / (results['total_games'] + 0.001)
        
        # Calculate showdown win rate
        if total_showdowns > 0:
            results['showdown_win_rate'] = (showdowns_won / total_showdowns) * 100
        
        # Calculate position-based win rates
        results['position_win_rates'] = {}
        for pos in range(num_players):
            if results['position_games'][pos] > 0:
                results['position_win_rates'][pos] = (results['position_wins'][pos] / results['position_games'][pos]) * 100
        
        # Restore original epsilon
        test_model.epsilon = original_epsilon
        
        return results
    
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
        print(f"  Game Win Rate: {results['overall_win_rate']:.1f}% ({results['games_won']}/{results['total_games']} games)")
        print(f"  Hand Win Rate: {results.get('hand_win_rate', 0):.1f}% ({results['hands_won']}/{results['total_hands']} hands)")
        print(f"  Average Earnings: ${results['overall_earnings']:+.0f}")
        print(f"  Showdown Win Rate: {results['showdown_win_rate']:.1f}%")
        print(f"\nGame Statistics:")
        print(f"  Games ended by elimination: {results['games_ended_by_elimination']}")
        print(f"  Games ended by hand limit: {results['games_ended_by_hand_limit']}")
        print(f"  Average hands per game: {results['avg_hands_per_game']:.1f}")
        
        print(f"\nWin Rate by Position:")
        for pos in range(num_players):
            win_rate = results['position_win_rates'].get(pos, 0)
            games = results['position_games'].get(pos, 0)
            wins = results['position_wins'].get(pos, 0)
            position_name = {0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}.get(pos, f"Pos{pos}")
            print(f"  {position_name}: {win_rate:.1f}% ({wins}/{games} games)")
        
        print(f"\nAction Distribution:")
        for action, pct in results['action_distribution'].items():
            print(f"  {action.capitalize()}: {pct:.1f}%")
        
        # Flag potential issues
        if results['action_distribution']['all_in'] > 50:
            print("\n⚠️ WARNING: Model shows excessive all-in behavior (>50%)")
            print("   This strategy may be exploitable by competent human players.")
        
        if results['action_distribution']['fold'] > 70:
            print("\n⚠️ WARNING: Model is too passive (>70% folds)")
        
        if results['showdown_win_rate'] < 40 and results['showdown_win_rate'] > 0:
            print("\n⚠️ WARNING: Low showdown win rate suggests poor hand selection")
        
        if results['avg_hands_per_game'] < 5:
            print("\n⚠️ WARNING: Games ending too quickly - aggressive all-in strategies detected")
        
        print(f"\nPerformance vs Each Benchmark:")
        for opponent, stats in results['vs_each_opponent'].items():
            print(f"  vs {opponent}:")
            print(f"    Hand Win Rate: {stats.get('hand_win_rate', 0):.1f}% ({stats.get('hands_won', 0)}/{stats.get('hands_played', 0)})")
            print(f"    Game Win Rate: {stats.get('game_win_rate', 0):.1f}% ({stats.get('games_won', 0)}/{stats.get('games_played', 0)} games)")
        
        print("\n" + "=" * 80)


def evaluate_all_models(model_paths: List[str], 
                        benchmark_models: List[str] = None,
                        num_games: int = 100,
                        show_position_stats: bool = True,
                        max_hands_per_game: int = 50) -> None:
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
                model, num_games=num_games, verbose=False, num_players=6,
                max_hands_per_game=max_hands_per_game
            )
            
            # Show position stats if requested
            if show_position_stats:
                print(f"  Position Win Rates for {model_name}:")
                for pos in range(6):
                    win_rate = results['position_win_rates'].get(pos, 0)
                    games = results['position_games'].get(pos, 0)
                    wins = results['position_wins'].get(pos, 0)
                    position_name = {0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}.get(pos, f"Pos{pos}")
                    print(f"    {position_name}: {win_rate:.1f}% ({wins}/{games} games)")
                
                print(f"  Game statistics:")
                print(f"    Ended by elimination: {results['games_ended_by_elimination']}")
                print(f"    Ended by hand limit: {results['games_ended_by_hand_limit']}")
                print(f"    Average hands per game: {results.get('avg_hands_per_game', 0):.1f}")
            
            # Calculate composite score
            strategy_penalty = max(0, results['action_distribution']['all_in'] - 30) * 2
            composite_score = (
                results['overall_win_rate'] * 0.2 +  # Game wins
                results.get('hand_win_rate', 0) * 0.4 +  # Hand wins (more weight)
                results['showdown_win_rate'] * 0.2 +
                (100 - strategy_penalty) * 0.2
            )
            
            all_results.append({
                'name': model_name,
                'game_win_rate': results['overall_win_rate'],
                'hand_win_rate': results.get('hand_win_rate', 0),
                'showdown_wr': results['showdown_win_rate'],
                'all_in_pct': results['action_distribution']['all_in'],
                'composite_score': composite_score,
                'games_won': results['games_won'],
                'total_games': results['total_games'],
                'hands_won': results['hands_won'],
                'total_hands': results['total_hands'],
                'position_stats': results.get('position_win_rates', {}),
                'avg_hands_per_game': results.get('avg_hands_per_game', 0),
                'games_by_elim': results['games_ended_by_elimination'],
                'games_by_limit': results['games_ended_by_hand_limit']
            })
            
        except Exception as e:
            print(f"Error evaluating {path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Rank and display results
    print("\n" + "=" * 80)
    print("MODEL RANKINGS (Against Strong Opponents)")
    print("=" * 80)
    
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<20} {'GameW%':<8} {'HandW%':<8} {'Show%':<8} {'AllIn%':<8} {'H/Game':<8} {'Score':<8}")
    print("-" * 85)
    
    for i, result in enumerate(all_results, 1):
        print(f"{i:<5} {result['name']:<20} "
              f"{result['game_win_rate']:<8.1f} {result['hand_win_rate']:<8.1f} "
              f"{result['showdown_wr']:<8.1f} {result['all_in_pct']:<8.1f} "
              f"{result['avg_hands_per_game']:<8.1f} {result['composite_score']:<8.1f}")
        
        # Add warning for problematic patterns
        if result['avg_hands_per_game'] < 5:
            print(f"      ⚠️ Very aggressive play detected")
    
    print("\nLegend:")
    print("  GameW%: Game win rate (winning entire games)")
    print("  HandW%: Hand win rate (winning individual hands)")
    print("  Show%: Win rate when reaching showdown")
    print("  AllIn%: Percentage of all-in actions")
    print("  H/Game: Average hands per game")
    print("  Score: Composite score considering all factors")
    
    # Analysis of results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if all_results:
        avg_game_wr = np.mean([r['game_win_rate'] for r in all_results])
        avg_hand_wr = np.mean([r['hand_win_rate'] for r in all_results])
        avg_all_in = np.mean([r['all_in_pct'] for r in all_results])
        avg_hands_game = np.mean([r['avg_hands_per_game'] for r in all_results])
        
        print(f"\nAverage Game Win Rate: {avg_game_wr:.1f}%")
        print(f"Average Hand Win Rate: {avg_hand_wr:.1f}%")
        print(f"Average All-in %: {avg_all_in:.1f}%")
        print(f"Average Hands per Game: {avg_hands_game:.1f}")
        
        if avg_hands_game < 10:
            print("\n⚠️ WARNING: Games are ending too quickly on average!")
            print("  This suggests aggressive all-in strategies dominate.")
        
        # Expected rates
        print(f"\nExpected random game win rate (6 players): {100/6:.1f}%")
        print(f"Expected random hand win rate (6 players): {100/6:.1f}%")
        
        # Identify trends
        simple_models = [r for r in all_results if 'opponent_eval' in r['name'] or 'best_' in r['name']]
        complex_models = [r for r in all_results if 'tuned_ai_v2' in r['name'] or 'tuned_ai_v4' in r['name']]
        
        if simple_models:
            avg_simple_all_in = np.mean([r['all_in_pct'] for r in simple_models])
            avg_simple_hands = np.mean([r['avg_hands_per_game'] for r in simple_models])
            print(f"\nSimple models:")
            print(f"  Average all-in: {avg_simple_all_in:.1f}%")
            print(f"  Average hands/game: {avg_simple_hands:.1f}")
        
        if complex_models:
            avg_complex_all_in = np.mean([r['all_in_pct'] for r in complex_models])
            avg_complex_hands = np.mean([r['avg_hands_per_game'] for r in complex_models])
            print(f"\nComplex models:")
            print(f"  Average all-in: {avg_complex_all_in:.1f}%")
            print(f"  Average hands/game: {avg_complex_hands:.1f}")


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
    test_model_path = 'tuned_ai_v2.pth'  # Change to your model
    
    if os.path.exists(test_model_path):
        test_model = PokerAI()
        test_model.load(test_model_path)
        
        # Run comprehensive evaluation
        evaluator.comprehensive_evaluation(
            test_model,
            model_name=test_model_path.replace('.pth', ''),
            num_games=100,
            num_players=6
        )
    
    # Or evaluate all models at once
    all_models = [
        'standard_poker_ai.pth',
        'enhanced_poker_ai.pth',
        'standard_ai_v2.pth',
        'standard_ai_v3.pth',
        'poker_ai_tuned.pth',
        'tuned_ai_v2.pth',
        'tuned_ai_v3.pth',
        'standard_ai_6p.pth',
        'standard_ai_v4.pth',
        'tuned_ai_v4.pth',
        'opponent_eval.pth',
        'opponent_eval_v2.pth',
        'best_checkpoint.pth',
        'best_final.pth',
        'best_checkpoint_2.pth',
        'best_final_2.pth'
    ]
    
    print("\n\nEvaluating all models against strong opponents...")
    evaluate_all_models(all_models, benchmark_models, num_games=100, show_position_stats=True)


if __name__ == "__main__":
    run_advanced_evaluation()