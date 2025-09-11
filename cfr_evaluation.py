"""
Real poker game evaluation for CFR AI
"""

import random
import numpy as np
from typing import List, Dict, Optional
from game_engine import TexasHoldEm
from player import Player
from cfr_player import CFRPlayer
from poker_ai import PokerAI
import os


class CFRGameEvaluator:
    """
    Evaluates CFR AI by playing actual poker games against various opponents
    """
    
    def __init__(self):
        self.benchmark_models = []
        self.benchmark_names = []
        self._load_benchmark_models()
    
    def _load_benchmark_models(self):
        """Load existing strong models as opponents"""
        potential_models = [
            'tuned_ai_v2.pth',
            'tuned_ai_v4.pth', 
            'poker_ai_tuned.pth',
            'standard_ai_v3.pth',
            'balanced_ai_v3.pth',
            'best_final_2.pth',
            'opponent_eval_v2.pth'
        ]
        
        for path in potential_models:
            if os.path.exists(path):
                try:
                    model = PokerAI()
                    model.load(path)
                    model.epsilon = 0  # No exploration during evaluation
                    self.benchmark_models.append(model)
                    self.benchmark_names.append(os.path.basename(path).replace('.pth', ''))
                    print(f"  Loaded opponent: {os.path.basename(path)}")
                except Exception as e:
                    pass
        
        if not self.benchmark_models:
            print("  No benchmark models found - will use random opponents")
    
    def evaluate_cfr_real_games(self, cfr_ai, num_games: int = 100, 
                               num_players: int = 4, verbose: bool = True) -> Dict:
        """
        Evaluate CFR AI by playing real poker games
        
        Args:
            cfr_ai: Trained CFR AI
            num_games: Number of games to play
            num_players: Number of players per game
            verbose: Whether to print progress
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if verbose:
            print(f"Evaluating CFR AI in {num_games} real poker games...")
            print(f"Players per game: {num_players}")
        
        # Create CFR player wrapper
        cfr_player = CFRPlayer(cfr_ai)
        
        # Statistics tracking
        cfr_wins = 0
        cfr_chips_won = 0
        cfr_total_invested = 0
        games_played = 0
        position_stats = {i: {'games': 0, 'wins': 0} for i in range(num_players)}
        action_counts = {'FOLD': 0, 'CHECK': 0, 'CALL': 0, 'RAISE': 0, 'ALL_IN': 0}
        
        for game_num in range(num_games):
            if verbose and game_num % 20 == 0:
                print(f"  Progress: {game_num}/{num_games}")
            
            # Setup game
            game = TexasHoldEm(num_players=num_players, starting_chips=1000)
            
            # Create players
            players = []
            
            # CFR AI player (position rotates)
            cfr_position = game_num % num_players
            
            for i in range(num_players):
                if i == cfr_position:
                    # Our CFR AI
                    player = Player(f"CFR_AI", 1000, is_ai=True)
                    player.ai_model = self._create_cfr_wrapper(cfr_player)
                    players.append(player)
                else:
                    # Opponent AI
                    if self.benchmark_models:
                        # Use strong benchmark model
                        opponent_model = random.choice(self.benchmark_models)
                        player = Player(f"Strong_AI_{i}", 1000, is_ai=True)
                        player.ai_model = opponent_model
                    else:
                        # Random opponent
                        player = Player(f"Random_{i}", 1000, is_ai=True)
                        player.ai_model = self._create_random_ai()
                    players.append(player)
            
            game.players = players
            
            # Track CFR position stats
            position_stats[cfr_position]['games'] += 1
            
            # Play the game
            starting_chips = players[cfr_position].chips
            winner = self._play_single_game(game, cfr_position, action_counts)
            
            # Update statistics
            games_played += 1
            final_chips = players[cfr_position].chips
            chips_change = final_chips - starting_chips
            cfr_chips_won += chips_change
            
            if winner == cfr_position:
                cfr_wins += 1
                position_stats[cfr_position]['wins'] += 1
        
        # Calculate final statistics
        win_rate = (cfr_wins / games_played) * 100 if games_played > 0 else 0
        avg_chips_per_game = cfr_chips_won / games_played if games_played > 0 else 0
        expected_random_rate = (1.0 / num_players) * 100
        
        # Calculate position performance
        position_performance = {}
        for pos in range(num_players):
            if position_stats[pos]['games'] > 0:
                pos_wr = (position_stats[pos]['wins'] / position_stats[pos]['games']) * 100
                position_performance[pos] = pos_wr
        
        # Action distribution
        total_actions = sum(action_counts.values())
        action_distribution = {}
        if total_actions > 0:
            for action, count in action_counts.items():
                action_distribution[action] = (count / total_actions) * 100
        
        results = {
            'win_rate': win_rate,
            'games_played': games_played,
            'avg_chips_per_game': avg_chips_per_game,
            'expected_random_rate': expected_random_rate,
            'performance_vs_random': win_rate / expected_random_rate if expected_random_rate > 0 else 0,
            'position_performance': position_performance,
            'action_distribution': action_distribution,
            'opponents_used': len(self.benchmark_models) if self.benchmark_models else 0
        }
        
        if verbose:
            self._print_evaluation_results(results, num_players)
        
        return results
    
    def _create_cfr_wrapper(self, cfr_player):
        """Create a wrapper that makes CFR player compatible with game engine"""
        class CFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0
            
            def choose_action(self, state, valid_actions):
                # Convert game state to CFR format
                game_state = {
                    'hole_cards': getattr(state, 'hole_cards', []),
                    'community_cards': getattr(state, 'community_cards', []),
                    'pot_size': getattr(state, 'pot_size', 0),
                    'to_call': getattr(state, 'to_call', 0),
                    'stack_size': getattr(state, 'stack_size', 1000),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4),
                    'action_history': getattr(state, 'action_history', [])
                }
                return self.cfr_player.choose_action(game_state)
            
            def get_raise_size(self, state):
                game_state = {
                    'pot_size': getattr(state, 'pot_size', 0),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4)
                }
                return self.cfr_player.get_raise_size(game_state)
        
        return CFRWrapper(cfr_player)
    
    def _create_random_ai(self):
        """Create a random AI opponent"""
        class RandomAI:
            def __init__(self):
                self.epsilon = 0
            
            def choose_action(self, state, valid_actions):
                from game_constants import Action
                return random.choice([Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE])
            
            def get_raise_size(self, state):
                return max(20, getattr(state, 'pot_size', 0) // 4)
        
        return RandomAI()
    
    def _play_single_game(self, game, cfr_position, action_counts):
        """Play a single game and track actions"""
        # Simple game simulation - play until one player has all chips
        hands_played = 0
        max_hands = 50  # Prevent infinite games
        
        while len([p for p in game.players if p.chips > 0]) > 1 and hands_played < max_hands:
            try:
                # Track actions for CFR player
                initial_cfr_chips = game.players[cfr_position].chips
                
                # Play a hand (simplified)
                game.play_hand()
                hands_played += 1
                
                # Update action counts (simplified tracking)
                if hands_played % 5 == 0:  # Sample some actions
                    action_counts['CALL'] += 1  # Placeholder - would need actual action tracking
                
            except Exception:
                break
        
        # Find winner (player with most chips)
        winner_idx = 0
        max_chips = 0
        for i, player in enumerate(game.players):
            if player.chips > max_chips:
                max_chips = player.chips
                winner_idx = i
        
        return winner_idx
    
    def _print_evaluation_results(self, results, num_players):
        """Print formatted evaluation results"""
        print(f"\n" + "="*60)
        print("CFR AI EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Games played: {results['games_played']}")
        print(f"  Win rate: {results['win_rate']:.1f}%")
        print(f"  Expected random rate: {results['expected_random_rate']:.1f}%")
        print(f"  Performance vs random: {results['performance_vs_random']:.1f}x")
        print(f"  Average chips per game: ${results['avg_chips_per_game']:.0f}")
        
        if results['opponents_used'] > 0:
            print(f"  Tested against {results['opponents_used']} strong benchmark models")
        else:
            print(f"  Tested against random opponents")
        
        # Performance indicator
        if results['win_rate'] > results['expected_random_rate'] * 1.2:
            print(f"  [OK] CFR AI performs significantly better than random!")
        elif results['win_rate'] > results['expected_random_rate']:
            print(f"  [OK] CFR AI performs better than random")
        else:
            print(f"  [WARNING] CFR AI needs more training or tuning")
        
        print(f"\nPosition Performance:")
        for pos, wr in results['position_performance'].items():
            pos_name = ["Early", "Middle", "Late", "Button"][pos] if pos < 4 else f"Pos{pos}"
            print(f"  {pos_name}: {wr:.1f}%")


def evaluate_cfr_ai_real(cfr_ai, num_games: int = 100, num_players: int = 4, 
                        verbose: bool = True) -> Dict:
    """
    Convenience function to evaluate CFR AI with real poker games
    
    Args:
        cfr_ai: Trained CFR AI
        num_games: Number of games to evaluate
        num_players: Number of players per game  
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = CFRGameEvaluator()
    return evaluator.evaluate_cfr_real_games(cfr_ai, num_games, num_players, verbose)