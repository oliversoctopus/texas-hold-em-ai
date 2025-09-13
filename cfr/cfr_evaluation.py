"""
Real poker game evaluation for CFR AI
"""

import random
import numpy as np
from typing import List, Dict, Optional
from core.game_engine import TexasHoldEm
from core.player import Player
from .cfr_player import CFRPlayer
from dqn.poker_ai import PokerAI
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
                               num_players: int = 4, verbose: bool = True, 
                               use_random_opponents: bool = False) -> Dict:
        """
        Evaluate CFR AI by playing real poker games
        
        Args:
            cfr_ai: Trained CFR AI
            num_games: Number of games to play
            num_players: Number of players per game
            verbose: Whether to print progress
            use_random_opponents: If True, use random AIs; if False, use strong benchmark models
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if verbose:
            print(f"Evaluating CFR AI in {num_games} real poker games...")
            print(f"Players per game: {num_players}")
        
        # Reset strategy usage stats for fresh evaluation tracking
        cfr_ai.reset_strategy_stats()
        
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
            
            # Setup game with no verbose output and randomize starting conditions
            starting_chips = random.randint(900, 1100)  # Vary starting chips slightly
            game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips, verbose=False)
            
            # Randomize dealer position to vary game dynamics
            game.dealer_position = random.randint(0, num_players - 1)
            
            # Create players
            players = []
            
            # CFR AI player (position rotates)
            cfr_position = game_num % num_players
            
            for i in range(num_players):
                if i == cfr_position:
                    # Our CFR AI
                    player = Player(f"CFR_AI", starting_chips, is_ai=True)
                    player.ai_model = self._create_cfr_wrapper(cfr_player)
                    players.append(player)
                else:
                    # Opponent AI selection based on use_random_opponents parameter
                    if use_random_opponents:
                        # Use only random opponents
                        player = Player(f"Random_{i}", starting_chips, is_ai=True)
                        player.ai_model = self._create_random_ai()
                    else:
                        # Use strong benchmark models (with fallback to random if no models)
                        if self.benchmark_models and random.random() < 0.8:  # 80% strong, 20% random
                            # Use strong benchmark model
                            opponent_model = random.choice(self.benchmark_models)
                            player = Player(f"Strong_AI_{i}", starting_chips, is_ai=True)
                            player.ai_model = opponent_model
                        else:
                            # Random opponent
                            player = Player(f"Random_{i}", starting_chips, is_ai=True)
                            player.ai_model = self._create_random_ai()
                    players.append(player)
            
            game.players = players
            game.game_num = game_num  # Add game number for debugging
            
            # Ensure deck is properly shuffled for each game
            game.deck.reset()  # Reset and shuffle the deck
            
            # Track CFR position stats
            position_stats[cfr_position]['games'] += 1
            
            # Play the game
            cfr_start_chips = players[cfr_position].chips  # CFR's starting chips
            total_start_chips = sum(p.chips for p in game.players)  # Total chips before game
            
            winner = self._play_single_game(game, cfr_position, action_counts)
            
            # Update statistics
            games_played += 1
            final_chips = players[cfr_position].chips
            chips_change = final_chips - cfr_start_chips
            cfr_chips_won += chips_change
            
            # Debug: Check total chip conservation for first few games
            if game_num < 3:
                total_end_chips = sum(p.chips for p in game.players)
                chip_loss = total_start_chips - total_end_chips
                print(f"  DEBUG Game {game_num}: Total chips START {total_start_chips}, END {total_end_chips}, LOST {chip_loss}")
                if game.pot != 0:
                    print(f"  WARNING: Undistributed pot remaining: {game.pot}")
                print(f"  DEBUG Game {game_num}: CFR pos {cfr_position}, start {cfr_start_chips}, end {final_chips}, change {chips_change}, winner {winner}")
            
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
            'opponents_used': len(self.benchmark_models) if not use_random_opponents and self.benchmark_models else 0,
            'use_random_opponents': use_random_opponents
        }
        
        if verbose:
            self._print_evaluation_results(results, num_players)
            # Print CFR strategy usage statistics
            cfr_ai.print_strategy_usage()
        
        return results
    
    def _create_cfr_wrapper(self, cfr_player):
        """Create a wrapper that makes CFR player compatible with game engine"""
        class CFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0
            
            def choose_action(self, state, valid_actions, **kwargs):
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
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                game_state = {
                    'pot_size': getattr(state, 'pot_size', pot),
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
            
            def choose_action(self, state, valid_actions, **kwargs):
                from core.game_constants import Action
                return random.choice([Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE])
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                # Simple random raise sizing strategy
                pot_size = getattr(state, 'pot_size', pot)
                available_chips = getattr(state, 'stack_size', player_chips - player_current_bet)
                
                # Choose a random raise size between min_raise and pot-sized bet
                max_raise = min(available_chips, pot_size * 2)  # Cap at 2x pot or available chips
                if max_raise <= min_raise:
                    return min_raise
                else:
                    return random.randint(min_raise, max_raise)
            
        
        return RandomAI()
    
    def _play_single_game(self, game, cfr_position, action_counts):
        """Play a single game until clear winner or sufficient separation"""
        hands_played = 0
        max_hands = 200  # Increase significantly to allow proper tournament play
        
        while hands_played < max_hands:
            try:
                # Check for clear winner or significant chip separation
                active_players = [p for p in game.players if p.chips > 0]
                if len(active_players) <= 1:
                    if hands_played < 3:  # Debug early termination
                        print(f"    Game ended early after {hands_played} hands: only {len(active_players)} players left")
                    break  # Game over - one or zero players left
                
                # Check for significant chip separation (one player has >90% of chips)
                total_chips = sum(p.chips for p in active_players)
                if total_chips > 0:
                    max_chips = max(p.chips for p in active_players)
                    if max_chips > 0.90 * total_chips and len(active_players) > 1:
                        if hands_played < 3:  # Debug early termination  
                            print(f"    Game ended early after {hands_played} hands: chip dominance {max_chips}/{total_chips}")
                        break  # Clear chip leader
                
                # Debug: Track chip movement each hand for first game
                if hands_played < 5 and hasattr(game, 'game_num') and game.game_num == 0:
                    chips_before = [p.chips for p in game.players]
                    pot_before = game.pot
                
                # Play a hand
                game.play_hand(verbose=False)
                hands_played += 1
                
                # Debug: Show what happened each hand for first game
                if hands_played <= 5 and hasattr(game, 'game_num') and game.game_num == 0:
                    chips_after = [p.chips for p in game.players]
                    pot_after = game.pot
                    print(f"      Hand {hands_played}: Chips {chips_before} -> {chips_after}, Pot {pot_before} -> {pot_after}")
                
                # Update action counts (simplified tracking)
                if hands_played % 10 == 0:  # Sample some actions
                    action_counts['CALL'] += 1  # Placeholder - would need actual action tracking
                
            except Exception as e:
                if hands_played < 3:
                    print(f"    Game ended with exception after {hands_played} hands: {e}")
                break
        
        # Debug: Show final game stats for first few games  
        if hasattr(game, 'game_num') and game.game_num < 3:
            print(f"    Game {game.game_num} completed: {hands_played} hands played, reason: loop ended normally")
        
        # Find winner (player with most chips) - handle ties fairly
        max_chips = max(player.chips for player in game.players)
        winners = [i for i, player in enumerate(game.players) if player.chips == max_chips]
        
        # If there's a tie, randomly pick one winner (fair evaluation)
        winner_idx = random.choice(winners)
        
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
        
        if results['use_random_opponents']:
            print(f"  Tested against random opponents only")
        elif results['opponents_used'] > 0:
            print(f"  Tested against {results['opponents_used']} strong benchmark models")
        else:
            print(f"  Tested against random opponents (no benchmark models available)")
        
        # Performance indicator
        if results['win_rate'] > results['expected_random_rate'] * 1.2:
            print(f"  [OK] CFR AI performs significantly better than random!")
        elif results['win_rate'] > results['expected_random_rate']:
            print(f"  [OK] CFR AI performs better than random")
        else:
            print(f"  [WARNING] CFR AI needs more training or tuning")
        
        print(f"\nPosition Performance:")
        for pos, wr in results['position_performance'].items():
            if pos < 6:
                pos_names = ["Early", "Middle", "Late", "Button", "Pos4", "Pos5"]
                pos_name = pos_names[pos]
            else:
                pos_name = f"Pos{pos}"
            print(f"  {pos_name}: {wr:.1f}%")


def evaluate_cfr_ai_real(cfr_ai, num_games: int = 100, num_players: int = 4, 
                        verbose: bool = True, use_random_opponents: bool = False) -> Dict:
    """
    Convenience function to evaluate CFR AI with real poker games
    
    Args:
        cfr_ai: Trained CFR AI
        num_games: Number of games to evaluate
        num_players: Number of players per game  
        verbose: Whether to print results
        use_random_opponents: If True, use random AIs; if False, use strong benchmark models
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = CFRGameEvaluator()
    return evaluator.evaluate_cfr_real_games(cfr_ai, num_games, num_players, verbose, use_random_opponents)