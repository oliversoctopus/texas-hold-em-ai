"""
Specialized evaluation for 2-player CFR
"""

import random
import numpy as np
from typing import List, Dict, Optional
from core.game_engine import TexasHoldEm
from core.player import Player
# Removed old import - evaluation now works with any CFR model with compatible interface


class TwoPlayerCFRPlayer:
    """
    Specialized player wrapper for 2-player CFR
    """
    
    def __init__(self, cfr_ai):
        self.cfr_ai = cfr_ai
        self.name = "TwoPlayer_CFR_AI"
        
    def choose_action(self, game_state: dict):
        """Choose action using 2-player CFR strategy"""
        # Extract necessary information
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        pot_size = game_state.get('pot_size', 0)
        to_call = game_state.get('to_call', 0)
        stack_size = game_state.get('stack_size', 1000)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 2)  # Force 2 players
        
        # Ensure we're in a 2-player game
        if num_players != 2:
            raise ValueError(f"TwoPlayerCFRPlayer only supports 2 players, got {num_players}")
        
        # Create simplified betting history for 2-player CFR
        betting_history = self._create_betting_history(game_state.get('action_history', []))
        
        # Get action from 2-player CFR AI
        return self.cfr_ai.get_action(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            pot_size=pot_size,
            to_call=to_call,
            stack_size=stack_size,
            position=position,
            num_players=2  # Always 2 for this implementation
        )
    
    def _create_betting_history(self, action_history: List) -> str:
        """Convert action history to simplified betting history string for 2-player"""
        history = ""
        for action in action_history:
            if hasattr(action, 'action'):
                from core.game_constants import Action
                if action.action == Action.FOLD:
                    history += "F"
                elif action.action == Action.CHECK:
                    history += "K"
                elif action.action == Action.CALL:
                    history += "C"
                elif action.action == Action.RAISE:
                    history += "R"
                elif action.action == Action.ALL_IN:
                    history += "A"
        return history
    
    def get_raise_size(self, game_state: dict) -> int:
        """Get raise size for 2-player CFR"""
        pot_size = game_state.get('pot_size', 0)
        position = game_state.get('position', 0)
        
        # 2-player specific sizing
        if position == 0:  # Button - more aggressive
            return max(40, pot_size // 2)
        else:  # Big blind - more defensive
            return max(30, pot_size // 3)


class TwoPlayerCFREvaluator:
    """
    Specialized evaluator for 2-player CFR
    """
    
    def __init__(self):
        pass
    
    def evaluate_two_player_cfr(self, cfr_ai,
                               num_games: int = 100, verbose: bool = True,
                               use_strong_opponents: bool = False) -> Dict:
        """
        Evaluate 2-player CFR against random or DQN opponents

        Args:
            cfr_ai: The CFR AI to evaluate
            num_games: Number of games to play
            verbose: Whether to print progress
            use_strong_opponents: If True, use DQN benchmark models; if False, use random
        """
        opponent_type = "DQN benchmark models" if use_strong_opponents else "random opponents"
        if verbose:
            print(f"Evaluating 2-Player CFR against {opponent_type} in {num_games} heads-up poker games...")

        # Load DQN benchmark models if requested
        strong_opponents = []
        if use_strong_opponents:
            import os
            from dqn.poker_ai import PokerAI

            # Use only the good benchmark models
            strong_model_paths = [
                'models/dqn/tuned_ai_v2.pth',      # Complex model with good strategy
                'models/dqn/tuned_ai_v4.pth',      # Another strong version
                'models/dqn/poker_ai_tuned.pth',   # Earlier strong model
            ]

            for path in strong_model_paths:
                if os.path.exists(path):
                    try:
                        opponent = PokerAI()
                        opponent.load(path)
                        opponent.epsilon = 0  # No exploration during evaluation
                        strong_opponents.append(opponent)
                        if verbose:
                            print(f"  Loaded DQN opponent: {path}")
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to load {path}: {e}")

            if not strong_opponents:
                if verbose:
                    print("  No DQN opponents loaded, falling back to random opponents")
                use_strong_opponents = False
            elif verbose:
                print(f"  Successfully loaded {len(strong_opponents)} DQN opponents")

        # Reset strategy usage stats
        cfr_ai.reset_strategy_stats()
        
        # Create 2-player CFR wrapper
        cfr_player = TwoPlayerCFRPlayer(cfr_ai)
        
        # Statistics tracking
        cfr_wins = 0
        cfr_chips_won = 0
        games_played = 0
        position_stats = {0: {'games': 0, 'wins': 0}, 1: {'games': 0, 'wins': 0}}
        
        for game_num in range(num_games):
            if verbose and game_num % 20 == 0:
                print(f"  Progress: {game_num}/{num_games}")
            
            # Setup 2-player game
            starting_chips = random.randint(950, 1050)
            game = TexasHoldEm(num_players=2, starting_chips=starting_chips, verbose=False)
            
            # Randomize dealer position
            game.dealer_position = random.randint(0, 1)
            
            # Create players - CFR vs Random/DQN
            cfr_position = game_num % 2  # Alternate positions

            players = []
            for i in range(2):
                if i == cfr_position:
                    # 2-player CFR AI
                    player = Player(f"TwoPlayerCFR", starting_chips, is_ai=True)
                    player.ai_model = self._create_cfr_wrapper(cfr_player)
                    players.append(player)
                else:
                    # Opponent - either DQN or random
                    if use_strong_opponents and strong_opponents:
                        # Use rotating DQN opponents
                        dqn_ai = strong_opponents[game_num % len(strong_opponents)]
                        player = Player(f"DQN_Opponent", starting_chips, is_ai=True)
                        player.ai_model = dqn_ai
                        players.append(player)
                    else:
                        # Random opponent
                        player = Player(f"Random", starting_chips, is_ai=True)
                        player.ai_model = self._create_random_ai()
                        players.append(player)
            
            game.players = players
            
            # Track position stats
            position_stats[cfr_position]['games'] += 1
            
            # Play the game
            cfr_start_chips = players[cfr_position].chips
            
            winner = self._play_single_game(game, cfr_position)
            
            # Update statistics
            games_played += 1
            final_chips = players[cfr_position].chips
            chips_change = final_chips - cfr_start_chips
            cfr_chips_won += chips_change
            
            if winner == cfr_position:
                cfr_wins += 1
                position_stats[cfr_position]['wins'] += 1
        
        # Calculate final statistics
        win_rate = (cfr_wins / games_played) * 100 if games_played > 0 else 0
        avg_chips_per_game = cfr_chips_won / games_played if games_played > 0 else 0
        expected_random_rate = 50.0  # 50% for 2-player
        
        # Position performance
        position_performance = {}
        for pos in range(2):
            if position_stats[pos]['games'] > 0:
                pos_wr = (position_stats[pos]['wins'] / position_stats[pos]['games']) * 100
                position_performance[pos] = pos_wr
        
        results = {
            'win_rate': win_rate,
            'games_played': games_played,
            'avg_chips_per_game': avg_chips_per_game,
            'expected_random_rate': expected_random_rate,
            'performance_vs_random': win_rate / expected_random_rate if expected_random_rate > 0 else 0,
            'position_performance': position_performance,
            'strategy_usage': cfr_ai.get_strategy_usage_stats()
        }
        
        if verbose:
            self._print_evaluation_results(results)
            cfr_ai.print_strategy_usage()
        
        return results
    
    def _create_cfr_wrapper(self, cfr_player):
        """Create wrapper compatible with game engine"""
        class TwoPlayerCFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0
            
            def choose_action(self, state, valid_actions, **kwargs):
                game_state = {
                    'hole_cards': getattr(state, 'hole_cards', []),
                    'community_cards': getattr(state, 'community_cards', []),
                    'pot_size': getattr(state, 'pot_size', 0),
                    'to_call': getattr(state, 'to_call', 0),
                    'stack_size': getattr(state, 'stack_size', 1000),
                    'position': getattr(state, 'position', 0),
                    'num_players': 2,  # Force 2 players
                    'action_history': getattr(state, 'action_history', [])
                }
                return self.cfr_player.choose_action(game_state)
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                game_state = {
                    'pot_size': getattr(state, 'pot_size', pot),
                    'position': getattr(state, 'position', 0),
                    'num_players': 2
                }
                return self.cfr_player.get_raise_size(game_state)
        
        return TwoPlayerCFRWrapper(cfr_player)
    
    def _create_random_ai(self):
        """Create random AI for 2-player"""
        class RandomAI:
            def __init__(self):
                self.epsilon = 0
            
            def choose_action(self, state, valid_actions, **kwargs):
                from core.game_constants import Action
                return random.choice([Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE])
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                pot_size = getattr(state, 'pot_size', pot)
                return random.randint(min_raise, max(min_raise, pot_size))
        
        return RandomAI()
    
    def _play_single_game(self, game, cfr_position):
        """Play a single 2-player game"""
        hands_played = 0
        max_hands = 100  # Reasonable limit for heads-up
        
        while hands_played < max_hands:
            try:
                # Check for winner
                active_players = [p for p in game.players if p.chips > 0]
                if len(active_players) <= 1:
                    break
                
                # Check for chip dominance (95% of chips)
                total_chips = sum(p.chips for p in active_players)
                if total_chips > 0:
                    max_chips = max(p.chips for p in active_players)
                    if max_chips > 0.95 * total_chips:
                        break
                
                # Play a hand
                game.play_hand(verbose=False)
                hands_played += 1
                
            except Exception as e:
                break
        
        # Find winner (player with most chips)
        max_chips = max(player.chips for player in game.players)
        winners = [i for i, player in enumerate(game.players) if player.chips == max_chips]
        
        # Return winner (random choice if tie)
        return random.choice(winners)
    
    def _print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print(f"\n" + "="*60)
        print("2-PLAYER CFR EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Games played: {results['games_played']}")
        print(f"  Win rate: {results['win_rate']:.1f}%")
        print(f"  Expected random rate: {results['expected_random_rate']:.1f}%")
        print(f"  Performance vs random: {results['performance_vs_random']:.1f}x")
        print(f"  Average chips per game: ${results['avg_chips_per_game']:.0f}")
        
        # Performance indicator
        if results['win_rate'] > results['expected_random_rate'] * 1.2:
            print(f"  [EXCELLENT] 2-Player CFR significantly outperforms random!")
        elif results['win_rate'] > results['expected_random_rate']:
            print(f"  [GOOD] 2-Player CFR beats random opponents")
        else:
            print(f"  [NEEDS WORK] 2-Player CFR needs more training")
        
        print(f"\nPosition Performance:")
        pos_names = ["Button", "Big Blind"]
        for pos, wr in results['position_performance'].items():
            print(f"  {pos_names[pos]}: {wr:.1f}%")


def evaluate_two_player_cfr_ai(cfr_ai,
                              num_games: int = 100, verbose: bool = True,
                              use_strong_opponents: bool = False) -> Dict:
    """
    Convenience function to evaluate 2-player CFR

    Args:
        cfr_ai: The CFR AI to evaluate
        num_games: Number of games to play
        verbose: Whether to print progress
        use_strong_opponents: If True, use DQN benchmark models; if False, use random
    """
    evaluator = TwoPlayerCFREvaluator()
    return evaluator.evaluate_two_player_cfr(cfr_ai, num_games, verbose, use_strong_opponents)