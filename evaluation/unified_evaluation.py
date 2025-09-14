"""
Unified evaluation module for all AI models (DQN, CFR, Deep CFR)
Consolidates evaluation logic with consistent move distribution tracking
"""

import os
import random
import numpy as np
from typing import List, Dict, Optional, Any
from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action
from dqn.poker_ai import PokerAI


class UnifiedEvaluator:
    """
    Unified evaluator for all poker AI models with consistent metrics
    """

    def __init__(self):
        self.benchmark_models = []
        self.benchmark_names = []
        self._load_benchmark_models()

    def _load_benchmark_models(self):
        """Load existing strong DQN models as opponents"""
        # Check both direct paths and models/dqn directory
        potential_paths = [
            'models/dqn/tuned_ai_v2.pth',
            'models/dqn/tuned_ai_v4.pth',
            'models/dqn/poker_ai_tuned.pth',
            'tuned_ai_v2.pth',
            'tuned_ai_v4.pth',
            'poker_ai_tuned.pth',
            'standard_ai_v3.pth',
            'balanced_ai_v3.pth',
            'best_final_2.pth',
            'opponent_eval_v2.pth'
        ]

        for path in potential_paths:
            if os.path.exists(path):
                try:
                    model = PokerAI()
                    model.load(path)
                    model.epsilon = 0  # No exploration during evaluation
                    self.benchmark_models.append(model)
                    self.benchmark_names.append(os.path.basename(path).replace('.pth', ''))
                    print(f"  Loaded benchmark opponent: {os.path.basename(path)}")
                except Exception as e:
                    pass

        if not self.benchmark_models:
            print("  No benchmark models found - will use random opponents")

    def evaluate_dqn_model(self, ai_model: PokerAI, num_games: int = 100,
                          num_players: int = 4, use_strong_opponents: bool = True,
                          verbose: bool = True) -> Dict:
        """
        Evaluate DQN model with detailed action distribution tracking
        """
        if verbose:
            print(f"Evaluating DQN model over {num_games} games...")
            print(f"  Players per game: {num_players}")
            print(f"  Opponents: {'Strong AI' if use_strong_opponents else 'Random'}")

        original_epsilon = ai_model.epsilon
        ai_model.epsilon = 0  # No exploration during evaluation

        # Statistics tracking
        wins = 0
        total_earnings = 0
        games_survived = 0
        action_distribution = {'FOLD': 0, 'CHECK': 0, 'CALL': 0, 'RAISE': 0, 'ALL_IN': 0}
        total_actions = 0
        position_stats = {i: {'games': 0, 'wins': 0} for i in range(num_players)}

        for game_num in range(num_games):
            if verbose and (game_num + 1) % 20 == 0:
                print(f"  Progress: {game_num + 1}/{num_games}")

            # Randomize test position
            test_position = game_num % num_players
            position_stats[test_position]['games'] += 1

            # Setup game
            starting_chips = 1000
            game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips, verbose=False)

            # Create players
            players = []
            for i in range(num_players):
                if i == test_position:
                    # Our test DQN model
                    player = Player("DQN_Test", starting_chips, is_ai=True)
                    player.ai_model = ai_model
                    players.append(player)
                else:
                    # Opponent
                    if use_strong_opponents and self.benchmark_models:
                        if random.random() < 0.8:  # 80% strong, 20% random for variety
                            opponent_model = random.choice(self.benchmark_models)
                            player = Player(f"Strong_AI_{i}", starting_chips, is_ai=True)
                            player.ai_model = opponent_model
                        else:
                            player = Player(f"Random_{i}", starting_chips, is_ai=True)
                            player.ai_model = self._create_random_ai()
                    else:
                        player = Player(f"Random_{i}", starting_chips, is_ai=True)
                        player.ai_model = self._create_random_ai()
                    players.append(player)

            game.players = players

            # Play game and track actions
            test_start_chips = players[test_position].chips
            winner_idx, game_actions = self._play_single_game_with_tracking(game, test_position)

            # Update action distribution from tracked actions
            for action in game_actions:
                if action in action_distribution:
                    action_distribution[action] += 1
                    total_actions += 1

            # Update statistics
            final_chips = players[test_position].chips
            chips_change = final_chips - test_start_chips
            total_earnings += chips_change

            if final_chips > 0:
                games_survived += 1

            if winner_idx == test_position:
                wins += 1
                position_stats[test_position]['wins'] += 1

        ai_model.epsilon = original_epsilon

        # Calculate statistics
        win_rate = (wins / num_games) * 100 if num_games > 0 else 0
        survival_rate = (games_survived / num_games) * 100 if num_games > 0 else 0
        avg_earnings = total_earnings / num_games if num_games > 0 else 0
        expected_random_rate = (1.0 / num_players) * 100

        # Calculate action distribution percentages
        action_distribution_pct = {}
        if total_actions > 0:
            for action, count in action_distribution.items():
                action_distribution_pct[action] = (count / total_actions) * 100

        results = {
            'model_type': 'DQN',
            'win_rate': win_rate,
            'survival_rate': survival_rate,
            'games_played': num_games,
            'avg_earnings': avg_earnings,
            'expected_random_rate': expected_random_rate,
            'performance_vs_random': win_rate / expected_random_rate if expected_random_rate > 0 else 0,
            'action_distribution': action_distribution_pct,
            'total_actions': total_actions,
            'position_performance': {pos: (stats['wins']/stats['games']*100 if stats['games'] > 0 else 0)
                                    for pos, stats in position_stats.items()},
            'opponents_used': 'Strong AI' if use_strong_opponents and self.benchmark_models else 'Random'
        }

        if verbose:
            self._print_evaluation_results(results, num_players)

        return results

    def evaluate_cfr_model(self, cfr_ai: Any, num_games: int = 100,
                          num_players: int = 4, use_random_opponents: bool = False,
                          verbose: bool = True) -> Dict:
        """
        Evaluate CFR model with detailed action distribution tracking
        """
        if verbose:
            print(f"Evaluating CFR model over {num_games} games...")
            print(f"  Players per game: {num_players}")
            print(f"  Opponents: {'Random' if use_random_opponents else 'Strong AI'}")

        # Import CFR player wrapper
        from cfr.cfr_player import CFRPlayer
        cfr_player = CFRPlayer(cfr_ai)

        # Reset strategy stats if available
        if hasattr(cfr_ai, 'reset_strategy_stats'):
            cfr_ai.reset_strategy_stats()

        # Statistics tracking
        wins = 0
        total_earnings = 0
        games_survived = 0
        action_distribution = {'FOLD': 0, 'CHECK': 0, 'CALL': 0, 'RAISE': 0, 'ALL_IN': 0}
        total_actions = 0
        position_stats = {i: {'games': 0, 'wins': 0} for i in range(num_players)}

        for game_num in range(num_games):
            if verbose and (game_num + 1) % 20 == 0:
                print(f"  Progress: {game_num + 1}/{num_games}")

            # Randomize test position
            test_position = game_num % num_players
            position_stats[test_position]['games'] += 1

            # Setup game with slight randomization
            starting_chips = random.randint(900, 1100)
            game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips, verbose=False)
            game.dealer_position = random.randint(0, num_players - 1)

            # Create players
            players = []
            for i in range(num_players):
                if i == test_position:
                    # Our test CFR model
                    player = Player("CFR_Test", starting_chips, is_ai=True)
                    player.ai_model = self._create_cfr_wrapper(cfr_player)
                    players.append(player)
                else:
                    # Opponent
                    if use_random_opponents:
                        player = Player(f"Random_{i}", starting_chips, is_ai=True)
                        player.ai_model = self._create_random_ai()
                    else:
                        # Use strong opponents when available
                        if self.benchmark_models and random.random() < 0.8:
                            opponent_model = random.choice(self.benchmark_models)
                            player = Player(f"Strong_AI_{i}", starting_chips, is_ai=True)
                            player.ai_model = opponent_model
                        else:
                            player = Player(f"Random_{i}", starting_chips, is_ai=True)
                            player.ai_model = self._create_random_ai()
                    players.append(player)

            game.players = players

            # Play game and track actions
            test_start_chips = players[test_position].chips
            winner_idx, game_actions = self._play_single_game_with_tracking(game, test_position)

            # Update action distribution
            for action in game_actions:
                if action in action_distribution:
                    action_distribution[action] += 1
                    total_actions += 1

            # Update statistics
            final_chips = players[test_position].chips
            chips_change = final_chips - test_start_chips
            total_earnings += chips_change

            if final_chips > 0:
                games_survived += 1

            if winner_idx == test_position:
                wins += 1
                position_stats[test_position]['wins'] += 1

        # Calculate statistics
        win_rate = (wins / num_games) * 100 if num_games > 0 else 0
        survival_rate = (games_survived / num_games) * 100 if num_games > 0 else 0
        avg_earnings = total_earnings / num_games if num_games > 0 else 0
        expected_random_rate = (1.0 / num_players) * 100

        # Calculate action distribution percentages
        action_distribution_pct = {}
        if total_actions > 0:
            for action, count in action_distribution.items():
                action_distribution_pct[action] = (count / total_actions) * 100

        results = {
            'model_type': 'CFR',
            'win_rate': win_rate,
            'survival_rate': survival_rate,
            'games_played': num_games,
            'avg_earnings': avg_earnings,
            'expected_random_rate': expected_random_rate,
            'performance_vs_random': win_rate / expected_random_rate if expected_random_rate > 0 else 0,
            'action_distribution': action_distribution_pct,
            'total_actions': total_actions,
            'position_performance': {pos: (stats['wins']/stats['games']*100 if stats['games'] > 0 else 0)
                                    for pos, stats in position_stats.items()},
            'opponents_used': 'Random' if use_random_opponents else 'Strong AI/Mixed'
        }

        if verbose:
            self._print_evaluation_results(results, num_players)
            # Print CFR-specific strategy stats if available
            if hasattr(cfr_ai, 'print_strategy_usage'):
                cfr_ai.print_strategy_usage()

        return results

    def _play_single_game_with_tracking(self, game: TexasHoldEm, test_position: int):
        """
        Play a single game and track actions taken by the test player

        Returns:
            tuple: (winner_index, list_of_test_player_actions)
        """
        hands_played = 0
        max_hands = 200
        test_actions = []

        while hands_played < max_hands:
            try:
                # Check for game end conditions
                active_players = [p for p in game.players if p.chips > 0]
                if len(active_players) <= 1:
                    break

                # Check for chip dominance
                total_chips = sum(p.chips for p in active_players)
                if total_chips > 0:
                    max_chips = max(p.chips for p in active_players)
                    if max_chips > 0.90 * total_chips and len(active_players) > 1:
                        break

                # Track action history before hand
                actions_before = len(game.action_history) if hasattr(game, 'action_history') else 0

                # Play a hand
                game.play_hand(verbose=False)
                hands_played += 1

                # Extract actions taken by test player this hand
                if hasattr(game, 'action_history') and len(game.action_history) > actions_before:
                    # Analyze new actions
                    new_actions = game.action_history[actions_before:]

                    # Track which player took each action based on position
                    current_player_idx = 0
                    for action in new_actions:
                        # Simple heuristic: actions cycle through active players
                        # This is approximate but good enough for statistics
                        active_count = len([p for p in game.players if p.chips > 0 and not p.folded])
                        if active_count > 0:
                            if current_player_idx % active_count == test_position % active_count:
                                # This action was likely from our test player
                                action_name = action.name.upper()
                                if action_name in ['FOLD', 'CHECK', 'CALL', 'RAISE', 'ALL_IN']:
                                    test_actions.append(action_name)
                            current_player_idx += 1

            except Exception as e:
                # Game ended with error
                break

        # Find winner
        max_chips = max(player.chips for player in game.players)
        winners = [i for i, player in enumerate(game.players) if player.chips == max_chips]
        winner_idx = random.choice(winners) if winners else -1

        return winner_idx, test_actions

    def _create_cfr_wrapper(self, cfr_player):
        """Create a wrapper that makes CFR player compatible with game engine"""
        class CFRWrapper:
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
                    'num_players': getattr(state, 'num_players', 4),
                    'action_history': getattr(state, 'action_history', [])
                }
                return self.cfr_player.choose_action(game_state)

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000,
                             player_current_bet=0, min_raise=20):
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
                return random.choice([Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE])

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000,
                             player_current_bet=0, min_raise=20):
                pot_size = getattr(state, 'pot_size', pot)
                available_chips = getattr(state, 'stack_size', player_chips - player_current_bet)

                max_raise = min(available_chips, pot_size * 2)
                if max_raise <= min_raise:
                    return min_raise
                else:
                    return random.randint(min_raise, max_raise)

        return RandomAI()

    def _print_evaluation_results(self, results: Dict, num_players: int):
        """Print formatted evaluation results"""
        print(f"\n" + "="*60)
        print(f"{results['model_type']} MODEL EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Performance:")
        print(f"  Games played: {results['games_played']}")
        print(f"  Win rate: {results['win_rate']:.1f}%")
        print(f"  Survival rate: {results.get('survival_rate', 0):.1f}%")
        print(f"  Expected random rate: {results['expected_random_rate']:.1f}%")
        print(f"  Performance vs random: {results['performance_vs_random']:.1f}x")
        print(f"  Average earnings: ${results['avg_earnings']:+.0f}")
        print(f"  Opponents: {results['opponents_used']}")

        # Performance indicator
        if results['win_rate'] > results['expected_random_rate'] * 1.5:
            print(f"  ✓ Excellent performance! {results['model_type']} significantly outperforms baseline")
        elif results['win_rate'] > results['expected_random_rate'] * 1.2:
            print(f"  ✓ Good performance! {results['model_type']} performs well above baseline")
        elif results['win_rate'] > results['expected_random_rate']:
            print(f"  ✓ {results['model_type']} performs better than random baseline")
        else:
            print(f"  ⚠ {results['model_type']} needs more training or tuning")

        # Action distribution
        print(f"\nAction Distribution ({results['total_actions']} total actions):")
        if results['action_distribution']:
            for action in ['FOLD', 'CHECK', 'CALL', 'RAISE', 'ALL_IN']:
                pct = results['action_distribution'].get(action, 0)
                bar_length = int(pct / 2)  # Scale to max 50 chars
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"  {action:8s}: {bar} {pct:5.1f}%")

            # Strategy analysis
            print(f"\nStrategy Analysis:")
            fold_pct = results['action_distribution'].get('FOLD', 0)
            raise_pct = results['action_distribution'].get('RAISE', 0)
            all_in_pct = results['action_distribution'].get('ALL_IN', 0)
            call_pct = results['action_distribution'].get('CALL', 0)
            check_pct = results['action_distribution'].get('CHECK', 0)

            if all_in_pct > 40:
                print("  ⚠ Hyper-aggressive: Excessive all-in behavior detected")
            elif raise_pct + all_in_pct > 50:
                print("  • Aggressive: High raising frequency")
            elif fold_pct > 60:
                print("  • Conservative: High folding frequency")
            elif call_pct + check_pct > 60:
                print("  • Passive: Mostly calling/checking")
            else:
                print("  • Balanced: Mixed strategy observed")

        # Position performance
        if results.get('position_performance'):
            print(f"\nPosition Performance:")
            pos_names = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
            for pos, win_rate in results['position_performance'].items():
                if pos < len(pos_names):
                    pos_name = pos_names[pos]
                else:
                    pos_name = f"Pos{pos}"
                print(f"  {pos_name}: {win_rate:.1f}%")


# Convenience functions for backward compatibility
def evaluate_dqn_full(ai_model: PokerAI, num_games: int = 100,
                      num_players: int = 4, use_strong_opponents: bool = True,
                      verbose: bool = True) -> Dict:
    """
    Evaluate DQN model (backward compatible function)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_dqn_model(ai_model, num_games, num_players,
                                        use_strong_opponents, verbose)


def evaluate_cfr_full(cfr_ai: Any, num_games: int = 100,
                      num_players: int = 4, use_random_opponents: bool = False,
                      verbose: bool = True) -> Dict:
    """
    Evaluate CFR model (backward compatible function)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_cfr_model(cfr_ai, num_games, num_players,
                                        use_random_opponents, verbose)


def evaluate_deep_cfr_full(deep_cfr_ai: Any, num_games: int = 100,
                          num_players: int = 4, use_random_opponents: bool = False,
                          verbose: bool = True) -> Dict:
    """
    Evaluate Deep CFR model
    """
    evaluator = UnifiedEvaluator()
    # Deep CFR uses the same evaluation as regular CFR
    return evaluator.evaluate_cfr_model(deep_cfr_ai, num_games, num_players,
                                        use_random_opponents, verbose)