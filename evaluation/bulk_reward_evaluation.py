"""
Bulk evaluation of all reward-based AI models
Analyzes performance and move distributions
"""

import os
import glob
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import time

from reward_nn.reward_based_ai import RewardBasedAI
from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action
from dqn.poker_ai import PokerAI


class BulkRewardEvaluator:
    """Evaluate all reward-based AI models in bulk"""

    def __init__(self, models_dir: str = "models/reward_nn"):
        self.models_dir = models_dir
        self.results = {}

    def load_model(self, model_path: str) -> RewardBasedAI:
        """Load a reward-based AI model, handling compatibility issues"""
        model_name = os.path.basename(model_path)
        try:
            # First, detect the architecture from the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Detect hidden dimension
            hidden_dim = checkpoint.get('hidden_dim', None)
            if hidden_dim is None:
                # Try to infer from layer shapes
                state_dict = checkpoint.get('network_state', {})
                if 'input_projection.weight' in state_dict:
                    hidden_dim = state_dict['input_projection.weight'].shape[0]
                else:
                    hidden_dim = 256  # Default fallback

            # Create AI instance with correct architecture
            ai = RewardBasedAI(hidden_dim=hidden_dim)
            ai.load(model_path)
            ai.epsilon = 0  # No exploration during evaluation
            return ai
        except Exception as e:
            # Check for specific compatibility issues
            error_str = str(e)
            if "Missing key(s)" in error_str or "size mismatch" in error_str:
                print(f"  {model_name} is incompatible with the current version")
            else:
                print(f"  Warning: Could not load {model_name}: {str(e)[:50]}")
            return None

    def evaluate_model(self, model_path: str, num_games: int = 100, verbose: bool = False) -> Dict:
        """Evaluate a single model"""
        model_name = os.path.basename(model_path)

        # Try to load the model
        ai_model = self.load_model(model_path)
        if ai_model is None:
            return {
                'model_name': model_name,
                'status': 'incompatible',
                'error': 'Failed to load model (possibly older version)'
            }

        print(f"  Evaluating {model_name}...")

        # Track statistics
        stats = {
            'model_name': model_name,
            'status': 'success',
            'games_played': 0,
            'wins': 0,
            'total_chips_won': 0,
            'action_counts': defaultdict(int),
            'action_distribution': {},
            'avg_bb_per_hand': 0,
            'survival_rate': 0,
            'hands_played': 0,
            'preflop_actions': defaultdict(int),
            'postflop_actions': defaultdict(int),
        }

        # Load a benchmark opponent (DQN model if available)
        opponent_ai = None
        try:
            opponent_path = "models/dqn/poker_ai_tuned.pth"
            if os.path.exists(opponent_path):
                opponent_ai = PokerAI()
                opponent_ai.load(opponent_path)
                opponent_ai.epsilon = 0
                opponent_type = "DQN"
            else:
                opponent_type = "Random"
        except:
            opponent_type = "Random"

        # Play evaluation games
        starting_chips = 1000
        total_hands = 0
        survived_games = 0

        for game_idx in range(num_games):
            # Create game
            game = TexasHoldEm(num_players=2, starting_chips=starting_chips, verbose=False)

            # Setup players
            game.players = []
            test_player = Player("TestAI", starting_chips, is_ai=True)
            test_player.ai_model = ai_model
            opponent_player = Player("Opponent", starting_chips, is_ai=True)
            if opponent_ai:
                opponent_player.ai_model = opponent_ai

            game.players = [test_player, opponent_player]

            # Play the game (max hands to prevent infinite games)
            max_hands = 100
            hands_this_game = 0

            for hand_num in range(max_hands):
                # Check if game is over
                if test_player.chips <= 0 or opponent_player.chips <= 0:
                    break

                # Track if this is preflop
                is_preflop = True

                # Reset for new hand
                game.deck.reset()
                game.community_cards = []
                game.pot = 0
                game.current_bet = 0

                for player in game.players:
                    player.reset_hand()

                # Deal cards
                for player in game.players:
                    if player.chips > 0:
                        player.hand = game.deck.draw(2)

                # Post blinds
                game.pot += game.players[0].bet(min(10, game.players[0].chips))
                game.pot += game.players[1].bet(min(20, game.players[1].chips))
                game.current_bet = 20

                # Play betting rounds
                for street in ['preflop', 'flop', 'turn', 'river']:
                    if street == 'flop':
                        game.community_cards.extend(game.deck.draw(3))
                        is_preflop = False
                    elif street == 'turn':
                        game.community_cards.append(game.deck.draw())
                    elif street == 'river':
                        game.community_cards.append(game.deck.draw())

                    # Simple betting round
                    for _ in range(2):  # Max 2 actions per player per street
                        for player_idx, player in enumerate(game.players):
                            if player.folded or player.chips == 0:
                                continue

                            # Get valid actions
                            valid_actions = []
                            if player.current_bet < game.current_bet:
                                valid_actions.extend([Action.FOLD, Action.CALL])
                                if player.chips > game.current_bet - player.current_bet:
                                    valid_actions.append(Action.RAISE)
                            else:
                                valid_actions.extend([Action.CHECK])
                                if player.chips > 0:
                                    valid_actions.append(Action.RAISE)

                            if player.chips > 0:
                                valid_actions.append(Action.ALL_IN)

                            # Get action from AI
                            if player == test_player:
                                # Create state for the AI
                                state = ai_model.get_state_features(
                                    player.hand, game.community_cards, game.pot,
                                    game.current_bet, player.chips, player.current_bet,
                                    2, sum(1 for p in game.players if not p.folded),
                                    player_idx, [], None,
                                    hand_phase=0 if street == 'preflop' else (1 if street == 'flop' else (2 if street == 'turn' else 3))
                                )

                                action = ai_model.choose_action(state, valid_actions, training=False)

                                # Track action
                                stats['action_counts'][action] += 1
                                if is_preflop:
                                    stats['preflop_actions'][action] += 1
                                else:
                                    stats['postflop_actions'][action] += 1

                                # Execute action
                                if action == Action.FOLD:
                                    player.folded = True
                                elif action == Action.CALL:
                                    call_amount = min(game.current_bet - player.current_bet, player.chips)
                                    game.pot += player.bet(call_amount)
                                elif action == Action.RAISE:
                                    raise_amount = ai_model.get_raise_size(
                                        state, game.pot, game.current_bet,
                                        player.chips, player.current_bet, 20
                                    )
                                    total_bet = min(game.current_bet - player.current_bet + raise_amount, player.chips)
                                    game.pot += player.bet(total_bet)
                                    game.current_bet = player.current_bet
                                elif action == Action.ALL_IN:
                                    game.pot += player.bet(player.chips)
                                    if player.current_bet > game.current_bet:
                                        game.current_bet = player.current_bet
                            else:
                                # Opponent plays
                                if opponent_ai:
                                    # Use opponent AI
                                    state = opponent_ai.get_state_features(
                                        player.hand, game.community_cards, game.pot,
                                        game.current_bet, player.chips, player.current_bet,
                                        2, sum(1 for p in game.players if not p.folded),
                                        player_idx, [], None
                                    )
                                    action = opponent_ai.choose_action(state, valid_actions, training=False)
                                else:
                                    # Random opponent
                                    action = np.random.choice(valid_actions)

                                # Execute opponent action (simplified)
                                if action == Action.FOLD:
                                    player.folded = True
                                elif action == Action.CALL:
                                    call_amount = min(game.current_bet - player.current_bet, player.chips)
                                    game.pot += player.bet(call_amount)
                                elif action == Action.CHECK:
                                    pass
                                elif action == Action.RAISE:
                                    raise_amount = 40  # Fixed raise for opponent
                                    total_bet = min(game.current_bet - player.current_bet + raise_amount, player.chips)
                                    game.pot += player.bet(total_bet)
                                    game.current_bet = player.current_bet
                                elif action == Action.ALL_IN:
                                    game.pot += player.bet(player.chips)
                                    if player.current_bet > game.current_bet:
                                        game.current_bet = player.current_bet

                    # Check if hand is over
                    active = [p for p in game.players if not p.folded]
                    if len(active) <= 1:
                        break

                # Determine winner (simplified)
                active = [p for p in game.players if not p.folded]
                if len(active) == 1:
                    winner = active[0]
                else:
                    # Simple showdown - random winner for simplicity
                    winner = active[0] if np.random.random() < 0.5 else active[1]

                winner.chips += game.pot

                hands_this_game += 1
                total_hands += 1

            # Record game results
            stats['games_played'] += 1
            stats['hands_played'] += hands_this_game

            if test_player.chips > opponent_player.chips:
                stats['wins'] += 1

            if test_player.chips > 0:
                survived_games += 1

            stats['total_chips_won'] += (test_player.chips - starting_chips)

        # Calculate final statistics
        if stats['games_played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['games_played']
            stats['survival_rate'] = survived_games / stats['games_played']
            stats['avg_bb_per_hand'] = stats['total_chips_won'] / max(1, total_hands) / 20  # 20 = big blind

        # Calculate action distribution
        total_actions = sum(stats['action_counts'].values())
        if total_actions > 0:
            stats['action_distribution'] = {
                action.name: count / total_actions
                for action, count in stats['action_counts'].items()
            }

        # Calculate preflop/postflop distributions
        preflop_total = sum(stats['preflop_actions'].values())
        if preflop_total > 0:
            stats['preflop_distribution'] = {
                action.name: count / preflop_total
                for action, count in stats['preflop_actions'].items()
            }

        postflop_total = sum(stats['postflop_actions'].values())
        if postflop_total > 0:
            stats['postflop_distribution'] = {
                action.name: count / postflop_total
                for action, count in stats['postflop_actions'].items()
            }

        stats['opponent_type'] = opponent_type

        return stats

    def evaluate_all_models(self, num_games: int = 50):
        """Evaluate all models in the directory"""
        print(f"\n{'='*80}")
        print("BULK EVALUATION OF REWARD-BASED AI MODELS")
        print(f"{'='*80}")
        print(f"Models directory: {self.models_dir}")
        print(f"Games per model: {num_games}")
        print(f"{'='*80}\n")

        # Find all model files
        model_files = glob.glob(os.path.join(self.models_dir, "*.pth"))
        model_files.extend(glob.glob(os.path.join(self.models_dir, "*.pkl")))

        if not model_files:
            print(f"No models found in {self.models_dir}")
            return

        print(f"Found {len(model_files)} models to evaluate\n")

        # Evaluate each model
        all_results = []
        start_time = time.time()

        for i, model_path in enumerate(model_files, 1):
            print(f"[{i}/{len(model_files)}] Processing {os.path.basename(model_path)}...")
            result = self.evaluate_model(model_path, num_games)
            all_results.append(result)
            print()

        # Print summary report
        self.print_summary_report(all_results)

        elapsed = time.time() - start_time
        print(f"\nTotal evaluation time: {elapsed:.1f} seconds")

        return all_results

    def print_summary_report(self, results: List[Dict]):
        """Print a comprehensive summary report"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY REPORT")
        print(f"{'='*80}\n")

        # Separate compatible and incompatible models
        compatible = [r for r in results if r['status'] == 'success']
        incompatible = [r for r in results if r['status'] == 'incompatible']

        if incompatible:
            print("INCOMPATIBLE MODELS:")
            for r in incompatible:
                print(f"  - {r['model_name']}")
            print()

        if not compatible:
            print("No compatible models found.")
            return

        # Sort by win rate
        compatible.sort(key=lambda x: x.get('win_rate', 0), reverse=True)

        # Performance table
        print("PERFORMANCE METRICS:")
        print(f"{'Model':<30} {'Win Rate':>10} {'BB/Hand':>10} {'Survival':>10} {'Games':>8}")
        print("-" * 70)

        for r in compatible:
            print(f"{r['model_name'][:30]:<30} "
                  f"{r.get('win_rate', 0)*100:>9.1f}% "
                  f"{r.get('avg_bb_per_hand', 0):>10.2f} "
                  f"{r.get('survival_rate', 0)*100:>9.1f}% "
                  f"{r.get('games_played', 0):>8}")

        # Action distribution table
        print(f"\n{'='*80}")
        print("ACTION DISTRIBUTIONS (Overall):")
        print(f"{'Model':<30} {'FOLD':>10} {'CHECK':>10} {'CALL':>10} {'RAISE':>10} {'ALL_IN':>10}")
        print("-" * 80)

        for r in compatible:
            dist = r.get('action_distribution', {})
            print(f"{r['model_name'][:30]:<30} "
                  f"{dist.get('FOLD', 0)*100:>9.1f}% "
                  f"{dist.get('CHECK', 0)*100:>9.1f}% "
                  f"{dist.get('CALL', 0)*100:>9.1f}% "
                  f"{dist.get('RAISE', 0)*100:>9.1f}% "
                  f"{dist.get('ALL_IN', 0)*100:>9.1f}%")

        # Preflop vs Postflop comparison for top models
        print(f"\n{'='*80}")
        print("PREFLOP VS POSTFLOP BEHAVIOR (Top 5 Models):")
        print("-" * 80)

        for r in compatible[:5]:
            print(f"\n{r['model_name']}:")

            pre_dist = r.get('preflop_distribution', {})
            if pre_dist:
                print("  Preflop:  ", end="")
                for action in ['FOLD', 'CALL', 'RAISE', 'ALL_IN']:
                    if action in pre_dist:
                        print(f"{action}: {pre_dist[action]*100:.1f}%  ", end="")
                print()

            post_dist = r.get('postflop_distribution', {})
            if post_dist:
                print("  Postflop: ", end="")
                for action in ['FOLD', 'CHECK', 'CALL', 'RAISE', 'ALL_IN']:
                    if action in post_dist:
                        print(f"{action}: {post_dist[action]*100:.1f}%  ", end="")
                print()

        # Summary statistics
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS:")
        print(f"  Total models evaluated: {len(compatible)}")
        print(f"  Incompatible models: {len(incompatible)}")

        if compatible:
            avg_win_rate = np.mean([r.get('win_rate', 0) for r in compatible])
            avg_bb = np.mean([r.get('avg_bb_per_hand', 0) for r in compatible])
            avg_all_in = np.mean([r.get('action_distribution', {}).get('ALL_IN', 0) for r in compatible])

            print(f"  Average win rate: {avg_win_rate*100:.1f}%")
            print(f"  Average BB/hand: {avg_bb:.2f}")
            print(f"  Average all-in rate: {avg_all_in*100:.1f}%")

            # Find best and worst performers
            best_model = compatible[0]
            worst_model = compatible[-1]

            print(f"\n  Best performer: {best_model['model_name']} ({best_model.get('win_rate', 0)*100:.1f}% win rate)")
            print(f"  Worst performer: {worst_model['model_name']} ({worst_model.get('win_rate', 0)*100:.1f}% win rate)")


def main():
    """Run bulk evaluation"""
    evaluator = BulkRewardEvaluator()

    # Check if models directory exists
    if not os.path.exists(evaluator.models_dir):
        print(f"Creating models directory: {evaluator.models_dir}")
        os.makedirs(evaluator.models_dir, exist_ok=True)
        print("No models found to evaluate. Train some models first!")
        return

    # Run evaluation
    results = evaluator.evaluate_all_models(num_games=50)

    # Optionally save results to file
    save_choice = input("\nSave results to file? (y/n): ")
    if save_choice.lower() == 'y':
        import json
        filename = f"reward_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"

        # Convert defaultdicts to regular dicts for JSON serialization
        for r in results:
            if 'action_counts' in r:
                r['action_counts'] = {k.name if hasattr(k, 'name') else str(k): v
                                     for k, v in r['action_counts'].items()}
            if 'preflop_actions' in r:
                r['preflop_actions'] = {k.name if hasattr(k, 'name') else str(k): v
                                       for k, v in r['preflop_actions'].items()}
            if 'postflop_actions' in r:
                r['postflop_actions'] = {k.name if hasattr(k, 'name') else str(k): v
                                        for k, v in r['postflop_actions'].items()}

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()