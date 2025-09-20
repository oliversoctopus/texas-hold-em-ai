"""
Bulk evaluation of all reward-based AI models
Analyzes performance and move distributions
"""

import os
import glob
import torch
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import time

from reward_nn.reward_based_ai import RewardBasedAI
from evaluation.unified_evaluation import evaluate_reward_based_ai


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
        """Evaluate a single model using unified evaluation logic"""
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

        # Use unified evaluation for consistency
        # Test vs random opponents
        random_results = evaluate_reward_based_ai(
            ai_model,
            num_games=num_games // 2,  # Split games between random and DQN
            num_players=2,
            use_random_opponents=True,
            verbose=False
        )

        # Test vs DQN opponents
        dqn_results = evaluate_reward_based_ai(
            ai_model,
            num_games=num_games // 2,
            num_players=2,
            use_random_opponents=False,
            verbose=False
        )

        # Combine results from both opponent types
        total_games = num_games
        combined_wins = (random_results['win_rate'] * (num_games // 2) +
                        dqn_results['win_rate'] * (num_games // 2)) / total_games

        # Prepare stats dictionary matching expected format
        stats = {
            'model_name': model_name,
            'status': 'success',
            'games_played': total_games,
            'win_rate': combined_wins,
            'win_rate_vs_random': random_results['win_rate'],
            'win_rate_vs_dqn': dqn_results['win_rate'],
            'bb_per_hand': (random_results.get('bb_per_hand', 0) + dqn_results.get('bb_per_hand', 0)) / 2,
            'action_distribution': dqn_results.get('action_distribution', {}),
            'survival_rate': (random_results.get('survival_rate', 0) + dqn_results.get('survival_rate', 0)) / 2,
        }

        return stats

    def evaluate_all_models(self, num_games: int = 50):
        """Evaluate all models in the directory"""
        print(f"\n{'='*80}")
        print("BULK EVALUATION OF REWARD-BASED AI MODELS")
        print(f"{'='*80}\n")

        # Find all model files
        model_files = glob.glob(os.path.join(self.models_dir, "*.pth"))
        model_files.extend(glob.glob(os.path.join(self.models_dir, "*.pkl")))

        if not model_files:
            print(f"No models found in {self.models_dir}")
            return []

        print(f"Found {len(model_files)} models to evaluate")
        print(f"Each model will play {num_games} games total:")
        print(f"  - {num_games // 2} vs Random opponents")
        print(f"  - {num_games // 2} vs DQN pool (tuned_ai_v2, tuned_ai_v4, poker_ai_tuned)")
        print()

        all_results = []
        start_time = time.time()

        for i, model_path in enumerate(model_files, 1):
            model_name = os.path.basename(model_path)
            print(f"[{i}/{len(model_files)}] {model_name}")

            result = self.evaluate_model(model_path, num_games, verbose=False)
            all_results.append(result)

            if result['status'] == 'success':
                print(f"  ✓ Win rate: {result['win_rate']*100:.1f}% overall "
                      f"(Random: {result['win_rate_vs_random']*100:.1f}%, "
                      f"DQN: {result['win_rate_vs_dqn']*100:.1f}%)")
            else:
                print(f"  ✗ {result.get('error', 'Unknown error')}")

        elapsed = time.time() - start_time
        print(f"\nTotal evaluation time: {elapsed:.1f} seconds")

        return all_results

    def print_summary_report(self, results: List[Dict]):
        """Print a comprehensive summary report"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY REPORT")
        print(f"{'='*80}\n")
        print("Note: Models play full games until elimination (matching post-training evaluation)")
        print("      Win rates should now accurately reflect model performance\n")

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

        # Sort by combined win rate
        compatible.sort(key=lambda x: x.get('win_rate', 0), reverse=True)

        # Performance table
        print("PERFORMANCE METRICS:")
        print(f"{'Model':<30} {'Overall':>10} {'vs Random':>12} {'vs DQN':>10} {'BB/Hand':>10}")
        print("-" * 75)

        for r in compatible:
            print(f"{r['model_name'][:30]:<30} "
                  f"{r.get('win_rate', 0)*100:>9.1f}% "
                  f"{r.get('win_rate_vs_random', 0)*100:>11.1f}% "
                  f"{r.get('win_rate_vs_dqn', 0)*100:>9.1f}% "
                  f"{r.get('bb_per_hand', 0):>9.3f}")

        # Action distribution for top models
        print("\nACTION DISTRIBUTIONS (Top 3 Models):")
        for i, r in enumerate(compatible[:3]):
            if 'action_distribution' in r and r['action_distribution']:
                print(f"\n{r['model_name']}:")
                for action, freq in sorted(r['action_distribution'].items()):
                    print(f"  {action:8s}: {freq*100:5.1f}%")

        # Summary statistics
        if compatible:
            avg_win_rate = np.mean([r.get('win_rate', 0) for r in compatible])
            avg_win_vs_random = np.mean([r.get('win_rate_vs_random', 0) for r in compatible])
            avg_win_vs_dqn = np.mean([r.get('win_rate_vs_dqn', 0) for r in compatible])

            print(f"\n{'='*75}")
            print(f"AVERAGE PERFORMANCE:")
            print(f"  Overall win rate:      {avg_win_rate*100:.1f}%")
            print(f"  Win rate vs Random:    {avg_win_vs_random*100:.1f}%")
            print(f"  Win rate vs DQN:       {avg_win_vs_dqn*100:.1f}%")