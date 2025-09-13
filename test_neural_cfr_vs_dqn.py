#!/usr/bin/env python3
"""
Test Neural-Enhanced CFR vs DQN benchmarks
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_neural_cfr_vs_dqn():
    """Test Neural-Enhanced CFR vs DQN performance"""
    print("Testing Neural-Enhanced CFR vs DQN Benchmarks...")
    print("=" * 60)

    try:
        # Create and train Neural-Enhanced CFR
        from cfr.neural_enhanced_cfr import NeuralEnhancedTwoPlayerCFR
        print("Creating Neural-Enhanced CFR with quick training (1000 iterations)...")

        cfr = NeuralEnhancedTwoPlayerCFR(
            iterations=1000,
            use_neural_networks=True,
            progressive_training=True
        )

        print("Training CFR (this may take ~1-2 minutes)...")
        cfr.train(verbose=False)  # Silent training to avoid spam
        print(f"Training complete! Created {len(cfr.nodes):,} information sets")

        # Test vs random opponents (baseline)
        print("\nTesting vs random opponents (baseline)...")
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai

        random_results = evaluate_two_player_cfr_ai(
            cfr,
            num_games=50,
            verbose=False,
            use_strong_opponents=False
        )

        print(f"vs Random: {random_results['win_rate']:.1f}% win rate")

        # Test vs DQN benchmarks (main test)
        print("\nTesting vs DQN benchmark models (challenge)...")
        dqn_results = evaluate_two_player_cfr_ai(
            cfr,
            num_games=50,
            verbose=True,  # Show DQN evaluation details
            use_strong_opponents=True
        )

        print(f"vs DQN benchmarks: {dqn_results['win_rate']:.1f}% win rate")

        # Analysis
        print("\n" + "=" * 60)
        print("NEURAL-ENHANCED CFR PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"vs Random opponents:     {random_results['win_rate']:.1f}%")
        print(f"vs DQN benchmark models: {dqn_results['win_rate']:.1f}%")
        print(f"Information sets:        {len(cfr.nodes):,}")

        strategy_usage = dqn_results.get('strategy_usage', {})
        learned_pct = strategy_usage.get('learned_percentage', 0)
        print(f"Strategy coverage:       {learned_pct:.1f}%")

        # Performance assessment
        if dqn_results['win_rate'] >= 55:
            print("\n[EXCELLENT] Neural-Enhanced CFR is competitive with DQN models!")
        elif dqn_results['win_rate'] >= 45:
            print("\n[GOOD] Neural-Enhanced CFR shows strong performance vs DQN models!")
        elif dqn_results['win_rate'] >= 35:
            print("\n[FAIR] Neural-Enhanced CFR needs more training to compete with DQN models.")
        else:
            print("\n[NEEDS WORK] Neural-Enhanced CFR requires significant improvement.")

        improvement = dqn_results['win_rate'] - 40  # Compare to old CFR's 40%
        if improvement > 0:
            print(f"IMPROVEMENT: +{improvement:.1f}% better than old CFR implementations")
        else:
            print(f"REGRESSION: {abs(improvement):.1f}% worse than old CFR implementations")

        print("\n" + "=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_neural_cfr_vs_dqn()