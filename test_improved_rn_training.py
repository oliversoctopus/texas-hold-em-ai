"""
Quick test of improved Raw Neural CFR training
"""

import sys
from cfr.raw_neural_cfr import RawNeuralCFR
from evaluation.unified_evaluation import UnifiedEvaluator

def test_improved_training():
    print("Testing improved Raw Neural CFR with mixed training")
    print("=" * 60)

    # Create model with mixed training focused on DQN opponents
    model = RawNeuralCFR(
        iterations=1000,  # Quick test
        learning_rate=0.001,
        batch_size=32,
        initial_epsilon=0.3,
        final_epsilon=0.05,
        initial_temperature=1.0,
        final_temperature=0.1,
        hidden_dim=512,
        use_mixed_training=True,
        mixed_training_weights={'self': 0.3, 'random': 0.2, 'dqn': 0.5}  # 50% against DQN
    )

    # Train the model
    print("\nStarting training...")
    model.train(verbose=True)

    # Save the model
    model_path = "models/cfr/rn_improved_test.pkl"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Quick evaluation
    print("\n" + "=" * 60)
    print("Quick evaluation against DQN benchmarks (10 games):")
    evaluator = UnifiedEvaluator()
    results = evaluator.evaluate_cfr_model(
        model,
        num_games=10,
        num_players=2,
        use_random_opponents=False,  # Use strong DQN opponents
        verbose=True
    )

    print("\nAction distribution:")
    for action, pct in results['action_distribution'].items():
        print(f"  {action}: {pct:.1f}%")

    print(f"\nWin rate: {results['win_rate']:.1f}%")
    print(f"Average earnings: {results['avg_earnings']:.1f} chips")

if __name__ == "__main__":
    test_improved_training()