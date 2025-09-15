"""
Debug script to investigate survival rate anomaly
"""

from evaluation.unified_evaluation import evaluate_cfr_full
from cfr.raw_neural_cfr import RawNeuralCFR

# Create a small model for testing
print("Creating small CFR model for testing...")
model = RawNeuralCFR(iterations=10)
print("Training for 10 iterations...")
model.train(verbose=False)

print("\nRunning 2-player evaluation with debug output...")
print("=" * 60)
# Force 2-player games to reproduce the original issue
from evaluation.unified_evaluation import UnifiedEvaluator
evaluator = UnifiedEvaluator()
results = evaluator.evaluate_cfr_model(model, num_games=100, num_players=2,
                                       use_random_opponents=True, verbose=True)