"""
Test CFR AI with real poker game evaluation
"""

from cfr_poker import CFRPokerAI
from cfr_evaluation import evaluate_cfr_ai_real

print("=" * 60)
print("Testing CFR AI with REAL poker game evaluation")
print("=" * 60)

# Load the saved CFR model
print("\nLoading CFR model...")
cfr_ai = CFRPokerAI()
cfr_ai.load("cfr_poker_v2.pkl")
print(f"Loaded CFR AI with {len(cfr_ai.nodes)} information sets")

# Test with real games
print("\nTesting with real poker games (this will take a moment)...")
results = evaluate_cfr_ai_real(cfr_ai, num_games=50, num_players=6, verbose=True)

print("\n" + "=" * 60)
print("COMPARISON: Estimated vs Real Performance")
print("=" * 60)
print(f"Previous estimated win rate: 13.7%")
print(f"Real game win rate: {results['win_rate']:.1f}%")
print(f"Expected for random play: {results['expected_random_rate']:.1f}%")
print(f"Performance multiplier: {results['performance_vs_random']:.1f}x")

if results['win_rate'] > results['expected_random_rate']:
    print("\n✓ CFR AI performs better than random!")
    print("The CFR implementation is working correctly.")
else:
    print("\n⚠ CFR AI may need more training or better integration.")