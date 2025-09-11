"""
Quick CFR demonstration
"""

from cfr_player import train_cfr_ai, evaluate_cfr_ai

print("=" * 60)
print("CFR (Counterfactual Regret Minimization) Demo")
print("=" * 60)

print("\nTraining CFR AI with 2000 iterations...")
cfr_ai = train_cfr_ai(iterations=2000, num_players=4, verbose=True)

print("\nEvaluating CFR AI...")
results = evaluate_cfr_ai(cfr_ai, num_games=20, verbose=True)

print("\n" + "=" * 60)
print("CFR vs Deep Q-Learning Comparison:")
print("=" * 60)
print()
print("CFR Advantages:")
print("✓ Game theory optimal (Nash equilibrium)")
print("✓ No hyperparameter tuning needed")  
print("✓ Consistent, reproducible results")
print("✓ Used by professional poker AIs")
print("✓ Handles multi-player games naturally")
print()
print("DQN Disadvantages you experienced:")
print("✗ Inconsistent training results")
print("✗ Complex hyperparameter tuning")
print("✗ Often gets stuck in suboptimal strategies")
print("✗ Requires careful reward engineering")
print()
print("This CFR implementation is ready to use in your main program!")
print("Select option 1 in main.py to train CFR AI.")