"""
Debug script to identify cause of extreme losses
"""

from cfr.raw_neural_cfr import RawNeuralCFR

print("Debugging extreme loss values...")
print("=" * 60)

# Create model with very small iteration count
model = RawNeuralCFR(iterations=20, learning_rate=0.001, batch_size=8)

print("Training for 20 iterations with debug output...")
print("Looking for:")
print("- Large utilities (>40 BB)")
print("- Large losses (>50)")
print()

try:
    model.train(verbose=True)
    print("\nTraining completed.")
except Exception as e:
    print(f"\nTraining failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Analysis:")
print("If you see large utilities, the game is reaching extreme chip swings.")
print("If you see extreme losses, check the value predictions vs actual utilities.")
print("Large advantages indicate the value network is very wrong.")