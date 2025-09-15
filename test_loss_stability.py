"""
Test that the loss is now stable and not growing exponentially
"""

from cfr.raw_neural_cfr import RawNeuralCFR
import numpy as np

print("Testing loss stability with fixed training...")
print("=" * 60)

# Create model with small iteration count
model = RawNeuralCFR(iterations=50, learning_rate=0.001, batch_size=8)

print("Training for 50 iterations to check loss stability...")
print("Expected: Loss should stabilize, not grow exponentially\n")

# Store losses for analysis
losses = []

# Override the train method to capture losses
original_train_on_batch = model._train_on_batch

def track_loss():
    loss = original_train_on_batch()
    losses.append(loss)
    return loss

model._train_on_batch = track_loss

# Train
model.train(verbose=True)

# Analyze loss trajectory
print("\n" + "=" * 60)
print("Loss Analysis:")
print(f"First 10 losses: {losses[:10]}")
print(f"Last 10 losses: {losses[-10:]}")
print(f"Mean of first 10: {np.mean(losses[:10]):.2f}")
print(f"Mean of last 10: {np.mean(losses[-10:]):.2f}")

# Check if loss is exploding
if len(losses) > 20:
    early_avg = np.mean(losses[:10])
    late_avg = np.mean(losses[-10:])

    if abs(late_avg) > abs(early_avg) * 10:
        print("\nWARNING: Loss appears to be growing exponentially!")
    else:
        print("\nGOOD: Loss appears stable or converging!")

print("\nKey fixes applied:")
print("1. Each action gets credit only for future rewards (temporal credit)")
print("2. Value network trained on accuracy, not manipulation")
print("3. Advantages calculated with stop-gradient")
print("4. Returns discounted by 0.95 per step")