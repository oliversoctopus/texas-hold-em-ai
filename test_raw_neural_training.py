"""
Test Raw Neural CFR training to ensure no errors
"""

from cfr.raw_neural_cfr import RawNeuralCFR

print("Testing Raw Neural CFR training...")
print("=" * 60)

# Create model with small iteration count for testing
model = RawNeuralCFR(iterations=20, learning_rate=0.001, batch_size=4)

print("Starting training with 20 iterations...")
try:
    model.train(verbose=True)
    print("\nTraining completed successfully!")

    # Test that the model can make decisions
    from core.card_deck import Card
    print("\nTesting action selection...")
    action = model.get_action(
        hole_cards=[Card('A', '♠'), Card('K', '♠')],
        community_cards=[],
        pot_size=30,
        to_call=20,
        stack_size=980,
        position=0,
        num_players=2
    )
    print(f"Model selected action: {action}")

except Exception as e:
    print(f"\nERROR during training: {e}")
    import traceback
    traceback.print_exc()