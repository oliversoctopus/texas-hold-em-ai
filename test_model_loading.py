"""
Test that Raw Neural CFR models can be loaded correctly
"""

from cfr.raw_neural_cfr import RawNeuralCFR
from core.card_deck import Card
import os

def test_model_loading(filepath):
    """Test loading a model and making a prediction"""
    print(f"\nTesting: {filepath}")

    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return False

    try:
        # Create a new model instance
        model = RawNeuralCFR()

        # Load the saved model
        model.load(filepath)
        print(f"  Model loaded successfully")

        # Test that the model can make predictions
        action = model.get_action(
            hole_cards=[Card('A', '♠'), Card('K', '♠')],
            community_cards=[],
            pot_size=30,
            to_call=20,
            stack_size=980,
            position=0,
            num_players=2
        )

        print(f"  Test prediction: {action}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("Testing Raw Neural CFR model loading...")
    print("=" * 60)

    models_dir = "models/cfr"

    # Test the fixed models
    test_models = [
        "raw_neural_quick.pkl",
        "raw_neural_tiny.pkl"
    ]

    success_count = 0
    for model_file in test_models:
        filepath = os.path.join(models_dir, model_file)
        if test_model_loading(filepath):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(test_models)} models loaded successfully")

    if success_count == len(test_models):
        print("SUCCESS: All models can be loaded!")
    else:
        print("WARNING: Some models failed to load")

if __name__ == "__main__":
    main()