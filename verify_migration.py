"""Verify that migrated models work correctly"""

import pickle
from cfr.raw_neural_cfr import RawNeuralCFR
from core.game_constants import Action

def verify_model(filepath):
    """Verify a migrated model works"""
    print(f"Testing {filepath}...")

    try:
        # Check the checkpoint has hidden_dim
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        if 'hidden_dim' not in checkpoint:
            print(f"  [FAIL] Missing hidden_dim parameter")
            return False

        print(f"  Checkpoint has hidden_dim={checkpoint['hidden_dim']}")

        # Try loading with RawNeuralCFR
        model = RawNeuralCFR()
        model.load(filepath)
        print(f"  Loaded successfully with hidden_dim={model.hidden_dim}")

        # Test get_action with dummy cards
        from core.card_deck import Card

        # Create dummy cards
        hole_cards = [Card('A', '♠'), Card('K', '♥')]
        community_cards = [Card('Q', '♦'), Card('J', '♣'), Card('10', '♠')]

        action = model.get_action(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=100,
            stack_size=1000,
            position=0,
            valid_actions=[Action.FOLD, Action.CALL, Action.RAISE]
        )
        print(f"  get_action returned: {action.name}")

        print(f"  [PASS] Model works correctly")
        return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def main():
    print("Verifying migrated Raw Neural CFR models")
    print("=" * 60)

    models = [
        "models/cfr/raw_neural_professional.pkl",
        "models/cfr/raw_neural_quick.pkl",
        "models/cfr/raw_neural_tiny.pkl",
        "models/cfr/rn_resume100.pkl",
        "models/cfr/rn_resumequick.pkl",
        "models/cfr/rn_resume_extended.pkl"
    ]

    passed = 0
    failed = 0

    for model_path in models:
        if verify_model(model_path):
            passed += 1
        else:
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All models verified successfully!")
    else:
        print("Some models failed verification. Check the errors above.")

if __name__ == "__main__":
    main()