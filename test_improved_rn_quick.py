"""
Very quick test of improved Raw Neural CFR
"""

import sys
from cfr.raw_neural_cfr import RawNeuralCFR

def test_quick():
    print("Quick test of improved Raw Neural CFR")
    print("=" * 60)

    # Create model with minimal iterations
    model = RawNeuralCFR(
        iterations=100,  # Very quick test
        learning_rate=0.001,
        batch_size=16,
        initial_epsilon=0.3,
        final_epsilon=0.05,
        initial_temperature=1.0,
        final_temperature=0.1,
        hidden_dim=256,  # Smaller network for faster training
        use_mixed_training=True,
        mixed_training_weights={'self': 0.4, 'random': 0.3, 'dqn': 0.3}
    )

    # Train the model
    print("\nStarting quick training (100 iterations)...")
    model.train(verbose=True)

    # Test action selection with different scenarios
    print("\n" + "=" * 60)
    print("Testing action selection in different scenarios:")

    from core.card_deck import Card
    from core.game_constants import Action

    # Test 1: Strong hand
    print("\n1. Strong hand (AA):")
    action = model.get_action(
        hole_cards=[Card('A', '♠'), Card('A', '♥')],
        community_cards=[],
        pot_size=20,
        to_call=10,
        stack_size=990,
        opponent_stack=980,
        valid_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        position=0,
        num_players=2
    )
    print(f"   Action chosen: {action}")
    if action == Action.RAISE and hasattr(model, '_last_bet_action'):
        bet_amount = model.get_bet_amount(20, 990, 980)
        print(f"   Bet amount: {bet_amount}")

    # Test 2: Weak hand
    print("\n2. Weak hand (72o):")
    action = model.get_action(
        hole_cards=[Card('7', '♠'), Card('2', '♥')],
        community_cards=[],
        pot_size=20,
        to_call=10,
        stack_size=990,
        opponent_stack=980,
        valid_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        position=0,
        num_players=2
    )
    print(f"   Action chosen: {action}")

    # Test 3: No chips to call (can only check/fold)
    print("\n3. Facing large bet with medium hand:")
    action = model.get_action(
        hole_cards=[Card('J', '♠'), Card('10', '♠')],
        community_cards=[Card('9', '♠'), Card('8', '♥'), Card('2', '♣')],
        pot_size=200,
        to_call=150,
        stack_size=500,
        opponent_stack=500,
        valid_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        position=0,
        num_players=2
    )
    print(f"   Action chosen: {action}")
    if action == Action.RAISE and hasattr(model, '_last_bet_action'):
        bet_amount = model.get_bet_amount(200, 500, 500)
        print(f"   Bet amount: {bet_amount}")

    print("\n" + "=" * 60)
    print("Quick test completed successfully!")

if __name__ == "__main__":
    test_quick()