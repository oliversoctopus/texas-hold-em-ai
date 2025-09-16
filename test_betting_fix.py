"""Test script to verify Raw Neural CFR now properly handles betting information"""

import os
from core.game_constants import Action
from core.card_deck import Card
from cfr.raw_neural_cfr import RawNeuralCFR
from utils.model_loader import load_cfr_model_by_type, create_game_wrapper_for_model

def test_betting_scenarios():
    """Test that the model now considers call amounts properly"""
    print("Testing Raw Neural CFR Betting Fix")
    print("=" * 60)

    # Load a trained model
    model_path = "models/cfr/rn_resume_extended.pkl"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Using a new untrained model instead...")
        model = RawNeuralCFR(hidden_dim=256)  # Use smaller model for faster testing
        model_info = {'model_type': 'raw_neural_cfr'}
    else:
        print(f"Loading model: {model_path}")
        model, model_info = load_cfr_model_by_type(model_path, verbose=False)

    ai_wrapper = create_game_wrapper_for_model(model, model_info)
    print("Model loaded successfully!")
    print()

    # Test scenarios with different betting situations
    test_cases = [
        {
            "name": "No bet to call (should be more willing to bet/raise)",
            "hole_cards": [Card('A', '♠'), Card('K', '♥')],
            "community_cards": [Card('Q', '♦'), Card('J', '♣'), Card('10', '♠')],
            "pot_size": 100,
            "to_call": 0,
            "stack_size": 900,
            "valid_actions": [Action.CHECK, Action.RAISE, Action.ALL_IN]
        },
        {
            "name": "Small bet to call (might call or raise)",
            "hole_cards": [Card('A', '♠'), Card('K', '♥')],
            "community_cards": [Card('Q', '♦'), Card('J', '♣'), Card('10', '♠')],
            "pot_size": 100,
            "to_call": 50,
            "stack_size": 900,
            "valid_actions": [Action.FOLD, Action.CALL, Action.RAISE, Action.ALL_IN]
        },
        {
            "name": "Large bet to call (should be more selective)",
            "hole_cards": [Card('A', '♠'), Card('K', '♥')],
            "community_cards": [Card('Q', '♦'), Card('J', '♣'), Card('10', '♠')],
            "pot_size": 100,
            "to_call": 200,
            "stack_size": 900,
            "valid_actions": [Action.FOLD, Action.CALL, Action.RAISE, Action.ALL_IN]
        },
        {
            "name": "Massive all-in to call (should often fold)",
            "hole_cards": [Card('7', '♠'), Card('2', '♥')],  # Weak hand
            "community_cards": [Card('A', '♦'), Card('K', '♣'), Card('Q', '♠')],
            "pot_size": 200,
            "to_call": 800,  # Huge bet!
            "stack_size": 900,
            "valid_actions": [Action.FOLD, Action.CALL, Action.ALL_IN]
        },
        {
            "name": "Good pot odds with decent hand",
            "hole_cards": [Card('J', '♠'), Card('J', '♥')],  # Pocket jacks
            "community_cards": [Card('9', '♦'), Card('5', '♣'), Card('2', '♠')],
            "pot_size": 300,
            "to_call": 75,  # Good pot odds: 75 to win 375
            "stack_size": 900,
            "valid_actions": [Action.FOLD, Action.CALL, Action.RAISE, Action.ALL_IN]
        }
    ]

    # Test each scenario
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Cards: {test_case['hole_cards']} | Board: {test_case['community_cards']}")
        print(f"  Pot: ${test_case['pot_size']}, To call: ${test_case['to_call']}")
        print(f"  Stack: ${test_case['stack_size']}")

        # Calculate pot odds
        if test_case['to_call'] > 0:
            pot_odds = test_case['to_call'] / (test_case['pot_size'] + test_case['to_call'])
            print(f"  Pot odds: {pot_odds:.1%} (need to call ${test_case['to_call']} to win ${test_case['pot_size'] + test_case['to_call']})")

        # Create state for the wrapper
        class TestState:
            def __init__(self, tc):
                self.hole_cards = tc['hole_cards']
                self.community_cards = tc['community_cards']
                self.pot_size = tc['pot_size']
                self.to_call = tc['to_call']
                self.stack_size = tc['stack_size']
                self.position = 0
                self.num_players = 2
                self.action_history = []

        state = TestState(test_case)

        # Get AI's decision
        action = ai_wrapper.choose_action(state, test_case['valid_actions'])

        print(f"  AI chose: {action.name}")

        # Analyze the decision
        if test_case['to_call'] == 0 and action == Action.CHECK:
            print("  ✓ Reasonable: Checking when no bet to call")
        elif test_case['to_call'] > 0 and action == Action.FOLD:
            if test_case['to_call'] > test_case['pot_size']:
                print("  ✓ Good: Folding to overbet")
            else:
                print("  ? Folded with pot odds")
        elif action == Action.CALL:
            if test_case['to_call'] < test_case['pot_size'] * 0.5:
                print("  ✓ Good: Calling with good pot odds")
            else:
                print("  ? Calling large bet")
        elif action == Action.RAISE:
            if test_case['to_call'] == 0:
                print("  ✓ Good: Betting when checked to")
            else:
                print("  ! Aggressive: Raising over existing bet")
        elif action == Action.ALL_IN:
            print("  !! Very aggressive: All-in")

        print()

    print("=" * 60)
    print("Testing complete!")
    print("\nSummary:")
    print("The model should now be more aware of betting costs and pot odds.")
    print("With the fix, it should:")
    print("  - Be more selective when facing large bets")
    print("  - Consider pot odds when deciding to call")
    print("  - Not raise as aggressively when it's expensive to do so")

if __name__ == "__main__":
    test_betting_scenarios()