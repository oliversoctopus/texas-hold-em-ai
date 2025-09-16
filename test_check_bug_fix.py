"""Test script to verify the check bug fix for Raw Neural CFR"""

import sys
import os
from core.game_engine import TexasHoldEm
from core.game_constants import Action
from utils.model_loader import load_cfr_model_by_type, create_game_wrapper_for_model

def test_check_bug():
    """Test that Raw Neural CFR cannot check when there's a bet to call"""
    print("Testing check bug fix for Raw Neural CFR...")
    print("=" * 60)

    # Load the Raw Neural CFR model
    model_path = "models/cfr/rn_resume_standard.pkl"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please ensure the model exists before running this test.")
        return

    print(f"Loading model: {model_path}")
    try:
        model, model_info = load_cfr_model_by_type(model_path, verbose=False)
        ai_wrapper = create_game_wrapper_for_model(model, model_info)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded successfully!")
    print()

    # Create a simple test scenario
    print("Creating test scenario:")
    print("- Setting up a game where human raises")
    print("- Verifying AI cannot check in response")
    print()

    # Create game
    game = TexasHoldEm(num_players=2)
    game.verbose = False

    # Initialize players first
    game.setup_players([("Human", 1000), ("RawNeuralCFR", 1000)])

    # Setup players
    game.players[0].name = "Human"
    game.players[0].is_ai = False
    game.players[1].name = "RawNeuralCFR"
    game.players[1].is_ai = True
    game.players[1].ai_model = ai_wrapper

    # Deal cards
    game.deck.reset()
    for player in game.players:
        player.reset_hand()
        player.hand = game.deck.draw(2)

    # Post blinds
    game.pot = 30  # Small blind (10) + Big blind (20)
    game.players[0].current_bet = 10  # Small blind
    game.players[1].current_bet = 20  # Big blind
    game.players[0].chips = 990
    game.players[1].chips = 980
    game.current_bet = 20

    print("Initial state:")
    print(f"  Pot: ${game.pot}")
    print(f"  Current bet: ${game.current_bet}")
    print(f"  Human chips: ${game.players[0].chips}, bet: ${game.players[0].current_bet}")
    print(f"  AI chips: ${game.players[1].chips}, bet: ${game.players[1].current_bet}")
    print()

    # Human raises
    print("Human raises to $100...")
    raise_amount = 100
    human_player = game.players[0]

    # Execute the raise
    call_amount = game.current_bet - human_player.current_bet
    total_bet = call_amount + raise_amount
    amount_bet = human_player.bet(total_bet)
    game.pot += amount_bet
    game.current_bet = human_player.current_bet
    game.last_raise_amount = raise_amount

    print(f"After human raise:")
    print(f"  Pot: ${game.pot}")
    print(f"  Current bet: ${game.current_bet}")
    print(f"  Human chips: ${game.players[0].chips}, bet: ${game.players[0].current_bet}")
    print(f"  AI chips: ${game.players[1].chips}, bet: ${game.players[1].current_bet}")
    print()

    # Get AI's valid actions
    ai_player = game.players[1]
    valid_actions = game.get_valid_actions(ai_player)

    print("Valid actions for AI:")
    for action in valid_actions:
        if action == Action.CALL:
            print(f"  - {action.name} (${game.current_bet - ai_player.current_bet})")
        else:
            print(f"  - {action.name}")
    print()

    # Verify CHECK is not in valid actions
    if Action.CHECK in valid_actions:
        print("ERROR: CHECK is in valid actions when there's a bet to call!")
        print("    The bug still exists in get_valid_actions")
        return
    else:
        print("CHECK is correctly NOT in valid actions (PASS)")
    print()

    # Get AI's action
    print("Getting AI's action...")

    # Create state for AI
    class DummyState:
        def __init__(self, player, game):
            self.hole_cards = player.hand
            self.community_cards = game.community_cards
            self.pot_size = game.pot
            self.to_call = max(0, game.current_bet - player.current_bet)
            self.stack_size = player.chips
            self.position = game.players.index(player)
            self.num_players = len(game.players)
            self.action_history = []

    state = DummyState(ai_player, game)

    # Get action from AI
    try:
        action = ai_wrapper.choose_action(state, valid_actions)
        print(f"AI chose: {action.name}")

        if action == Action.CHECK:
            print("ERROR: AI returned CHECK action when it's not valid!")
            print("    The wrapper is not properly filtering actions")
        else:
            print("AI correctly chose a valid action (PASS)")

        # Test the failsafe in execute_action
        print()
        print("Testing failsafe in execute_action...")

        # Try to force a CHECK action
        print("Attempting to execute CHECK action (should be blocked)...")
        game.execute_action(ai_player, Action.CHECK, verbose=True)

        if ai_player.folded:
            print("Failsafe worked: Invalid CHECK was converted to FOLD (PASS)")
        else:
            print("ERROR: Invalid CHECK was not blocked!")

    except Exception as e:
        print(f"Error getting AI action: {e}")
        return

    print()
    print("=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_check_bug()