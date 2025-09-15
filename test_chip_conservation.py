"""
Test chip conservation in the game engine
"""

from core.game_engine import TexasHoldEm
from core.player import Player
from cfr.raw_neural_cfr import RawNeuralCFR

print("Testing chip conservation...")
print("=" * 60)

# Create a simple game
game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=False)

# Create simple players
model = RawNeuralCFR(iterations=10)
model.train(verbose=False)

players = []
for i in range(2):
    player = Player(f"Player_{i}", 1000, is_ai=True)
    # Create wrapper for CFR model
    class CFRWrapper:
        def __init__(self, cfr_model):
            self.cfr_model = cfr_model
            self.epsilon = 0

        def choose_action(self, state, valid_actions, **kwargs):
            return self.cfr_model.get_action(
                hole_cards=state.hole_cards,
                community_cards=state.community_cards,
                pot_size=state.pot_size,
                to_call=state.to_call,
                stack_size=state.stack_size,
                position=state.position,
                num_players=2
            )

    player.ai_model = CFRWrapper(model)
    players.append(player)

game.players = players

print(f"Initial chips: {[p.chips for p in game.players]}")
print(f"Total: {sum(p.chips for p in game.players)}")
print()

# Play 500 hands and check conservation
errors_found = 0
games_played = 0
total_hands = 0

for hand_num in range(500):
    initial_total = sum(p.chips for p in game.players)

    # Check if game is over and restart if needed
    active_players = [p for p in game.players if p.chips > 0]
    if len(active_players) <= 1:
        if hand_num < 20 or hand_num % 50 == 0:
            print(f"\nGame {games_played + 1} ended. Restarting...")
        games_played += 1
        # Reset all players to starting chips
        for player in game.players:
            player.chips = 1000
        initial_total = sum(p.chips for p in game.players)

    if hand_num < 20 or hand_num % 50 == 0:  # Print first 20 and every 50th
        print(f"Hand {hand_num + 1}: ", end='')

    game.play_hand(verbose=False)

    final_total = sum(p.chips for p in game.players)

    if initial_total != final_total:
        errors_found += 1
        print(f"\nERROR at hand {hand_num + 1}! Chips changed from {initial_total} to {final_total}")
        print(f"  Player chips: {[p.chips for p in game.players]}")
        print(f"  Difference: {final_total - initial_total}")
    elif hand_num < 20 or hand_num % 50 == 0:
        print(f"OK (Total: {final_total}, Players: {[p.chips for p in game.players]})")

print(f"\nFinal chips: {[p.chips for p in game.players]}")
print(f"Final total: {sum(p.chips for p in game.players)}")
print(f"Expected total: 2000")
print(f"Hands played: {hand_num + 1}")
print(f"Games completed: {games_played}")
print(f"Errors found: {errors_found}")

if errors_found == 0 and sum(p.chips for p in game.players) == 2000:
    print("\nChip conservation test PASSED! No errors in any hands.")
else:
    print(f"\nChip conservation test FAILED! Found {errors_found} errors.")