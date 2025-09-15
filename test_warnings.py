"""
Test that warnings are triggered for impossible game states
"""

from core.game_engine import TexasHoldEm
from core.player import Player

print("Testing warnings for impossible game states...")
print("=" * 60)

# Test 1: Try to play with all players having 0 chips
print("\nTest 1: All players with 0 chips")
game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=False)
game.players = [
    Player("Player1", 0, is_ai=False),
    Player("Player2", 0, is_ai=False)
]
print("Attempting to play hand with all players at 0 chips...")
game.play_hand(verbose=False)

# Test 2: Force deck to run out (not really possible in normal 2-player game)
print("\nTest 2: Deck running out")
print("In a normal 2-player game, deck should never run out.")
print("(2 players * 2 cards + 5 community = 9 cards max)")

# Test 3: Test chip conservation with our fix
print("\nTest 3: Chip conservation after fix")
game2 = TexasHoldEm(num_players=2, starting_chips=1000, verbose=False)
game2.players = [
    Player("Player1", 1000, is_ai=False),
    Player("Player2", 1000, is_ai=False)
]

# Manually simulate a hand end where we forget to reset pot
print("Simulating pot distribution...")
game2.pot = 100
winners = [game2.players[0]]
game2.distribute_pot(winners, verbose=False)
print(f"After distribution - Pot: {game2.pot} (should be 0)")
print(f"Player chips: {[p.chips for p in game2.players]}")

if game2.pot == 0:
    print("SUCCESS: Pot correctly reset to 0 after distribution")
else:
    print("ERROR: Pot not reset after distribution!")

print("\n" + "=" * 60)
print("Warning tests complete.")