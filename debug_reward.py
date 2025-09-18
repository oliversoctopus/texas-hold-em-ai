"""Debug reward calculation issue"""

from reward_nn.reward_based_ai import RewardBasedAI
from reward_nn.training import RewardBasedTrainer
from core.game_engine import TexasHoldEmTraining
from core.player import Player

# Create AI and trainer
ai = RewardBasedAI(hidden_dim=256)
trainer = RewardBasedTrainer(ai, num_players=2)

# Manually simulate a hand with detailed tracking
game = TexasHoldEmTraining(num_players=2)

# Setup players
players = []
for i in range(2):
    player = Player(f"Player_{i}", 1000, is_ai=True)
    if i == 0:
        player.ai_model = ai
    else:
        player.ai_model = trainer._create_random_ai()
    players.append(player)

game.players = players

print("Initial state:")
print(f"  Player 0: {game.players[0].chips} chips")
print(f"  Player 1: {game.players[1].chips} chips")
print(f"  Total: {sum(p.chips for p in game.players)} chips")

# Deal cards
for player in game.players:
    player.hand = game.deck.draw(2)
    player.reset_hand()

print("\nAfter dealing:")
print(f"  Player 0: {game.players[0].chips} chips")
print(f"  Player 1: {game.players[1].chips} chips")

# Post blinds
sb_pos = (game.dealer_position + 1) % 2
bb_pos = (game.dealer_position + 2) % 2
print(f"\nDealer position: {game.dealer_position}")
print(f"SB position: {sb_pos}, BB position: {bb_pos}")

initial_p0_chips = game.players[0].chips
initial_p1_chips = game.players[1].chips

sb_bet = game.players[sb_pos].bet(10)
bb_bet = game.players[bb_pos].bet(20)
game.pot = sb_bet + bb_bet

print(f"\nAfter blinds:")
print(f"  Player 0: {game.players[0].chips} chips (posted {'SB' if sb_pos == 0 else 'BB' if bb_pos == 0 else 'nothing'})")
print(f"  Player 1: {game.players[1].chips} chips (posted {'SB' if sb_pos == 1 else 'BB' if bb_pos == 1 else 'nothing'})")
print(f"  Pot: {game.pot}")
print(f"  Total: {sum(p.chips for p in game.players) + game.pot} chips")

# Simulate one betting round
print("\nSimulating preflop...")
# For simplicity, just have both players call/check
if game.players[0].current_bet < 20:
    call_amount = 20 - game.players[0].current_bet
    game.pot += game.players[0].bet(call_amount)
    print(f"  Player 0 calls {call_amount}")

if game.players[1].current_bet < 20:
    call_amount = 20 - game.players[1].current_bet
    game.pot += game.players[1].bet(call_amount)
    print(f"  Player 1 calls {call_amount}")

print(f"\nAfter preflop:")
print(f"  Player 0: {game.players[0].chips} chips")
print(f"  Player 1: {game.players[1].chips} chips")
print(f"  Pot: {game.pot}")

# Simple showdown - player 0 wins
print("\nShowdown: Player 0 wins")
game.players[0].chips += game.pot

final_p0_chips = game.players[0].chips
final_p1_chips = game.players[1].chips

print(f"\nFinal state:")
print(f"  Player 0: {final_p0_chips} chips")
print(f"  Player 1: {final_p1_chips} chips")
print(f"  Total: {final_p0_chips + final_p1_chips} chips")

# Calculate reward as trainer does
chips_won = final_p0_chips - initial_p0_chips
reward_bb = chips_won / 20

print(f"\nReward calculation:")
print(f"  Initial P0 chips (after blinds): {initial_p0_chips}")
print(f"  Final P0 chips: {final_p0_chips}")
print(f"  Chips won: {chips_won}")
print(f"  Reward in BB: {reward_bb}")

# But the ACTUAL starting chips were 1000!
actual_chips_won = final_p0_chips - 1000
actual_reward_bb = actual_chips_won / 20
print(f"\nActual reward (from 1000 starting):")
print(f"  Actual chips won: {actual_chips_won}")
print(f"  Actual reward in BB: {actual_reward_bb}")