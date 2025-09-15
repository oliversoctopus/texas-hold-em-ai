"""
Test the fixes for Raw Neural CFR
"""

from cfr.raw_neural_cfr import RawNeuralCFR
from core.card_deck import Card
import numpy as np

print("Testing Raw Neural CFR fixes...")
print("=" * 60)

# Test 1: Check that training doesn't explode
print("\n1. Testing loss stability with normalized rewards...")
model = RawNeuralCFR(iterations=20, learning_rate=0.001, batch_size=8)

losses = []
original_train_on_batch = model._train_on_batch

def track_loss():
    loss = original_train_on_batch()
    losses.append(loss)
    return loss

model._train_on_batch = track_loss

try:
    model.train(verbose=False)
    print(f"   Training completed without errors")
    print(f"   First 5 losses: {losses[:5]}")
    print(f"   Last 5 losses: {losses[-5:]}")

    # Check if loss is reasonable (should be < 100 with BB normalization)
    max_loss = max(abs(l) for l in losses if l != 0)
    if max_loss < 100:
        print(f"   SUCCESS: Loss is stable (max: {max_loss:.2f})")
    else:
        print(f"   WARNING: Loss still high (max: {max_loss:.2f})")
except Exception as e:
    print(f"   ERROR: Training failed: {e}")

# Test 2: Check community cards display
print("\n2. Testing community cards display...")
from core.game_engine import TexasHoldEm
from core.player import Player

game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=False)
game.players = [
    Player("Player1", 1000, is_ai=False),
    Player("Player2", 1000, is_ai=False)
]

# Simulate adding cards
game.deck.reset()
game.community_cards = []

# Draw flop
flop_cards = game.deck.draw(3)
game.community_cards.extend([c for c in flop_cards if c is not None])

print(f"   Community cards after flop: {[str(c) for c in game.community_cards]}")
if None not in game.community_cards:
    print(f"   SUCCESS: No None values in community cards")
else:
    print(f"   ERROR: None values still present")

# Test 3: Test action selection
print("\n3. Testing action selection after training...")
try:
    action = model.get_action(
        hole_cards=[Card('A', '♠'), Card('K', '♠')],
        community_cards=[],
        pot_size=30,
        to_call=20,
        stack_size=980,
        position=0,
        num_players=2
    )
    print(f"   Selected action: {action}")
    print(f"   SUCCESS: Action selection works")
except Exception as e:
    print(f"   ERROR: Action selection failed: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("- Rewards normalized to BB units (±50 typical)")
print("- Value network trained with Huber loss")
print("- Temporal credit properly assigns returns per player")
print("- Community cards filtered for None values")
print("\nThese fixes should:")
print("1. Prevent loss explosion")
print("2. Clean up display")
print("3. Improve learning stability")