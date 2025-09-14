"""
Test script for Strategy Selector CFR
"""

from cfr.strategy_selector_cfr import StrategySelectorCFR
from core.card_deck import Card

# Create and load model
print("Loading Strategy Selector CFR...")
cfr = StrategySelectorCFR()

try:
    cfr.load("test_stats.pkl")
    print("Model loaded successfully")
except:
    print("No saved model, using untrained")

# Test get_action
print("\nTesting get_action...")
hole_cards = [Card('A', '♠'), Card('K', '♠')]
community_cards = [Card('Q', '♠'), Card('J', '♥'), Card('10', '♦')]

action = cfr.get_action(
    hole_cards=hole_cards,
    community_cards=community_cards,
    betting_history='br',
    pot_size=100,
    to_call=50,
    stack_size=900,
    position=0,
    opponent_stack=850
)

print(f"Selected action: {action}")
print(f"Last strategy used: {cfr.last_selected_strategy.name}")

# Test get_raise_size
print("\nTesting get_raise_size...")
for i in range(5):
    size = cfr.get_raise_size(pot_size=100, current_bet=20, player_chips=900, min_raise=40)
    print(f"  Raise size {i+1}: ${size}")

# Show statistics
print("\nStrategy usage after tests:")
stats = cfr.get_strategy_usage_stats()
print(f"  Total decisions: {stats['total_used']}")
print(f"  Most used: {stats['most_used_strategy']}")
for strategy, pct in stats['strategy_distribution'].items():
    if pct > 0:
        print(f"    {strategy}: {pct:.1f}%")

print("\nAll tests completed successfully!")