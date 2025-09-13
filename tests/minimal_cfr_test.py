#!/usr/bin/env python3
"""
Minimal CFR test to verify improvements work
"""

from cfr_poker import CFRPokerAI

def minimal_test():
    print("Minimal CFR Test")
    print("=" * 30)
    
    # Create CFR AI with very few iterations
    print("Creating CFR AI with 100 iterations...")
    cfr_ai = CFRPokerAI(iterations=100)
    
    print("Starting training...")
    cfr_ai.train(verbose=True)
    
    print(f"Training complete! Generated {len(cfr_ai.nodes)} nodes")
    
    # Test a simple decision
    from card_deck import Card
    test_cards = [Card(14, 'hearts'), Card(13, 'hearts')]  # Ace-King suited
    community = [Card(12, 'hearts'), Card(11, 'hearts'), Card(10, 'spades')]  # Strong flush draw
    
    action = cfr_ai.get_action(
        hole_cards=test_cards,
        community_cards=community,
        betting_history="",
        pot_size=100,
        to_call=20,
        stack_size=1000,
        position=0,
        num_players=4
    )
    
    print(f"CFR decision with strong hand: {action}")
    
    return cfr_ai

if __name__ == "__main__":
    minimal_test()