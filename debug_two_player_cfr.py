#!/usr/bin/env python3
"""
Debug script to check what keys the 2-player CFR generates during evaluation
"""

from cfr.cfr_two_player import TwoPlayerCFRPokerAI, TwoPlayerInformationSet
from core.card_deck import Card
import pickle

def debug_loaded_model():
    """Debug what keys are in a loaded 2-player CFR model"""
    print("Loading 2-player CFR model...")
    
    try:
        cfr_ai = TwoPlayerCFRPokerAI()
        cfr_ai.load('models/cfr/two_player_v1.pkl')
        
        print(f"Model loaded with {len(cfr_ai.nodes)} information sets")
        
        # Show sample keys from the model
        print("\nSample keys in trained model:")
        sample_keys = list(cfr_ai.nodes.keys())[:10]
        for key in sample_keys:
            print(f"  {key}")
        
        # Test creating some information sets that might be encountered in evaluation
        print("\nTesting information set key generation:")
        
        # Test case 1: Preflop with decent cards
        hole_cards = [Card(14, 'hearts'), Card(13, 'spades')]  # A-K
        community_cards = []
        
        info_set = TwoPlayerInformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history="",
            pot_size=30,
            to_call=20,
            stack_size=1000,
            position=0
        )
        
        key1 = info_set.get_key()
        print(f"  Test key 1 (preflop A-K): {key1}")
        print(f"  In model? {key1 in cfr_ai.nodes}")
        
        # Test case 2: Flop with pair
        hole_cards = [Card(10, 'hearts'), Card(10, 'spades')]  # Pocket 10s
        community_cards = [Card(7, 'diamonds'), Card(2, 'clubs'), Card(14, 'spades')]  # 7-2-A flop
        
        info_set = TwoPlayerInformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history="",
            pot_size=60,
            to_call=0,
            stack_size=950,
            position=1
        )
        
        key2 = info_set.get_key()
        print(f"  Test key 2 (flop pair): {key2}")
        print(f"  In model? {key2 in cfr_ai.nodes}")
        
        # Check if any keys match patterns
        print(f"\nKeys starting with '2p_preflop': {len([k for k in cfr_ai.nodes.keys() if k.startswith('2p_preflop')])}")
        print(f"Keys starting with '2p_flop': {len([k for k in cfr_ai.nodes.keys() if k.startswith('2p_flop')])}")
        print(f"Keys with 'btn': {len([k for k in cfr_ai.nodes.keys() if 'btn' in k])}")
        print(f"Keys with 'bb': {len([k for k in cfr_ai.nodes.keys() if 'bb' in k])}")
        
        return cfr_ai
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_key_generation_scenarios():
    """Test various key generation scenarios"""
    print("\nTesting key generation for common scenarios...")
    
    # Common 2-player scenarios
    scenarios = [
        {
            'name': 'Button preflop raise',
            'hole_cards': [Card(14, 'hearts'), Card(13, 'hearts')],
            'community_cards': [],
            'betting_history': '',
            'pot_size': 30,
            'to_call': 20,
            'position': 0  # Button
        },
        {
            'name': 'BB preflop call',
            'hole_cards': [Card(9, 'spades'), Card(8, 'spades')],
            'community_cards': [],
            'betting_history': 'C',
            'pot_size': 40,
            'to_call': 0,
            'position': 1  # Big blind
        },
        {
            'name': 'Flop continuation bet',
            'hole_cards': [Card(12, 'hearts'), Card(12, 'spades')],
            'community_cards': [Card(7, 'diamonds'), Card(5, 'clubs'), Card(2, 'hearts')],
            'betting_history': 'CR',
            'pot_size': 80,
            'to_call': 30,
            'position': 1
        }
    ]
    
    for scenario in scenarios:
        info_set = TwoPlayerInformationSet(
            hole_cards=scenario['hole_cards'],
            community_cards=scenario['community_cards'],
            betting_history=scenario['betting_history'],
            pot_size=scenario['pot_size'],
            to_call=scenario['to_call'],
            stack_size=1000,
            position=scenario['position']
        )
        
        key = info_set.get_key()
        print(f"  {scenario['name']}: {key}")

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING 2-PLAYER CFR KEY MATCHING")
    print("=" * 60)
    
    debug_loaded_model()
    test_key_generation_scenarios()