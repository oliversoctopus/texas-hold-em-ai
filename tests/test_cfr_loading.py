"""
Test CFR model loading functionality
"""

print("Testing CFR model loading...")

# Test loading the CFR model directly
from cfr_poker import CFRPokerAI
from cfr_player import CFRPlayer

try:
    print("1. Loading CFR model...")
    cfr_ai = CFRPokerAI()
    cfr_ai.load("cfr_poker_v2.pkl")
    print(f"   [OK] CFR model loaded with {len(cfr_ai.nodes)} information sets")
    
    print("2. Creating CFR player...")
    cfr_player = CFRPlayer(cfr_ai)
    print(f"   [OK] CFR player created: {cfr_player.name}")
    
    print("3. Testing action selection...")
    from card_deck import Card
    
    test_state = {
        'hole_cards': [Card('A', 'spades'), Card('K', 'spades')],
        'community_cards': [Card('Q', 'hearts'), Card('J', 'diamonds'), Card('10', 'clubs')],
        'pot_size': 100,
        'to_call': 20,
        'stack_size': 1000,
        'position': 0,
        'num_players': 4,
        'action_history': []
    }
    
    action = cfr_player.choose_action(test_state)
    print(f"   [OK] Action selected: {action}")
    
    print("\n[SUCCESS] CFR loading functionality works correctly!")
    print("The main.py fix should now work for loading CFR models.")
    
except Exception as e:
    print(f"[ERROR] CFR loading failed: {e}")
    import traceback
    traceback.print_exc()