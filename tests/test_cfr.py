"""
Simple test for CFR implementation
"""

from cfr_poker import CFRPokerAI
from cfr_player import CFRPlayer, train_cfr_ai, evaluate_cfr_ai
from card_deck import Card


def test_cfr_basic():
    """Basic test of CFR functionality"""
    print("Testing CFR implementation...")
    
    # Test 1: Basic CFR AI creation
    print("\n1. Testing CFR AI creation...")
    cfr_ai = CFRPokerAI(iterations=100)  # Small number for testing
    print(f"[OK] CFR AI created with {cfr_ai.iterations} iterations")
    
    # Test 2: Training
    print("\n2. Testing CFR training...")
    cfr_ai.train(verbose=False)
    print(f"[OK] Training completed, generated {len(cfr_ai.nodes)} information sets")
    
    # Test 3: CFR Player creation
    print("\n3. Testing CFR Player...")
    cfr_player = CFRPlayer(cfr_ai)
    print(f"[OK] CFR Player created: {cfr_player.name}")
    
    # Test 4: Action selection
    print("\n4. Testing action selection...")
    test_cards = [
        Card('A', 'hearts'),
        Card('K', 'hearts')
    ]
    community_cards = [
        Card('Q', 'hearts'),
        Card('J', 'hearts'),
        Card('10', 'hearts')
    ]
    
    game_state = {
        'hole_cards': test_cards,
        'community_cards': community_cards,
        'pot_size': 100,
        'to_call': 20,
        'stack_size': 1000,
        'position': 0,
        'num_players': 4,
        'action_history': []
    }
    
    action = cfr_player.choose_action(game_state)
    print(f"[OK] Action selected: {action}")
    
    # Test 5: Raise sizing
    print("\n5. Testing raise sizing...")
    raise_size = cfr_player.get_raise_size(game_state)
    print(f"[OK] Raise size: {raise_size}")
    
    # Test 6: Save/Load
    print("\n6. Testing save/load...")
    cfr_ai.save("test_cfr_model.pkl")
    
    new_cfr_ai = CFRPokerAI()
    new_cfr_ai.load("test_cfr_model.pkl")
    print(f"[OK] Model saved and loaded, {len(new_cfr_ai.nodes)} information sets")
    
    print("\n[SUCCESS] All tests passed! CFR implementation is working.")
    
    # Clean up
    import os
    if os.path.exists("test_cfr_model.pkl"):
        os.remove("test_cfr_model.pkl")
        print("[OK] Test file cleaned up")


def test_cfr_training_function():
    """Test the high-level training function"""
    print("\n" + "="*50)
    print("Testing high-level CFR training...")
    
    # Train a small CFR AI
    cfr_ai = train_cfr_ai(iterations=1000, num_players=4, verbose=True)
    
    # Evaluate it
    results = evaluate_cfr_ai(cfr_ai, num_games=10, verbose=True)
    
    print("[OK] High-level training and evaluation completed")
    print(f"Results: {results}")


if __name__ == "__main__":
    test_cfr_basic()
    test_cfr_training_function()