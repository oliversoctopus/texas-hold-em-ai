#!/usr/bin/env python3
"""
Test script to verify the new reorganized codebase structure
"""

def test_imports():
    """Test that all imports work correctly"""
    print("Testing reorganized codebase imports...")
    
    try:
        # Test core imports
        from core import TexasHoldEm, Action, Card, Deck, Player
        print("[OK] Core imports successful")
        
        # Test DQN imports (deprecated but should work)
        from dqn import PokerAI
        print("[OK] DQN imports successful (deprecated)")
        
        # Test CFR imports  
        from cfr import CFRPokerAI, train_cfr_ai, TwoPlayerCFRPokerAI, train_two_player_cfr_ai
        print("[OK] CFR imports successful")
        
        # Test Deep CFR imports
        from deepcfr import DeepCFRPokerAI, train_deep_cfr_ai
        print("[OK] Deep CFR imports successful")
        
        print("\n[SUCCESS] All imports successful! Codebase reorganization complete.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Import classes for testing
        from core import Card, Deck, TexasHoldEm
        from cfr import CFRPokerAI, TwoPlayerCFRPokerAI
        from deepcfr import DeepCFRPokerAI
        
        # Test card creation
        card = Card(14, 'hearts')  # Ace of hearts
        deck = Deck()
        deck.reset()
        print(f"[OK] Created card: {card}")
        
        # Test game creation
        game = TexasHoldEm(num_players=2, verbose=False)
        print("[OK] Game engine created")
        
        # Test CFR AI creation (without training)
        cfr_ai = CFRPokerAI(iterations=10)  # Very small for testing
        print("[OK] CFR AI created")
        
        # Test 2-player CFR AI creation
        two_player_cfr = TwoPlayerCFRPokerAI(iterations=10)
        print("[OK] 2-Player CFR AI created")
        
        # Test Deep CFR AI creation
        deep_cfr = DeepCFRPokerAI()
        print("[OK] Deep CFR AI created")
        
        print("[OK] All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING REORGANIZED CODEBASE")
    print("=" * 60)
    
    imports_ok = test_imports()
    functionality_ok = test_functionality()
    
    if imports_ok and functionality_ok:
        print("\n[SUCCESS] SUCCESS: Codebase reorganization is working correctly!")
        print("\nNew structure:")
        print("- core/: Game engine and utilities")
        print("- dqn/: DQN implementation (deprecated, evaluation only)")
        print("- cfr/: Standard and 2-player CFR implementations") 
        print("- deepcfr/: Neural network enhanced CFR for 3+ players")
        print("- models/: Saved model files (dqn/ and cfr/ subdirectories)")
        print("- evaluation/: Evaluation frameworks")
        print("- utils/: Utility functions")
    else:
        print("\n[ERROR] FAILURE: Issues with codebase reorganization")
        print("Check the import errors above and fix them.")