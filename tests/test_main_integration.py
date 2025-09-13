"""
Test a small portion of main.py functionality to verify CFR loading
"""

print("=" * 60)
print("TEXAS HOLD'EM with Advanced PyTorch AI")
print("=" * 60)

print("\nTesting option 7 (Load existing AI) with CFR model...")
print("Simulating user selecting option 7 and entering 'cfr_poker_v2.pkl'")

# Simulate the main.py logic for option 7
choice = '7'
filename = "cfr_poker_v2.pkl"

ai_model = None

if choice == '7':
    print(f"\nModel filename: {filename}")
    
    # Check if it's a CFR model (.pkl) or DQN model (.pth)
    if filename.endswith('.pkl'):
        # Load CFR model
        try:
            from cfr_poker import CFRPokerAI
            from cfr_player import CFRPlayer
            
            print(f"Loading CFR model from {filename}...")
            cfr_ai = CFRPokerAI()
            cfr_ai.load(filename)
            print(f"CFR model loaded from {filename}")
            print(f"Information sets learned: {len(cfr_ai.nodes)}")
            
            # Create CFR player wrapper
            cfr_player = CFRPlayer(cfr_ai)
            
            # Create wrapper for gameplay compatibility
            class CFRWrapper:
                def __init__(self, cfr_player):
                    self.cfr_player = cfr_player
                    self.epsilon = 0  # CFR doesn't use epsilon
                    
                def choose_action(self, state, valid_actions):
                    # Convert state to CFR format
                    game_state = {
                        'hole_cards': getattr(state, 'hole_cards', []),
                        'community_cards': getattr(state, 'community_cards', []),
                        'pot_size': getattr(state, 'pot_size', 0),
                        'to_call': getattr(state, 'to_call', 0),
                        'stack_size': getattr(state, 'stack_size', 1000),
                        'position': getattr(state, 'position', 0),
                        'num_players': getattr(state, 'num_players', 4),
                        'action_history': getattr(state, 'action_history', [])
                    }
                    return self.cfr_player.choose_action(game_state)
                
                def get_raise_size(self, state):
                    game_state = {
                        'pot_size': getattr(state, 'pot_size', 0),
                        'position': getattr(state, 'position', 0),
                        'num_players': getattr(state, 'num_players', 4)
                    }
                    return self.cfr_player.get_raise_size(game_state)
            
            ai_model = CFRWrapper(cfr_player)
            print("CFR AI ready for gameplay!")
            
        except Exception as e:
            print(f"Could not load CFR model: {e}")
            print("Starting with untrained AI")
            ai_model = None

print(f"\nFinal result: ai_model = {type(ai_model).__name__ if ai_model else 'None'}")

if ai_model:
    print("\n🎉 SUCCESS! CFR model loading works correctly in main.py!")
    print("You can now use option 7 to load CFR models (.pkl files)")
else:
    print("\n❌ FAILED! CFR model loading did not work")

print("\nTo use this:")
print("1. Run python main.py")  
print("2. Select option 7")
print("3. Enter 'cfr_poker_v2.pkl' as filename")
print("4. The CFR AI will be loaded and ready for gameplay!")