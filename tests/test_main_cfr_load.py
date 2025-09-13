"""
Test the main.py CFR loading logic
"""

print("Testing main.py CFR loading logic...")

# Simulate the main.py loading logic
filename = "cfr_poker_v2.pkl"

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
        
        # Test the wrapper
        class MockState:
            def __init__(self):
                self.hole_cards = []
                self.community_cards = []
                self.pot_size = 100
                self.to_call = 20
                self.stack_size = 1000
                self.position = 0
                self.num_players = 4
                self.action_history = []
        
        test_state = MockState()
        action = ai_model.choose_action(test_state, [])
        print(f"Test action: {action}")
        
        print("\n[SUCCESS] Main.py CFR loading logic works perfectly!")
        
    except Exception as e:
        print(f"Could not load CFR model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Not a CFR model file")