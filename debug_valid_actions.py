"""
Debug what valid actions are actually being passed to AlwaysFoldPlayer
"""

from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action
import random

class DebugFoldPlayer(Player):
    """A player that logs everything about valid actions"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)
        self.action_log = []
        self.call_count = 0

    def choose_action(self, game_state):
        self.call_count += 1

        # Log raw game_state
        print(f"\n{self.name} - choose_action call #{self.call_count}")
        print(f"  Raw game_state type: {type(game_state)}")
        print(f"  game_state keys: {game_state.keys() if isinstance(game_state, dict) else 'Not a dict!'}")

        valid_actions = game_state.get('valid_actions', [])
        print(f"  valid_actions type: {type(valid_actions)}")
        print(f"  valid_actions content: {valid_actions}")
        print(f"  valid_actions names: {[a.name if hasattr(a, 'name') else str(a) for a in valid_actions]}")

        # Check if FOLD is in valid_actions
        has_fold = Action.FOLD in valid_actions
        print(f"  Action.FOLD in valid_actions: {has_fold}")

        # Try to fold
        if Action.FOLD in valid_actions:
            print(f"  -> Choosing FOLD")
            return Action.FOLD
        elif Action.CHECK in valid_actions:
            print(f"  -> Cannot fold, choosing CHECK")
            return Action.CHECK
        else:
            choice = valid_actions[0] if valid_actions else Action.CHECK
            print(f"  -> No fold/check available, choosing {choice.name if hasattr(choice, 'name') else choice}")
            return choice

    def get_raise_size(self, *args, **kwargs):
        print(f"  WARNING: {self.name}.get_raise_size called!")
        return 20

class AllInPlayer(Player):
    """A player that always goes all-in"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])
        if Action.ALL_IN in valid_actions:
            return Action.ALL_IN
        elif Action.RAISE in valid_actions:
            return Action.RAISE
        elif Action.CALL in valid_actions:
            return Action.CALL
        elif Action.CHECK in valid_actions:
            return Action.CHECK
        else:
            return valid_actions[0] if valid_actions else Action.CHECK

    def get_raise_size(self, *args, **kwargs):
        return 1000  # Go all-in

print("\n" + "="*60)
print("TEST: Debug valid_actions in 2-player game")
print("="*60)

# Create game
game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=True)

# Create players
folder = DebugFoldPlayer("DebugFolder", 1000)
aggressive = AllInPlayer("AllInPlayer", 1000)

game.players = [folder, aggressive]

print(f"\nInitial setup:")
print(f"  Player 0: {folder.name} (chips: {folder.chips})")
print(f"  Player 1: {aggressive.name} (chips: {aggressive.chips})")

# Play one hand
print(f"\nPlaying hand...")
winner_idx = game.play_hand(verbose=True)

print(f"\n" + "="*60)
print(f"RESULTS:")
print(f"  Final chips: {[p.chips for p in game.players]}")
print(f"  Winner: {game.players[winner_idx].name if winner_idx is not None else 'None'}")