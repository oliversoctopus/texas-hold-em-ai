"""
Simple test to see what actions AlwaysFoldPlayer actually takes
"""

from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action

class DetailedFoldPlayer(Player):
    """A player that tries to always fold but logs everything"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)
        self.action_log = []

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])

        # Log the decision process
        log_entry = {
            'valid_actions': [a.name for a in valid_actions],
            'has_fold': Action.FOLD in valid_actions,
            'has_check': Action.CHECK in valid_actions,
        }

        # Decision logic
        if Action.FOLD in valid_actions:
            chosen = Action.FOLD
            log_entry['reason'] = 'Can fold, so folding'
        elif Action.CHECK in valid_actions:
            chosen = Action.CHECK
            log_entry['reason'] = 'Cannot fold, checking instead'
        else:
            chosen = valid_actions[0] if valid_actions else Action.CHECK
            log_entry['reason'] = f'No fold/check, choosing {chosen.name}'

        log_entry['chosen'] = chosen.name
        self.action_log.append(log_entry)

        print(f"  {self.name} decision: Valid={[a.name for a in valid_actions]}, Chosen={chosen.name}, Reason={log_entry['reason']}")

        return chosen

    def get_raise_size(self, *args, **kwargs):
        print(f"  WARNING: {self.name} get_raise_size was called!")
        return 20

# Test 1: Simple 2-player game
print("\n" + "="*60)
print("TEST: 2-player game action sequence")
print("="*60)

game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=False)

folder1 = DetailedFoldPlayer("Folder1", 1000)
folder2 = DetailedFoldPlayer("Folder2", 1000)

game.players = [folder1, folder2]

print(f"Initial setup:")
print(f"  Player 0: {folder1.name}")
print(f"  Player 1: {folder2.name}")
print()

# Manually step through the betting
print("Starting betting round...")

# Get first action
current_player_idx = 0
for i in range(10):  # Limit iterations to prevent infinite loop
    current_player = game.players[current_player_idx]

    if current_player.folded:
        print(f"  {current_player.name} has already folded")
        current_player_idx = (current_player_idx + 1) % 2
        continue

    # Build game state for the player
    game_state = {
        'valid_actions': game.get_valid_actions(current_player),
        'current_bet': game.current_bet,
        'player_bet': current_player.current_bet,
        'pot': game.pot,
        'player_chips': current_player.chips
    }

    print(f"\nTurn {i+1}: {current_player.name}'s turn")
    print(f"  Current bet: ${game.current_bet}, Player bet: ${current_player.current_bet}, To call: ${game.current_bet - current_player.current_bet}")

    action = current_player.choose_action(game_state)

    # Execute action
    game.execute_action(current_player, action, verbose=True)

    # Check if betting is complete
    if all(p.folded or p.current_bet == game.current_bet for p in game.players):
        if not any(p.current_bet == 0 and not p.folded for p in game.players):
            print("\nBetting round complete!")
            break

    # Next player
    current_player_idx = (current_player_idx + 1) % 2

print(f"\nFinal state:")
print(f"  Pot: ${game.pot}")
print(f"  Player chips: {[p.chips for p in game.players]}")
print(f"  Player bets: {[p.current_bet for p in game.players]}")
print(f"  Folded: {[p.folded for p in game.players]}")

print("\n" + "="*60)
print("Action logs:")
for i, player in enumerate([folder1, folder2]):
    print(f"\n{player.name}:")
    for j, entry in enumerate(player.action_log):
        print(f"  Action {j+1}: {entry}")