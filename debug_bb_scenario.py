"""
Debug what happens when Folder is BB and everyone folds
"""

from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action
import random

class AlwaysFoldPlayer(Player):
    """A player that always folds when possible"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)
        self.action_log = []

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])

        # Log what we're choosing
        chosen = None
        if Action.FOLD in valid_actions:
            chosen = Action.FOLD
        elif Action.CHECK in valid_actions:
            chosen = Action.CHECK
        else:
            chosen = valid_actions[0] if valid_actions else Action.CHECK

        self.action_log.append({
            'valid': [a.name for a in valid_actions],
            'chosen': chosen.name,
            'bet_to_call': game_state.get('current_bet', 0) - game_state.get('player_bet', 0)
        })

        return chosen

    def get_raise_size(self, *args, **kwargs):
        return 20

class SpecificPlayer(Player):
    """A player with specific programmed actions"""
    def __init__(self, name, chips, actions_to_take):
        super().__init__(name, chips, is_ai=True)
        self.actions_to_take = actions_to_take  # List of actions to take in order
        self.action_index = 0

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])

        if self.action_index < len(self.actions_to_take):
            desired_action = self.actions_to_take[self.action_index]
            self.action_index += 1

            if desired_action in valid_actions:
                return desired_action
            elif Action.CHECK in valid_actions:
                return Action.CHECK
            else:
                return valid_actions[0]
        else:
            # Default to check/fold
            if Action.CHECK in valid_actions:
                return Action.CHECK
            elif Action.FOLD in valid_actions:
                return Action.FOLD
            else:
                return valid_actions[0]

    def get_raise_size(self, game_state, pot, current_bet, player_chips, player_current_bet, min_raise):
        # Always go all-in when raising
        return player_chips

def test_bb_wins_by_default():
    """Test scenario where BB wins because everyone folds"""
    print("\n" + "="*60)
    print("SCENARIO 1: BB wins when everyone folds")
    print("="*60)

    # Create a 2-player game
    game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=True)

    # Player 0 is dealer/SB, Player 1 is BB
    # We want Player 1 (BB) to be our folder
    folder = AlwaysFoldPlayer("Folder_BB", 1000)
    aggressive = SpecificPlayer("Aggressive_SB", 1000, [Action.FOLD])  # SB folds

    # In 2-player game, dealer is SB, other player is BB
    # So player 0 is SB, player 1 is BB
    game.players = [aggressive, folder]

    print(f"\nInitial chips: {[p.chips for p in game.players]}")
    print(f"Player positions: {[p.name for p in game.players]}")
    print(f"Player 0 (SB): {game.players[0].name}")
    print(f"Player 1 (BB): {game.players[1].name}")

    # Play one hand
    winner_idx = game.play_hand(verbose=True)

    print(f"\nFinal chips: {[p.chips for p in game.players]}")
    print(f"Winner: {game.players[winner_idx].name if winner_idx is not None and winner_idx >= 0 else 'None'}")

    # Check folder's action log
    for player in game.players:
        if isinstance(player, AlwaysFoldPlayer):
            print(f"\n{player.name}'s actions:")
            for i, action in enumerate(player.action_log):
                print(f"  Turn {i+1}: Valid={action['valid']}, Chosen={action['chosen']}, To_call={action['bet_to_call']}")

def test_bb_vs_all_in():
    """Test scenario where SB goes all-in against BB"""
    print("\n" + "="*60)
    print("SCENARIO 2: SB goes all-in, BB must respond")
    print("="*60)

    # Create a 2-player game
    game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=True)

    folder = AlwaysFoldPlayer("Folder", 1000)
    aggressive = SpecificPlayer("Aggressive", 1000, [Action.ALL_IN])

    # Set players - folder at position 0, aggressive at position 1
    game.players = [folder, aggressive]

    print(f"\nInitial chips: {[p.chips for p in game.players]}")
    print(f"Player positions: {[p.name for p in game.players]}")
    print(f"BB position: {game.current_bet_idx}")

    # Play one hand
    winner_idx = game.play_hand(verbose=True)

    print(f"\nFinal chips: {[p.chips for p in game.players]}")
    print(f"Winner: {game.players[winner_idx].name if winner_idx is not None and winner_idx >= 0 else 'None'}")

    # Check folder's action log
    print(f"\n{folder.name}'s actions:")
    for i, action in enumerate(folder.action_log):
        print(f"  Turn {i+1}: Valid={action['valid']}, Chosen={action['chosen']}, To_call={action['bet_to_call']}")

def test_check_check_showdown():
    """Test where both players just check to showdown"""
    print("\n" + "="*60)
    print("SCENARIO 3: Both check to showdown")
    print("="*60)

    game = TexasHoldEm(num_players=2, starting_chips=1000, verbose=True)

    folder = AlwaysFoldPlayer("Folder", 1000)
    passive = SpecificPlayer("Passive", 1000,
                            [Action.CALL,   # SB calls BB
                             Action.CHECK, Action.CHECK,  # Flop
                             Action.CHECK, Action.CHECK,  # Turn
                             Action.CHECK, Action.CHECK]) # River

    game.players = [folder, passive]

    print(f"\nInitial chips: {[p.chips for p in game.players]}")

    winner_idx = game.play_hand(verbose=True)

    print(f"\nFinal chips: {[p.chips for p in game.players]}")
    print(f"Winner: {game.players[winner_idx].name if winner_idx is not None and winner_idx >= 0 else 'None'}")

    print(f"\n{folder.name}'s actions:")
    for i, action in enumerate(folder.action_log):
        print(f"  Turn {i+1}: Valid={action['valid']}, Chosen={action['chosen']}, To_call={action['bet_to_call']}")

if __name__ == "__main__":
    # Run all test scenarios
    test_bb_wins_by_default()
    test_bb_vs_all_in()
    test_check_check_showdown()