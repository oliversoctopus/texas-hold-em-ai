"""
Debug script to understand why constant folding leads to high win rates
"""

import random
import numpy as np
from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action

class AlwaysFoldPlayer(Player):
    """A player that always folds"""
    def __init__(self, name, chips):
        super().__init__(name, chips)

    def choose_action(self, game_state):
        valid_actions = game_state['valid_actions']
        # Always fold if possible, otherwise check
        if Action.FOLD in valid_actions:
            return Action.FOLD
        elif Action.CHECK in valid_actions:
            return Action.CHECK
        else:
            return valid_actions[0]

    def get_raise_size(self, game_state, pot, current_bet, player_chips, player_current_bet, min_raise):
        return min_raise

def simulate_games(num_games=100, num_players=6):
    """Simulate games with one always-folding player"""

    wins = 0
    chips_history = []
    start_chips = 1000

    for game_num in range(num_games):
        # Create players - first one always folds, others play randomly
        players = []
        players.append(AlwaysFoldPlayer("Folder", start_chips))

        for i in range(1, num_players):
            players.append(Player(f"Random_{i}", start_chips))

        # Play one hand
        game = TexasHoldEm(players)
        winner_idx = game.play_hand(verbose=False)

        # Track results
        if winner_idx == 0:
            wins += 1

        chips_history.append(players[0].chips)

    # Calculate statistics
    win_rate = (wins / num_games) * 100
    avg_chips = np.mean(chips_history)
    chip_change = avg_chips - start_chips

    print(f"\n=== Simulation Results ===")
    print(f"Games played: {num_games}")
    print(f"Players per game: {num_players}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Average chips after game: {avg_chips:.1f}")
    print(f"Average chip change: {chip_change:+.1f}")
    print(f"BB lost per game: {chip_change/20:.2f}")

    # Theoretical analysis
    print(f"\n=== Theoretical Analysis ===")
    print(f"Expected win rate with random play: {100/num_players:.1f}%")
    print(f"Actual win rate with constant folding: {win_rate:.1f}%")

    # Track position-based folding
    print(f"\n=== Position Analysis ===")
    position_folds = {
        'BB': 0,
        'SB': 0,
        'Other': 0
    }

    for game_num in range(20):  # Sample 20 games for position analysis
        players = []
        players.append(AlwaysFoldPlayer("Folder", start_chips))
        for i in range(1, num_players):
            players.append(Player(f"Random_{i}", start_chips))

        game = TexasHoldEm(players)

        # Check folder's position
        folder_idx = 0
        if game.current_bet_idx == folder_idx:
            position = 'BB'
        elif (game.current_bet_idx - 1) % num_players == folder_idx:
            position = 'SB'
        else:
            position = 'Other'

        position_folds[position] += 1

        winner_idx = game.play_hand(verbose=False)

        # Show one example hand in detail
        if game_num == 0:
            print(f"\nExample hand (Folder in {position} position):")
            print(f"  Folder chips before: {start_chips}")
            print(f"  Folder chips after: {players[0].chips}")
            print(f"  Chips lost: {start_chips - players[0].chips}")
            print(f"  Winner: Player {winner_idx}")

    print(f"\nPosition distribution (20 games):")
    for pos, count in position_folds.items():
        print(f"  {pos}: {count}")

if __name__ == "__main__":
    # Run simulations with different player counts
    for num_players in [2, 4, 6]:
        simulate_games(num_games=100, num_players=num_players)
        print("\n" + "="*50)