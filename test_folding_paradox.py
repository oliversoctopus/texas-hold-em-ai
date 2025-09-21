"""
Test to understand why a folding AI can win 30% of games
"""

import random
from core.game_engine import TexasHoldEm
from core.player import Player
from core.game_constants import Action

class AlwaysFoldPlayer(Player):
    """A player that always folds"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])
        if Action.FOLD in valid_actions:
            return Action.FOLD
        elif Action.CHECK in valid_actions:
            return Action.CHECK
        else:
            return valid_actions[0] if valid_actions else Action.CHECK

    def get_raise_size(self, *args, **kwargs):
        return 20

class AggressivePlayer(Player):
    """A player that plays aggressively"""
    def __init__(self, name, chips):
        super().__init__(name, chips, is_ai=True)

    def choose_action(self, game_state):
        valid_actions = game_state.get('valid_actions', [])
        # 40% raise, 40% call, 20% fold
        choices = []
        if Action.RAISE in valid_actions:
            choices.extend([Action.RAISE] * 4)
        if Action.CALL in valid_actions:
            choices.extend([Action.CALL] * 4)
        if Action.CHECK in valid_actions:
            choices.extend([Action.CHECK] * 3)
        if Action.FOLD in valid_actions:
            choices.extend([Action.FOLD] * 2)

        return random.choice(choices) if choices else valid_actions[0]

    def get_raise_size(self, game_state, pot, current_bet, player_chips, player_current_bet, min_raise):
        # Bet 50-100% of pot
        pot_size = pot + current_bet
        bet_size = int(random.uniform(0.5, 1.0) * pot_size)
        return max(min_raise, min(bet_size, player_chips))


def simulate_single_game(num_players=6, verbose=True):
    """Simulate a single game and track what happens"""

    # Create game
    game = TexasHoldEm(num_players=num_players, starting_chips=1000, verbose=False)

    # Create players - first one always folds
    players = []
    folder = AlwaysFoldPlayer("Folder", 1000)
    players.append(folder)

    for i in range(1, num_players):
        players.append(AggressivePlayer(f"Aggressive_{i}", 1000))

    game.players = players

    # Track chips before each hand
    hands_played = 0
    max_hands = 200

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting game with {num_players} players")
        print(f"Initial chips: {[p.chips for p in game.players]}")
        print(f"{'='*60}")

    while hands_played < max_hands:
        # Check end conditions
        active_players = [p for p in game.players if p.chips > 0]
        if len(active_players) <= 1:
            if verbose:
                print(f"\nGame ended: Only {len(active_players)} players with chips")
            break

        # Check for dominance
        total_chips = sum(p.chips for p in active_players)
        if total_chips > 0:
            max_chips = max(p.chips for p in active_players)
            if max_chips > 0.90 * total_chips and len(active_players) > 1:
                if verbose:
                    print(f"\nGame ended: Player dominance (90% of chips)")
                break

        # Play hand
        if verbose and hands_played < 5:  # Show first 5 hands in detail
            print(f"\n--- Hand {hands_played + 1} ---")
            chips_before = [p.chips for p in game.players]
            print(f"Chips before: {chips_before}")

        try:
            game.play_hand(verbose=False)
            hands_played += 1

            if verbose and hands_played <= 5:
                chips_after = [p.chips for p in game.players]
                print(f"Chips after:  {chips_after}")
                changes = [after - before for after, before in zip(chips_after, chips_before)]
                print(f"Changes:      {changes}")

                # Track who won the pot
                pot_winner_idx = changes.index(max(changes)) if max(changes) > 0 else -1
                if pot_winner_idx >= 0:
                    print(f"Pot winner:   {game.players[pot_winner_idx].name} (+{max(changes)} chips)")

        except Exception as e:
            if verbose:
                print(f"Error during hand: {e}")
            break

    # Determine winner
    final_chips = [p.chips for p in game.players]
    max_chips = max(final_chips)
    winner_idx = final_chips.index(max_chips) if max_chips > 0 else -1

    if verbose:
        print(f"\n{'='*60}")
        print(f"GAME RESULTS after {hands_played} hands:")
        print(f"Final chips: {final_chips}")
        print(f"Winner: {game.players[winner_idx].name if winner_idx >= 0 else 'None'}")
        print(f"Folder's final chips: {game.players[0].chips}")
        print(f"Folder won: {winner_idx == 0}")
        print(f"{'='*60}")

    return winner_idx == 0, game.players[0].chips, hands_played


def run_simulation(num_games=20, num_players=6):
    """Run multiple simulations"""

    wins = 0
    total_final_chips = 0
    total_hands = 0

    print(f"\nRunning {num_games} games with {num_players} players...")
    print("(Folder vs Aggressive players)")

    for i in range(num_games):
        folder_won, final_chips, hands = simulate_single_game(
            num_players=num_players,
            verbose=(i < 2)  # Show first 2 games in detail
        )

        if folder_won:
            wins += 1
        total_final_chips += final_chips
        total_hands += hands

    win_rate = (wins / num_games) * 100
    avg_chips = total_final_chips / num_games
    avg_hands = total_hands / num_games

    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY:")
    print(f"  Games played: {num_games}")
    print(f"  Folder win rate: {win_rate:.1f}%")
    print(f"  Folder avg final chips: {avg_chips:.1f}")
    print(f"  Average hands per game: {avg_hands:.1f}")
    print(f"  Expected random win rate: {100/num_players:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test different player counts
    for num_players in [2, 4, 6]:
        run_simulation(num_games=20, num_players=num_players)