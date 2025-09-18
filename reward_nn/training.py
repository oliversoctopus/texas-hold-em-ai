"""
Training system for reward-based neural network AI
"""

import torch
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from core.game_engine import TexasHoldEmTraining
from core.player import Player
from core.game_constants import Action
from core.card_deck import evaluate_hand
from .reward_based_ai import RewardBasedAI
from dqn.poker_ai import PokerAI


class RewardBasedTrainer:
    """Trainer for reward-based AI using self-play and PPO"""

    def __init__(self, ai_model: RewardBasedAI, num_players: int = 2,
                 big_blind: int = 20, starting_chips: int = 1000):
        """Initialize trainer"""
        self.ai_model = ai_model
        self.num_players = num_players
        self.big_blind = big_blind
        self.starting_chips = starting_chips

        # Training statistics
        self.hand_count = 0
        self.episode_rewards = []
        self.win_rates = []
        self.bb_per_hand = []

        # Opponent models for mixed training
        self.opponent_models = []
        self._load_opponent_models()

    def _load_opponent_models(self):
        """Load existing models as opponents for diversity"""
        # Try to load some DQN models if available
        model_paths = [
            'models/dqn/tuned_ai_v2.pth',
            'models/dqn/tuned_ai_v4.pth',
            'models/dqn/poker_ai_tuned.pth'
        ]

        for path in model_paths:
            try:
                import os
                if os.path.exists(path):
                    model = PokerAI()
                    model.load(path)
                    model.epsilon = 0
                    self.opponent_models.append(('DQN', model))
                    break  # Just load one for now
            except:
                pass

    def train(self, num_hands: int = 1000, batch_size: int = 32,
              update_every: int = 10, verbose: bool = True):
        """Train the AI for a specified number of hands"""
        if verbose:
            print(f"Training Reward-Based AI for {num_hands} hands...")
            print(f"Players: {self.num_players}, Big Blind: ${self.big_blind}")
            print("-" * 60)

        start_time = time.time()
        wins = 0
        total_bb_won = 0
        hands_per_update = []

        for hand_idx in range(num_hands):
            # Play a hand and collect experiences
            hand_reward_bb, won = self.simulate_hand(hand_idx)

            # Track statistics
            total_bb_won += hand_reward_bb
            if won:
                wins += 1

            self.bb_per_hand.append(hand_reward_bb)
            self.hand_count += 1

            # Train every N hands
            if (hand_idx + 1) % update_every == 0:
                self.ai_model.train_on_batch(batch_size=batch_size, epochs=4)
                hands_per_update.append(hand_idx + 1)

            # Progress update
            if verbose and (hand_idx + 1) % 100 == 0:
                win_rate = (wins / (hand_idx + 1)) * 100
                avg_bb = total_bb_won / (hand_idx + 1)
                elapsed = time.time() - start_time
                hands_per_sec = (hand_idx + 1) / elapsed

                print(f"Hands: {hand_idx + 1}/{num_hands} | "
                      f"Win Rate: {win_rate:.1f}% | "
                      f"Avg BB/hand: {avg_bb:+.2f} | "
                      f"Speed: {hands_per_sec:.1f} hands/sec")

                # Clear old experiences to prevent memory issues
                if len(self.ai_model.memory) > 5000:
                    # Keep only recent experiences
                    recent_experiences = list(self.ai_model.memory)[-2500:]
                    self.ai_model.memory.clear()
                    self.ai_model.memory.extend(recent_experiences)

        # Final statistics
        if verbose:
            total_time = time.time() - start_time
            final_win_rate = (wins / num_hands) * 100
            final_avg_bb = total_bb_won / num_hands

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Total Hands: {num_hands}")
            print(f"Win Rate: {final_win_rate:.1f}%")
            print(f"Avg BB/hand: {final_avg_bb:+.3f}")
            print(f"Training Time: {total_time:.1f} seconds")
            print(f"Hands/second: {num_hands / total_time:.1f}")

            # Show loss trend if available
            if self.ai_model.loss_history:
                recent_loss = np.mean(self.ai_model.loss_history[-10:])
                print(f"Recent Avg Loss: {recent_loss:.4f}")

    def simulate_hand(self, hand_idx: int) -> Tuple[float, bool]:
        """Simulate a single hand and return reward in big blinds"""
        game = TexasHoldEmTraining(num_players=self.num_players)

        # Setup players first, before reset
        players = []
        for i in range(self.num_players):
            player = Player(f"Player_{i}", self.starting_chips, is_ai=True)

            if i == 0:
                # Our training AI
                player.ai_model = self.ai_model
            else:
                # Opponent - mix of self-play, random, and other models
                opponent_type = self._choose_opponent(hand_idx)

                if opponent_type == 'self':
                    player.ai_model = self.ai_model
                elif opponent_type == 'random':
                    player.ai_model = self._create_random_ai()
                elif opponent_type == 'model' and self.opponent_models:
                    _, model = random.choice(self.opponent_models)
                    player.ai_model = model
                else:
                    player.ai_model = self._create_random_ai()

            players.append(player)

        # Set players before reset
        game.players = players
        # Don't call reset_game as it clears players

        # Deal cards
        for player in game.players:
            player.hand = game.deck.draw(2)
            player.reset_hand()

        # Post blinds
        sb_pos = (game.dealer_position + 1) % self.num_players
        bb_pos = (game.dealer_position + 2) % self.num_players
        sb_bet = game.players[sb_pos].bet(self.big_blind // 2)
        bb_bet = game.players[bb_pos].bet(self.big_blind)
        game.pot = sb_bet + bb_bet  # Start with clean integer pot
        game.current_bet = self.big_blind

        initial_chips = game.players[0].chips

        # Play through streets
        streets = ['preflop', 'flop', 'turn', 'river']
        for street_idx, street in enumerate(streets):
            # Deal community cards
            if street == 'flop':
                game.community_cards.extend(game.deck.draw(3))
            elif street in ['turn', 'river']:
                game.community_cards.append(game.deck.draw())

            # Betting round
            self.betting_round(game, street_idx)

            # Check if hand is over
            active = [p for p in game.players if not p.folded]
            if len(active) <= 1:
                break

        # Determine winner and calculate reward
        winners = self.determine_winners(game)
        our_player = game.players[0]

        # Calculate reward in big blinds
        final_chips = our_player.chips
        chips_won = final_chips - initial_chips
        reward_bb = chips_won / self.big_blind

        # Store experience with reward
        if hasattr(our_player.ai_model, 'remember'):
            # Create final state
            final_state = our_player.ai_model.get_state_features(
                our_player.hand, game.community_cards, game.pot, 0,
                our_player.chips, 0, self.num_players, 1, 0,
                game.action_history, [], hand_phase=3
            )

            # Store with the BB-based reward
            our_player.ai_model.remember(
                final_state, Action.CHECK, reward_bb, None, True
            )

        won = our_player in winners
        return reward_bb, won

    def betting_round(self, game: TexasHoldEmTraining, street_idx: int):
        """Execute a betting round"""
        if not game.players or len(game.players) == 0:
            return

        num_active_players = len(game.players)
        first_to_act = (game.dealer_position + 3) % num_active_players if street_idx == 0 \
            else (game.dealer_position + 1) % num_active_players

        current = first_to_act
        players_acted = set()
        last_raiser = None

        for _ in range(num_active_players * 3):  # Max iterations
            if current >= num_active_players:
                current = current % num_active_players
            player = game.players[current]

            if player.folded or player.all_in or player.chips == 0:
                current = (current + 1) % num_active_players
                continue

            # Check if betting is complete
            if player.current_bet >= game.current_bet and player in players_acted:
                if last_raiser is None or player == last_raiser:
                    break

            # Get state for the AI
            state = player.ai_model.get_state_features(
                player.hand, game.community_cards, game.pot, game.current_bet,
                player.chips, player.current_bet, self.num_players,
                sum(1 for p in game.players if not p.folded),
                current, game.action_history[-10:], [], hand_phase=street_idx
            )

            # Get valid actions
            valid_actions = self.get_valid_actions(game, player)

            # Choose action
            action = player.ai_model.choose_action(state, valid_actions, training=True)

            # Execute action
            old_bet = player.current_bet
            self.execute_action(game, player, action)

            # Store experience
            if hasattr(player.ai_model, 'remember') and player == game.players[0]:
                next_state = player.ai_model.get_state_features(
                    player.hand, game.community_cards, game.pot, game.current_bet,
                    player.chips, player.current_bet, self.num_players,
                    sum(1 for p in game.players if not p.folded),
                    current, game.action_history[-10:], [], hand_phase=street_idx
                )

                # Intermediate reward (small shaping rewards)
                intermediate_reward = 0
                if action == Action.FOLD and player.total_invested < self.big_blind * 2:
                    intermediate_reward = -0.1  # Small penalty for early fold
                elif action in [Action.RAISE, Action.ALL_IN] and player.current_bet > old_bet:
                    intermediate_reward = 0.05  # Small bonus for aggression

                player.ai_model.remember(state, action, intermediate_reward, next_state, False)

            # Update tracking
            game.action_history.append(action)
            players_acted.add(player)

            if action in [Action.RAISE, Action.ALL_IN] and player.current_bet > old_bet:
                last_raiser = player
                players_acted = {player}

            # Check for end conditions
            active = [p for p in game.players if not p.folded and not p.all_in]
            if len(active) <= 1 or sum(1 for p in game.players if not p.folded) <= 1:
                break

            current = (current + 1) % num_active_players

    def get_valid_actions(self, game, player) -> List[Action]:
        """Get valid actions for a player"""
        valid = []

        if player.current_bet < game.current_bet:
            valid.append(Action.FOLD)
            if player.chips > 0:
                valid.append(Action.CALL)
                if player.chips > game.current_bet - player.current_bet:
                    valid.append(Action.RAISE)
        else:
            valid.append(Action.CHECK)
            if player.chips > 0:
                valid.append(Action.RAISE)

        if player.chips > 0:
            valid.append(Action.ALL_IN)

        return valid

    def execute_action(self, game, player, action):
        """Execute a player's action"""
        if action == Action.FOLD:
            player.folded = True
        elif action == Action.CHECK:
            pass  # No action needed
        elif action == Action.CALL:
            amount = min(game.current_bet - player.current_bet, player.chips)
            actual_bet = player.bet(amount)
            game.pot += actual_bet
            game.pot = int(round(game.pot))  # Ensure pot stays integer
        elif action == Action.RAISE:
            # Calculate how much we need to call first
            call_amount = game.current_bet - player.current_bet

            # Get the raise amount from AI (this returns the raise amount, not total)
            raise_amount = player.ai_model.get_raise_size(
                None, game.pot, game.current_bet, player.chips,
                player.current_bet, self.big_blind
            )

            # Total amount to bet is call + raise
            total_bet = min(call_amount + raise_amount, player.chips)

            # Execute the bet
            actual_bet = player.bet(total_bet)
            game.pot += actual_bet
            game.pot = int(round(game.pot))  # Ensure pot stays integer
            game.current_bet = player.current_bet
        elif action == Action.ALL_IN:
            actual_bet = player.bet(player.chips)
            game.pot += actual_bet
            game.pot = int(round(game.pot))  # Ensure pot stays integer
            if player.current_bet > game.current_bet:
                game.current_bet = player.current_bet
            player.all_in = True

    def determine_winners(self, game) -> List[Player]:
        """Determine the winners of a hand"""
        active_players = [p for p in game.players if not p.folded]

        if len(active_players) == 1:
            return active_players

        # Evaluate hands
        player_hands = []
        for player in active_players:
            if player.hand:
                all_cards = player.hand + game.community_cards
                hand_value = evaluate_hand(all_cards)
                player_hands.append((player, hand_value))

        if not player_hands:
            return active_players

        # Find best hand
        best_value = max(hand[1] for hand in player_hands)
        winners = [player for player, value in player_hands if value == best_value]

        # Distribute pot
        if winners:
            # Use integer division to avoid fractional chips
            split_pot = game.pot // len(winners)
            remainder = game.pot % len(winners)

            for i, winner in enumerate(winners):
                # Give the remainder chips to first winners
                if i < remainder:
                    winner.chips += split_pot + 1
                else:
                    winner.chips += split_pot

        return winners

    def _choose_opponent(self, hand_idx: int) -> str:
        """Choose opponent type based on curriculum"""
        # Start with more random, gradually increase self-play
        progress = min(1.0, hand_idx / 1000)

        weights = {
            'random': 0.3 * (1 - progress) + 0.1,
            'self': 0.5 + 0.3 * progress,
            'model': 0.2 if self.opponent_models else 0
        }

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        # Sample opponent type
        rand = random.random()
        cumulative = 0
        for opponent_type, weight in weights.items():
            cumulative += weight
            if rand < cumulative:
                return opponent_type

        return 'random'

    def _create_random_ai(self):
        """Create a random AI opponent"""
        class RandomAI:
            def __init__(self):
                self.epsilon = 0

            def get_state_features(self, *args, **kwargs):
                return {}

            def choose_action(self, state, valid_actions, **kwargs):
                # Prefer passive play
                weights = []
                for action in valid_actions:
                    if action == Action.FOLD:
                        weights.append(0.1)
                    elif action == Action.CHECK:
                        weights.append(0.4)
                    elif action == Action.CALL:
                        weights.append(0.3)
                    elif action == Action.RAISE:
                        weights.append(0.15)
                    elif action == Action.ALL_IN:
                        weights.append(0.05)

                # Normalize and sample
                total = sum(weights)
                weights = [w/total for w in weights]
                return np.random.choice(valid_actions, p=weights)

            def get_raise_size(self, *args, **kwargs):
                return random.randint(40, 100)

            def remember(self, *args, **kwargs):
                pass

        return RandomAI()