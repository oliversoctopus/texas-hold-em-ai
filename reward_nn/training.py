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
                 big_blind: int = 20, starting_chips: int = 1000,
                 variable_stacks: bool = True, min_stack_bb: int = 20, max_stack_bb: int = 200,
                 progressive_stacks: bool = True, variable_player_count: bool = True):
        """Initialize trainer"""
        self.ai_model = ai_model
        self.max_players = num_players  # Maximum number of players
        self.big_blind = big_blind
        self.starting_chips = starting_chips
        self.variable_stacks = variable_stacks
        self.min_stack_bb = min_stack_bb  # Minimum stack in big blinds
        self.max_stack_bb = max_stack_bb  # Maximum stack in big blinds
        self.progressive_stacks = progressive_stacks  # Gradually increase variation
        self.variable_player_count = variable_player_count  # Train with varying player counts
        self.total_training_hands = None  # Will be set during train()

        # Training statistics
        self.hand_count = 0
        self.episode_rewards = []
        self.win_rates = []
        self.bb_per_hand = []

        # Fold tracking for penalty
        self.recent_folds = []  # Track last N hands for fold rate
        self.fold_window_size = 100  # Window for calculating fold rate
        self.excessive_fold_threshold = 0.75  # Penalty if folding > 75%
        self.fold_penalty = -1.0  # Penalty in BBs for excessive folding

        # Opponent models for mixed training
        self.opponent_models = []
        self._load_opponent_models()

        # Snapshot pool for preventing overfitting
        self.opponent_snapshots = []
        self.snapshot_interval = 200  # Save snapshot every 200 hands
        self.max_snapshots = 5  # Keep only recent snapshots to save memory

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
                    #break  # Just load one for now
            except:
                pass

    def train(self, num_hands: int = 1000, batch_size: int = 32,
              update_every: int = 10, verbose: bool = True):
        """Train the AI for a specified number of hands"""
        # Store total hands for progressive stacks calculation
        self.total_training_hands = num_hands

        if verbose:
            print(f"Training Reward-Based AI for {num_hands} hands...")
            if self.variable_player_count and self.max_players > 2:
                print(f"Players: 2-{self.max_players} (variable), Big Blind: ${self.big_blind}")
                print(f"  Training includes scenarios with fewer players (tournament-style)")
            else:
                print(f"Players: {self.max_players} (fixed), Big Blind: ${self.big_blind}")
            if self.variable_stacks:
                if self.progressive_stacks:
                    print(f"Progressive Stacks: Starting uniform, increasing variation over time")
                    print(f"Total chips scale with active players (1000 × num_players)")
                else:
                    print(f"Variable Stacks: {self.min_stack_bb}-{self.max_stack_bb} BBs")
            else:
                print(f"Fixed Stack: {self.starting_chips // self.big_blind} BBs")
            print("-" * 60)

        start_time = time.time()
        wins = 0
        total_bb_won = 0
        hands_per_update = []
        total_all_ins = 0  # Track all-in frequency
        total_folds = 0  # Track fold frequency

        # Rolling window tracking for display stats (last 100 hands)
        recent_wins = []  # 1 for win, 0 for loss
        recent_bb = []  # BB won/lost per hand
        recent_all_ins = []  # 1 if went all-in, 0 otherwise
        recent_folds_display = []  # 1 if folded, 0 otherwise
        window_size = 100  # Display stats for last 100 hands

        for hand_idx in range(num_hands):
            # Track all-ins and folds for this hand
            self._current_training_all_ins = 0
            self._current_training_folded = False

            # Play a hand and collect experiences
            hand_reward_bb, won = self.simulate_hand(hand_idx)

            # Update total all-in and fold tracking
            if self._current_training_all_ins > 0:
                total_all_ins += 1
            if self._current_training_folded:
                total_folds += 1

            # Track statistics
            total_bb_won += hand_reward_bb
            if won:
                wins += 1

            # Track for rolling windows
            recent_wins.append(1 if won else 0)
            recent_bb.append(hand_reward_bb)
            recent_all_ins.append(1 if self._current_training_all_ins > 0 else 0)
            recent_folds_display.append(1 if self._current_training_folded else 0)

            # Keep windows at max size
            if len(recent_wins) > window_size:
                recent_wins.pop(0)
            if len(recent_bb) > window_size:
                recent_bb.pop(0)
            if len(recent_all_ins) > window_size:
                recent_all_ins.pop(0)
            if len(recent_folds_display) > window_size:
                recent_folds_display.pop(0)

            self.bb_per_hand.append(hand_reward_bb)
            self.hand_count += 1

            # Train every N hands
            if (hand_idx + 1) % update_every == 0:
                self.ai_model.train_on_batch(batch_size=batch_size, epochs=4)
                hands_per_update.append(hand_idx + 1)

            # Save snapshot for opponent pool
            if (hand_idx + 1) % self.snapshot_interval == 0:
                self._save_snapshot()

            # Progress update
            if verbose and (hand_idx + 1) % 100 == 0:
                # Calculate stats for the recent window
                recent_win_rate = (sum(recent_wins) / len(recent_wins)) * 100 if recent_wins else 0
                recent_avg_bb = sum(recent_bb) / len(recent_bb) if recent_bb else 0
                recent_all_in_rate = (sum(recent_all_ins) / len(recent_all_ins)) * 100 if recent_all_ins else 0
                recent_fold_rate = (sum(recent_folds_display) / len(recent_folds_display)) * 100 if recent_folds_display else 0

                elapsed = time.time() - start_time
                hands_per_sec = (hand_idx + 1) / elapsed

                print(f"Hands: {hand_idx + 1}/{num_hands} | "
                      f"Win Rate: {recent_win_rate:.1f}% | "
                      f"Avg BB/hand: {recent_avg_bb:+.2f} | "
                      f"Fold: {recent_fold_rate:.1f}% | "
                      f"All-in: {recent_all_in_rate:.1f}% | "
                      f"Speed: {hands_per_sec:.1f} hands/sec")
                print(f"  (Last {len(recent_wins)} hands)")

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
            final_all_in_rate = (total_all_ins / num_hands) * 100
            final_fold_rate = (total_folds / num_hands) * 100

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Total Hands: {num_hands}")
            print(f"Overall Statistics:")
            print(f"  Win Rate: {final_win_rate:.1f}%")
            print(f"  Avg BB/hand: {final_avg_bb:+.3f}")
            print(f"  Fold Rate: {final_fold_rate:.1f}%")
            print(f"  All-in Rate: {final_all_in_rate:.1f}%")
            print(f"Training Time: {total_time:.1f} seconds")
            print(f"Hands/second: {num_hands / total_time:.1f}")

            # Show loss trend if available
            if self.ai_model.loss_history:
                recent_loss = np.mean(self.ai_model.loss_history[-10:])
                print(f"Recent Avg Loss: {recent_loss:.4f}")

    def _get_stack_distribution(self, hand_idx: int, num_players: int) -> List[int]:
        """Get stack distribution for the current hand with progressive variation

        Args:
            hand_idx: Current hand index for progression calculation
            num_players: Actual number of players in this hand (may be less than max)
        """
        if not self.variable_stacks:
            # Fixed stacks for all players
            return [self.starting_chips] * num_players

        # Total chips scales with actual number of players in this hand
        total_chips = 1000 * num_players

        # Calculate progress for progressive variation
        if self.progressive_stacks and self.total_training_hands:
            # Progressive variation based on training progress
            # Full variation after 75% of training
            variation_threshold = max(1000, self.total_training_hands * 0.75)  # At least 1000 hands
            progress = min(1.0, hand_idx / variation_threshold)
        elif self.progressive_stacks:
            # Fallback if total_training_hands not set
            progress = min(1.0, hand_idx / 1000)  # Default to 1000 hands
        else:
            progress = 1.0  # Full variation immediately if not progressive

        # For 2 players, use the original logic
        if num_players == 2:
            # Determine stack scenario based on hand index
            scenario_cycle = hand_idx % 100  # Cycle through scenarios

            if scenario_cycle < 25 or progress < 0.1:
                # Uniform stacks (first 25% of cycle or very early training)
                stack1 = total_chips // 2
                stack2 = total_chips - stack1
            elif scenario_cycle < 50:
                # Small stack vs big stack scenario
                # Variation increases with progress
                variation = int(400 * progress)  # Max 400 chip difference
                stack1 = (total_chips // 2) - variation
                stack2 = total_chips - stack1
            elif scenario_cycle < 75:
                # Medium variation scenario
                variation = int(200 * progress)  # Max 200 chip difference
                if random.random() < 0.5:
                    stack1 = (total_chips // 2) + variation
                    stack2 = total_chips - stack1
                else:
                    stack1 = (total_chips // 2) - variation
                    stack2 = total_chips - stack1
            else:
                # Random variation (but controlled by progress)
                max_variation = int(600 * progress)  # Max 600 chip difference
                variation = random.randint(-max_variation, max_variation) if max_variation > 0 else 0
                stack1 = (total_chips // 2) + variation
                # Ensure minimum stack size
                stack1 = max(200, min(1800, stack1))  # At least 10BB, at most 90BB
                stack2 = total_chips - stack1

            return [stack1, stack2]

        # For 3+ players, implement progressive variation
        else:
            avg_stack = total_chips // num_players
            min_stack = int(avg_stack * 0.2)  # Minimum 20% of average

            # Maximum possible stack (leaving minimum for others)
            theoretical_max = total_chips - (min_stack * (num_players - 1))

            scenario_cycle = hand_idx % 100  # Cycle through scenarios

            if scenario_cycle < 25 or progress < 0.1:
                # Uniform stacks early in training
                return [avg_stack] * num_players

            elif scenario_cycle < 50:
                # Chip leader vs others scenario with exponentially decreasing probability for extreme stacks
                stacks = []

                # Determine leader stack size with exponential distribution
                # Use random to determine how extreme the chip leader is
                extremeness = random.random()  # 0 to 1

                if extremeness < 0.5:
                    # 50% chance: moderate chip leader (1.5-2.5x average)
                    leader_multiplier = 1.5 + random.random() * 1.0 * progress
                elif extremeness < 0.8:
                    # 30% chance: strong chip leader (2.5-4x average)
                    leader_multiplier = 2.5 + random.random() * 1.5 * progress
                elif extremeness < 0.95:
                    # 15% chance: dominant chip leader (4-6x average)
                    leader_multiplier = 4.0 + random.random() * 2.0 * progress
                else:
                    # 5% chance: extreme chip leader (6x-near max)
                    # For 6 players, this could be up to ~80% of all chips
                    leader_multiplier = 6.0 + random.random() * (theoretical_max / avg_stack - 6.0) * progress

                leader_chips = int(avg_stack * leader_multiplier)
                leader_chips = min(leader_chips, theoretical_max)
                leader_chips = max(leader_chips, avg_stack)  # At least average
                stacks.append(leader_chips)

                # Distribute remaining chips among other players
                remaining_chips = total_chips - leader_chips
                remaining_players = num_players - 1

                for i in range(remaining_players):
                    if i == remaining_players - 1:
                        # Last player gets whatever is left
                        stacks.append(remaining_chips)
                    else:
                        # Random variation for middle stacks
                        avg_remaining = remaining_chips / (remaining_players - i)
                        variation_factor = 0.3 * progress
                        stack = int(avg_remaining * random.uniform(1 - variation_factor, 1 + variation_factor))
                        stack = max(min_stack, min(stack, remaining_chips - min_stack * (remaining_players - i - 1)))
                        stacks.append(stack)
                        remaining_chips -= stack

                random.shuffle(stacks)  # Randomize positions
                return stacks

            elif scenario_cycle < 75:
                # Short stack vs medium stacks scenario (including extreme short stack scenarios)
                stacks = []

                # Occasionally create extreme scenarios where AI is very short
                short_extremeness = random.random()

                if short_extremeness < 0.1 and progress > 0.5:
                    # 10% chance: AI gets very short stack (10-20% of average)
                    num_short = 1
                    short_stack = int(avg_stack * (0.1 + 0.1 * random.random()))
                    short_stack = max(min_stack, short_stack)
                    stacks.append(short_stack)
                else:
                    # Normal short stack scenario
                    num_short = min(2, num_players - 1)
                    for _ in range(num_short):
                        short_stack = int(avg_stack * (0.3 + 0.3 * (1 - progress)))  # 30-60% of average
                        short_stack = max(min_stack, short_stack)
                        stacks.append(short_stack)

                # Distribute remaining chips among other players
                remaining_chips = total_chips - sum(stacks)
                remaining_players = num_players - num_short

                for i in range(remaining_players):
                    if i == remaining_players - 1:
                        stacks.append(remaining_chips)
                    else:
                        avg_remaining = remaining_chips / (remaining_players - i)
                        # Add some variation
                        stack = int(avg_remaining * random.uniform(0.7, 1.3))
                        stack = max(min_stack, min(stack, remaining_chips - min_stack * (remaining_players - i - 1)))
                        stacks.append(stack)
                        remaining_chips -= stack

                random.shuffle(stacks)  # Randomize positions
                return stacks

            else:
                # Random variation scenario with possibility of extreme stacks
                stacks = []
                remaining_chips = total_chips

                # Small chance for extreme random distributions
                if random.random() < 0.05 and progress > 0.5:
                    # 5% chance: Create one very large stack randomly
                    extreme_multiplier = 3.0 + random.random() * (theoretical_max / avg_stack - 3.0) * 0.7
                    extreme_stack = int(avg_stack * extreme_multiplier)
                    extreme_stack = min(extreme_stack, theoretical_max)
                    extreme_position = random.randint(0, num_players - 1)

                    for i in range(num_players):
                        if i == extreme_position:
                            stacks.append(extreme_stack)
                            remaining_chips -= extreme_stack
                        elif i == num_players - 1:
                            stacks.append(remaining_chips)
                        else:
                            avg_remaining = remaining_chips / (num_players - len(stacks))
                            stack = int(avg_remaining * random.uniform(0.5, 1.5))
                            stack = max(min_stack, min(stack, remaining_chips - min_stack * (num_players - len(stacks) - 1)))
                            stacks.append(stack)
                            remaining_chips -= stack
                else:
                    # Normal random distribution
                    for i in range(num_players):
                        if i == num_players - 1:
                            # Last player gets whatever is left
                            stacks.append(remaining_chips)
                        else:
                            # Random stack with progressive variation
                            avg_remaining = remaining_chips / (num_players - i)
                            variation_range = 0.6 * progress  # Up to 60% variation when fully progressed

                            stack = int(avg_remaining * random.uniform(1 - variation_range, 1 + variation_range))
                            # Allow stacks to go higher but with decreasing probability
                            if random.random() < 0.1:  # 10% chance for larger variation
                                stack = int(stack * random.uniform(1.0, 1.5))

                            # Ensure we don't violate minimum constraints
                            max_allowed = remaining_chips - min_stack * (num_players - i - 1)
                            stack = max(min_stack, min(stack, max_allowed))

                            stacks.append(stack)
                            remaining_chips -= stack

                return stacks

    def _get_num_players_for_hand(self, hand_idx: int) -> int:
        """Determine the number of players for this training hand

        Simulates tournament-style progression where players get eliminated
        """
        if not self.variable_player_count or self.max_players <= 2:
            return self.max_players

        # Calculate training progress
        progress = min(1.0, hand_idx / max(1000, self.total_training_hands or 1000))

        # Distribution of player counts (weighted by likelihood)
        # Most hands should be at max players, but include smaller games
        rand = random.random()

        if self.max_players == 3:
            if rand < 0.7:
                return 3  # 70% full table
            else:
                return 2  # 30% heads-up

        elif self.max_players == 4:
            if rand < 0.6:
                return 4  # 60% full table
            elif rand < 0.85:
                return 3  # 25% three-handed
            else:
                return 2  # 15% heads-up

        elif self.max_players == 5:
            if rand < 0.5:
                return 5  # 50% full table
            elif rand < 0.75:
                return 4  # 25% four-handed
            elif rand < 0.9:
                return 3  # 15% three-handed
            else:
                return 2  # 10% heads-up

        else:  # 6 players
            if rand < 0.45:
                return 6  # 45% full table
            elif rand < 0.7:
                return 5  # 25% five-handed
            elif rand < 0.85:
                return 4  # 15% four-handed
            elif rand < 0.95:
                return 3  # 10% three-handed
            else:
                return 2  # 5% heads-up

        # Increase heads-up frequency slightly as training progresses
        # (simulating getting deeper in tournaments)
        if progress > 0.7 and rand < 0.1:
            return 2  # Extra 10% chance for heads-up in late training

    def simulate_hand(self, hand_idx: int) -> Tuple[float, bool]:
        """Simulate a single hand and return reward in big blinds"""
        # Determine actual number of players for this hand
        num_players = self._get_num_players_for_hand(hand_idx)
        game = TexasHoldEmTraining(num_players=num_players)

        # Track all-ins and folds for penalty/bonus
        our_all_in_count = 0
        our_player_folded = False

        # Get stack distribution for this hand
        stack_distribution = self._get_stack_distribution(hand_idx, num_players)

        # Setup players first, before reset
        players = []
        for i in range(num_players):
            player_chips = stack_distribution[i]
            player = Player(f"Player_{i}", player_chips, is_ai=True)

            if i == 0:
                # Our training AI
                player.ai_model = self.ai_model
            else:
                # Opponent - mix of self-play, random, and other models
                opponent_type = self._choose_opponent(hand_idx)

                if opponent_type == 'self':
                    player.ai_model = self.ai_model
                elif opponent_type == 'random_passive':
                    player.ai_model = self._create_random_ai(aggressive=False)
                elif opponent_type == 'random_aggressive':
                    player.ai_model = self._create_random_ai(aggressive=True)
                elif opponent_type == 'tight_aggressive':
                    player.ai_model = self._create_tight_aggressive_ai()
                elif opponent_type == 'loose_passive':
                    player.ai_model = self._create_loose_passive_ai()
                elif opponent_type == 'calling_station':
                    player.ai_model = self._create_calling_station_ai()
                elif opponent_type == 'maniac':
                    player.ai_model = self._create_maniac_ai()
                elif opponent_type == 'all_in_bot':
                    player.ai_model = self._create_all_in_bot()
                elif opponent_type == 'uniform_random':
                    player.ai_model = self._create_uniform_random_ai()
                elif opponent_type == 'snapshot' and hasattr(self, 'opponent_snapshots') and self.opponent_snapshots:
                    snapshot_model = random.choice(self.opponent_snapshots)
                    player.ai_model = snapshot_model
                elif opponent_type == 'model' and self.opponent_models:
                    _, model = random.choice(self.opponent_models)
                    player.ai_model = model
                else:
                    player.ai_model = self._create_random_ai(aggressive=False)

            players.append(player)

        # Set players before reset
        game.players = players
        # Don't call reset_game as it clears players

        # Deal cards
        for player in game.players:
            player.hand = game.deck.draw(2)
            player.reset_hand()

        # Post blinds
        sb_pos = (game.dealer_position + 1) % num_players
        bb_pos = (game.dealer_position + 2) % num_players
        sb_bet = game.players[sb_pos].bet(self.big_blind // 2)
        bb_bet = game.players[bb_pos].bet(self.big_blind)
        game.pot = sb_bet + bb_bet  # Start with clean integer pot
        game.current_bet = self.big_blind

        initial_chips = game.players[0].chips
        initial_stack_bbs = initial_chips / self.big_blind  # Track initial stack in BBs
        # Debug output removed

        # Play through streets
        streets = ['preflop', 'flop', 'turn', 'river']
        for street_idx, street in enumerate(streets):
            # Deal community cards
            if street == 'flop':
                game.community_cards.extend(game.deck.draw(3))
            elif street in ['turn', 'river']:
                game.community_cards.append(game.deck.draw())

            # Betting round
            all_ins_this_round, folded_this_round = self.betting_round(game, street_idx, num_players)
            our_all_in_count += all_ins_this_round
            if folded_this_round:
                our_player_folded = True  # Track if folded in ANY round

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

        # Apply stronger penalty for all-ins to discourage over-aggressive play
        if our_all_in_count > 0:
            # Much stronger penalty, especially for early all-ins
            # Base penalty increases with stack size (all-in with big stack = bad)
            stack_size_factor = initial_stack_bbs / 50  # Normalize by 50BB

            # Penalty based on when the all-in happened
            # Preflop all-ins get maximum penalty
            if len(game.community_cards) == 0:  # Preflop all-in
                all_in_penalty = 10.0 * our_all_in_count * stack_size_factor
            elif len(game.community_cards) == 3:  # Flop all-in
                all_in_penalty = 6.0 * our_all_in_count * stack_size_factor
            elif len(game.community_cards) == 4:  # Turn all-in
                all_in_penalty = 4.0 * our_all_in_count * stack_size_factor
            else:  # River all-in
                all_in_penalty = 2.0 * our_all_in_count * stack_size_factor

            reward_bb -= all_in_penalty

            # Additional penalty for going all-in too frequently
            # Track recent all-ins in a window
            if not hasattr(self, 'recent_all_ins'):
                self.recent_all_ins = []
            self.recent_all_ins.append(1)
            if len(self.recent_all_ins) > 20:
                self.recent_all_ins.pop(0)

            # If all-in rate > 50% in recent hands, apply extra penalty
            if len(self.recent_all_ins) >= 10:
                all_in_rate = sum(self.recent_all_ins) / len(self.recent_all_ins)
                if all_in_rate > 0.5:
                    frequency_penalty = (all_in_rate - 0.5) * 10  # Up to 5 BB extra penalty
                    reward_bb -= frequency_penalty
        else:
            # Track that we didn't go all-in this hand
            if not hasattr(self, 'recent_all_ins'):
                self.recent_all_ins = []
            self.recent_all_ins.append(0)
            if len(self.recent_all_ins) > 20:
                self.recent_all_ins.pop(0)

        # Track folding behavior
        self.recent_folds.append(1 if our_player_folded else 0)
        if len(self.recent_folds) > self.fold_window_size:
            self.recent_folds.pop(0)

        # Apply penalty for excessive folding
        if len(self.recent_folds) >= 20:  # Need some history before applying penalty
            recent_fold_rate = sum(self.recent_folds) / len(self.recent_folds)
            if recent_fold_rate > self.excessive_fold_threshold:
                # Penalty proportional to how much over the threshold
                excess_folding = recent_fold_rate - self.excessive_fold_threshold
                fold_penalty = self.fold_penalty * excess_folding * 4  # Scale penalty
                reward_bb += fold_penalty  # Negative penalty reduces reward

        won = our_player in winners

        # Update tracking (add all-in and fold counts to parent method's tracking)
        if hasattr(self, '_current_training_all_ins'):
            self._current_training_all_ins += our_all_in_count
        if hasattr(self, '_current_training_folded'):
            self._current_training_folded = our_player_folded


        # Store experience with reward
        if hasattr(our_player.ai_model, 'remember'):
            # Collect final opponent information
            final_opponent_info = []
            for p in game.players:
                if p != our_player and not p.folded:
                    final_opponent_info.append({
                        'chips': p.chips,
                        'current_bet': p.current_bet,
                        'all_in': p.all_in
                    })

            # Create final state
            final_state = our_player.ai_model.get_state_features(
                our_player.hand, game.community_cards, game.pot, 0,
                our_player.chips, 0, num_players, 1, 0,
                game.action_history, final_opponent_info, hand_phase=3
            )

            # Store with the BB-based reward
            our_player.ai_model.remember(
                final_state, Action.CHECK, reward_bb, None, True
            )

        won = our_player in winners
        return reward_bb, won

    def betting_round(self, game: TexasHoldEmTraining, street_idx: int, num_players: int):
        """Execute a betting round, returns (all-ins, folded) for our player"""
        if not game.players or len(game.players) == 0:
            return 0, False

        our_all_ins = 0  # Track our player's all-ins
        our_player_folded = False  # Track if our player folded

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
            # Collect opponent information
            opponent_info = []
            for p in game.players:
                if p != player and not p.folded:
                    opponent_info.append({
                        'chips': p.chips,
                        'current_bet': p.current_bet,
                        'all_in': p.all_in
                    })

            state = player.ai_model.get_state_features(
                player.hand, game.community_cards, game.pot, game.current_bet,
                player.chips, player.current_bet, num_players,
                sum(1 for p in game.players if not p.folded),
                current, game.action_history[-10:], opponent_info, hand_phase=street_idx
            )

            # Get valid actions
            valid_actions = self.get_valid_actions(game, player)

            # Choose action
            # Only the main training player (position 0) should be in training mode
            is_training_player = (player == game.players[0])
            action = player.ai_model.choose_action(state, valid_actions, training=is_training_player)

            # Execute action
            old_bet = player.current_bet
            self.execute_action(game, player, action, num_players)

            # Track if our player went all-in or folded
            if player == game.players[0]:
                if action == Action.ALL_IN:
                    our_all_ins += 1
                elif action == Action.FOLD:
                    our_player_folded = True

            # Store experience
            if hasattr(player.ai_model, 'remember') and player == game.players[0]:
                # Collect opponent information for next state
                next_opponent_info = []
                for p in game.players:
                    if p != player and not p.folded:
                        next_opponent_info.append({
                            'chips': p.chips,
                            'current_bet': p.current_bet,
                            'all_in': p.all_in
                        })

                next_state = player.ai_model.get_state_features(
                    player.hand, game.community_cards, game.pot, game.current_bet,
                    player.chips, player.current_bet, num_players,
                    sum(1 for p in game.players if not p.folded),
                    current, game.action_history[-10:], next_opponent_info, hand_phase=street_idx
                )

                # Intermediate reward (small shaping rewards)
                intermediate_reward = 0

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

        return our_all_ins, our_player_folded

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

    def execute_action(self, game, player, action, num_players):
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
            # For DQN models, we need to provide the state
            if hasattr(player.ai_model, 'q_network'):
                # DQN model needs state tensor
                state = player.ai_model.get_state_features(
                    player.hand, game.community_cards, game.pot, game.current_bet,
                    player.chips, player.current_bet, num_players,
                    sum(1 for p in game.players if not p.folded),
                    0, game.action_history[-10:], [], hand_phase=0
                )
                raise_amount = player.ai_model.get_raise_size(
                    state, game.pot, game.current_bet, player.chips,
                    player.current_bet, self.big_blind
                )
            else:
                # Other models don't need state for raise sizing
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
            # Single winner gets the entire pot
            winner = active_players[0]
            winner.chips += game.pot
            game.pot = 0
            return active_players

        # Evaluate hands
        player_hands = []
        for player in active_players:
            if player.hand:
                all_cards = player.hand + game.community_cards
                hand_value = evaluate_hand(all_cards)
                player_hands.append((player, hand_value))

        if not player_hands:
            # If no valid hands, split pot equally among active players
            if active_players and game.pot > 0:
                split_pot = game.pot // len(active_players)
                remainder = game.pot % len(active_players)
                for i, player in enumerate(active_players):
                    if i < remainder:
                        player.chips += split_pot + 1
                    else:
                        player.chips += split_pot
                game.pot = 0
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

            # Clear the pot after distribution
            game.pot = 0

        return winners

    def _choose_opponent(self, hand_idx: int) -> str:
        """Choose opponent type based on curriculum"""
        # Stronger curriculum with less exploitable opponents
        progress = min(1.0, hand_idx / 1000)

        weights = {
            'random_passive': 0.05,  # Reduced
            'random_aggressive': 0.05,  # Reduced
            'tight_aggressive': 0.15,  # Increased - harder opponent
            'loose_passive': 0.05,  # Reduced
            'calling_station': 0.03,  # Reduced - too exploitable
            'maniac': 0.07,
            'all_in_bot': 0.02,  # Greatly reduced - too exploitable
            #'uniform_random': 0.2,  # Use uniform random only for now (testing)
            'self': 0.25 - 0.15 * progress,  # More self-play for better learning
            'snapshot': 0.15 * progress if hasattr(self, 'opponent_snapshots') and self.opponent_snapshots else 0,
            'model': 0.15 if self.opponent_models else 0  # More games against strong models
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

        return 'random_passive'

    def _create_random_ai(self, aggressive=False):
        """Create a random AI opponent with configurable play style"""
        # Define weight profiles for different play styles
        if aggressive:
            action_weights = {
                Action.FOLD: 0.05,
                Action.CHECK: 0.1,
                Action.CALL: 0.2,
                Action.RAISE: 0.4,
                Action.ALL_IN: 0.25
            }
            raise_range = (60, 150)  # Aggressive players bet bigger
        else:
            action_weights = {
                Action.FOLD: 0.1,
                Action.CHECK: 0.4,
                Action.CALL: 0.3,
                Action.RAISE: 0.15,
                Action.ALL_IN: 0.05
            }
            raise_range = (40, 100)  # Passive players bet smaller

        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_weighted_random_ai(self, action_weights, raise_range=(40, 100)):
        """Create a random AI with specified action weights"""
        class RandomAI:
            def __init__(self, action_weights, raise_range):
                self.epsilon = 0
                self.action_weights = action_weights
                self.raise_range = raise_range

            def get_state_features(self, *args, **kwargs):
                return {}

            def choose_action(self, state, valid_actions, **kwargs):
                # Build weights for valid actions
                weights = []
                for action in valid_actions:
                    weights.append(self.action_weights.get(action, 0.1))

                # Normalize and sample
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                    return np.random.choice(valid_actions, p=weights)
                else:
                    # Fallback to uniform random if weights are all 0
                    return np.random.choice(valid_actions)

            def get_raise_size(self, *args, **kwargs):
                return random.randint(*self.raise_range)

            def remember(self, *args, **kwargs):
                pass

        return RandomAI(action_weights, raise_range)

    def _create_uniform_random_ai(self):
        """Create a uniform random AI (equal chance for all actions)"""
        action_weights = {
            Action.FOLD: 0.2,
            Action.CHECK: 0.2,
            Action.CALL: 0.2,
            Action.RAISE: 0.2,
            Action.ALL_IN: 0.2
        }
        raise_range = (40, 100)  # Moderate bets when raising
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_tight_aggressive_ai(self):
        """Create a tight-aggressive AI (plays few hands but plays them aggressively)"""
        action_weights = {
            Action.FOLD: 0.35,   # Folds often (tight)
            Action.CHECK: 0.05,  # Rarely checks
            Action.CALL: 0.1,    # Rarely calls
            Action.RAISE: 0.35,  # Raises frequently when playing
            Action.ALL_IN: 0.15  # Occasional all-ins
        }
        raise_range = (80, 200)  # Bigger bets when playing
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_loose_passive_ai(self):
        """Create a loose-passive AI (plays many hands but passively)"""
        action_weights = {
            Action.FOLD: 0.05,   # Rarely folds (loose)
            Action.CHECK: 0.35,  # Checks often
            Action.CALL: 0.45,   # Calls frequently (passive)
            Action.RAISE: 0.1,   # Rarely raises
            Action.ALL_IN: 0.05  # Very rarely goes all-in
        }
        raise_range = (30, 60)  # Small bets when raising
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_calling_station_ai(self):
        """Create a calling station AI (calls almost everything)"""
        action_weights = {
            Action.FOLD: 0.02,   # Almost never folds
            Action.CHECK: 0.2,   # Checks when possible
            Action.CALL: 0.7,    # Calls most of the time
            Action.RAISE: 0.06,  # Very rarely raises
            Action.ALL_IN: 0.02  # Almost never all-in
        }
        raise_range = (20, 40)  # Minimal raises
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_maniac_ai(self):
        """Create a maniac AI (extremely aggressive, raises constantly)"""
        action_weights = {
            Action.FOLD: 0.02,   # Almost never folds
            Action.CHECK: 0.03,  # Rarely checks
            Action.CALL: 0.1,    # Sometimes calls
            Action.RAISE: 0.5,   # Raises very frequently
            Action.ALL_IN: 0.35  # Goes all-in often
        }
        raise_range = (100, 300)  # Large, erratic bets
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _create_all_in_bot(self):
        """Create an AI that always goes all-in or folds"""
        action_weights = {
            Action.FOLD: 0.15,   # Sometimes folds bad hands
            Action.CHECK: 0.05,  # Rarely checks (only when no chips)
            Action.CALL: 0.0,    # Never calls
            Action.RAISE: 0.0,   # Never normal raises
            Action.ALL_IN: 0.8   # Almost always all-in
        }
        raise_range = (1000, 1000)  # Doesn't matter, always all-in
        return self._create_weighted_random_ai(action_weights, raise_range)

    def _save_snapshot(self):
        """Save a snapshot of the current model for the opponent pool"""
        import copy

        # Create a frozen copy of the current model
        snapshot = copy.deepcopy(self.ai_model)

        # Disable training for the snapshot
        snapshot.epsilon = 0

        # Add to pool, maintaining max size
        self.opponent_snapshots.append(snapshot)
        if len(self.opponent_snapshots) > self.max_snapshots:
            self.opponent_snapshots.pop(0)  # Remove oldest snapshot

        if hasattr(self, 'verbose') and self.verbose:
            print(f"  [Snapshot saved, pool size: {len(self.opponent_snapshots)}]")