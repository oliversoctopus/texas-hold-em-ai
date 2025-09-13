"""
Fixed Enhanced CFR Two-Player Implementation
Replaces the broken iterative approach with proven Monte Carlo CFR
Includes comprehensive progress monitoring and timeout protection
"""

import numpy as np
import random
import pickle
import os
import time
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class CFRAction(Enum):
    """Simple, proven action set"""
    FOLD = 0
    CHECK_CALL = 1
    BET_HALF_POT = 2
    BET_POT = 3
    ALL_IN = 4


@dataclass
class GameState:
    """Simple game state representation"""
    hole_cards: List[List[Card]]
    community_cards: List[Card]
    pot: int
    player_bets: List[int]
    to_act: int
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    is_terminal: bool = False
    winner: int = -1


class ProgressTracker:
    """Simple progress tracker without threading issues"""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = time.time()
        self.iteration = 0
        self.total_iterations = 0
        self.info_sets_created = 0
        self.nodes_processed = 0

    def start_iteration(self, iteration: int, total: int, info_sets: int):
        """Start a new iteration"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        if iteration == 0:
            print(f"\nStarting Enhanced CFR training for {total} iterations...")
            print("Using proven Monte Carlo CFR with timeout protection\n")

        # Print progress every iteration (they should be fast)
        if iteration % max(1, total // 20) == 0 or elapsed - (self.last_update if hasattr(self, 'last_update') else 0) > 10:
            progress_pct = (iteration / total * 100) if total > 0 else 0
            eta_seconds = (elapsed / max(1, iteration)) * (total - iteration) if iteration > 0 else 0
            eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

            print(f"Iteration {iteration:,}/{total:,} ({progress_pct:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta_str}")
            print(f"  Info sets: {info_sets:,} | "
                  f"Nodes processed: {self.nodes_processed:,}")

            self.last_update = current_time

        self.iteration = iteration
        self.total_iterations = total
        self.info_sets_created = info_sets

    def record_node(self):
        """Record a node being processed"""
        self.nodes_processed += 1


class EnhancedInformationSet:
    """Enhanced information set with comprehensive abstractions"""

    def __init__(self, hole_cards: List[Card], community_cards: List[Card],
                 betting_history: str, position: int, pot: int, effective_stack: int):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.betting_history = betting_history
        self.position = position
        self.pot = pot
        self.effective_stack = effective_stack
        self._key = self._create_key()

    def _create_key(self) -> str:
        """Create comprehensive information set key"""
        # Hand strength abstraction (fine-grained)
        hand_strength = self._get_hand_strength_category()

        # Board texture
        board_texture = self._get_board_texture()

        # Betting abstraction
        betting_pattern = self._get_betting_abstraction()

        # Stack depth
        stack_depth = self._get_stack_depth_category()

        # Position
        pos = f"pos{self.position}"

        return f"{hand_strength}_{board_texture}_{betting_pattern}_{stack_depth}_{pos}"

    def _get_hand_strength_category(self) -> str:
        """Get detailed hand strength category"""
        if not self.hole_cards or len(self.hole_cards) < 2:
            return "trash"

        try:
            if len(self.community_cards) >= 3:
                # Post-flop: use actual hand evaluation
                full_hand = self.hole_cards + self.community_cards
                hand_value = evaluate_hand(full_hand)

                if hand_value >= 8000000:    return "quads_plus"
                elif hand_value >= 7000000:  return "full_house"
                elif hand_value >= 6000000:  return "flush"
                elif hand_value >= 5000000:  return "straight"
                elif hand_value >= 4000000:  return "trips"
                elif hand_value >= 3000000:  return "two_pair"
                elif hand_value >= 2500000:  return "strong_pair"
                elif hand_value >= 2000000:  return "weak_pair"
                else:                        return "high_card"
            else:
                # Preflop: hand category
                return self._preflop_strength()
        except:
            return self._preflop_strength()

    def _preflop_strength(self) -> str:
        """Preflop hand strength categorization"""
        ranks = sorted([card.value for card in self.hole_cards], reverse=True)
        high, low = ranks[0], ranks[1]
        suited = self.hole_cards[0].suit == self.hole_cards[1].suit

        # Pocket pairs
        if high == low:
            if high >= 11: return "premium_pair"  # JJ+
            elif high >= 7: return "medium_pair"   # 77-TT
            else: return "small_pair"              # 22-66

        # Suited cards
        if suited:
            if high == 14:  # Ace suited
                if low >= 10: return "premium_suited"   # AK, AQ, AJ, AT
                else: return "ace_suited"               # A9-A2
            elif high >= 11 and low >= 10: return "broadway_suited"  # KQ, KJ, QJ
            elif abs(high - low) <= 3: return "suited_connector"     # Suited connectors
            else: return "weak_suited"

        # Offsuit cards
        else:
            if high == 14 and low >= 11: return "premium_offsuit"    # AK, AQ, AJ
            elif high >= 11 and low >= 10: return "broadway_offsuit" # KQ, KJ, QJ
            elif abs(high - low) <= 2: return "connector"            # Connected
            else: return "random"

    def _get_board_texture(self) -> str:
        """Get board texture description"""
        num_cards = len(self.community_cards)

        if num_cards == 0:
            return "preflop"
        elif num_cards == 3:
            return self._analyze_flop()
        elif num_cards == 4:
            return "turn"
        else:
            return "river"

    def _analyze_flop(self) -> str:
        """Analyze flop texture"""
        if len(self.community_cards) != 3:
            return "flop"

        ranks = [card.value for card in self.community_cards]
        suits = [card.suit for card in self.community_cards]

        # Check for pairs
        if len(set(ranks)) == 2:
            return "paired_board"

        # Check suit coordination
        if len(set(suits)) == 1:
            texture = "monotone"
        elif len(set(suits)) == 2:
            texture = "two_tone"
        else:
            texture = "rainbow"

        # Check connectivity
        sorted_ranks = sorted(ranks)
        max_gap = max(sorted_ranks[i+1] - sorted_ranks[i] for i in range(len(sorted_ranks)-1))

        if max_gap <= 2:
            return texture + "_connected"
        else:
            return texture + "_dry"

    def _get_betting_abstraction(self) -> str:
        """Get betting pattern abstraction"""
        if not self.betting_history:
            return "no_betting"

        # Count aggressive actions
        aggressive = self.betting_history.count('b') + self.betting_history.count('r')
        passive = self.betting_history.count('c') + self.betting_history.count('k')

        if aggressive >= 3:
            return "aggressive_war"
        elif aggressive == 2:
            return "bet_raise"
        elif aggressive == 1:
            return "single_bet"
        elif passive > 0:
            return "passive_line"
        else:
            return "no_action"

    def _get_stack_depth_category(self) -> str:
        """Get stack depth relative to pot"""
        if self.pot <= 0:
            return "deep"

        spr = self.effective_stack / self.pot

        if spr <= 1:
            return "spr_committed"
        elif spr <= 3:
            return "spr_small"
        elif spr <= 6:
            return "spr_medium"
        else:
            return "spr_deep"

    def get_key(self) -> str:
        return self._key


class CFRNode:
    """Standard CFR node"""

    def __init__(self, num_actions: int = 5):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy_count = 0

    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        strategy = np.maximum(self.regret_sum, 0)

        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()
        else:
            strategy = np.ones(self.num_actions) / self.num_actions

        # Update strategy sum for averaging
        self.strategy_sum += realization_weight * strategy
        self.strategy_count += 1

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Get time-averaged strategy"""
        if self.strategy_sum.sum() > 0:
            return self.strategy_sum / self.strategy_sum.sum()
        else:
            return np.ones(self.num_actions) / self.num_actions


class FixedEnhancedTwoPlayerCFR:
    """Fixed Enhanced CFR using proven Monte Carlo CFR algorithms"""

    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.nodes: Dict[str, CFRNode] = {}
        self.progress_tracker = ProgressTracker()

        # Timeout protection
        self.max_time_per_iteration = 30  # seconds
        self.max_cfr_depth = 8  # Prevent infinite recursion

        # Action mapping
        self.action_mapping = {
            CFRAction.FOLD: Action.FOLD,
            CFRAction.CHECK_CALL: Action.CALL,
            CFRAction.BET_HALF_POT: Action.RAISE,
            CFRAction.BET_POT: Action.RAISE,
            CFRAction.ALL_IN: Action.RAISE
        }

        # Strategy statistics
        self.strategy_stats = {
            'learned_strategy_count': 0,
            'fallback_strategy_count': 0,
            'total_decisions': 0
        }

    def train(self, verbose: bool = True):
        """Train using Monte Carlo CFR with timeout protection"""
        if verbose:
            print("Training Fixed Enhanced 2-Player CFR...")
            print("Using proven Monte Carlo CFR algorithms with timeout protection")

        for iteration in range(self.iterations):
            iteration_start_time = time.time()
            initial_info_sets = len(self.nodes)

            if verbose:
                self.progress_tracker.start_iteration(iteration, self.iterations, len(self.nodes))

            # Generate random game scenario
            game_state = self._generate_random_game()

            # Run CFR on this game from both players' perspectives
            try:
                for training_player in [0, 1]:
                    reach_probs = [1.0, 1.0]
                    self._cfr_recursive(game_state, reach_probs, training_player, 0, iteration_start_time)

            except TimeoutError:
                if verbose:
                    print(f"  Iteration {iteration} timed out - moving to next iteration")
                continue

            # Validate progress
            new_info_sets = len(self.nodes) - initial_info_sets
            if iteration > 0 and new_info_sets == 0 and len(self.nodes) < 100:
                if verbose:
                    print(f"  Warning: No new information sets created in iteration {iteration}")

            # Early stopping if we have reasonable coverage
            if len(self.nodes) > 5000 and iteration > self.iterations // 4:
                if verbose:
                    print(f"  Early stopping at iteration {iteration} - sufficient coverage ({len(self.nodes):,} info sets)")
                break

        if verbose:
            print(f"\nTraining complete!")
            print(f"Generated {len(self.nodes):,} information sets")
            self._print_analysis()

    def _generate_random_game(self) -> GameState:
        """Generate a random poker game state"""
        deck = Deck()
        deck.reset()

        # Deal hole cards
        hole_cards = [deck.draw(2), deck.draw(2)]

        # Randomly choose street (bias toward earlier streets for more training)
        street = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]

        # Deal community cards based on street
        if street == 0:
            community_cards = []
        elif street == 1:
            community_cards = deck.draw(3)  # Flop
        elif street == 2:
            community_cards = deck.draw(4)  # Turn
        else:
            community_cards = deck.draw(5)  # River

        # Random pot size and position
        pot = random.choice([30, 60, 120, 240, 480])
        to_act = random.choice([0, 1])

        return GameState(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot=pot,
            player_bets=[pot//4, pot//4],  # Some betting has occurred
            to_act=to_act,
            street=street
        )

    def _cfr_recursive(self, game_state: GameState, reach_probs: List[float],
                      training_player: int, depth: int, start_time: float) -> float:
        """Recursive CFR with timeout and depth protection"""

        # Timeout protection
        if time.time() - start_time > self.max_time_per_iteration:
            raise TimeoutError("Iteration timeout")

        # Depth protection
        if depth > self.max_cfr_depth:
            return self._estimate_payoff(game_state, training_player)

        self.progress_tracker.record_node()

        # Terminal node check
        if self._is_terminal(game_state):
            return self._get_terminal_payoff(game_state, training_player)

        # Chance node (deal community cards)
        if self._is_chance_node(game_state):
            return self._handle_chance_node(game_state, reach_probs, training_player, depth, start_time)

        current_player = game_state.to_act

        # Create information set
        info_set = self._create_information_set(game_state, current_player)
        key = info_set.get_key()

        if key not in self.nodes:
            self.nodes[key] = CFRNode()

        node = self.nodes[key]
        strategy = node.get_strategy(reach_probs[current_player])

        # Get valid actions
        valid_actions = self._get_valid_actions(game_state)

        # Calculate action utilities
        action_utilities = np.zeros(len(valid_actions))
        node_utility = 0.0

        for i, action in enumerate(valid_actions):
            next_state = self._apply_action(game_state, action, current_player)

            if current_player == training_player:
                action_utilities[i] = -self._cfr_recursive(
                    next_state, reach_probs, training_player, depth + 1, start_time
                )
            else:
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy[i]
                action_utilities[i] = -self._cfr_recursive(
                    next_state, new_reach_probs, training_player, depth + 1, start_time
                )

            node_utility += strategy[i] * action_utilities[i]

        # Update regrets for training player
        if current_player == training_player:
            opponent_reach = reach_probs[1 - current_player]
            for i in range(len(valid_actions)):
                regret = action_utilities[i] - node_utility
                node.regret_sum[i] += opponent_reach * regret

        return node_utility

    def _create_information_set(self, game_state: GameState, player: int) -> EnhancedInformationSet:
        """Create information set for current game state"""
        # Simple betting history from bets
        betting_history = ""
        if max(game_state.player_bets) > min(game_state.player_bets):
            betting_history = "b"  # Someone has bet/raised

        effective_stack = 1000 - max(game_state.player_bets)  # Remaining chips

        return EnhancedInformationSet(
            hole_cards=game_state.hole_cards[player],
            community_cards=game_state.community_cards,
            betting_history=betting_history,
            position=player,
            pot=game_state.pot,
            effective_stack=effective_stack
        )

    def _is_terminal(self, game_state: GameState) -> bool:
        """Check if game state is terminal"""
        return game_state.is_terminal or game_state.street > 3

    def _is_chance_node(self, game_state: GameState) -> bool:
        """Check if this is a chance node (community card dealing)"""
        # For simplicity, we'll treat all states as decision nodes
        # In a full implementation, you'd handle community card dealing here
        return False

    def _handle_chance_node(self, game_state: GameState, reach_probs: List[float],
                          training_player: int, depth: int, start_time: float) -> float:
        """Handle chance nodes (community card dealing)"""
        # Simplified - just continue with current state
        return self._cfr_recursive(game_state, reach_probs, training_player, depth, start_time)

    def _get_valid_actions(self, game_state: GameState) -> List[CFRAction]:
        """Get valid actions for current game state"""
        actions = []

        # Always can fold (except when no bet to call)
        opponent_bet = game_state.player_bets[1 - game_state.to_act]
        current_bet = game_state.player_bets[game_state.to_act]

        if opponent_bet > current_bet:
            actions.append(CFRAction.FOLD)

        # Always can check/call
        actions.append(CFRAction.CHECK_CALL)

        # Can bet/raise if have chips
        remaining_chips = 1000 - current_bet  # Simplified
        if remaining_chips > 0:
            actions.extend([CFRAction.BET_HALF_POT, CFRAction.BET_POT, CFRAction.ALL_IN])

        return actions

    def _apply_action(self, game_state: GameState, action: CFRAction, player: int) -> GameState:
        """Apply action and return new game state"""
        new_state = GameState(
            hole_cards=game_state.hole_cards.copy(),
            community_cards=game_state.community_cards.copy(),
            pot=game_state.pot,
            player_bets=game_state.player_bets.copy(),
            to_act=1 - player,  # Switch players
            street=game_state.street
        )

        if action == CFRAction.FOLD:
            new_state.is_terminal = True
            new_state.winner = 1 - player

        elif action == CFRAction.CHECK_CALL:
            # Call any bet
            opponent_bet = game_state.player_bets[1 - player]
            call_amount = max(0, opponent_bet - game_state.player_bets[player])
            new_state.player_bets[player] += call_amount
            new_state.pot += call_amount

        elif action in [CFRAction.BET_HALF_POT, CFRAction.BET_POT, CFRAction.ALL_IN]:
            # Make a bet/raise
            if action == CFRAction.BET_HALF_POT:
                bet_size = max(20, game_state.pot // 2)
            elif action == CFRAction.BET_POT:
                bet_size = max(40, game_state.pot)
            else:  # ALL_IN
                bet_size = 500  # Simplified

            new_state.player_bets[player] += bet_size
            new_state.pot += bet_size

        # Check if betting round is complete (simplified)
        if new_state.player_bets[0] == new_state.player_bets[1] and not new_state.is_terminal:
            # Advance to next street or end game
            if new_state.street >= 3:
                new_state.is_terminal = True
            else:
                new_state.street += 1
                new_state.to_act = 0  # Reset to first player

        return new_state

    def _get_terminal_payoff(self, game_state: GameState, player: int) -> float:
        """Calculate payoff at terminal node"""
        if game_state.winner >= 0:
            # Someone folded
            return game_state.pot if player == game_state.winner else -game_state.player_bets[player]

        # Showdown
        try:
            player_hand = game_state.hole_cards[player] + game_state.community_cards
            opponent_hand = game_state.hole_cards[1 - player] + game_state.community_cards

            if len(player_hand) >= 5 and len(opponent_hand) >= 5:
                player_value = evaluate_hand(player_hand)
                opponent_value = evaluate_hand(opponent_hand)

                if player_value > opponent_value:
                    return game_state.pot // 2
                elif player_value == opponent_value:
                    return 0
                else:
                    return -game_state.pot // 2
        except:
            pass

        return 0  # Tie as fallback

    def _estimate_payoff(self, game_state: GameState, player: int) -> float:
        """Estimate payoff when depth limit reached"""
        # Very simple estimation
        return 0

    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int, stack_size: int,
                   position: int, **kwargs) -> Action:
        """Get action using trained strategy"""

        info_set = EnhancedInformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            position=position,
            pot=pot_size,
            effective_stack=stack_size
        )

        key = info_set.get_key()
        self.strategy_stats['total_decisions'] += 1

        if key in self.nodes:
            self.strategy_stats['learned_strategy_count'] += 1
            node = self.nodes[key]
            strategy = node.get_average_strategy()

            # Sample action from strategy
            action_idx = np.random.choice(len(strategy), p=strategy)
            cfr_action = list(CFRAction)[action_idx]
        else:
            self.strategy_stats['fallback_strategy_count'] += 1
            # Simple fallback
            if to_call == 0:
                cfr_action = CFRAction.CHECK_CALL
            elif to_call > stack_size // 3:
                cfr_action = CFRAction.FOLD
            else:
                cfr_action = CFRAction.CHECK_CALL

        return self._map_action(cfr_action, to_call)

    def _map_action(self, cfr_action: CFRAction, to_call: int) -> Action:
        """Map CFR action to game action"""
        if cfr_action == CFRAction.FOLD:
            return Action.FOLD
        elif cfr_action == CFRAction.CHECK_CALL:
            return Action.CHECK if to_call == 0 else Action.CALL
        else:
            return Action.RAISE

    def get_strategy_usage_stats(self) -> Dict:
        """Get strategy usage statistics"""
        total = self.strategy_stats['total_decisions']
        learned = self.strategy_stats['learned_strategy_count']
        fallback = self.strategy_stats['fallback_strategy_count']

        return {
            'total_decisions': total,
            'learned_count': learned,
            'fallback_count': fallback,
            'learned_percentage': (learned / total * 100) if total > 0 else 0,
            'fallback_percentage': (fallback / total * 100) if total > 0 else 0
        }

    def reset_strategy_stats(self):
        """Reset strategy statistics"""
        self.strategy_stats = {
            'learned_strategy_count': 0,
            'fallback_strategy_count': 0,
            'total_decisions': 0
        }

    def print_strategy_usage(self):
        """Print strategy usage statistics"""
        stats = self.get_strategy_usage_stats()
        total = stats['total_decisions']
        learned_pct = stats['learned_percentage']
        fallback_pct = stats['fallback_percentage']

        print(f"\nStrategy Usage Statistics:")
        print(f"  Total decisions: {total:,}")
        print(f"  Learned strategy: {learned_pct:.1f}%")
        print(f"  Fallback strategy: {fallback_pct:.1f}%")

        if learned_pct >= 90:
            print(f"  [EXCELLENT] High strategy coverage!")
        elif learned_pct >= 70:
            print(f"  [GOOD] Decent strategy coverage")
        else:
            print(f"  [NEEDS WORK] Low strategy coverage - needs more training")

    def _print_analysis(self):
        """Print training analysis"""
        print(f"\nTraining Analysis:")

        # Sample information set keys
        sample_keys = list(self.nodes.keys())[:10]
        print(f"  Sample information sets:")
        for i, key in enumerate(sample_keys, 1):
            print(f"    {i}. {key}")

        # Key diversity analysis
        if len(self.nodes) > 0:
            avg_key_length = np.mean([len(k) for k in sample_keys])
            print(f"  Average key length: {avg_key_length:.1f} characters")
            print(f"  Key diversity: {'Good' if len(set(sample_keys)) == len(sample_keys) else 'Poor'}")

    def save(self, filename: str):
        """Save model"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'model_type': 'FixedEnhancedTwoPlayerCFR',
                'model_version': '1.0',
                'nodes': self.nodes,
                'iterations': self.iterations,
                'strategy_stats': self.strategy_stats
            }, f)
        print(f"Fixed Enhanced 2-Player CFR model saved to {filename}")

    def load(self, filename: str):
        """Load model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.nodes = data.get('nodes', {})
        self.iterations = data.get('iterations', self.iterations)
        self.strategy_stats = data.get('strategy_stats', self.strategy_stats)

        print(f"Fixed Enhanced 2-Player CFR model loaded from {filename}")
        print(f"  Loaded {len(self.nodes):,} information sets")


def train_fixed_enhanced_two_player_cfr_ai(iterations: int = 1000,
                                          save_filename: Optional[str] = None,
                                          verbose: bool = True) -> FixedEnhancedTwoPlayerCFR:
    """
    Train fixed enhanced 2-player CFR AI with proper algorithms
    """
    cfr_ai = FixedEnhancedTwoPlayerCFR(iterations=iterations)
    cfr_ai.train(verbose=verbose)

    if save_filename:
        cfr_ai.save(save_filename)

    return cfr_ai