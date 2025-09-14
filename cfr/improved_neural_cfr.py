"""
Improved Neural-Enhanced CFR for 2-Player Texas Hold'em
Fixes key issues with hand strength awareness and convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class ImprovedAction(Enum):
    """Actions with better granularity"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_SMALL = 3    # 0.25-0.4 pot
    BET_MEDIUM = 4   # 0.5-0.75 pot
    BET_LARGE = 5    # 0.9-1.1 pot
    BET_OVERBET = 6  # 1.5-2.0 pot
    ALL_IN = 7


class HandStrengthCategory(Enum):
    """Hand strength categories for strategy selection"""
    MONSTER = 8      # Straight flush, quads
    VERY_STRONG = 7  # Full house, flush
    STRONG = 6       # Straight, set
    GOOD = 5         # Two pair, overpair
    MEDIUM = 4       # Top pair, middle pair
    WEAK = 3         # Bottom pair, ace high
    VERY_WEAK = 2    # King high, queen high
    TRASH = 1        # Complete air
    DRAWING = 0      # Strong draws


class StrategyNetworkEnsemble(nn.Module):
    """Ensemble of networks for different hand strengths"""

    def __init__(self, feature_dim: int = 50, num_actions: int = 8, hidden_size: int = 256):
        super(StrategyNetworkEnsemble, self).__init__()

        # Separate networks for different situations
        self.value_network = self._create_network(feature_dim + 4, num_actions, hidden_size)
        self.bluff_network = self._create_network(feature_dim + 4, num_actions, hidden_size // 2)
        self.balanced_network = self._create_network(feature_dim + 4, num_actions, hidden_size)
        self.drawing_network = self._create_network(feature_dim + 4, num_actions, hidden_size // 2)

        # Meta-network to blend strategies
        self.blend_network = nn.Sequential(
            nn.Linear(feature_dim + 4 + num_actions * 4, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions),
            nn.Softmax(dim=-1)
        )

    def _create_network(self, input_dim: int, output_dim: int, hidden_size: int):
        """Create a strategy network"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, features, hand_strength, pot_odds, position, board_texture):
        """Get blended strategy based on context"""
        # Augment features with explicit poker information
        augmented = torch.cat([features, hand_strength, pot_odds, position, board_texture], dim=-1)

        # Get strategies from each network
        value_strategy = self.value_network(augmented)
        bluff_strategy = self.bluff_network(augmented)
        balanced_strategy = self.balanced_network(augmented)
        drawing_strategy = self.drawing_network(augmented)

        # Stack all strategies
        all_strategies = torch.cat([value_strategy, bluff_strategy, balanced_strategy, drawing_strategy], dim=-1)

        # Blend based on situation
        blend_input = torch.cat([augmented, all_strategies], dim=-1)
        final_strategy = self.blend_network(blend_input)

        return final_strategy


class HandEvaluator:
    """Contextual hand strength evaluation"""

    def __init__(self):
        self.preflop_rankings = self._init_preflop_rankings()

    def _init_preflop_rankings(self):
        """Initialize preflop hand rankings"""
        # Simplified - top hands get high scores
        rankings = {}
        premium = ['AA', 'KK', 'QQ', 'AKs', 'JJ', 'AKo']
        strong = ['TT', '99', 'AQs', 'AJs', 'KQs', 'AQo']
        playable = ['88', '77', 'ATs', 'KJs', 'QJs', 'JTs', 'AJo', 'KQo']

        for hand in premium:
            rankings[hand] = 0.95
        for hand in strong:
            rankings[hand] = 0.80
        for hand in playable:
            rankings[hand] = 0.65

        return rankings

    def get_hand_category(self, hole_cards: List[Card], community_cards: List[Card]) -> HandStrengthCategory:
        """Categorize hand strength"""
        if not hole_cards or len(hole_cards) < 2:
            return HandStrengthCategory.TRASH

        if len(community_cards) == 0:
            # Preflop
            hand_str = self._cards_to_string(hole_cards)
            ranking = self.preflop_rankings.get(hand_str, 0.3)
            if ranking > 0.9:
                return HandStrengthCategory.VERY_STRONG
            elif ranking > 0.7:
                return HandStrengthCategory.STRONG
            elif ranking > 0.5:
                return HandStrengthCategory.MEDIUM
            else:
                return HandStrengthCategory.WEAK
        else:
            # Postflop - evaluate made hand
            hand_value = evaluate_hand(hole_cards + community_cards)

            # Categorize based on hand value (simplified)
            if hand_value > 8000000:  # Straight flush
                return HandStrengthCategory.MONSTER
            elif hand_value > 7000000:  # Four of a kind
                return HandStrengthCategory.MONSTER
            elif hand_value > 6000000:  # Full house
                return HandStrengthCategory.VERY_STRONG
            elif hand_value > 5000000:  # Flush
                return HandStrengthCategory.VERY_STRONG
            elif hand_value > 4000000:  # Straight
                return HandStrengthCategory.STRONG
            elif hand_value > 3000000:  # Three of a kind
                return HandStrengthCategory.STRONG
            elif hand_value > 2000000:  # Two pair
                return HandStrengthCategory.GOOD
            elif hand_value > 1000000:  # One pair
                return self._evaluate_pair_strength(hole_cards, community_cards)
            else:
                return HandStrengthCategory.WEAK

    def _evaluate_pair_strength(self, hole_cards: List[Card], community_cards: List[Card]) -> HandStrengthCategory:
        """Evaluate the strength of a pair"""
        # Check if we have top pair, middle pair, etc.
        if not community_cards:
            return HandStrengthCategory.MEDIUM

        board_ranks = sorted([c.value for c in community_cards], reverse=True)
        hole_ranks = [c.value for c in hole_cards]

        # Check for overpair
        if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > board_ranks[0]:
            return HandStrengthCategory.GOOD

        # Check for top pair
        if any(hr == board_ranks[0] for hr in hole_ranks):
            return HandStrengthCategory.MEDIUM

        # Bottom pair or worse
        return HandStrengthCategory.WEAK

    def _cards_to_string(self, cards: List[Card]) -> str:
        """Convert cards to string notation"""
        if len(cards) != 2:
            return ""

        # Sort by rank
        sorted_cards = sorted(cards, key=lambda c: c.value, reverse=True)
        suited = 's' if sorted_cards[0].suit == sorted_cards[1].suit else 'o'

        rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        r1 = rank_map.get(sorted_cards[0].value, str(sorted_cards[0].value))
        r2 = rank_map.get(sorted_cards[1].value, str(sorted_cards[1].value))

        if r1 == r2:
            return f"{r1}{r2}"
        else:
            return f"{r1}{r2}{suited}"

    def get_hand_percentile(self, hole_cards: List[Card], community_cards: List[Card],
                           num_opponents: int = 1, samples: int = 100) -> float:
        """Calculate hand strength percentile vs random opponents"""
        if not hole_cards or len(hole_cards) < 2:
            return 0.0

        wins = 0
        ties = 0

        for _ in range(samples):
            deck = Deck()
            deck.reset()

            # Remove known cards (match by value and suit)
            for card in hole_cards + community_cards:
                for deck_card in deck.cards[:]:  # Use slice to avoid modification issues
                    if deck_card.value == card.value and deck_card.suit == card.suit:
                        deck.cards.remove(deck_card)
                        break

            # Deal opponent cards
            opponent_hands = []
            for _ in range(num_opponents):
                if len(deck.cards) >= 2:
                    opponent_hands.append([deck.draw(), deck.draw()])

            # Complete the board if needed
            remaining_community = 5 - len(community_cards)
            future_community = []
            for _ in range(remaining_community):
                if deck.cards:
                    future_community.append(deck.draw())

            # Evaluate hands
            full_community = community_cards + future_community
            our_strength = evaluate_hand(hole_cards + full_community)

            # Check against opponents
            best_opponent = 0
            for opp_hand in opponent_hands:
                opp_strength = evaluate_hand(opp_hand + full_community)
                best_opponent = max(best_opponent, opp_strength)

            if our_strength > best_opponent:
                wins += 1
            elif our_strength == best_opponent:
                ties += 0.5

        return (wins + ties) / samples

    def get_contextual_strength(self, hole_cards: List[Card], community_cards: List[Card],
                               betting_history: str, pot_size: int, to_call: int) -> Dict[str, float]:
        """Get comprehensive hand strength analysis"""
        base_percentile = self.get_hand_percentile(hole_cards, community_cards)
        category = self.get_hand_category(hole_cards, community_cards)

        # Adjust for betting patterns
        aggression_factor = betting_history.count('r') + betting_history.count('b')
        if aggression_factor > 2:
            # Heavy action suggests strong hands
            adjusted_percentile = base_percentile * 0.9
        else:
            adjusted_percentile = base_percentile

        # Calculate pot odds
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0

        # Determine if we're drawing
        is_drawing = self._has_strong_draw(hole_cards, community_cards)

        return {
            'percentile': base_percentile,
            'adjusted_percentile': adjusted_percentile,
            'category': category,
            'pot_odds': pot_odds,
            'is_drawing': is_drawing,
            'hand_rank': category.value / 8.0  # Normalized 0-1
        }

    def _has_strong_draw(self, hole_cards: List[Card], community_cards: List[Card]) -> bool:
        """Check if hand has strong drawing potential"""
        if len(community_cards) >= 5:
            return False  # No more cards to come

        all_cards = hole_cards + community_cards

        # Check for flush draw (4 cards of same suit)
        suits = {}
        for card in all_cards:
            suits[card.suit] = suits.get(card.suit, 0) + 1
        if any(count >= 4 for count in suits.values()):
            return True

        # Check for open-ended straight draw
        ranks = sorted([c.value for c in all_cards])
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] <= 4:
                return True

        return False


class ImprovedNeuralCFRNode:
    """Improved CFR node with better strategy tracking"""

    def __init__(self, num_actions: int = 8):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.visits = 0

        # Track actual money won/lost for better convergence
        self.action_ev = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)

    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching+"""
        # Use regret matching+ (don't let regrets go negative)
        strategy = np.maximum(self.regret_sum, 0)

        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            # Use uniform strategy if no positive regrets
            strategy = np.ones(self.num_actions) / self.num_actions

        # Update strategy sum for averaging
        self.strategy_sum += realization_weight * strategy
        self.visits += 1

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        if np.sum(self.strategy_sum) > 0:
            return self.strategy_sum / np.sum(self.strategy_sum)
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, utilities: np.ndarray, node_utility: float, weight: float):
        """Update regrets with CFR+"""
        for i in range(self.num_actions):
            regret = utilities[i] - node_utility
            # CFR+ : keep regrets non-negative
            self.regret_sum[i] = max(self.regret_sum[i] + weight * regret, 0)


class ImprovedNeuralEnhancedCFR:
    """Improved Neural-Enhanced CFR with hand strength awareness"""

    def __init__(self, iterations: int = 10000, use_neural_networks: bool = True,
                 use_hand_strength: bool = True):
        self.iterations = iterations
        self.use_neural_networks = use_neural_networks
        self.use_hand_strength = use_hand_strength

        # Core components
        self.nodes = {}
        self.hand_evaluator = HandEvaluator()

        # Neural networks
        if use_neural_networks:
            self.strategy_ensemble = StrategyNetworkEnsemble()
            self.optimizer = optim.Adam(self.strategy_ensemble.parameters(), lr=0.001)

        # Training tracking
        self.iteration_count = 0
        self.training_start_time = None

        # Strategy statistics
        self.strategy_stats = defaultdict(lambda: {'used': 0, 'learned': 0})

    def train(self, verbose: bool = True):
        """Train using improved Monte Carlo CFR"""
        self.training_start_time = time.time()

        if verbose:
            print("Starting Improved Neural-Enhanced CFR Training")
            print(f"Iterations: {self.iterations}")
            print(f"Hand strength awareness: {self.use_hand_strength}")
            print(f"Neural networks: {self.use_neural_networks}")
            print("=" * 60)

        # Progressive training stages
        stages = [
            (self.iterations // 4, "Bootstrap", self._bootstrap_iteration),
            (self.iterations // 2, "Self-play", self._self_play_iteration),
            (self.iterations // 4, "Refinement", self._refinement_iteration)
        ]

        total_iterations = 0
        for stage_iterations, stage_name, iteration_fn in stages:
            if verbose:
                print(f"\nStage: {stage_name} ({stage_iterations} iterations)")

            for i in range(stage_iterations):
                total_iterations += 1
                self.iteration_count = total_iterations

                # Run iteration
                iteration_fn()

                # Update neural networks periodically
                if self.use_neural_networks and i % 100 == 0:
                    self._update_neural_networks()

                # Progress update
                progress_interval = max(1, stage_iterations // 10)
                if verbose and i % progress_interval == 0:
                    self._print_progress(i, stage_iterations, stage_name)

        if verbose:
            self._print_final_stats()

    def _bootstrap_iteration(self):
        """Bootstrap with random play to explore game tree"""
        game_state = self._sample_game_state()

        # External sampling for both players
        for player in [0, 1]:
            self._external_sampling_cfr(game_state.copy(), player, 1.0, 1.0)

    def _self_play_iteration(self):
        """Self-play iteration for strategy refinement"""
        game_state = self._sample_game_state()

        # Outcome sampling with importance sampling
        sampled_player = random.randint(0, 1)
        self._outcome_sampling_cfr(game_state, sampled_player)

    def _refinement_iteration(self):
        """Refinement iteration to patch exploits"""
        game_state = self._sample_game_state()

        # Use best response computation
        for player in [0, 1]:
            self._best_response_cfr(game_state.copy(), player)

    def _external_sampling_cfr(self, game_state: Dict, player: int,
                               reach_p0: float, reach_p1: float, depth: int = 0) -> float:
        """External sampling Monte Carlo CFR"""
        # Check terminal
        if self._is_terminal(game_state) or depth > 15:
            return self._evaluate_terminal(game_state, player)

        # Get current player
        current_player = game_state['current_player']

        # Create info set
        info_set = self._get_info_set(game_state, current_player)

        # Get or create node
        if info_set not in self.nodes:
            self.nodes[info_set] = ImprovedNeuralCFRNode()
        node = self.nodes[info_set]

        # Get strategy
        if current_player == player:
            # For the sampling player, use current strategy
            strategy = self._get_strategy_with_hand_strength(game_state, current_player, node)
        else:
            # For opponent, use average strategy
            strategy = node.get_average_strategy()

        # Get valid actions
        valid_actions = self._get_valid_actions(game_state)

        if current_player == player:
            # Sample action for update player
            action_probs = strategy[valid_actions]
            action_probs = action_probs / np.sum(action_probs)
            action_idx = np.random.choice(len(valid_actions), p=action_probs)
            action = valid_actions[action_idx]

            # Apply action
            next_state = self._apply_action(game_state.copy(), action)

            # Recurse
            utility = self._external_sampling_cfr(
                next_state, player,
                reach_p0 * (strategy[action] if current_player == 0 else 1.0),
                reach_p1 * (strategy[action] if current_player == 1 else 1.0),
                depth + 1
            )

            # Update regrets
            action_utilities = np.zeros(len(ImprovedAction))
            action_utilities[action] = utility

            # Calculate counterfactual values for all actions
            for a in valid_actions:
                if a != action:
                    next_state_cf = self._apply_action(game_state.copy(), a)
                    action_utilities[a] = self._external_sampling_cfr(
                        next_state_cf, player, reach_p0, reach_p1, depth + 1
                    )

            # Update node
            cf_reach = reach_p1 if player == 0 else reach_p0
            node.update_regrets(action_utilities, utility, cf_reach)

            return utility
        else:
            # Sample action for opponent
            action_probs = strategy[valid_actions]
            action_probs = action_probs / np.sum(action_probs)
            action_idx = np.random.choice(len(valid_actions), p=action_probs)
            action = valid_actions[action_idx]

            # Apply and recurse
            next_state = self._apply_action(game_state.copy(), action)
            return self._external_sampling_cfr(
                next_state, player,
                reach_p0 * (strategy[action] if current_player == 0 else 1.0),
                reach_p1 * (strategy[action] if current_player == 1 else 1.0),
                depth + 1
            )

    def _outcome_sampling_cfr(self, game_state: Dict, player: int) -> float:
        """Outcome sampling with importance sampling"""
        # Similar to external sampling but with importance sampling correction
        return self._external_sampling_cfr(game_state, player, 1.0, 1.0)

    def _best_response_cfr(self, game_state: Dict, player: int) -> float:
        """Compute best response value"""
        # Use current strategy and compute best response
        return self._external_sampling_cfr(game_state, player, 1.0, 1.0)

    def _get_strategy_with_hand_strength(self, game_state: Dict, player: int,
                                        node: ImprovedNeuralCFRNode) -> np.ndarray:
        """Get strategy considering hand strength"""
        base_strategy = node.get_strategy()

        if not self.use_hand_strength:
            return base_strategy

        # Get hand strength information
        hole_cards = game_state['hole_cards'][player]
        community_cards = game_state['community_cards']
        pot_size = game_state['pot']
        to_call = game_state.get('to_call', 0)

        strength_info = self.hand_evaluator.get_contextual_strength(
            hole_cards, community_cards,
            game_state.get('betting_history', ''),
            pot_size, to_call
        )

        # Adjust strategy based on hand strength
        adjusted_strategy = self._adjust_strategy_for_strength(
            base_strategy, strength_info, pot_size, to_call
        )

        return adjusted_strategy

    def _adjust_strategy_for_strength(self, base_strategy: np.ndarray,
                                     strength_info: Dict, pot_size: int, to_call: int) -> np.ndarray:
        """Adjust strategy based on hand strength without hard blocking"""
        adjusted = base_strategy.copy()
        hand_percentile = strength_info['percentile']
        pot_odds = strength_info['pot_odds']
        category = strength_info['category']

        # Create adjustment weights (not hard blocks)
        if category == HandStrengthCategory.MONSTER:
            # With monsters, increase aggressive actions but keep some slow-play
            adjusted[ImprovedAction.BET_LARGE.value] *= 1.5
            adjusted[ImprovedAction.BET_OVERBET.value] *= 1.3
            adjusted[ImprovedAction.CHECK.value] *= 0.7  # Can still trap

        elif category == HandStrengthCategory.VERY_STRONG:
            # Strong hands lean toward value betting
            adjusted[ImprovedAction.BET_MEDIUM.value] *= 1.4
            adjusted[ImprovedAction.BET_LARGE.value] *= 1.2
            adjusted[ImprovedAction.FOLD.value] *= 0.3

        elif category == HandStrengthCategory.WEAK or category == HandStrengthCategory.TRASH:
            # Weak hands can still bluff but prefer checking/folding
            adjusted[ImprovedAction.FOLD.value] *= 1.3
            adjusted[ImprovedAction.CHECK.value] *= 1.2
            adjusted[ImprovedAction.CALL.value] *= 0.8
            # Keep some bluff frequency
            adjusted[ImprovedAction.BET_LARGE.value] *= 0.5  # Reduced but not zero

        # Drawing hands
        if strength_info['is_drawing']:
            # Semi-bluff more with draws
            adjusted[ImprovedAction.BET_SMALL.value] *= 1.2
            adjusted[ImprovedAction.BET_MEDIUM.value] *= 1.1

        # Pot odds adjustment for calling
        if pot_odds > 0 and to_call > 0:
            if hand_percentile > pot_odds:
                # Good odds to call
                adjusted[ImprovedAction.CALL.value] *= 1.5
            else:
                # Bad odds to call
                adjusted[ImprovedAction.CALL.value] *= 0.7
                adjusted[ImprovedAction.FOLD.value] *= 1.2

        # Renormalize
        total = np.sum(adjusted)
        if total > 0:
            adjusted /= total

        return adjusted

    def _sample_game_state(self) -> Dict:
        """Sample a realistic game state"""
        deck = Deck()
        deck.reset()

        # Deal hole cards
        hole_cards = [
            [deck.draw(), deck.draw()],
            [deck.draw(), deck.draw()]
        ]

        # Random street
        street = random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.25, 0.15])[0]

        # Deal community cards
        community_cards = []
        if street >= 1:
            community_cards.extend([deck.draw() for _ in range(3)])
        if street >= 2:
            community_cards.append(deck.draw())
        if street >= 3:
            community_cards.append(deck.draw())

        # Random stacks and pot
        stacks = [random.randint(800, 1200) for _ in range(2)]
        pot = random.randint(20, 100)

        return {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot': pot,
            'stacks': stacks,
            'current_player': random.randint(0, 1),
            'street': street,
            'betting_history': '',
            'to_call': 0
        }

    def _get_info_set(self, game_state: Dict, player: int) -> str:
        """Create information set identifier"""
        hole_cards = game_state['hole_cards'][player]
        community_cards = game_state['community_cards']
        betting_history = game_state.get('betting_history', '')

        # Include hand strength category for better bucketing
        category = self.hand_evaluator.get_hand_category(hole_cards, community_cards)

        # Create key
        hole_str = ''.join([f"{c.rank}{c.suit[0]}" for c in hole_cards])
        comm_str = ''.join([f"{c.rank}{c.suit[0]}" for c in community_cards])

        # Bucket pot and stacks
        pot_bucket = min(game_state['pot'] // 50, 10)
        stack_bucket = min(game_state['stacks'][player] // 100, 10)

        return f"{hole_str}|{comm_str}|{betting_history}|{category.value}|{pot_bucket}|{stack_bucket}"

    def _get_valid_actions(self, game_state: Dict) -> List[int]:
        """Get valid actions in current state"""
        valid = []
        player = game_state['current_player']
        stack = game_state['stacks'][player]
        pot = game_state['pot']
        to_call = game_state.get('to_call', 0)

        # Always can fold
        valid.append(ImprovedAction.FOLD.value)

        # Check/Call
        if to_call == 0:
            valid.append(ImprovedAction.CHECK.value)
        elif stack >= to_call:
            valid.append(ImprovedAction.CALL.value)

        # Betting actions (if we have chips)
        if stack > to_call:
            remaining = stack - to_call

            # Different bet sizes
            if remaining > pot * 0.25:
                valid.append(ImprovedAction.BET_SMALL.value)
            if remaining > pot * 0.5:
                valid.append(ImprovedAction.BET_MEDIUM.value)
            if remaining > pot * 0.9:
                valid.append(ImprovedAction.BET_LARGE.value)
            if remaining > pot * 1.5:
                valid.append(ImprovedAction.BET_OVERBET.value)

            # All-in always available if we have chips
            valid.append(ImprovedAction.ALL_IN.value)

        return valid

    def _apply_action(self, game_state: Dict, action: int) -> Dict:
        """Apply action to game state"""
        new_state = game_state.copy()
        player = new_state['current_player']

        if action == ImprovedAction.FOLD.value:
            new_state['is_terminal'] = True
            new_state['winner'] = 1 - player

        elif action == ImprovedAction.CHECK.value:
            new_state['betting_history'] += 'k'
            new_state['current_player'] = 1 - player

        elif action == ImprovedAction.CALL.value:
            call_amount = new_state['to_call']
            new_state['stacks'][player] -= call_amount
            new_state['pot'] += call_amount
            new_state['betting_history'] += 'c'
            new_state['current_player'] = 1 - player
            new_state['to_call'] = 0

        elif action in [ImprovedAction.BET_SMALL.value, ImprovedAction.BET_MEDIUM.value,
                       ImprovedAction.BET_LARGE.value, ImprovedAction.BET_OVERBET.value]:
            # Calculate bet size
            bet_sizes = {
                ImprovedAction.BET_SMALL.value: 0.33,
                ImprovedAction.BET_MEDIUM.value: 0.66,
                ImprovedAction.BET_LARGE.value: 1.0,
                ImprovedAction.BET_OVERBET.value: 1.75
            }

            bet_fraction = bet_sizes[action]
            bet_amount = int(new_state['pot'] * bet_fraction)
            bet_amount = min(bet_amount, new_state['stacks'][player])

            new_state['stacks'][player] -= bet_amount
            new_state['pot'] += bet_amount
            new_state['betting_history'] += 'b'
            new_state['current_player'] = 1 - player
            new_state['to_call'] = bet_amount

        elif action == ImprovedAction.ALL_IN.value:
            all_in_amount = new_state['stacks'][player]
            new_state['stacks'][player] = 0
            new_state['pot'] += all_in_amount
            new_state['betting_history'] += 'a'
            new_state['current_player'] = 1 - player
            new_state['to_call'] = all_in_amount

        # Check if we should move to next street
        if self._should_advance_street(new_state):
            new_state['street'] += 1
            new_state['to_call'] = 0

            # Deal more community cards if needed
            if new_state['street'] == 1 and len(new_state['community_cards']) == 0:
                # Deal flop
                deck = self._get_remaining_deck(new_state)
                new_state['community_cards'] = [deck.draw() for _ in range(3)]
            elif new_state['street'] == 2 and len(new_state['community_cards']) == 3:
                # Deal turn
                deck = self._get_remaining_deck(new_state)
                new_state['community_cards'].append(deck.draw())
            elif new_state['street'] == 3 and len(new_state['community_cards']) == 4:
                # Deal river
                deck = self._get_remaining_deck(new_state)
                new_state['community_cards'].append(deck.draw())

        return new_state

    def _should_advance_street(self, game_state: Dict) -> bool:
        """Check if we should move to next street"""
        history = game_state.get('betting_history', '')
        if not history:
            return False

        # Both players checked
        if history[-2:] == 'kk':
            return True

        # Call after bet/raise
        if 'c' in history and history[-1] == 'c':
            return True

        return False

    def _get_remaining_deck(self, game_state: Dict) -> Deck:
        """Get deck with remaining cards"""
        deck = Deck()
        deck.reset()

        # Remove known cards (match by value and suit)
        for hand in game_state['hole_cards']:
            for card in hand:
                for deck_card in deck.cards[:]:
                    if deck_card.value == card.value and deck_card.suit == card.suit:
                        deck.cards.remove(deck_card)
                        break

        for card in game_state['community_cards']:
            for deck_card in deck.cards[:]:
                if deck_card.value == card.value and deck_card.suit == card.suit:
                    deck.cards.remove(deck_card)
                    break

        return deck

    def _is_terminal(self, game_state: Dict) -> bool:
        """Check if game state is terminal"""
        if game_state.get('is_terminal', False):
            return True

        # Check if someone is all-in
        if any(s == 0 for s in game_state['stacks']):
            return True

        # Check if we're at showdown
        if game_state['street'] >= 3 and self._should_advance_street(game_state):
            return True

        return False

    def _evaluate_terminal(self, game_state: Dict, player: int) -> float:
        """Evaluate terminal node with proper payoffs"""
        if game_state.get('winner') is not None:
            # Someone folded
            winner = game_state['winner']
            pot = game_state['pot']

            # Calculate actual profit (pot minus our investment)
            starting_stack = 1000  # Assumed starting stack
            our_investment = starting_stack - game_state['stacks'][player]

            if winner == player:
                return pot - our_investment
            else:
                return -our_investment
        else:
            # Showdown
            hole_cards = game_state['hole_cards']
            community_cards = game_state['community_cards']

            # Evaluate both hands
            hand_values = []
            for i in range(2):
                if len(community_cards) >= 3:
                    hand_values.append(evaluate_hand(hole_cards[i] + community_cards))
                else:
                    # Preflop all-in, evaluate with future cards
                    deck = self._get_remaining_deck(game_state)
                    future_cards = []
                    for _ in range(5 - len(community_cards)):
                        if deck.cards:
                            future_cards.append(deck.draw())
                    hand_values.append(evaluate_hand(hole_cards[i] + community_cards + future_cards))

            # Determine winner
            pot = game_state['pot']
            starting_stack = 1000
            our_investment = starting_stack - game_state['stacks'][player]

            if hand_values[player] > hand_values[1-player]:
                return pot - our_investment
            elif hand_values[player] < hand_values[1-player]:
                return -our_investment
            else:
                # Split pot
                return (pot / 2) - our_investment

    def _update_neural_networks(self):
        """Update neural networks from CFR data"""
        if not self.use_neural_networks or len(self.nodes) < 100:
            return

        # Sample batch of nodes
        batch_size = min(32, len(self.nodes))
        sampled_nodes = random.sample(list(self.nodes.items()), batch_size)

        # Prepare training data
        features_batch = []
        targets_batch = []

        for info_set, node in sampled_nodes:
            # Parse info set to get features
            # This is simplified - in practice would need proper parsing
            strategy = node.get_average_strategy()

            # Create dummy features for now
            features = torch.randn(1, 50)
            hand_strength = torch.tensor([[0.5]])
            pot_odds = torch.tensor([[0.3]])
            position = torch.tensor([[0.5]])
            board_texture = torch.tensor([[0.5]])

            features_batch.append((features, hand_strength, pot_odds, position, board_texture))
            targets_batch.append(torch.tensor(strategy).unsqueeze(0))

        # Update network
        self.optimizer.zero_grad()

        total_loss = 0
        for (features, hs, po, pos, bt), target in zip(features_batch, targets_batch):
            output = self.strategy_ensemble(features, hs, po, pos, bt)
            loss = F.kl_div(torch.log(output + 1e-8), target, reduction='batchmean')
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()

    def _print_progress(self, current: int, total: int, stage: str):
        """Print training progress"""
        progress = current / total * 100
        elapsed = time.time() - self.training_start_time
        eta = elapsed / (current + 1) * (total - current)

        print(f"Progress: {progress:.1f}% | Nodes: {len(self.nodes):,} | "
              f"Stage: {stage} | ETA: {eta/60:.1f}m")

    def _print_final_stats(self):
        """Print final training statistics"""
        elapsed = time.time() - self.training_start_time

        print("\n" + "=" * 60)
        print("IMPROVED NEURAL-ENHANCED CFR TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Information sets: {len(self.nodes):,}")
        print(f"Iterations: {self.iteration_count}")
        print(f"Hand strength awareness: Active")
        print(f"Neural networks: {'Active' if self.use_neural_networks else 'Disabled'}")
        print("=" * 60)

    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int,
                   stack_size: int, position: int, opponent_stack: int = 1000) -> Action:
        """Get action for gameplay with hand strength awareness"""

        # Get hand strength information
        strength_info = self.hand_evaluator.get_contextual_strength(
            hole_cards, community_cards, betting_history, pot_size, to_call
        )

        # Create game state for info set
        game_state = {
            'hole_cards': [hole_cards, []],
            'community_cards': community_cards,
            'pot': pot_size,
            'stacks': [stack_size, opponent_stack],
            'current_player': 0,
            'street': len(community_cards) // 2,
            'betting_history': betting_history,
            'to_call': to_call
        }

        # Get info set
        info_set = self._get_info_set(game_state, 0)

        # Get strategy
        if info_set in self.nodes:
            node = self.nodes[info_set]
            base_strategy = node.get_average_strategy()

            # Adjust for hand strength
            strategy = self._adjust_strategy_for_strength(
                base_strategy, strength_info, pot_size, to_call
            )
        else:
            # Use hand-strength-aware default
            strategy = self._get_default_strategy(strength_info, pot_size, to_call)

        # Get valid actions and sample
        valid_actions = self._get_valid_actions(game_state)
        valid_probs = [strategy[a] for a in valid_actions]
        valid_probs = np.array(valid_probs) / np.sum(valid_probs)

        chosen_action = np.random.choice(valid_actions, p=valid_probs)

        # Convert to game engine action
        action_map = {
            ImprovedAction.FOLD.value: Action.FOLD,
            ImprovedAction.CHECK.value: Action.CHECK,
            ImprovedAction.CALL.value: Action.CALL,
            ImprovedAction.BET_SMALL.value: Action.RAISE,
            ImprovedAction.BET_MEDIUM.value: Action.RAISE,
            ImprovedAction.BET_LARGE.value: Action.RAISE,
            ImprovedAction.BET_OVERBET.value: Action.RAISE,
            ImprovedAction.ALL_IN.value: Action.ALL_IN
        }

        # Update statistics
        self.strategy_stats[info_set]['used'] += 1
        if info_set in self.nodes:
            self.strategy_stats[info_set]['learned'] += 1

        return action_map.get(chosen_action, Action.CALL)

    def _get_default_strategy(self, strength_info: Dict, pot_size: int, to_call: int) -> np.ndarray:
        """Get hand-strength-aware default strategy"""
        strategy = np.zeros(len(ImprovedAction))
        category = strength_info['category']
        percentile = strength_info['percentile']
        pot_odds = strength_info['pot_odds']

        if category in [HandStrengthCategory.MONSTER, HandStrengthCategory.VERY_STRONG]:
            # Strong hands - value bet heavy
            strategy[ImprovedAction.FOLD.value] = 0.02
            strategy[ImprovedAction.CHECK.value] = 0.1
            strategy[ImprovedAction.CALL.value] = 0.15
            strategy[ImprovedAction.BET_SMALL.value] = 0.15
            strategy[ImprovedAction.BET_MEDIUM.value] = 0.25
            strategy[ImprovedAction.BET_LARGE.value] = 0.25
            strategy[ImprovedAction.BET_OVERBET.value] = 0.05
            strategy[ImprovedAction.ALL_IN.value] = 0.03

        elif category in [HandStrengthCategory.STRONG, HandStrengthCategory.GOOD]:
            # Good hands - balanced
            strategy[ImprovedAction.FOLD.value] = 0.05
            strategy[ImprovedAction.CHECK.value] = 0.2
            strategy[ImprovedAction.CALL.value] = 0.25
            strategy[ImprovedAction.BET_SMALL.value] = 0.2
            strategy[ImprovedAction.BET_MEDIUM.value] = 0.2
            strategy[ImprovedAction.BET_LARGE.value] = 0.08
            strategy[ImprovedAction.BET_OVERBET.value] = 0.01
            strategy[ImprovedAction.ALL_IN.value] = 0.01

        elif category == HandStrengthCategory.MEDIUM:
            # Medium hands - cautious
            strategy[ImprovedAction.FOLD.value] = 0.15
            strategy[ImprovedAction.CHECK.value] = 0.35
            strategy[ImprovedAction.CALL.value] = 0.25
            strategy[ImprovedAction.BET_SMALL.value] = 0.15
            strategy[ImprovedAction.BET_MEDIUM.value] = 0.08
            strategy[ImprovedAction.BET_LARGE.value] = 0.01
            strategy[ImprovedAction.BET_OVERBET.value] = 0.005
            strategy[ImprovedAction.ALL_IN.value] = 0.005

        else:
            # Weak hands - defensive with some bluffs
            strategy[ImprovedAction.FOLD.value] = 0.35
            strategy[ImprovedAction.CHECK.value] = 0.35
            strategy[ImprovedAction.CALL.value] = 0.15
            strategy[ImprovedAction.BET_SMALL.value] = 0.08  # Some bluffs
            strategy[ImprovedAction.BET_MEDIUM.value] = 0.04
            strategy[ImprovedAction.BET_LARGE.value] = 0.02  # Occasional big bluff
            strategy[ImprovedAction.BET_OVERBET.value] = 0.005
            strategy[ImprovedAction.ALL_IN.value] = 0.005

        # Adjust for pot odds when facing a bet
        if to_call > 0 and percentile > pot_odds:
            strategy[ImprovedAction.CALL.value] *= 1.5
            strategy[ImprovedAction.FOLD.value] *= 0.7
        elif to_call > 0:
            strategy[ImprovedAction.FOLD.value] *= 1.3
            strategy[ImprovedAction.CALL.value] *= 0.7

        # Normalize
        total = np.sum(strategy)
        if total > 0:
            strategy /= total

        return strategy

    def get_raise_size(self, pot_size: int, current_bet: int = 0,
                       player_chips: int = 1000, min_raise: int = 20) -> int:
        """Get raise size based on strategy"""
        # Use pot-relative sizing
        if pot_size <= 0:
            return min_raise

        # Random selection weighted by situation
        sizes = [
            max(min_raise, int(pot_size * 0.33)),   # Small
            max(min_raise, int(pot_size * 0.66)),   # Medium
            max(min_raise, int(pot_size * 1.0)),    # Large
            max(min_raise, int(pot_size * 1.5))     # Overbet
        ]

        # Weight based on stack depth
        if player_chips > pot_size * 3:
            # Deep stacked - can use all sizes
            weights = [0.3, 0.35, 0.25, 0.1]
        elif player_chips > pot_size * 1.5:
            # Medium stacked
            weights = [0.35, 0.4, 0.2, 0.05]
        else:
            # Short stacked
            weights = [0.5, 0.35, 0.1, 0.05]

        return random.choices(sizes, weights=weights)[0]

    def save(self, filename: str):
        """Save the model"""
        data = {
            'model_type': 'ImprovedNeuralEnhancedCFR',
            'model_version': '2.0',
            'nodes': self.nodes,
            'iterations': self.iteration_count,
            'strategy_stats': dict(self.strategy_stats)
        }

        if self.use_neural_networks:
            data['neural_state'] = self.strategy_ensemble.state_dict()

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Improved Neural-Enhanced CFR saved to {filename}")

    def load(self, filename: str):
        """Load the model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.nodes = data['nodes']
        self.iteration_count = data.get('iterations', 0)
        self.strategy_stats = defaultdict(lambda: {'used': 0, 'learned': 0},
                                        data.get('strategy_stats', {}))

        if 'neural_state' in data and self.use_neural_networks:
            self.strategy_ensemble.load_state_dict(data['neural_state'])

        print(f"Improved Neural-Enhanced CFR loaded from {filename}")
        print(f"  Information sets: {len(self.nodes):,}")
        print(f"  Iterations: {self.iteration_count}")

    def get_strategy_usage_stats(self) -> Dict:
        """Get strategy usage statistics"""
        total_used = sum(s['used'] for s in self.strategy_stats.values())
        total_learned = sum(s['learned'] for s in self.strategy_stats.values())

        return {
            'total_used': total_used,
            'total_learned': total_learned,
            'learned_percentage': (total_learned / total_used * 100) if total_used > 0 else 0,
            'unique_situations': len(self.strategy_stats)
        }

    def reset_strategy_stats(self):
        """Reset strategy usage statistics"""
        self.strategy_stats = defaultdict(lambda: {'used': 0, 'learned': 0})

    def print_strategy_usage(self):
        """Print strategy usage statistics"""
        stats = self.get_strategy_usage_stats()
        print(f"\nStrategy Usage Statistics:")
        print(f"  Unique situations: {stats['unique_situations']}")
        print(f"  Total decisions: {stats['total_used']}")
        print(f"  Using learned strategy: {stats['learned_percentage']:.1f}%")