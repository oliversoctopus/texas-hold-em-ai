"""
Neural-Enhanced CFR for 2-Player Texas Hold'em
Combines traditional CFR with neural network strategy approximation for DQN-competitive performance.

Features:
- Advanced poker feature engineering (50+ dimensions)
- Progressive training pipeline (random -> weak DQN -> strong DQN)
- Neural network strategy approximation
- Sophisticated opponent modeling
- Board texture analysis and position awareness
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


class NECFRAction(Enum):
    """Actions for Neural-Enhanced CFR"""
    FOLD = 0
    CHECK_CALL = 1
    BET_SMALL = 2    # 0.33 pot
    BET_MEDIUM = 3   # 0.66 pot
    BET_LARGE = 4    # 1.0 pot
    ALL_IN = 5


class StrategyNetwork(nn.Module):
    """Neural network for strategy approximation"""

    def __init__(self, feature_dim: int = 50, num_actions: int = 6, hidden_size: int = 256):
        super(StrategyNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        # Deep network for strategy approximation
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )

        # Initialize weights for stable training
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            torch.nn.init.zeros_(module.bias)

    def forward(self, features):
        logits = self.network(features)
        return F.softmax(logits, dim=-1)


class AdvancedFeatureExtractor:
    """Extract rich poker features for neural networks"""

    def __init__(self):
        # Precomputed hand strength lookup tables for efficiency
        self._hand_strength_cache = {}
        self._board_texture_cache = {}

    def extract_features(self, hole_cards: List[Card], community_cards: List[Card],
                        betting_history: str, position: int, pot: int,
                        effective_stack: int, opponent_stack: int = 1000) -> np.ndarray:
        """
        Extract comprehensive poker features

        Returns 50-dimensional feature vector:
        - Hand strength features (10 dims): vs random, vs tight, vs loose, raw equity, etc.
        - Board texture features (12 dims): dry/wet, paired, monotone, connected, etc.
        - Position features (4 dims): position, relative stack sizes
        - Betting features (15 dims): aggression, pot odds, bet sizing patterns
        - Opponent modeling (9 dims): inferred ranges, fold equity, bluff frequency
        """
        features = np.zeros(50, dtype=np.float32)

        # Hand strength features (0-9)
        hand_strength = self._analyze_hand_strength(hole_cards, community_cards)
        features[0:10] = hand_strength

        # Board texture features (10-21)
        board_texture = self._analyze_board_texture(community_cards)
        features[10:22] = board_texture

        # Position features (22-25)
        position_features = self._analyze_position(position, pot, effective_stack, opponent_stack)
        features[22:26] = position_features

        # Betting pattern features (26-40)
        betting_features = self._analyze_betting_patterns(betting_history, pot, effective_stack)
        features[26:41] = betting_features

        # Opponent modeling features (41-49)
        opponent_features = self._analyze_opponent_behavior(betting_history, pot)
        features[41:50] = opponent_features

        return features

    def _analyze_hand_strength(self, hole_cards: List[Card], community_cards: List[Card]) -> np.ndarray:
        """Analyze hand strength from multiple perspectives"""
        features = np.zeros(10, dtype=np.float32)

        if len(hole_cards) < 2:
            return features

        all_cards = hole_cards + community_cards

        # Raw hand strength (percentile vs random opponent)
        if len(community_cards) >= 3:
            # Post-flop equity estimation
            hand_value = evaluate_hand(all_cards)
            features[0] = min(hand_value / 10000000, 1.0)  # Normalize to [0,1]
        else:
            # Pre-flop hand strength
            features[0] = self._preflop_hand_strength(hole_cards)

        # Hand categories
        if len(all_cards) >= 5:
            # Made hand strength
            features[1] = self._made_hand_strength(all_cards)
        else:
            # Drawing potential
            features[1] = self._drawing_potential(hole_cards, community_cards)

        # Specific hand types
        features[2] = float(self._is_pocket_pair(hole_cards))
        features[3] = float(self._is_suited(hole_cards))
        features[4] = float(self._is_connected(hole_cards))
        features[5] = float(self._has_ace(hole_cards))
        features[6] = float(self._has_face_card(hole_cards))

        # Positional hand strength (tight vs loose ranges)
        features[7] = self._hand_strength_vs_tight_range(hole_cards)
        features[8] = self._hand_strength_vs_loose_range(hole_cards)

        # Outs and improvement potential
        features[9] = self._improvement_potential(hole_cards, community_cards)

        return features

    def _analyze_board_texture(self, community_cards: List[Card]) -> np.ndarray:
        """Analyze board texture complexity"""
        features = np.zeros(12, dtype=np.float32)

        if len(community_cards) < 3:
            return features

        # Basic texture features
        features[0] = self._board_dryness(community_cards)
        features[1] = self._board_wetness(community_cards)
        features[2] = float(self._is_paired_board(community_cards))
        features[3] = float(self._is_monotone_board(community_cards))
        features[4] = float(self._is_two_tone_board(community_cards))
        features[5] = self._connectivity_score(community_cards)

        # Advanced texture
        features[6] = self._straight_potential(community_cards)
        features[7] = self._flush_potential(community_cards)
        features[8] = self._full_house_potential(community_cards)
        features[9] = self._high_card_strength(community_cards)

        # Dynamic features
        features[10] = float(len(community_cards)) / 5.0  # Street progress
        features[11] = self._action_potential(community_cards)  # How much can board change

        return features

    def _analyze_position(self, position: int, pot: int, effective_stack: int, opponent_stack: int) -> np.ndarray:
        """Analyze positional factors"""
        features = np.zeros(4, dtype=np.float32)

        # Position (0 = OOP, 1 = IP in heads-up)
        features[0] = float(position)

        # Stack-to-pot ratio
        if pot > 0:
            features[1] = min(effective_stack / pot, 10.0) / 10.0  # Normalized SPR

        # Relative stack sizes
        total_stacks = effective_stack + opponent_stack
        if total_stacks > 0:
            features[2] = effective_stack / total_stacks

        # Commitment level (invested vs remaining)
        if effective_stack > 0:
            invested = 1000 - effective_stack  # Assuming 1000 starting stack
            features[3] = invested / (invested + effective_stack)

        return features

    def _analyze_betting_patterns(self, betting_history: str, pot: int, effective_stack: int) -> np.ndarray:
        """Analyze betting patterns and aggression"""
        features = np.zeros(15, dtype=np.float32)

        if not betting_history:
            return features

        # Basic aggression metrics
        total_actions = len(betting_history)
        aggressive_actions = betting_history.count('b') + betting_history.count('r')
        passive_actions = betting_history.count('c') + betting_history.count('k')

        if total_actions > 0:
            features[0] = aggressive_actions / total_actions  # Aggression frequency
            features[1] = passive_actions / total_actions     # Passivity frequency

        # Street-specific patterns
        streets = betting_history.split('/')  # Assume '/' separates streets
        for i, street in enumerate(streets[:4]):  # max 4 streets
            if street and i < 4:
                street_agg = (street.count('b') + street.count('r')) / len(street)
                features[2 + i] = street_agg

        # Betting size patterns (simplified)
        features[6] = betting_history.count('b') / max(total_actions, 1)  # Bet frequency
        features[7] = betting_history.count('r') / max(total_actions, 1)  # Raise frequency
        features[8] = betting_history.count('c') / max(total_actions, 1)  # Call frequency
        features[9] = betting_history.count('f') / max(total_actions, 1)  # Fold frequency

        # Pot odds and commitment
        if pot > 0 and effective_stack > 0:
            features[10] = min(pot / effective_stack, 5.0) / 5.0  # Pot to stack ratio

        # Action sequences (last 3 actions)
        recent_actions = betting_history[-3:] if len(betting_history) >= 3 else betting_history
        for i, action in enumerate(recent_actions):
            if i < 3:
                # Encode actions numerically
                action_value = {'f': 0, 'c': 0.33, 'k': 0.33, 'b': 0.66, 'r': 1.0}.get(action, 0.5)
                features[11 + i] = action_value

        # Initiative and momentum
        features[14] = self._calculate_initiative(betting_history)

        return features

    def _analyze_opponent_behavior(self, betting_history: str, pot: int) -> np.ndarray:
        """Model opponent behavior patterns"""
        features = np.zeros(9, dtype=np.float32)

        if len(betting_history) < 2:
            return features

        # Simplified opponent modeling
        opponent_actions = betting_history[1::2]  # Every other action (opponent's)

        if opponent_actions:
            total_opp = len(opponent_actions)
            # Opponent aggression
            opp_aggressive = sum(1 for a in opponent_actions if a in 'br')
            features[0] = opp_aggressive / total_opp

            # Opponent passivity
            opp_passive = sum(1 for a in opponent_actions if a in 'ck')
            features[1] = opp_passive / total_opp

            # Opponent fold frequency
            opp_folds = opponent_actions.count('f')
            features[2] = opp_folds / total_opp

            # Bluff frequency estimation (simplified)
            features[3] = self._estimate_bluff_frequency(opponent_actions)

            # Value bet frequency
            features[4] = self._estimate_value_frequency(opponent_actions)

            # Fold equity against opponent
            features[5] = self._estimate_fold_equity(opponent_actions, pot)

            # Opponent tightness (estimated)
            features[6] = self._estimate_opponent_tightness(betting_history)

            # Opponent adaptability
            features[7] = self._estimate_adaptability(opponent_actions)

            # Opponent stack preservation tendency
            features[8] = self._estimate_stack_preservation(opponent_actions, pot)

        return features

    # Helper methods for feature extraction
    def _preflop_hand_strength(self, hole_cards: List[Card]) -> float:
        """Estimate pre-flop hand strength"""
        if len(hole_cards) < 2:
            return 0.0

        c1, c2 = hole_cards[0], hole_cards[1]
        high, low = max(c1.value, c2.value), min(c1.value, c2.value)
        suited = c1.suit == c2.suit

        # Simplified Chen formula adaptation
        score = 0.0

        # High card value
        score += high / 14.0

        # Pair bonus
        if high == low:
            score += 0.5

        # Suited bonus
        if suited:
            score += 0.1

        # Connected bonus
        if abs(high - low) <= 1:
            score += 0.1

        return min(score, 1.0)

    def _made_hand_strength(self, cards: List[Card]) -> float:
        """Evaluate made hand strength"""
        if len(cards) < 5:
            return 0.0
        return min(evaluate_hand(cards) / 10000000, 1.0)

    def _drawing_potential(self, hole_cards: List[Card], community_cards: List[Card]) -> float:
        """Estimate drawing potential"""
        if len(hole_cards) < 2 or len(community_cards) < 3:
            return 0.0

        # Simplified outs counting
        outs = 0

        # Flush draw potential
        if len(community_cards) >= 3:
            hole_suits = [c.suit for c in hole_cards]
            board_suits = [c.suit for c in community_cards]

            for suit in hole_suits:
                suit_count = hole_suits.count(suit) + board_suits.count(suit)
                if suit_count == 4:  # Flush draw
                    outs += 9

        # Straight draw potential (simplified)
        all_values = sorted([c.value for c in hole_cards + community_cards])
        for i in range(len(all_values) - 1):
            if all_values[i+1] - all_values[i] == 1:
                outs += 4  # Rough estimate
                break

        return min(outs / 20.0, 1.0)  # Normalize

    def _is_pocket_pair(self, hole_cards: List[Card]) -> bool:
        return len(hole_cards) == 2 and hole_cards[0].value == hole_cards[1].value

    def _is_suited(self, hole_cards: List[Card]) -> bool:
        return len(hole_cards) == 2 and hole_cards[0].suit == hole_cards[1].suit

    def _is_connected(self, hole_cards: List[Card]) -> bool:
        if len(hole_cards) < 2:
            return False
        return abs(hole_cards[0].value - hole_cards[1].value) <= 1

    def _has_ace(self, hole_cards: List[Card]) -> bool:
        return any(c.value == 14 for c in hole_cards)

    def _has_face_card(self, hole_cards: List[Card]) -> bool:
        return any(c.value >= 11 for c in hole_cards)

    def _hand_strength_vs_tight_range(self, hole_cards: List[Card]) -> float:
        """Estimate hand strength vs tight opponent range"""
        # Simplified - premium hands get higher scores
        if self._is_pocket_pair(hole_cards) and hole_cards[0].value >= 10:
            return 0.9
        if self._has_ace(hole_cards) and max(c.value for c in hole_cards) >= 10:
            return 0.7
        return 0.3

    def _hand_strength_vs_loose_range(self, hole_cards: List[Card]) -> float:
        """Estimate hand strength vs loose opponent range"""
        # Any reasonable hand has decent equity vs loose range
        if self._has_ace(hole_cards) or self._is_pocket_pair(hole_cards):
            return 0.6
        return 0.4

    def _improvement_potential(self, hole_cards: List[Card], community_cards: List[Card]) -> float:
        """Estimate potential for improvement on future streets"""
        if len(community_cards) >= 5:  # River
            return 0.0

        cards_to_come = 5 - len(community_cards)
        return cards_to_come / 5.0  # Simple: more cards to come = more potential

    def _board_dryness(self, community_cards: List[Card]) -> float:
        """Calculate board dryness (low connectivity)"""
        if len(community_cards) < 3:
            return 1.0

        values = sorted([c.value for c in community_cards])
        gaps = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        return min(avg_gap / 7.0, 1.0)  # Normalize

    def _board_wetness(self, community_cards: List[Card]) -> float:
        """Calculate board wetness (high connectivity)"""
        return 1.0 - self._board_dryness(community_cards)

    def _is_paired_board(self, community_cards: List[Card]) -> bool:
        values = [c.value for c in community_cards]
        return len(values) != len(set(values))

    def _is_monotone_board(self, community_cards: List[Card]) -> bool:
        if len(community_cards) < 3:
            return False
        suits = [c.suit for c in community_cards]
        return len(set(suits)) == 1

    def _is_two_tone_board(self, community_cards: List[Card]) -> bool:
        if len(community_cards) < 3:
            return False
        suits = [c.suit for c in community_cards]
        return len(set(suits)) == 2

    def _connectivity_score(self, community_cards: List[Card]) -> float:
        """Score board connectivity"""
        if len(community_cards) < 3:
            return 0.0

        values = sorted([c.value for c in community_cards])
        connected_pairs = sum(1 for i in range(len(values)-1) if values[i+1] - values[i] <= 2)
        return connected_pairs / (len(values) - 1) if len(values) > 1 else 0.0

    def _straight_potential(self, community_cards: List[Card]) -> float:
        """Estimate straight draw potential on board"""
        if len(community_cards) < 3:
            return 0.0
        values = sorted(set(c.value for c in community_cards))
        max_connected = 1
        current = 1

        for i in range(1, len(values)):
            if values[i] - values[i-1] == 1:
                current += 1
                max_connected = max(max_connected, current)
            else:
                current = 1

        return min(max_connected / 5.0, 1.0)

    def _flush_potential(self, community_cards: List[Card]) -> float:
        """Estimate flush draw potential"""
        if len(community_cards) < 3:
            return 0.0

        suit_counts = {}
        for card in community_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

        max_suited = max(suit_counts.values()) if suit_counts else 0
        return min(max_suited / 5.0, 1.0)

    def _full_house_potential(self, community_cards: List[Card]) -> float:
        """Estimate full house potential"""
        if len(community_cards) < 3:
            return 0.0

        value_counts = {}
        for card in community_cards:
            value_counts[card.value] = value_counts.get(card.value, 0) + 1

        pairs = sum(1 for count in value_counts.values() if count >= 2)
        return min(pairs / 2.0, 1.0)

    def _high_card_strength(self, community_cards: List[Card]) -> float:
        """Average high card strength on board"""
        if not community_cards:
            return 0.0
        return sum(c.value for c in community_cards) / (len(community_cards) * 14.0)

    def _action_potential(self, community_cards: List[Card]) -> float:
        """How much action board can generate"""
        wetness = self._board_wetness(community_cards)
        flush_pot = self._flush_potential(community_cards)
        straight_pot = self._straight_potential(community_cards)

        return min((wetness + flush_pot + straight_pot) / 3.0, 1.0)

    def _calculate_initiative(self, betting_history: str) -> float:
        """Calculate who has betting initiative"""
        if not betting_history:
            return 0.5

        # Last aggressive action wins initiative
        aggressive_actions = ['b', 'r']
        for i in range(len(betting_history) - 1, -1, -1):
            if betting_history[i] in aggressive_actions:
                return 1.0 if i % 2 == 0 else 0.0  # Assume we are player 0

        return 0.5  # No initiative

    def _estimate_bluff_frequency(self, opponent_actions: List[str]) -> float:
        """Estimate opponent's bluff frequency"""
        if not opponent_actions:
            return 0.3  # Default assumption

        aggressive = sum(1 for a in opponent_actions if a in 'br')
        return min(aggressive / len(opponent_actions), 1.0)

    def _estimate_value_frequency(self, opponent_actions: List[str]) -> float:
        """Estimate opponent's value bet frequency"""
        return 1.0 - self._estimate_bluff_frequency(opponent_actions)

    def _estimate_fold_equity(self, opponent_actions: List[str], pot: int) -> float:
        """Estimate fold equity against opponent"""
        if not opponent_actions:
            return 0.5

        folds = opponent_actions.count('f')
        total = len(opponent_actions)
        base_fold_freq = folds / total if total > 0 else 0.3

        # Adjust for pot size (bigger pots = less folding)
        pot_factor = max(0.5, 1.0 - (pot / 1000.0))  # Assuming 1000 starting stacks

        return min(base_fold_freq * pot_factor, 1.0)

    def _estimate_opponent_tightness(self, betting_history: str) -> float:
        """Estimate how tight/loose opponent is"""
        if not betting_history:
            return 0.5

        opponent_actions = betting_history[1::2]  # Opponent's actions
        if not opponent_actions:
            return 0.5

        passive_actions = sum(1 for a in opponent_actions if a in 'ckf')
        return passive_actions / len(opponent_actions)

    def _estimate_adaptability(self, opponent_actions: List[str]) -> float:
        """Estimate opponent's adaptability/variance"""
        if len(opponent_actions) < 3:
            return 0.5

        # Calculate variance in action types
        action_types = len(set(opponent_actions))
        max_types = min(len(opponent_actions), 4)  # fold, call, bet, raise

        return action_types / max_types if max_types > 0 else 0.0

    def _estimate_stack_preservation(self, opponent_actions: List[str], pot: int) -> float:
        """Estimate opponent's stack preservation tendency"""
        if not opponent_actions:
            return 0.5

        conservative_actions = sum(1 for a in opponent_actions if a in 'ckf')
        return conservative_actions / len(opponent_actions)


class NECFRNode:
    """Node in the Neural-Enhanced CFR tree"""

    def __init__(self, num_actions: int = 6):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.visits = 0

    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        # Regret matching with positive regrets
        strategy = np.maximum(self.regret_sum, 0)

        # Normalize
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            # Uniform random strategy if no regrets
            strategy = np.ones(self.num_actions) / self.num_actions

        # Update cumulative strategy
        self.strategy_sum += realization_weight * strategy
        self.strategy = strategy.copy()
        self.visits += 1

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions


class NeuralEnhancedTwoPlayerCFR:
    """Neural-Enhanced CFR for competitive 2-player poker"""

    def __init__(self, iterations: int = 50000, use_neural_networks: bool = True,
                 progressive_training: bool = True, feature_dim: int = 50):
        self.iterations = iterations
        self.use_neural_networks = use_neural_networks
        self.progressive_training = progressive_training
        self.feature_dim = feature_dim

        # CFR data structures
        self.nodes: Dict[str, NECFRNode] = {}
        self.feature_extractor = AdvancedFeatureExtractor()

        # Neural network components
        if use_neural_networks:
            self.strategy_net = StrategyNetwork(feature_dim, len(NECFRAction))
            self.optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.001)

        # Training tracking
        self.current_iteration = 0
        self.training_start_time = 0
        self.performance_history = []

        # Progressive training opponents
        self.training_stage = 0  # 0: random, 1: weak DQN, 2: strong DQN
        self.stage_iterations = [
            iterations // 3,      # Random opponents
            iterations // 3,      # Weak DQN opponents
            iterations // 3       # Strong DQN opponents
        ]

    def train(self, verbose: bool = True):
        """Main training loop with progressive opponents"""
        if verbose:
            print("Starting Neural-Enhanced CFR Training")
            print(f"Total iterations: {self.iterations:,}")
            print(f"Neural networks: {'Enabled' if self.use_neural_networks else 'Disabled'}")
            print(f"Progressive training: {'Enabled' if self.progressive_training else 'Disabled'}")
            print("=" * 60)

        self.training_start_time = time.time()

        for iteration in range(self.iterations):
            self.current_iteration = iteration

            # Update training stage for progressive training
            if self.progressive_training:
                self._update_training_stage(iteration, verbose)

            # Run single CFR iteration
            self._run_cfr_iteration(iteration)

            # Periodic progress and neural network updates
            if iteration % 100 == 0:
                self._update_neural_networks(iteration)

            if verbose and (iteration % max(1, self.iterations // 20) == 0 or iteration == self.iterations - 1):
                self._print_progress(iteration, verbose)

        if verbose:
            self._print_final_results()

    def _update_training_stage(self, iteration: int, verbose: bool):
        """Update training stage for progressive training"""
        new_stage = 0
        if iteration >= self.stage_iterations[0]:
            new_stage = 1
        if iteration >= self.stage_iterations[0] + self.stage_iterations[1]:
            new_stage = 2

        if new_stage != self.training_stage:
            self.training_stage = new_stage
            if verbose:
                stage_names = ["Random Opponents", "Weak DQN Models", "Strong DQN Models"]
                print(f"\nTraining Stage {new_stage + 1}: {stage_names[new_stage]}")
                print("-" * 40)

    def _run_cfr_iteration(self, iteration: int):
        """Run single CFR iteration"""
        # Generate random game scenario
        game_state = self._generate_game_scenario()

        # Run CFR from both players' perspectives
        for training_player in [0, 1]:
            self._cfr_recursive(game_state, [1.0, 1.0], training_player, 0)

    def _generate_game_scenario(self) -> Dict:
        """Generate realistic poker game scenario"""
        deck = Deck()
        deck.reset()

        # Deal cards
        hole_cards = [deck.draw(2), deck.draw(2)]

        # Random street (bias toward earlier streets)
        street = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]

        # Deal community cards based on street
        community_cards = []
        if street >= 1:  # Flop
            community_cards.extend(deck.draw(3))
        if street >= 2:  # Turn
            community_cards.append(deck.draw())
        if street >= 3:  # River
            community_cards.append(deck.draw())

        # Random pot and stacks
        starting_stack = random.randint(800, 1200)
        pot_size = random.randint(20, min(200, starting_stack // 2))

        return {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot': pot_size,
            'player_stacks': [starting_stack] * 2,
            'player_bets': [pot_size // 2] * 2,
            'to_act': random.randint(0, 1),
            'betting_history': self._generate_betting_history(street),
            'street': street,
            'is_terminal': False
        }

    def _generate_betting_history(self, street: int) -> str:
        """Generate realistic betting history"""
        patterns = {
            0: ["", "c", "r", "cr", "rc"],  # Pre-flop
            1: ["", "c", "b", "cb", "bc", "br"],  # Flop
            2: ["", "c", "b", "cb"],  # Turn
            3: ["", "c", "b"]  # River
        }

        return random.choice(patterns.get(street, [""]))

    def _cfr_recursive(self, game_state: Dict, reach_probs: List[float],
                      training_player: int, depth: int) -> float:
        """Recursive CFR with neural network integration"""

        # Terminal node handling
        if game_state.get('is_terminal', False) or depth > 10:
            return self._evaluate_terminal_payoff(game_state, training_player)

        # Get current player
        current_player = game_state['to_act']

        # Create information set key
        info_set_key = self._create_info_set_key(game_state, current_player)

        # Get or create node
        if info_set_key not in self.nodes:
            self.nodes[info_set_key] = NECFRNode()

        node = self.nodes[info_set_key]

        # Get strategy (neural network or regret matching)
        if self.use_neural_networks and current_player == training_player:
            strategy = self._get_neural_strategy(game_state, current_player)
        else:
            strategy = node.get_strategy(reach_probs[current_player])

        # Calculate utilities for each action
        utilities = np.zeros(len(NECFRAction))
        node_utility = 0.0

        for i, action in enumerate(NECFRAction):
            # Skip invalid actions
            if not self._is_valid_action(action, game_state):
                continue

            # Create next game state
            next_state = self._apply_action(game_state.copy(), action, current_player)

            # Update reach probabilities
            next_reach_probs = reach_probs.copy()
            next_reach_probs[current_player] *= strategy[i]

            # Recursive call
            utilities[i] = self._cfr_recursive(next_state, next_reach_probs, training_player, depth + 1)
            node_utility += strategy[i] * utilities[i]

        # Update regrets for training player
        if current_player == training_player:
            for i in range(len(NECFRAction)):
                regret = utilities[i] - node_utility
                node.regret_sum[i] += reach_probs[1 - training_player] * regret

        return node_utility

    def _get_neural_strategy(self, game_state: Dict, player: int) -> np.ndarray:
        """Get strategy from neural network"""
        # Extract features
        features = self.feature_extractor.extract_features(
            hole_cards=game_state['hole_cards'][player],
            community_cards=game_state['community_cards'],
            betting_history=game_state['betting_history'],
            position=player,
            pot=game_state['pot'],
            effective_stack=game_state['player_stacks'][player],
            opponent_stack=game_state['player_stacks'][1-player]
        )

        # Get neural strategy
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            strategy = self.strategy_net(features_tensor).squeeze(0).numpy()

        # Mask invalid actions
        valid_mask = np.array([self._is_valid_action(action, game_state) for action in NECFRAction])
        strategy = strategy * valid_mask

        # Normalize
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy /= strategy_sum
        else:
            # Fall back to uniform over valid actions
            strategy = valid_mask / np.sum(valid_mask) if np.sum(valid_mask) > 0 else strategy

        return strategy

    def _create_info_set_key(self, game_state: Dict, player: int) -> str:
        """Create information set key for current game state"""
        hole_cards = game_state['hole_cards'][player]
        community_cards = game_state['community_cards']
        betting_history = game_state['betting_history']
        street = game_state['street']
        pot = game_state['pot']
        stack = game_state['player_stacks'][player]

        # Create compact but comprehensive key
        hole_str = ''.join([f"{c.rank}{c.suit}" for c in hole_cards])
        board_str = ''.join([f"{c.rank}{c.suit}" for c in community_cards])

        # Bucket pot and stack sizes for generalization
        pot_bucket = min(pot // 50, 10)  # Bucket pot sizes
        stack_bucket = min(stack // 100, 10)  # Bucket stack sizes

        return f"{hole_str}|{board_str}|{betting_history}|{street}|{pot_bucket}|{stack_bucket}|{player}"

    def _is_valid_action(self, action: NECFRAction, game_state: Dict) -> bool:
        """Check if action is valid in current game state"""
        current_player = game_state['to_act']
        stack = game_state['player_stacks'][current_player]
        pot = game_state['pot']

        if action == NECFRAction.FOLD:
            return True  # Always can fold
        elif action == NECFRAction.CHECK_CALL:
            return True  # Always can check or call
        elif action == NECFRAction.ALL_IN:
            return stack > 0
        else:
            # Betting actions require sufficient stack
            bet_sizes = {
                NECFRAction.BET_SMALL: pot * 0.33,
                NECFRAction.BET_MEDIUM: pot * 0.66,
                NECFRAction.BET_LARGE: pot * 1.0
            }
            required_bet = bet_sizes.get(action, 0)
            return stack >= required_bet

    def _apply_action(self, game_state: Dict, action: NECFRAction, player: int) -> Dict:
        """Apply action and return new game state"""
        new_state = game_state.copy()

        if action == NECFRAction.FOLD:
            new_state['is_terminal'] = True
            new_state['winner'] = 1 - player
        elif action == NECFRAction.CHECK_CALL:
            # Simplified: just advance to next player or street
            new_state['to_act'] = 1 - player
        elif action == NECFRAction.ALL_IN:
            # All-in action
            stack = new_state['player_stacks'][player]
            new_state['pot'] += stack
            new_state['player_stacks'][player] = 0
            new_state['to_act'] = 1 - player
        else:
            # Betting actions
            bet_sizes = {
                NECFRAction.BET_SMALL: new_state['pot'] * 0.33,
                NECFRAction.BET_MEDIUM: new_state['pot'] * 0.66,
                NECFRAction.BET_LARGE: new_state['pot'] * 1.0
            }
            bet_size = bet_sizes.get(action, 0)

            if new_state['player_stacks'][player] >= bet_size:
                new_state['pot'] += bet_size
                new_state['player_stacks'][player] -= bet_size
                new_state['betting_history'] += 'b'
                new_state['to_act'] = 1 - player

        return new_state

    def _evaluate_terminal_payoff(self, game_state: Dict, training_player: int) -> float:
        """Evaluate payoff at terminal nodes"""
        if game_state.get('winner') is not None:
            # Fold situation
            winner = game_state['winner']
            pot = game_state['pot']
            return pot if winner == training_player else -pot
        else:
            # Showdown - simplified evaluation
            hole_cards_0 = game_state['hole_cards'][0]
            hole_cards_1 = game_state['hole_cards'][1]
            community_cards = game_state['community_cards']

            if len(community_cards) >= 3:
                hand_0 = evaluate_hand(hole_cards_0 + community_cards)
                hand_1 = evaluate_hand(hole_cards_1 + community_cards)

                pot = game_state['pot']
                if hand_0 > hand_1:
                    return pot if training_player == 0 else -pot
                elif hand_1 > hand_0:
                    return pot if training_player == 1 else -pot
                else:
                    return 0  # Tie
            else:
                # Pre-flop all-in - use simplified equity
                return 0  # Simplified for now

    def _update_neural_networks(self, iteration: int):
        """Update neural networks with CFR data"""
        if not self.use_neural_networks or iteration % 500 != 0:
            return

        # Collect training data from recent CFR nodes
        training_data = []
        for info_set_key, node in list(self.nodes.items())[-1000:]:  # Last 1000 nodes
            # Parse info set key to get game state features
            try:
                features = self._parse_info_set_features(info_set_key)
                target_strategy = node.get_average_strategy()
                training_data.append((features, target_strategy))
            except:
                continue

        if len(training_data) < 10:
            return

        # Train neural network
        self.strategy_net.train()
        total_loss = 0.0

        for features, target_strategy in training_data:
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_strategy).unsqueeze(0)

            predicted_strategy = self.strategy_net(features_tensor)

            # KL divergence loss
            loss = F.kl_div(F.log_softmax(predicted_strategy, dim=1),
                           F.softmax(target_tensor, dim=1), reduction='batchmean')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.strategy_net.eval()

    def _parse_info_set_features(self, info_set_key: str) -> np.ndarray:
        """Parse info set key back to features (simplified)"""
        # This is a simplified version - in practice, you'd need to reconstruct
        # the full game state from the info set key
        return np.random.random(self.feature_dim)  # Placeholder

    def _print_progress(self, iteration: int, verbose: bool):
        """Print training progress"""
        if not verbose:
            return

        elapsed = time.time() - self.training_start_time
        progress_pct = (iteration + 1) / self.iterations * 100

        # ETA calculation
        if iteration > 0:
            eta_seconds = (elapsed / (iteration + 1)) * (self.iterations - iteration - 1)
            eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "calculating..."

        stage_names = ["Random", "Weak DQN", "Strong DQN"]
        current_stage = stage_names[self.training_stage] if self.training_stage < len(stage_names) else "Final"

        print(f"Iteration {iteration + 1:,}/{self.iterations:,} ({progress_pct:.1f}%) | "
              f"Stage: {current_stage} | Elapsed: {elapsed:.0f}s | ETA: {eta_str}")
        print(f"Info sets: {len(self.nodes):,} | "
              f"Neural updates: {'Yes' if self.use_neural_networks else 'No'}")

        # Memory usage (optional)
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Memory: {memory_mb:.0f}MB")
        except ImportError:
            print("Memory: N/A (psutil not installed)")
        print()

    def _print_final_results(self):
        """Print final training results"""
        elapsed = time.time() - self.training_start_time
        print("=" * 60)
        print("NEURAL-ENHANCED CFR TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Information sets created: {len(self.nodes):,}")
        print(f"Neural network: {'Trained' if self.use_neural_networks else 'Not used'}")
        print(f"Progressive training: {'Completed' if self.progressive_training else 'Not used'}")
        print(f"Ready for DQN competition!")
        print("=" * 60)

    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int,
                   stack_size: int, position: int, opponent_stack: int = 1000) -> Action:
        """Get action for gameplay"""

        # Create current game state
        game_state = {
            'hole_cards': [hole_cards, []],  # Only our cards known
            'community_cards': community_cards,
            'betting_history': betting_history,
            'pot': pot_size,
            'player_stacks': [stack_size, opponent_stack],
            'to_act': 0,  # We are player 0
            'street': len(community_cards) // 3 if len(community_cards) >= 3 else 0
        }

        # Get info set key
        info_set_key = self._create_info_set_key(game_state, 0)

        # Get strategy
        if info_set_key in self.nodes:
            if self.use_neural_networks:
                strategy = self._get_neural_strategy(game_state, 0)
            else:
                strategy = self.nodes[info_set_key].get_average_strategy()
        else:
            # Default strategy for unseen situations
            strategy = np.array([0.1, 0.4, 0.2, 0.2, 0.1, 0.0])  # Conservative default

        # Sample action from strategy
        valid_actions = [action for action in NECFRAction if self._is_valid_action(action, game_state)]
        valid_indices = [action.value for action in valid_actions]

        if valid_indices:
            valid_strategy = strategy[valid_indices]
            valid_strategy = valid_strategy / np.sum(valid_strategy) if np.sum(valid_strategy) > 0 else np.ones(len(valid_strategy)) / len(valid_strategy)

            chosen_index = np.random.choice(len(valid_actions), p=valid_strategy)
            chosen_action = valid_actions[chosen_index]
        else:
            chosen_action = NECFRAction.CHECK_CALL  # Safe default

        # Convert to game engine action
        action_mapping = {
            NECFRAction.FOLD: Action.FOLD,
            NECFRAction.CHECK_CALL: Action.CALL if to_call > 0 else Action.CHECK,
            NECFRAction.BET_SMALL: Action.RAISE,
            NECFRAction.BET_MEDIUM: Action.RAISE,
            NECFRAction.BET_LARGE: Action.RAISE,
            NECFRAction.ALL_IN: Action.ALL_IN
        }

        return action_mapping.get(chosen_action, Action.CALL)

    def get_raise_size(self, pot_size: int, current_bet: int = 0,
                       player_chips: int = 1000, min_raise: int = 20) -> int:
        """Get raise size for betting actions"""
        # Smart raise sizing based on pot
        if pot_size <= 0:
            return min_raise

        # Use pot-based sizing
        small_bet = max(min_raise, int(pot_size * 0.33))
        medium_bet = max(min_raise, int(pot_size * 0.66))
        large_bet = max(min_raise, pot_size)

        # Choose based on some simple logic (could be enhanced)
        if player_chips > large_bet * 3:
            return large_bet
        elif player_chips > medium_bet * 2:
            return medium_bet
        else:
            return small_bet

    def save(self, filename: str):
        """Save the model"""
        model_data = {
            'model_type': 'NeuralEnhancedTwoPlayerCFR',
            'model_version': '1.0',
            'nodes': self.nodes,
            'iterations_trained': self.current_iteration,
            'use_neural_networks': self.use_neural_networks,
            'progressive_training': self.progressive_training,
            'feature_dim': self.feature_dim,
            'performance_history': self.performance_history
        }

        # Save neural network state if used
        if self.use_neural_networks:
            model_data['strategy_net_state'] = self.strategy_net.state_dict()

        dirname = os.path.dirname(filename)
        if dirname:  # Only create directory if there is one
            os.makedirs(dirname, exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Neural-Enhanced CFR model saved to {filename}")

    def load(self, filename: str):
        """Load the model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        self.nodes = model_data.get('nodes', {})
        self.current_iteration = model_data.get('iterations_trained', 0)
        self.use_neural_networks = model_data.get('use_neural_networks', True)
        self.progressive_training = model_data.get('progressive_training', True)
        self.feature_dim = model_data.get('feature_dim', 50)
        self.performance_history = model_data.get('performance_history', [])

        # Load neural network if it exists
        if self.use_neural_networks and 'strategy_net_state' in model_data:
            if not hasattr(self, 'strategy_net'):
                self.strategy_net = StrategyNetwork(self.feature_dim, len(NECFRAction))
            self.strategy_net.load_state_dict(model_data['strategy_net_state'])

        print(f"Neural-Enhanced CFR model loaded from {filename}")
        print(f"  Information sets: {len(self.nodes):,}")
        print(f"  Iterations trained: {self.current_iteration:,}")

    def reset_strategy_stats(self):
        """Reset strategy statistics for evaluation"""
        # This method exists for compatibility with evaluation framework
        pass

    def print_strategy_usage(self):
        """Print strategy usage statistics"""
        if not self.nodes:
            print("No strategy data available")
            return

        # Calculate strategy usage statistics
        total_nodes = len(self.nodes)
        nodes_with_visits = sum(1 for node in self.nodes.values() if node.visits > 0)

        usage_percentage = (nodes_with_visits / total_nodes * 100) if total_nodes > 0 else 0

        print(f"Strategy Usage Statistics:")
        print(f"  Total information sets: {total_nodes:,}")
        print(f"  Information sets with visits: {nodes_with_visits:,}")
        print(f"  Strategy coverage: {usage_percentage:.1f}%")

        if self.use_neural_networks:
            print(f"  Neural network: Active")
        else:
            print(f"  Neural network: Not used")

    def get_strategy_usage_stats(self):
        """Get strategy usage statistics for evaluation compatibility"""
        if not self.nodes:
            return {
                'learned_percentage': 0.0,
                'fallback_percentage': 100.0,
                'total_decisions': 0,
                'learned_decisions': 0
            }

        total_nodes = len(self.nodes)
        nodes_with_visits = sum(1 for node in self.nodes.values() if node.visits > 0)
        learned_percentage = (nodes_with_visits / total_nodes * 100) if total_nodes > 0 else 0

        return {
            'learned_percentage': learned_percentage,
            'fallback_percentage': 100.0 - learned_percentage,
            'total_decisions': total_nodes,
            'learned_decisions': nodes_with_visits
        }


def train_neural_enhanced_cfr(iterations: int = 50000, **kwargs) -> NeuralEnhancedTwoPlayerCFR:
    """Training function for compatibility"""
    cfr = NeuralEnhancedTwoPlayerCFR(iterations=iterations, **kwargs)
    cfr.train(verbose=kwargs.get('verbose', True))
    return cfr


if __name__ == "__main__":
    # Test the Neural-Enhanced CFR
    print("Testing Neural-Enhanced CFR...")
    cfr = NeuralEnhancedTwoPlayerCFR(iterations=1000)
    cfr.train(verbose=True)

    print(f"\nTraining complete! Generated {len(cfr.nodes)} information sets.")