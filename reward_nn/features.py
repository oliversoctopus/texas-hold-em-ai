"""
Feature extraction utilities for reward-based AI
"""

import numpy as np
from typing import List, Optional
from core.card_deck import Card, evaluate_hand


def extract_hand_features(hole_cards: List[Card], community_cards: List[Card]) -> dict:
    """Extract comprehensive hand strength features"""
    features = {}

    if not hole_cards or len(hole_cards) < 2:
        return {
            'hand_strength': 0,
            'is_pair': 0,
            'is_suited': 0,
            'high_card': 0,
            'kicker': 0,
            'straight_potential': 0,
            'flush_potential': 0
        }

    # Basic hole card features
    values = sorted([c.value for c in hole_cards], reverse=True)
    features['high_card'] = values[0] / 14
    features['kicker'] = values[1] / 14
    features['is_pair'] = int(values[0] == values[1])
    features['is_suited'] = int(hole_cards[0].suit == hole_cards[1].suit)
    features['gap'] = (values[0] - values[1]) / 12

    # Calculate hand strength if community cards exist
    if community_cards:
        all_cards = hole_cards + community_cards
        hand_rank = evaluate_hand(all_cards)

        # Normalize hand rank (higher is better)
        # Hand ranks: high card=1, pair=2, two pair=3, etc.
        features['hand_strength'] = hand_rank / 10

        # Check for draws
        all_values = [c.value for c in all_cards]
        all_suits = [c.suit for c in all_cards]

        # Flush draw
        suit_counts = {}
        for suit in all_suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit = max(suit_counts.values())
        features['flush_potential'] = max_suit / 7

        # Straight draw
        unique_values = sorted(set(all_values))
        straight_potential = 0
        for i in range(len(unique_values) - 3):
            if unique_values[i+3] - unique_values[i] <= 4:
                straight_potential = 1
                break
        features['straight_potential'] = straight_potential

    else:
        features['hand_strength'] = 0
        features['flush_potential'] = features['is_suited']
        features['straight_potential'] = 1 if features['gap'] <= 0.33 else 0

    return features


def extract_pot_odds_features(pot_size: int, to_call: int, stack_size: int) -> dict:
    """Extract pot odds and implied odds features"""
    features = {}

    # Pot odds
    if to_call > 0:
        pot_odds = to_call / (pot_size + to_call + 1e-8)
        features['pot_odds'] = pot_odds

        # Required win percentage
        features['required_equity'] = to_call / (pot_size + 2 * to_call + 1e-8)
    else:
        features['pot_odds'] = 0
        features['required_equity'] = 0

    # Stack to pot ratio (SPR)
    features['spr'] = stack_size / (pot_size + 1e-8)

    # Implied odds potential (simplified)
    features['implied_odds'] = min(1.0, stack_size / (pot_size + to_call + 1e-8))

    return features


def extract_position_features(position: int, num_players: int, dealer_position: int) -> dict:
    """Extract positional features"""
    features = {}

    # Relative position to dealer
    relative_position = (position - dealer_position) % num_players
    features['relative_position'] = relative_position / (num_players - 1)

    # Early, middle, late position
    if num_players <= 3:
        position_category = relative_position / 2
    else:
        if relative_position <= 2:
            position_category = 0  # Early
        elif relative_position <= num_players - 2:
            position_category = 0.5  # Middle
        else:
            position_category = 1  # Late

    features['position_category'] = position_category

    # Is button, small blind, big blind
    features['is_button'] = int(relative_position == 0)
    features['is_sb'] = int(relative_position == 1)
    features['is_bb'] = int(relative_position == 2)

    return features


def extract_opponent_features(action_history: List, num_opponents: int) -> dict:
    """Extract opponent modeling features from action history"""
    features = {}

    if not action_history:
        return {
            'aggression_factor': 0.5,
            'fold_frequency': 0,
            'raise_frequency': 0,
            'opponents_remaining': num_opponents / 5
        }

    # Count action types
    fold_count = sum(1 for a in action_history if a == 0)
    check_count = sum(1 for a in action_history if a == 1)
    call_count = sum(1 for a in action_history if a == 2)
    raise_count = sum(1 for a in action_history if a in [3, 4])

    total_actions = len(action_history)

    # Aggression factor
    aggressive_actions = raise_count
    passive_actions = call_count + check_count
    features['aggression_factor'] = aggressive_actions / (passive_actions + aggressive_actions + 1e-8)

    # Action frequencies
    features['fold_frequency'] = fold_count / (total_actions + 1e-8)
    features['raise_frequency'] = raise_count / (total_actions + 1e-8)

    # Opponents remaining
    features['opponents_remaining'] = num_opponents / 5

    return features


def extract_board_texture_features(community_cards: List[Card]) -> dict:
    """Extract board texture features"""
    features = {}

    if not community_cards:
        return {
            'board_dryness': 0.5,
            'is_monotone': 0,
            'is_paired': 0,
            'straight_texture': 0
        }

    values = [c.value for c in community_cards]
    suits = [c.suit for c in community_cards]

    # Check for paired board
    value_counts = {}
    for v in values:
        value_counts[v] = value_counts.get(v, 0) + 1
    features['is_paired'] = int(max(value_counts.values()) >= 2)

    # Check for monotone/flush-heavy board
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values())
    features['is_monotone'] = int(max_suit >= 3)

    # Check for straight texture
    sorted_values = sorted(set(values))
    straight_texture = 0
    if len(sorted_values) >= 3:
        for i in range(len(sorted_values) - 2):
            if sorted_values[i+2] - sorted_values[i] <= 4:
                straight_texture = 1
                break
    features['straight_texture'] = straight_texture

    # Board dryness (inverse of coordination)
    coordination = features['is_paired'] + features['is_monotone'] + straight_texture
    features['board_dryness'] = 1 - (coordination / 3)

    return features


def combine_all_features(hole_cards, community_cards, pot_size, to_call, stack_size,
                         position, num_players, dealer_position, action_history,
                         num_opponents) -> np.ndarray:
    """Combine all features into a single vector"""
    all_features = {}

    # Extract all feature groups
    all_features.update(extract_hand_features(hole_cards, community_cards))
    all_features.update(extract_pot_odds_features(pot_size, to_call, stack_size))
    all_features.update(extract_position_features(position, num_players, dealer_position))
    all_features.update(extract_opponent_features(action_history, num_opponents))
    all_features.update(extract_board_texture_features(community_cards))

    # Convert to numpy array
    feature_vector = np.array(list(all_features.values()), dtype=np.float32)

    return feature_vector