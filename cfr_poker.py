"""
Counterfactual Regret Minimization (CFR) implementation for Texas Hold'em Poker
Based on the algorithms used in successful poker AIs like Libratus and Pluribus
"""

import numpy as np
import random
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

from game_constants import Action
from card_deck import Card, evaluate_hand


class BettingAction(Enum):
    """Betting actions for CFR"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2
    RAISE_MEDIUM = 3
    RAISE_LARGE = 4
    ALL_IN = 5


class InformationSet:
    """
    Represents an information set in poker CFR
    Contains the player's private information and observable game history
    """
    def __init__(self, hole_cards: List[Card], community_cards: List[Card], 
                 betting_history: str, pot_size: int, to_call: int, 
                 stack_size: int, position: int, num_players: int):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.betting_history = betting_history
        self.pot_size = pot_size
        self.to_call = to_call
        self.stack_size = stack_size
        self.position = position
        self.num_players = num_players
        
        # Create abstracted representation
        self._key = self._create_key()
    
    def _create_key(self) -> str:
        """Create a string key for this information set"""
        # Hand strength abstraction
        hand_strength = self._get_hand_strength_bucket()
        
        # Betting history (simplified)
        betting_abstract = self._abstract_betting_history()
        
        # Position abstraction (early/middle/late)
        pos_abstract = "early" if self.position < self.num_players // 3 else \
                      "middle" if self.position < 2 * self.num_players // 3 else "late"
        
        # Stack size abstraction
        stack_abstract = self._get_stack_bucket()
        
        # Pot odds abstraction
        pot_odds = self.to_call / max(self.pot_size, 1)
        pot_odds_bucket = min(3, int(pot_odds * 4))  # 0-3 buckets
        
        return f"{hand_strength}_{betting_abstract}_{pos_abstract}_{stack_abstract}_{pot_odds_bucket}"
    
    def _get_hand_strength_bucket(self) -> int:
        """Get abstracted hand strength (0-9, with 9 being strongest)"""
        if not self.hole_cards:
            return 0
            
        # Pre-flop hand strength
        if len(self.community_cards) == 0:
            return self._preflop_strength()
        
        # Post-flop hand strength using actual evaluation
        best_hand_value = evaluate_hand(self.hole_cards + self.community_cards)
        
        # Convert to 0-9 scale (rough approximation)
        if best_hand_value >= 8000000:  # Straight flush or better
            return 9
        elif best_hand_value >= 7000000:  # Four of a kind
            return 8
        elif best_hand_value >= 6000000:  # Full house
            return 7
        elif best_hand_value >= 5000000:  # Flush
            return 6
        elif best_hand_value >= 4000000:  # Straight
            return 5
        elif best_hand_value >= 3000000:  # Three of a kind
            return 4
        elif best_hand_value >= 2000000:  # Two pair
            return 3
        elif best_hand_value >= 1000000:  # One pair
            return 2
        elif best_hand_value >= 500000:   # High card (good)
            return 1
        else:
            return 0
    
    def _preflop_strength(self) -> int:
        """Pre-flop hand strength evaluation"""
        if len(self.hole_cards) != 2:
            return 0
            
        card1, card2 = self.hole_cards
        rank1, rank2 = card1.rank, card2.rank
        suited = card1.suit == card2.suit
        
        # Convert ranks to numerical values (2=2, ..., A=14)
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        val1 = rank_values.get(rank1, 2)
        val2 = rank_values.get(rank2, 2)
        high, low = max(val1, val2), min(val1, val2)
        
        # Premium pairs
        if val1 == val2:
            if high >= 13:  # AA, KK
                return 9
            elif high >= 11:  # QQ, JJ
                return 8
            elif high >= 9:   # TT, 99
                return 7
            elif high >= 7:   # 88, 77
                return 6
            else:             # 66 and below
                return 5
        
        # Suited hands
        if suited:
            if high == 14:  # Ace suited
                return 7 if low >= 10 else 6 if low >= 7 else 5
            elif high >= 12 and low >= 10:  # High suited connectors
                return 6
            elif abs(high - low) <= 2 and high >= 7:  # Suited connectors
                return 5
            else:
                return 3
        
        # Offsuit hands
        if high == 14:  # Ace offsuit
            return 6 if low >= 12 else 5 if low >= 10 else 4
        elif high >= 12 and low >= 10:  # High offsuit
            return 5
        elif abs(high - low) <= 2 and high >= 9:  # Offsuit connectors
            return 4
        else:
            return 2
    
    def _abstract_betting_history(self) -> str:
        """Abstract the betting history to reduce complexity"""
        if not self.betting_history:
            return "new"
        
        # Count aggressive actions (raises)
        raises = self.betting_history.count('R')
        calls = self.betting_history.count('C')
        
        if raises >= 3:
            return "very_aggressive"
        elif raises >= 2:
            return "aggressive"
        elif raises == 1:
            return "moderate"
        elif calls >= 2:
            return "passive"
        else:
            return "quiet"
    
    def _get_stack_bucket(self) -> str:
        """Abstract stack size relative to pot"""
        if self.pot_size == 0:
            return "deep"
        
        stack_to_pot = self.stack_size / max(self.pot_size, 1)
        if stack_to_pot >= 10:
            return "deep"
        elif stack_to_pot >= 3:
            return "medium"
        else:
            return "short"
    
    def get_key(self) -> str:
        """Get the information set key"""
        return self._key


class CFRNode:
    """
    Node in the CFR game tree
    Stores regret and strategy information for each action
    """
    def __init__(self, actions: List[BettingAction]):
        self.actions = actions
        self.regret_sum = np.zeros(len(actions))
        self.strategy_sum = np.zeros(len(actions))
        self.strategy = np.ones(len(actions)) / len(actions)  # Uniform initially
    
    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        # Regret matching: strategy proportional to positive regrets
        normalizing_sum = 0
        for i in range(len(self.actions)):
            self.strategy[i] = max(self.regret_sum[i], 0)
            normalizing_sum += self.strategy[i]
        
        # If no positive regrets, use uniform strategy
        if normalizing_sum > 0:
            self.strategy /= normalizing_sum
        else:
            self.strategy = np.ones(len(self.actions)) / len(self.actions)
        
        # Update strategy sum for average strategy calculation
        self.strategy_sum += realization_weight * self.strategy
        
        return self.strategy.copy()
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        avg_strategy = self.strategy_sum.copy()
        normalizing_sum = np.sum(avg_strategy)
        
        if normalizing_sum > 0:
            avg_strategy /= normalizing_sum
        else:
            avg_strategy = np.ones(len(self.actions)) / len(self.actions)
        
        return avg_strategy


class CFRPokerAI:
    """
    CFR-based Poker AI implementation
    """
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
        self.nodes: Dict[str, CFRNode] = {}
        self.util = 0
        self.action_mapping = {
            BettingAction.FOLD: Action.FOLD,
            BettingAction.CHECK_CALL: Action.CHECK,  # Will be converted to CALL if needed
            BettingAction.RAISE_SMALL: Action.RAISE,
            BettingAction.RAISE_MEDIUM: Action.RAISE,
            BettingAction.RAISE_LARGE: Action.RAISE,
            BettingAction.ALL_IN: Action.ALL_IN
        }
    
    def train(self, verbose: bool = True):
        """Train the CFR algorithm"""
        if verbose:
            print(f"Training CFR for {self.iterations} iterations...")
        
        for iteration in range(self.iterations):
            # Alternate between players for training
            for player in range(2):  # Start with 2-player for simplicity
                self._cfr_iteration(player, iteration)
            
            if verbose and iteration % 1000 == 0:
                print(f"  Iteration {iteration}/{self.iterations}, Nodes: {len(self.nodes)}")
        
        if verbose:
            print(f"CFR training complete! Generated {len(self.nodes)} information sets")
    
    def _cfr_iteration(self, training_player: int, iteration: int):
        """Single CFR iteration"""
        # Create a simplified game state for training
        # This is a simplified version - in practice you'd simulate full games
        cards = self._deal_random_cards()
        history = ""
        pot = 30  # Starting pot (blinds)
        
        # Run CFR on this game state
        self._cfr(cards, history, pot, [1.0, 1.0], training_player, iteration)
    
    def _deal_random_cards(self) -> Dict:
        """Deal random cards for training"""
        from card_deck import Deck
        deck = Deck()
        deck.reset()
        
        return {
            'player_0': deck.draw(2),
            'player_1': deck.draw(2),
            'community': deck.draw(5)[:3]  # Flop for now
        }
    
    def _cfr(self, cards: Dict, history: str, pot: int, reach_probs: List[float], 
             training_player: int, iteration: int) -> float:
        """Core CFR algorithm"""
        current_player = len(history) % 2
        
        # Terminal node check (simplified)
        if self._is_terminal(history):
            return self._get_payoff(cards, history, pot, training_player)
        
        # Create information set
        player_cards = cards[f'player_{current_player}']
        info_set = InformationSet(
            hole_cards=player_cards,
            community_cards=cards['community'],
            betting_history=history,
            pot_size=pot,
            to_call=self._get_to_call(history),
            stack_size=1000,  # Simplified
            position=current_player,
            num_players=2
        )
        
        info_set_key = info_set.get_key()
        
        # Get or create CFR node
        if info_set_key not in self.nodes:
            available_actions = self._get_available_actions(history, pot, 1000)
            self.nodes[info_set_key] = CFRNode(available_actions)
        
        node = self.nodes[info_set_key]
        strategy = node.get_strategy(reach_probs[current_player])
        
        # Calculate utility for each action
        util = np.zeros(len(node.actions))
        node_util = 0
        
        for i, action in enumerate(node.actions):
            next_history = history + self._action_to_char(action)
            next_pot = pot + self._get_action_cost(action, pot)
            
            if current_player == training_player:
                util[i] = -self._cfr(cards, next_history, next_pot, reach_probs, 
                                   training_player, iteration)
            else:
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy[i]
                util[i] = -self._cfr(cards, next_history, next_pot, new_reach_probs, 
                                   training_player, iteration)
            
            node_util += strategy[i] * util[i]
        
        # Update regrets for training player
        if current_player == training_player:
            for i in range(len(node.actions)):
                regret = util[i] - node_util
                node.regret_sum[i] += reach_probs[1 - current_player] * regret
        
        return node_util
    
    def _is_terminal(self, history: str) -> bool:
        """Check if game state is terminal"""
        return len(history) >= 4 or 'F' in history or history.endswith('CC')
    
    def _get_payoff(self, cards: Dict, history: str, pot: int, player: int) -> float:
        """Calculate payoff for terminal node"""
        if 'F' in history:
            # Someone folded
            if history[-1] == 'F':
                folder = (len(history) - 1) % 2
                return pot if player != folder else -self._get_invested(history, player)
        
        # Showdown
        player_0_hand = evaluate_hand(cards['player_0'] + cards['community'])
        player_1_hand = evaluate_hand(cards['player_1'] + cards['community'])
        
        if player_0_hand > player_1_hand:
            winner = 0
        elif player_1_hand > player_0_hand:
            winner = 1
        else:
            return 0  # Tie
        
        if player == winner:
            return pot / 2 - self._get_invested(history, player)
        else:
            return -self._get_invested(history, player)
    
    def _get_invested(self, history: str, player: int) -> int:
        """Calculate amount invested by player"""
        invested = 0
        for i, action in enumerate(history):
            if i % 2 == player and action == 'R':
                invested += 20  # Simplified bet size
        return invested
    
    def _get_to_call(self, history: str) -> int:
        """Calculate amount to call"""
        # Simplified - just check if there's a raise to call
        return 20 if 'R' in history and not history.endswith('C') else 0
    
    def _get_available_actions(self, history: str, pot: int, stack: int) -> List[BettingAction]:
        """Get available actions for current state"""
        actions = [BettingAction.FOLD]
        
        if self._get_to_call(history) == 0:
            actions.append(BettingAction.CHECK_CALL)
        else:
            actions.append(BettingAction.CHECK_CALL)  # Call
        
        # Add raise actions if stack allows
        if stack > self._get_to_call(history):
            actions.extend([
                BettingAction.RAISE_SMALL,
                BettingAction.RAISE_MEDIUM,
                BettingAction.ALL_IN
            ])
        
        return actions
    
    def _action_to_char(self, action: BettingAction) -> str:
        """Convert action to character for history"""
        if action == BettingAction.FOLD:
            return 'F'
        elif action == BettingAction.CHECK_CALL:
            return 'C'
        else:
            return 'R'
    
    def _get_action_cost(self, action: BettingAction, pot: int) -> int:
        """Get cost of action"""
        if action == BettingAction.FOLD:
            return 0
        elif action == BettingAction.CHECK_CALL:
            return self._get_to_call("")  # Simplified
        elif action == BettingAction.RAISE_SMALL:
            return 20
        elif action == BettingAction.RAISE_MEDIUM:
            return 40
        elif action == BettingAction.RAISE_LARGE:
            return 80
        else:  # ALL_IN
            return 1000
    
    def get_action(self, hole_cards: List[Card], community_cards: List[Card], 
                   betting_history: str, pot_size: int, to_call: int, 
                   stack_size: int, position: int, num_players: int) -> Action:
        """Get action for given game state"""
        # Create information set
        info_set = InformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            pot_size=pot_size,
            to_call=to_call,
            stack_size=stack_size,
            position=position,
            num_players=num_players
        )
        
        info_set_key = info_set.get_key()
        
        # If we've seen this info set during training, use learned strategy
        if info_set_key in self.nodes:
            node = self.nodes[info_set_key]
            strategy = node.get_average_strategy()
            
            # Sample action according to strategy
            action_idx = np.random.choice(len(node.actions), p=strategy)
            cfr_action = node.actions[action_idx]
            
            # Convert to game action
            game_action = self.action_mapping[cfr_action]
            
            # Adjust CHECK to CALL if needed
            if game_action == Action.CHECK and to_call > 0:
                game_action = Action.CALL
            
            return game_action
        
        # Fallback strategy for unseen states
        return self._get_fallback_action(to_call, stack_size, pot_size)
    
    def _get_fallback_action(self, to_call: int, stack_size: int, pot_size: int) -> Action:
        """Fallback strategy for unseen information sets"""
        if to_call == 0:
            return Action.CHECK
        elif to_call > stack_size * 0.5:
            return Action.FOLD
        elif pot_size > to_call * 3:  # Good pot odds
            return Action.CALL
        else:
            return random.choice([Action.FOLD, Action.CALL])
    
    def save(self, filename: str):
        """Save the trained CFR model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'iterations': self.iterations
            }, f)
        print(f"CFR model saved to {filename}")
    
    def load(self, filename: str):
        """Load a trained CFR model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.nodes = data['nodes']
            self.iterations = data['iterations']
        print(f"CFR model loaded from {filename}")