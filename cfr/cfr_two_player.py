"""
Specialized 2-Player (Heads-Up) CFR implementation for Texas Hold'em
Optimized for the simpler two-player case which is more easily solvable
"""

import numpy as np
import random
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class TwoPlayerBettingAction(Enum):
    """Simplified betting actions for 2-player CFR"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4


class TwoPlayerInformationSet:
    """Information set optimized for 2-player poker"""
    
    def __init__(self, hole_cards: List[Card], community_cards: List[Card], 
                 betting_history: str, pot_size: int, to_call: int, 
                 stack_size: int, position: int):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.betting_history = betting_history
        self.pot_size = pot_size
        self.to_call = to_call
        self.stack_size = stack_size
        self.position = position  # 0 = dealer/small blind, 1 = big blind
        
        self._key = self._create_key()
    
    def _create_key(self) -> str:
        """Create optimized key for 2-player scenario"""
        # Hand strength abstraction (more precise for 2-player)
        hand_strength = self._get_hand_strength()
        
        # Community card abstraction
        if len(self.community_cards) == 0:
            board_abstract = "preflop"
        elif len(self.community_cards) == 3:
            board_abstract = "flop"
        elif len(self.community_cards) == 4:
            board_abstract = "turn"
        else:
            board_abstract = "river"
        
        # Betting history abstraction (simpler for 2-player)
        history_abstract = self._abstract_betting_history()
        
        # Pot odds abstraction
        if self.to_call == 0:
            pot_odds = "free"
        else:
            odds_ratio = self.to_call / max(1, self.pot_size)
            if odds_ratio < 0.1:
                pot_odds = "small"
            elif odds_ratio < 0.5:
                pot_odds = "medium"
            else:
                pot_odds = "large"
        
        # Position (simpler for 2-player)
        pos = "btn" if self.position == 0 else "bb"
        
        return f"2p_{board_abstract}_{hand_strength}_{history_abstract}_{pot_odds}_{pos}"
    
    def _get_hand_strength(self) -> str:
        """Get hand strength category for 2-player"""
        if not self.hole_cards:
            return "unknown"
        
        # Combine hole cards with community cards
        full_hand = self.hole_cards + self.community_cards
        
        if len(full_hand) < 2:
            return "weak"
        
        try:
            hand_value = evaluate_hand(full_hand)
            
            # More granular hand strength for 2-player
            if hand_value >= 7000:  # Strong hands (straights and better)
                return "premium"
            elif hand_value >= 4000:  # Good pairs, two pairs
                return "strong"
            elif hand_value >= 2000:  # Weak pairs, high cards
                return "medium"
            else:
                return "weak"
        except:
            # Fallback: simple high card evaluation
            if len(self.hole_cards) >= 2:
                high_card = max(card.rank for card in self.hole_cards)
                if high_card >= 12:  # Ace or King
                    return "strong"
                elif high_card >= 10:  # Queen or Jack
                    return "medium"
                else:
                    return "weak"
            return "weak"
    
    def _abstract_betting_history(self) -> str:
        """Abstract betting history for 2-player"""
        if not self.betting_history:
            return "new"
        
        # Count actions
        raises = self.betting_history.count('R')
        calls = self.betting_history.count('C')
        
        if raises == 0:
            return "passive"
        elif raises == 1:
            return "single_raise"
        elif raises == 2:
            return "raised_back"
        else:
            return "aggressive"
    
    def get_key(self) -> str:
        return self._key


class TwoPlayerCFRNode:
    """CFR node optimized for 2-player scenarios"""
    
    def __init__(self, actions: List[TwoPlayerBettingAction]):
        self.actions = actions
        self.regret_sum = np.zeros(len(actions))
        self.strategy_sum = np.zeros(len(actions))
        
    def get_strategy(self, realization_weight: float) -> np.ndarray:
        """Get current strategy using regret matching"""
        # Normalize positive regrets
        strategy = np.maximum(self.regret_sum, 0)
        
        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()
        else:
            # Uniform random strategy if no positive regrets
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        # Update strategy sum for average strategy calculation
        self.strategy_sum += realization_weight * strategy
        
        return strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        if self.strategy_sum.sum() > 0:
            return self.strategy_sum / self.strategy_sum.sum()
        else:
            return np.ones(len(self.actions)) / len(self.actions)


class TwoPlayerCFRPokerAI:
    """Specialized CFR implementation for 2-player Texas Hold'em"""
    
    def __init__(self, iterations: int = 100000):  # Higher default for 2-player
        self.iterations = iterations
        self.nodes: Dict[str, TwoPlayerCFRNode] = {}
        self.util = 0
        self.action_mapping = {
            TwoPlayerBettingAction.FOLD: Action.FOLD,
            TwoPlayerBettingAction.CHECK_CALL: Action.CHECK,  # Will convert to CALL if needed
            TwoPlayerBettingAction.RAISE_HALF_POT: Action.RAISE,
            TwoPlayerBettingAction.RAISE_POT: Action.RAISE,
            TwoPlayerBettingAction.ALL_IN: Action.ALL_IN
        }
        
        # Strategy usage tracking
        self.strategy_stats = {
            'learned_strategy_count': 0,
            'fallback_strategy_count': 0,
            'total_decisions': 0
        }
        
    def train(self, verbose: bool = True):
        """Train the 2-player CFR algorithm"""
        if verbose:
            print(f"Training 2-Player CFR for {self.iterations} iterations...")
            print("Optimized for heads-up no-limit Texas Hold'em")
            
        for iteration in range(self.iterations):
            if verbose and (iteration == 0 or (iteration + 1) % (self.iterations // 10) == 0):
                print(f"  Iteration {iteration}/{self.iterations}, Nodes: {len(self.nodes)}")
            
            # Alternate training between both players
            for player in range(2):
                self._cfr_iteration(player, iteration)
        
        if verbose:
            print(f"2-Player CFR training complete! Generated {len(self.nodes)} information sets")
            self._print_training_summary()
    
    def _cfr_iteration(self, training_player: int, iteration: int):
        """Single CFR iteration for 2-player"""
        # Deal random cards for both players
        cards = self._deal_random_cards()
        
        # Random pot and stack sizes
        pot_sizes = [20, 40, 60, 100, 150, 200, 300]  # Common pot sizes
        pot = random.choice(pot_sizes)
        stack_sizes = [random.randint(800, 1200), random.randint(800, 1200)]
        
        # Create representative scenarios
        for street in range(4):  # preflop, flop, turn, river
            community_cards = cards['community'][:street*2] if street > 0 else []
            
            # Create information set
            info_set = TwoPlayerInformationSet(
                hole_cards=cards[f'player_{training_player}'],
                community_cards=community_cards,
                betting_history="",  # Start fresh each street
                pot_size=pot,
                to_call=random.choice([0, 20, 40, pot//2]),
                stack_size=stack_sizes[training_player],
                position=training_player
            )
            
            key = info_set.get_key()
            
            if key not in self.nodes:
                available_actions = [
                    TwoPlayerBettingAction.FOLD,
                    TwoPlayerBettingAction.CHECK_CALL,
                    TwoPlayerBettingAction.RAISE_HALF_POT,
                    TwoPlayerBettingAction.RAISE_POT
                ]
                if stack_sizes[training_player] < pot * 2:  # Add all-in option for small stacks
                    available_actions.append(TwoPlayerBettingAction.ALL_IN)
                
                self.nodes[key] = TwoPlayerCFRNode(available_actions)
        
        # Run traditional CFR on a simplified game tree
        history = ""
        reach_probs = [1.0, 1.0]
        self._cfr(cards, history, pot, reach_probs, training_player, iteration, stack_sizes)
    
    def _cfr(self, cards: Dict, history: str, pot: int, reach_probs: List[float], 
             training_player: int, iteration: int, stack_sizes: List[int], depth: int = 0) -> float:
        """Core CFR algorithm for 2-player"""
        current_player = len(history) % 2
        
        # Depth limit (can be higher for 2-player)
        if depth > 6:
            return self._get_payoff(cards, history, pot, training_player, stack_sizes)
        
        # Terminal check
        if self._is_terminal(history):
            return self._get_payoff(cards, history, pot, training_player, stack_sizes)
        
        # Create information set
        community_cards = cards['community']
        if len(history) <= 2:  # preflop
            board = []
        elif len(history) <= 6:  # flop
            board = community_cards[:3]
        elif len(history) <= 8:  # turn  
            board = community_cards[:4]
        else:  # river
            board = community_cards[:5]
            
        info_set = TwoPlayerInformationSet(
            hole_cards=cards[f'player_{current_player}'],
            community_cards=board,
            betting_history=history,
            pot_size=pot,
            to_call=self._get_to_call(history, pot),
            stack_size=stack_sizes[current_player],
            position=current_player
        )
        
        key = info_set.get_key()
        
        if key not in self.nodes:
            actions = [TwoPlayerBettingAction.FOLD, TwoPlayerBettingAction.CHECK_CALL, 
                      TwoPlayerBettingAction.RAISE_HALF_POT]
            self.nodes[key] = TwoPlayerCFRNode(actions)
        
        node = self.nodes[key]
        strategy = node.get_strategy(reach_probs[current_player])
        
        util = np.zeros(len(node.actions))
        node_util = 0
        
        # Evaluate all actions (simpler for 2-player)
        for i, action in enumerate(node.actions):
            next_history = history + self._action_to_char(action)
            next_pot = pot + self._get_action_cost(action, pot, stack_sizes[current_player])
            
            if current_player == training_player:
                util[i] = -self._cfr(cards, next_history, next_pot, reach_probs, 
                                   training_player, iteration, stack_sizes, depth + 1)
            else:
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy[i]
                util[i] = -self._cfr(cards, next_history, next_pot, new_reach_probs, 
                                   training_player, iteration, stack_sizes, depth + 1)
            
            node_util += strategy[i] * util[i]
        
        # Update regrets for training player
        if current_player == training_player:
            counterfactual_reach = reach_probs[1 - current_player]
            
            for i in range(len(node.actions)):
                regret = util[i] - node_util
                node.regret_sum[i] += counterfactual_reach * regret
        
        return node_util
    
    def _is_terminal(self, history: str) -> bool:
        """Check terminal conditions for 2-player"""
        if 'F' in history:  # Someone folded
            return True
        if len(history) >= 8:  # Hard limit
            return True
        
        # Check betting round completion
        if len(history) >= 2:
            # Both players acted
            if history[-2:] == "CC":  # Both checked
                return True
            if len(history) >= 4 and history[-2:] == "CR" and history[-4] == "R":
                return False  # Raise after call-raise, continue
            if 'R' in history:
                # Find last raise
                last_raise = len(history) - 1 - history[::-1].index('R')
                actions_after = history[last_raise + 1:]
                if len(actions_after) >= 1 and all(a == 'C' for a in actions_after):
                    return True
        
        return False
    
    def _get_payoff(self, cards: Dict, history: str, pot: int, player: int, stack_sizes: List[int]) -> float:
        """Calculate payoff for 2-player terminal node"""
        if 'F' in history:
            # Someone folded
            folder = history.rindex('F') % 2
            if player != folder:
                return pot - self._get_invested(history, player)
            else:
                return -self._get_invested(history, player)
        
        # Showdown between 2 players
        hand_values = {}
        for p in range(2):
            if f'player_{p}' in cards and cards[f'player_{p}']:
                full_hand = cards[f'player_{p}'] + cards['community']
                try:
                    hand_values[p] = evaluate_hand(full_hand)
                except:
                    hand_values[p] = sum(card.rank for card in cards[f'player_{p}'])
            else:
                hand_values[p] = 0
        
        if hand_values[player] > hand_values[1 - player]:
            return pot - self._get_invested(history, player)  # Win
        elif hand_values[player] == hand_values[1 - player]:
            return (pot / 2) - self._get_invested(history, player)  # Tie
        else:
            return -self._get_invested(history, player)  # Lose
    
    def _get_invested(self, history: str, player: int) -> int:
        """Calculate amount invested by player"""
        invested = 0
        for i, action in enumerate(history):
            if i % 2 == player:
                if action == 'R':
                    invested += 50  # Standard raise
                elif action == 'C':
                    invested += 25  # Call
                elif action == 'A':
                    invested += 200  # All-in
        return invested
    
    def _get_to_call(self, history: str, pot: int) -> int:
        """Calculate amount to call in 2-player"""
        if not history:
            return 0
        
        last_raise = history.rfind('R')
        if last_raise == -1:
            return 0
        
        # Simple: assume standard bet sizes
        return 25
    
    def _action_to_char(self, action: TwoPlayerBettingAction) -> str:
        """Convert action to character"""
        if action == TwoPlayerBettingAction.FOLD:
            return 'F'
        elif action == TwoPlayerBettingAction.CHECK_CALL:
            return 'C'
        elif action in [TwoPlayerBettingAction.RAISE_HALF_POT, TwoPlayerBettingAction.RAISE_POT]:
            return 'R'
        elif action == TwoPlayerBettingAction.ALL_IN:
            return 'A'
        return 'C'
    
    def _get_action_cost(self, action: TwoPlayerBettingAction, pot: int, stack_size: int) -> int:
        """Get cost of action"""
        if action == TwoPlayerBettingAction.FOLD:
            return 0
        elif action == TwoPlayerBettingAction.CHECK_CALL:
            return 25  # Standard call
        elif action == TwoPlayerBettingAction.RAISE_HALF_POT:
            return max(50, pot // 2)
        elif action == TwoPlayerBettingAction.RAISE_POT:
            return max(100, pot)
        elif action == TwoPlayerBettingAction.ALL_IN:
            return min(stack_size, pot * 2)
        return 0
    
    def _deal_random_cards(self) -> Dict:
        """Deal random cards for 2-player game"""
        deck = Deck()
        deck.reset()
        
        return {
            'player_0': deck.draw(2),
            'player_1': deck.draw(2),
            'community': deck.draw(5)  # All 5 community cards
        }
    
    def _print_training_summary(self):
        """Print training summary"""
        position_counts = {}
        for key in self.nodes.keys():
            parts = key.split('_')
            if len(parts) > 5:
                pos = parts[5]
                position_counts[pos] = position_counts.get(pos, 0) + 1
        
        print(f"  Position distribution in trained nodes:")
        for pos, count in position_counts.items():
            print(f"    {pos}: {count} nodes")
        
        print(f"  Sample information set keys:")
        sample_keys = list(self.nodes.keys())[:10]
        for key in sample_keys:
            print(f"    {key}")
    
    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int, stack_size: int,
                   position: int, num_players: int = 2) -> Action:
        """Get action using trained 2-player CFR strategy"""
        if num_players != 2:
            raise ValueError("TwoPlayerCFRPokerAI only supports 2 players")
        
        # Create information set
        info_set = TwoPlayerInformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            pot_size=pot_size,
            to_call=to_call,
            stack_size=stack_size,
            position=position
        )
        
        key = info_set.get_key()
        self.strategy_stats['total_decisions'] += 1
        
        if key in self.nodes:
            # Use learned strategy
            self.strategy_stats['learned_strategy_count'] += 1
            node = self.nodes[key]
            strategy = node.get_average_strategy()
            
            # Sample action based on strategy
            action_idx = np.random.choice(len(node.actions), p=strategy)
            cfr_action = node.actions[action_idx]
        else:
            # Fallback strategy for unseen information sets
            self.strategy_stats['fallback_strategy_count'] += 1
            cfr_action = TwoPlayerBettingAction.CHECK_CALL
        
        # Convert to game engine action
        game_action = self.action_mapping[cfr_action]
        
        # Adjust CHECK to CALL if needed
        if game_action == Action.CHECK and to_call > 0:
            game_action = Action.CALL
        
        return game_action
    
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
        """Reset strategy usage statistics"""
        self.strategy_stats = {
            'learned_strategy_count': 0,
            'fallback_strategy_count': 0,
            'total_decisions': 0
        }
    
    def print_strategy_usage(self):
        """Print current strategy usage statistics"""
        stats = self.get_strategy_usage_stats()
        print(f"\n[STATS] 2-Player CFR Strategy Usage Statistics:")
        print(f"  Total decisions made: {stats['total_decisions']}")
        print(f"  Learned strategies: {stats['learned_count']} ({stats['learned_percentage']:.1f}%)")
        print(f"  Fallback strategies: {stats['fallback_count']} ({stats['fallback_percentage']:.1f}%)")
        
        if stats['learned_percentage'] < 20:
            print(f"  [WARNING] Using mostly fallback strategy - AI may need more training!")
        elif stats['learned_percentage'] > 80:
            print(f"  [GOOD] Using mostly learned strategies")
        else:
            print(f"  [MIXED] Using combination of learned and fallback strategies")
    
    def save(self, filename: str):
        """Save the trained 2-player CFR model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'iterations': self.iterations,
                'strategy_stats': self.strategy_stats
            }, f)
        print(f"2-Player CFR model saved to {filename}")
    
    def load(self, filename: str):
        """Load a trained 2-player CFR model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.nodes = data['nodes']
        self.iterations = data.get('iterations', self.iterations)
        self.strategy_stats = data.get('strategy_stats', self.strategy_stats)
        print(f"2-Player CFR model loaded from {filename}")


def train_two_player_cfr_ai(iterations: int = 100000, save_filename: Optional[str] = None, 
                           verbose: bool = True) -> TwoPlayerCFRPokerAI:
    """
    Train a specialized 2-player CFR AI
    
    Args:
        iterations: Number of CFR iterations (higher for better quality)
        save_filename: Optional filename to save the model
        verbose: Whether to print training progress
        
    Returns:
        Trained TwoPlayerCFRPokerAI instance
    """
    cfr_ai = TwoPlayerCFRPokerAI(iterations=iterations)
    cfr_ai.train(verbose=verbose)
    
    if save_filename:
        cfr_ai.save(save_filename)
        
    return cfr_ai