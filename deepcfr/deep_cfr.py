"""
Deep CFR implementation for 3+ player Texas Hold'em
Uses neural networks to approximate regret and strategy functions
Based on the DeepCFR algorithm from "Deep Counterfactual Regret Minimization" (Brown et al.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class DeepCFRAction(Enum):
    """Actions for Deep CFR"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2
    RAISE_MEDIUM = 3
    RAISE_LARGE = 4
    ALL_IN = 5


class RegretNetwork(nn.Module):
    """Neural network to approximate regret function"""
    
    def __init__(self, input_size: int, num_actions: int, hidden_size: int = 512):
        super(RegretNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class StrategyNetwork(nn.Module):
    """Neural network to approximate strategy function"""
    
    def __init__(self, input_size: int, num_actions: int, hidden_size: int = 512):
        super(StrategyNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Output probabilities using softmax
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class DeepCFRInformationSet:
    """Information set representation for Deep CFR"""
    
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
        
        # Create feature vector for neural networks
        self.features = self._create_features()
    
    def _create_features(self) -> np.ndarray:
        """Create feature vector for neural network input"""
        features = []
        
        # Card features (52 dimensions - one-hot encoding)
        card_features = np.zeros(52)
        for card in self.hole_cards:
            card_idx = (card.value - 2) * 4 + ['♠', '♥', '♦', '♣'].index(card.suit)
            card_features[card_idx] = 1
        
        for card in self.community_cards:
            card_idx = (card.value - 2) * 4 + ['♠', '♥', '♦', '♣'].index(card.suit)
            card_features[card_idx] = 1
        
        features.extend(card_features)
        
        # Betting round features (4 dimensions)
        betting_round = [0, 0, 0, 0]
        if len(self.community_cards) == 0:
            betting_round[0] = 1  # preflop
        elif len(self.community_cards) == 3:
            betting_round[1] = 1  # flop
        elif len(self.community_cards) == 4:
            betting_round[2] = 1  # turn
        else:
            betting_round[3] = 1  # river
        features.extend(betting_round)
        
        # Position features (one-hot encoding for up to 6 players)
        position_features = [0] * 6
        if self.position < 6:
            position_features[self.position] = 1
        features.extend(position_features)
        
        # Pot and stack features (normalized)
        max_pot = 10000  # Normalization constant
        max_stack = 5000
        
        features.append(min(self.pot_size / max_pot, 1.0))
        features.append(min(self.to_call / max_pot, 1.0))
        features.append(min(self.stack_size / max_stack, 1.0))
        features.append(self.num_players / 6.0)  # Normalized number of players
        
        # Betting history features (simplified)
        history_features = [0, 0, 0, 0]  # [folds, checks, calls, raises]
        for action in self.betting_history:
            if action == 'F':
                history_features[0] += 1
            elif action == 'K':
                history_features[1] += 1
            elif action == 'C':
                history_features[2] += 1
            elif action == 'R':
                history_features[3] += 1
        
        # Normalize by number of actions
        if len(self.betting_history) > 0:
            history_features = [f / len(self.betting_history) for f in history_features]
        features.extend(history_features)
        
        # Hand strength feature (if possible to calculate)
        if len(self.hole_cards) >= 2:
            try:
                full_hand = self.hole_cards + self.community_cards
                if len(full_hand) >= 2:
                    hand_value = evaluate_hand(full_hand)
                    features.append(min(hand_value / 10000.0, 1.0))  # Normalized
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_features(self) -> np.ndarray:
        return self.features


class DeepCFRPokerAI:
    """Deep CFR implementation for multi-player poker"""
    
    def __init__(self, num_actions: int = 6, learning_rate: float = 0.001, 
                 hidden_size: int = 512, device: str = None):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"DeepCFR using device: {self.device}")
        
        # Feature dimension (calculated based on _create_features)
        self.feature_dim = 52 + 4 + 6 + 4 + 4 + 1  # cards + round + position + pot/stack + history + hand_strength
        
        # Initialize networks
        self.regret_net = RegretNetwork(self.feature_dim, num_actions, hidden_size).to(self.device)
        self.strategy_net = StrategyNetwork(self.feature_dim, num_actions, hidden_size).to(self.device)
        
        # Optimizers
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
        
        # Action mapping
        self.action_mapping = {
            DeepCFRAction.FOLD: Action.FOLD,
            DeepCFRAction.CHECK_CALL: Action.CHECK,
            DeepCFRAction.RAISE_SMALL: Action.RAISE,
            DeepCFRAction.RAISE_MEDIUM: Action.RAISE,
            DeepCFRAction.RAISE_LARGE: Action.RAISE,
            DeepCFRAction.ALL_IN: Action.ALL_IN
        }
        
        # Training data storage
        self.regret_buffer = deque(maxlen=100000)
        self.strategy_buffer = deque(maxlen=100000)
        
        # Statistics
        self.iterations_trained = 0
        self.strategy_stats = {
            'learned_strategy_count': 0,
            'fallback_strategy_count': 0,
            'total_decisions': 0
        }
    
    def train(self, iterations: int = 10000, verbose: bool = True):
        """Train Deep CFR using neural network approximation"""
        if verbose:
            print(f"Training Deep CFR for {iterations} iterations...")
            print(f"Using neural networks with {self.feature_dim} input features")
            print(f"Networks have {self.hidden_size} hidden units each")
        
        for iteration in range(iterations):
            if verbose and (iteration == 0 or (iteration + 1) % (iterations // 20) == 0):
                print(f"  Iteration {iteration}/{iterations}")
            
            # Generate training data through self-play
            self._generate_training_data(iteration)
            
            # Train networks periodically
            if len(self.regret_buffer) > 1000 and iteration % 100 == 0:
                self._train_networks()
        
        self.iterations_trained += iterations
        
        if verbose:
            print(f"Deep CFR training complete!")
            print(f"  Total iterations: {self.iterations_trained}")
            print(f"  Regret buffer size: {len(self.regret_buffer)}")
            print(f"  Strategy buffer size: {len(self.strategy_buffer)}")
    
    def _generate_training_data(self, iteration: int):
        """Generate training data through CFR self-play"""
        # Random number of players (3-6)
        num_players = random.randint(3, 6)
        
        # Deal random cards
        cards = self._deal_random_cards(num_players)
        
        # Random initial conditions
        pot = random.randint(50, 500)
        stack_sizes = [random.randint(800, 1500) for _ in range(num_players)]
        
        # Run CFR on this scenario
        history = ""
        reach_probs = [1.0] * num_players
        training_player = iteration % num_players
        
        self._deep_cfr(cards, history, pot, reach_probs, training_player, 
                      iteration, stack_sizes, num_players)
    
    def _deep_cfr(self, cards: Dict, history: str, pot: int, reach_probs: List[float],
                  training_player: int, iteration: int, stack_sizes: List[int], 
                  num_players: int, depth: int = 0) -> float:
        """Deep CFR algorithm using neural network approximation"""
        current_player = len(history) % num_players
        
        # Depth limit (reduce for faster training)
        if depth > 4:
            return self._get_payoff(cards, history, pot, training_player, stack_sizes, num_players)
        
        # Terminal check
        if self._is_terminal(history, num_players):
            return self._get_payoff(cards, history, pot, training_player, stack_sizes, num_players)
        
        # Create information set
        info_set = DeepCFRInformationSet(
            hole_cards=cards[f'player_{current_player}'],
            community_cards=cards['community'],
            betting_history=history,
            pot_size=pot,
            to_call=self._get_to_call(history, pot),
            stack_size=stack_sizes[current_player],
            position=current_player,
            num_players=num_players
        )
        
        features = torch.tensor(info_set.get_features(), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get current strategy from neural network
        with torch.no_grad():
            strategy_probs = self.strategy_net(features).cpu().numpy().flatten()
        
        # Ensure valid probability distribution
        strategy_probs = np.maximum(strategy_probs, 1e-8)
        strategy_probs = strategy_probs / strategy_probs.sum()
        
        # Available actions
        available_actions = list(DeepCFRAction)
        
        # Calculate utilities for each action
        action_utilities = np.zeros(len(available_actions))
        
        # Sample a subset of actions for efficiency (reduce for faster training)
        num_actions_to_sample = min(2, len(available_actions))
        sampled_indices = np.random.choice(len(available_actions), 
                                         size=num_actions_to_sample, 
                                         replace=False, p=strategy_probs)
        
        node_utility = 0
        for i in sampled_indices:
            action = available_actions[i]
            next_history = history + self._action_to_char(action)
            next_pot = pot + self._get_action_cost(action, pot, stack_sizes[current_player])
            
            if current_player == training_player:
                action_utilities[i] = -self._deep_cfr(cards, next_history, next_pot, reach_probs,
                                                     training_player, iteration, stack_sizes, 
                                                     num_players, depth + 1)
            else:
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy_probs[i]
                action_utilities[i] = -self._deep_cfr(cards, next_history, next_pot, new_reach_probs,
                                                     training_player, iteration, stack_sizes, 
                                                     num_players, depth + 1)
            
            node_utility += strategy_probs[i] * action_utilities[i]
        
        # Store training data for the training player
        if current_player == training_player:
            # Calculate regrets
            regrets = action_utilities - node_utility
            counterfactual_reach = 1.0
            for p in range(num_players):
                if p != current_player:
                    counterfactual_reach *= reach_probs[p]
            
            weighted_regrets = counterfactual_reach * regrets
            
            # Store regret training data
            self.regret_buffer.append((features.cpu().numpy().flatten(), weighted_regrets))
            
            # Store strategy training data (for average strategy learning)
            self.strategy_buffer.append((features.cpu().numpy().flatten(), strategy_probs))
        
        return node_utility
    
    def _train_networks(self):
        """Train regret and strategy networks"""
        batch_size = min(64, len(self.regret_buffer))
        
        # Train regret network
        if len(self.regret_buffer) >= batch_size:
            # Sample batch
            batch_indices = random.sample(range(len(self.regret_buffer)), batch_size)
            regret_batch = [self.regret_buffer[i] for i in batch_indices]
            
            features_batch = torch.tensor(np.array([item[0] for item in regret_batch]), 
                                        dtype=torch.float32).to(self.device)
            regrets_batch = torch.tensor(np.array([item[1] for item in regret_batch]), 
                                       dtype=torch.float32).to(self.device)
            
            # Train regret network
            self.regret_optimizer.zero_grad()
            predicted_regrets = self.regret_net(features_batch)
            regret_loss = F.mse_loss(predicted_regrets, regrets_batch)
            regret_loss.backward()
            self.regret_optimizer.step()
        
        # Train strategy network
        if len(self.strategy_buffer) >= batch_size:
            # Sample batch
            batch_indices = random.sample(range(len(self.strategy_buffer)), batch_size)
            strategy_batch = [self.strategy_buffer[i] for i in batch_indices]
            
            features_batch = torch.tensor(np.array([item[0] for item in strategy_batch]), 
                                        dtype=torch.float32).to(self.device)
            strategies_batch = torch.tensor(np.array([item[1] for item in strategy_batch]), 
                                          dtype=torch.float32).to(self.device)
            
            # Train strategy network
            self.strategy_optimizer.zero_grad()
            predicted_strategies = self.strategy_net(features_batch)
            strategy_loss = F.kl_div(torch.log(predicted_strategies + 1e-8), 
                                   strategies_batch, reduction='batchmean')
            strategy_loss.backward()
            self.strategy_optimizer.step()
    
    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int, stack_size: int,
                   position: int, num_players: int) -> Action:
        """Get action using trained Deep CFR"""
        # Create information set
        info_set = DeepCFRInformationSet(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            pot_size=pot_size,
            to_call=to_call,
            stack_size=stack_size,
            position=position,
            num_players=num_players
        )
        
        self.strategy_stats['total_decisions'] += 1
        
        # Get features
        features = torch.tensor(info_set.get_features(), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get strategy from neural network
        with torch.no_grad():
            if self.iterations_trained > 0:
                # Use learned strategy
                self.strategy_stats['learned_strategy_count'] += 1
                strategy_probs = self.strategy_net(features).cpu().numpy().flatten()
                
                # Sample action based on strategy
                action_idx = np.random.choice(len(strategy_probs), p=strategy_probs)
                deep_cfr_action = list(DeepCFRAction)[action_idx]
            else:
                # Fallback strategy if not trained
                self.strategy_stats['fallback_strategy_count'] += 1
                deep_cfr_action = DeepCFRAction.CHECK_CALL
        
        # Convert to game engine action
        game_action = self.action_mapping[deep_cfr_action]
        
        # Adjust CHECK to CALL if needed
        if game_action == Action.CHECK and to_call > 0:
            game_action = Action.CALL
        
        return game_action
    
    def _deal_random_cards(self, num_players: int) -> Dict:
        """Deal random cards for training"""
        deck = Deck()
        deck.reset()
        
        cards = {}
        
        # Deal hole cards
        for i in range(num_players):
            cards[f'player_{i}'] = deck.draw(2)
        
        # Deal community cards
        num_community = random.choice([0, 3, 4, 5])
        cards['community'] = deck.draw(5)[:num_community] if num_community > 0 else []
        
        return cards
    
    def _is_terminal(self, history: str, num_players: int) -> bool:
        """Check if game state is terminal"""
        # Early termination for efficiency
        if len(history) >= 6:  # Shorter games for training efficiency
            return True
        if 'F' in history:
            return True
        
        # Simple completion check
        if len(history) >= num_players * 2:
            return True
        
        return False
    
    def _get_payoff(self, cards: Dict, history: str, pot: int, player: int, 
                   stack_sizes: List[int], num_players: int) -> float:
        """Calculate payoff for terminal node"""
        # Simplified payoff calculation
        active_players = []
        for i in range(num_players):
            folded = False
            for action_idx, action in enumerate(history):
                if action_idx % num_players == i and action == 'F':
                    folded = True
                    break
            if not folded:
                active_players.append(i)
        
        if len(active_players) == 1:
            if player == active_players[0]:
                return pot - self._get_invested(history, player, num_players)
            else:
                return -self._get_invested(history, player, num_players)
        
        # Showdown
        hand_values = {}
        for p in active_players:
            if f'player_{p}' in cards and cards[f'player_{p}']:
                try:
                    full_hand = cards[f'player_{p}'] + cards['community']
                    hand_values[p] = evaluate_hand(full_hand)
                except:
                    hand_values[p] = sum(card.value for card in cards[f'player_{p}'])
            else:
                hand_values[p] = 0
        
        if hand_values:
            max_value = max(hand_values.values())
            winners = [p for p, value in hand_values.items() if value == max_value]
            
            if player in winners:
                return (pot / len(winners)) - self._get_invested(history, player, num_players)
            else:
                return -self._get_invested(history, player, num_players)
        
        return -self._get_invested(history, player, num_players)
    
    def _get_invested(self, history: str, player: int, num_players: int) -> int:
        """Calculate amount invested by player"""
        invested = 0
        for i, action in enumerate(history):
            if i % num_players == player:
                if action == 'R':
                    invested += 50
                elif action == 'C':
                    invested += 25
                elif action == 'A':
                    invested += 150
        return invested
    
    def _get_to_call(self, history: str, pot: int) -> int:
        """Calculate amount to call"""
        if not history:
            return 0
        
        last_raise = history.rfind('R')
        if last_raise == -1:
            return 0
        
        return 25  # Simplified
    
    def _action_to_char(self, action: DeepCFRAction) -> str:
        """Convert action to character"""
        if action == DeepCFRAction.FOLD:
            return 'F'
        elif action == DeepCFRAction.CHECK_CALL:
            return 'C'
        elif action in [DeepCFRAction.RAISE_SMALL, DeepCFRAction.RAISE_MEDIUM, DeepCFRAction.RAISE_LARGE]:
            return 'R'
        elif action == DeepCFRAction.ALL_IN:
            return 'A'
        return 'C'
    
    def _get_action_cost(self, action: DeepCFRAction, pot: int, stack_size: int) -> int:
        """Get cost of action"""
        if action == DeepCFRAction.FOLD:
            return 0
        elif action == DeepCFRAction.CHECK_CALL:
            return 25
        elif action == DeepCFRAction.RAISE_SMALL:
            return 50
        elif action == DeepCFRAction.RAISE_MEDIUM:
            return 100
        elif action == DeepCFRAction.RAISE_LARGE:
            return 200
        elif action == DeepCFRAction.ALL_IN:
            return min(stack_size, 500)
        return 0
    
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
        """Print strategy usage statistics"""
        stats = self.get_strategy_usage_stats()
        print(f"\n[STATS] Deep CFR Strategy Usage Statistics:")
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
        """Save the Deep CFR model with type flag"""
        torch.save({
            'model_type': 'DeepCFR',  # Model type flag
            'model_version': '1.0',
            'regret_net_state_dict': self.regret_net.state_dict(),
            'strategy_net_state_dict': self.strategy_net.state_dict(),
            'regret_optimizer_state_dict': self.regret_optimizer.state_dict(),
            'strategy_optimizer_state_dict': self.strategy_optimizer.state_dict(),
            'iterations_trained': self.iterations_trained,
            'strategy_stats': self.strategy_stats,
            'feature_dim': self.feature_dim,
            'num_actions': self.num_actions,
            'hidden_size': self.hidden_size
        }, filename)
        print(f"Deep CFR model saved to {filename}")
    
    def load(self, filename: str):
        """Load a Deep CFR model"""
        checkpoint = torch.load(filename, map_location=self.device)

        # Verify it's a Deep CFR model
        if checkpoint.get('model_type') != 'DeepCFR':
            print(f"Warning: Expected DeepCFR model, got {checkpoint.get('model_type', 'Unknown')}")

        # Recreate networks if needed
        feature_dim = checkpoint.get('feature_dim', self.feature_dim)
        num_actions = checkpoint.get('num_actions', self.num_actions)
        hidden_size = checkpoint.get('hidden_size', self.hidden_size)
        
        if feature_dim != self.feature_dim or num_actions != self.num_actions:
            self.regret_net = RegretNetwork(feature_dim, num_actions, hidden_size).to(self.device)
            self.strategy_net = StrategyNetwork(feature_dim, num_actions, hidden_size).to(self.device)
            self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=self.learning_rate)
            self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=self.learning_rate)
        
        # Load state dicts
        self.regret_net.load_state_dict(checkpoint['regret_net_state_dict'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net_state_dict'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer_state_dict'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer_state_dict'])
        
        self.iterations_trained = checkpoint.get('iterations_trained', 0)
        self.strategy_stats = checkpoint.get('strategy_stats', self.strategy_stats)
        
        print(f"Deep CFR model loaded from {filename}")
        print(f"Iterations trained: {self.iterations_trained}")


def train_deep_cfr_ai(iterations: int = 10000, save_filename: Optional[str] = None, 
                     verbose: bool = True) -> DeepCFRPokerAI:
    """
    Train a Deep CFR AI for multi-player poker
    
    Args:
        iterations: Number of training iterations
        save_filename: Optional filename to save the model
        verbose: Whether to print training progress
        
    Returns:
        Trained DeepCFRPokerAI instance
    """
    deep_cfr_ai = DeepCFRPokerAI()
    deep_cfr_ai.train(iterations=iterations, verbose=verbose)
    
    if save_filename:
        deep_cfr_ai.save(save_filename)
        
    return deep_cfr_ai