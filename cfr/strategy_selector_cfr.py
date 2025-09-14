"""
Strategy Selector CFR for 2-Player Texas Hold'em
Uses a neural network to select between multiple specialized strategy networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class StrategyAction(Enum):
    """Actions for Strategy Selector CFR"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_SMALL = 3    # 0.25-0.4 pot
    BET_MEDIUM = 4   # 0.5-0.75 pot
    BET_LARGE = 5    # 0.9-1.1 pot
    BET_OVERBET = 6  # 1.5-2.0 pot
    ALL_IN = 7


class StrategyType(Enum):
    """Different playing strategies"""
    AGGRESSIVE = 0    # High betting, frequent bluffs
    CONSERVATIVE = 1  # Tight play, value betting
    BALANCED = 2      # GTO-inspired mixed strategy
    EXPLOITATIVE = 3  # Adapts to opponent patterns
    DECEPTIVE = 4     # Trapping, slow-playing


class StrategyNetwork(nn.Module):
    """Individual strategy network for a specific playing style"""

    def __init__(self, feature_dim: int = 50, num_actions: int = 8, hidden_size: int = 256, strategy_type: StrategyType = StrategyType.BALANCED):
        super(StrategyNetwork, self).__init__()
        self.strategy_type = strategy_type

        # Different architectures for different strategies
        if strategy_type == StrategyType.AGGRESSIVE:
            self.network = nn.Sequential(
                nn.Linear(feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),  # More dropout for aggressive style
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_actions)
            )
        elif strategy_type == StrategyType.CONSERVATIVE:
            self.network = nn.Sequential(
                nn.Linear(feature_dim, hidden_size // 2),  # Smaller network
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, num_actions)
            )
        else:  # BALANCED, EXPLOITATIVE, DECEPTIVE
            self.network = nn.Sequential(
                nn.Linear(feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_actions)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights based on strategy type"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.strategy_type == StrategyType.AGGRESSIVE:
                    # Bias toward betting actions
                    nn.init.normal_(module.weight, mean=0, std=0.1)
                    if module.out_features == 8:  # Output layer
                        module.bias.data[3:7] += 0.2  # Betting actions
                elif self.strategy_type == StrategyType.CONSERVATIVE:
                    # Bias toward checking/calling
                    nn.init.normal_(module.weight, mean=0, std=0.05)
                    if module.out_features == 8:  # Output layer
                        module.bias.data[1:3] += 0.2  # Check/call
                else:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.network(features)
        return F.softmax(logits, dim=-1)


class StrategySelectorNetwork(nn.Module):
    """Network that selects which strategy to use based on game context"""

    def __init__(self, context_dim: int = 20, num_strategies: int = 5):
        super(StrategySelectorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_strategies)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize for balanced strategy selection"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Get strategy selection probabilities"""
        logits = self.network(context)
        return F.softmax(logits, dim=-1)


class CFRNode:
    """CFR node for a specific strategy"""

    def __init__(self, num_actions: int = 8):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.visits = 0

    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching+"""
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)

        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions

        self.strategy_sum += realization_weight * strategy
        self.visits += 1

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy"""
        if np.sum(self.strategy_sum) > 0:
            return self.strategy_sum / np.sum(self.strategy_sum)
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, utilities: np.ndarray, node_utility: float, weight: float):
        """Update regrets with CFR+"""
        for i in range(self.num_actions):
            regret = utilities[i] - node_utility
            self.regret_sum[i] = max(self.regret_sum[i] + weight * regret, 0)


class StrategySelectorCFR:
    """CFR that uses multiple strategy networks selected by a meta-network"""

    def __init__(self, iterations: int = 10000):
        self.iterations = iterations

        # Strategy networks for different playing styles
        self.strategy_networks = {
            StrategyType.AGGRESSIVE: StrategyNetwork(strategy_type=StrategyType.AGGRESSIVE),
            StrategyType.CONSERVATIVE: StrategyNetwork(strategy_type=StrategyType.CONSERVATIVE),
            StrategyType.BALANCED: StrategyNetwork(strategy_type=StrategyType.BALANCED),
            StrategyType.EXPLOITATIVE: StrategyNetwork(strategy_type=StrategyType.EXPLOITATIVE),
            StrategyType.DECEPTIVE: StrategyNetwork(strategy_type=StrategyType.DECEPTIVE)
        }

        # Strategy selector network
        self.selector_network = StrategySelectorNetwork()

        # Optimizers
        self.strategy_optimizers = {
            strategy: optim.Adam(net.parameters(), lr=0.001)
            for strategy, net in self.strategy_networks.items()
        }
        self.selector_optimizer = optim.Adam(self.selector_network.parameters(), lr=0.001)

        # CFR nodes for each strategy
        self.strategy_nodes = {
            strategy: {} for strategy in StrategyType
        }

        # Performance tracking
        self.strategy_performance = {
            strategy: {'wins': 0, 'games': 0, 'total_reward': 0}
            for strategy in StrategyType
        }

        # Training statistics
        self.iteration_count = 0
        self.training_start_time = None
        self.games_won = 0
        self.games_played = 0

        # Track last selected strategy
        self.last_selected_strategy = StrategyType.BALANCED

        # Memory for experience replay
        self.experience_buffer = deque(maxlen=10000)

    def train(self, verbose: bool = True):
        """Train the strategy selector CFR"""
        self.training_start_time = time.time()

        if verbose:
            print("Starting Strategy Selector CFR Training")
            print(f"Iterations: {self.iterations}")
            print("Training 5 specialized strategy networks...")
            print("=" * 60)

        # Phase 1: Pre-train individual strategies
        if verbose:
            print("\nPhase 1: Pre-training individual strategies...")

        for strategy_type in StrategyType:
            self._pretrain_strategy(strategy_type, iterations=self.iterations // 10)

        # Phase 2: Train strategy selector
        if verbose:
            print("\nPhase 2: Training strategy selector through self-play...")

        for iteration in range(self.iterations):
            self.iteration_count = iteration

            # Sample game and play with strategy selection
            game_state = self._sample_game_state()

            # Select strategy based on context (training mode)
            context = self._extract_context(game_state)
            selected_strategy = self._select_strategy(context, training=True)

            # Run CFR iteration with selected strategy
            utility = self._cfr_iteration(game_state, 0, selected_strategy)

            # Update performance tracking
            won = utility > 0
            self.games_played += 1
            if won:
                self.games_won += 1

            # Update strategy performance
            self.strategy_performance[selected_strategy]['games'] += 1
            if won:
                self.strategy_performance[selected_strategy]['wins'] += 1
            self.strategy_performance[selected_strategy]['total_reward'] += utility

            # Store experience for training
            self.experience_buffer.append({
                'context': context,
                'strategy': selected_strategy,
                'reward': utility,
                'game_state': game_state
            })

            # Update networks periodically
            if iteration % 100 == 0 and iteration > 0:
                self._update_networks()

            # Progress update
            if verbose and iteration % (self.iterations // 20) == 0:
                self._print_progress(iteration)

        if verbose:
            self._print_final_results()

    def _pretrain_strategy(self, strategy_type: StrategyType, iterations: int):
        """Pre-train a specific strategy with biased rewards"""
        for _ in range(iterations):
            game_state = self._sample_game_state()

            # Use strategy-specific reward shaping
            if strategy_type == StrategyType.AGGRESSIVE:
                # Reward aggressive plays
                utility = self._cfr_iteration_biased(game_state, 0, strategy_type, aggression_bonus=0.2)
            elif strategy_type == StrategyType.CONSERVATIVE:
                # Reward conservative plays
                utility = self._cfr_iteration_biased(game_state, 0, strategy_type, safety_bonus=0.2)
            elif strategy_type == StrategyType.DECEPTIVE:
                # Reward trapping plays
                utility = self._cfr_iteration_biased(game_state, 0, strategy_type, deception_bonus=0.2)
            else:
                # Normal training for balanced/exploitative
                utility = self._cfr_iteration(game_state, 0, strategy_type)

    def _cfr_iteration(self, game_state: Dict, player: int, strategy_type: StrategyType) -> float:
        """Run CFR iteration with specified strategy"""
        # Terminal check
        if self._is_terminal(game_state):
            return self._evaluate_terminal(game_state, player)

        # Get current player
        current_player = game_state['current_player']

        # Create info set
        info_set = self._get_info_set(game_state, current_player)

        # Get or create node for this strategy
        if info_set not in self.strategy_nodes[strategy_type]:
            self.strategy_nodes[strategy_type][info_set] = CFRNode()
        node = self.strategy_nodes[strategy_type][info_set]

        # Get strategy from neural network
        if current_player == player:
            features = self._extract_features(game_state, current_player)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                nn_strategy = self.strategy_networks[strategy_type](features_tensor).squeeze(0).numpy()

            # Blend with CFR strategy
            cfr_strategy = node.get_strategy()
            strategy = 0.7 * nn_strategy + 0.3 * cfr_strategy
            strategy /= np.sum(strategy)
        else:
            strategy = node.get_average_strategy()

        # Get valid actions
        valid_actions = self._get_valid_actions(game_state)

        # Sample action
        action_probs = strategy[valid_actions]
        if np.sum(action_probs) <= 0:
            action_probs = np.ones(len(valid_actions)) / len(valid_actions)
        else:
            action_probs = action_probs / np.sum(action_probs)

        action_idx = np.random.choice(len(valid_actions), p=action_probs)
        action = valid_actions[action_idx]

        # Apply action and recurse
        next_state = self._apply_action(game_state.copy(), action)
        utility = self._cfr_iteration(next_state, player, strategy_type)

        # Update regrets if our turn
        if current_player == player:
            action_utilities = np.zeros(len(StrategyAction))
            action_utilities[action] = utility
            node.update_regrets(action_utilities, utility, 1.0)

        return utility

    def _cfr_iteration_biased(self, game_state: Dict, player: int, strategy_type: StrategyType,
                              aggression_bonus: float = 0, safety_bonus: float = 0,
                              deception_bonus: float = 0) -> float:
        """CFR iteration with biased rewards for pre-training"""
        base_utility = self._cfr_iteration(game_state, player, strategy_type)

        # Add strategy-specific bonuses
        if strategy_type == StrategyType.AGGRESSIVE and aggression_bonus > 0:
            # Bonus for betting actions
            if 'last_action' in game_state and game_state['last_action'] in [3, 4, 5, 6]:
                base_utility += aggression_bonus

        elif strategy_type == StrategyType.CONSERVATIVE and safety_bonus > 0:
            # Bonus for careful play
            if 'last_action' in game_state and game_state['last_action'] in [1, 2]:
                base_utility += safety_bonus

        elif strategy_type == StrategyType.DECEPTIVE and deception_bonus > 0:
            # Bonus for check-raise patterns
            if 'action_history' in game_state and len(game_state['action_history']) > 1:
                if game_state['action_history'][-2] == 1 and game_state['action_history'][-1] in [3, 4, 5]:
                    base_utility += deception_bonus

        return base_utility

    def _select_strategy(self, context: np.ndarray, training: bool = True) -> StrategyType:
        """Select strategy using the selector network"""
        context_tensor = torch.FloatTensor(context).unsqueeze(0)

        with torch.no_grad():
            strategy_probs = self.selector_network(context_tensor).squeeze(0).numpy()

        # Add exploration only during training
        if training and self.iteration_count < self.iterations * 0.3:  # 30% exploration phase
            epsilon = 0.2
            if np.random.random() < epsilon:
                return np.random.choice(list(StrategyType))

        # Select based on probabilities
        strategy_idx = np.random.choice(len(StrategyType), p=strategy_probs)
        return list(StrategyType)[strategy_idx]

    def _update_networks(self):
        """Update strategy and selector networks"""
        if len(self.experience_buffer) < 32:
            return

        # Sample batch
        batch = random.sample(self.experience_buffer, 32)

        # Update selector network
        contexts = torch.FloatTensor([exp['context'] for exp in batch])
        strategies = torch.LongTensor([exp['strategy'].value for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute loss for selector (policy gradient)
        strategy_logits = self.selector_network(contexts)
        log_probs = F.log_softmax(strategy_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, strategies.unsqueeze(1)).squeeze(1)

        selector_loss = -(selected_log_probs * rewards).mean()

        self.selector_optimizer.zero_grad()
        selector_loss.backward()
        self.selector_optimizer.step()

        # Update individual strategy networks (supervised learning on good experiences)
        for strategy_type in StrategyType:
            strategy_experiences = [exp for exp in batch if exp['strategy'] == strategy_type and exp['reward'] > 0]

            if len(strategy_experiences) > 4:
                # Train on successful experiences
                states = [self._extract_features(exp['game_state'], 0) for exp in strategy_experiences]
                features = torch.FloatTensor(states)

                # Use the CFR strategy as target
                targets = []
                for exp in strategy_experiences:
                    info_set = self._get_info_set(exp['game_state'], 0)
                    if info_set in self.strategy_nodes[strategy_type]:
                        targets.append(self.strategy_nodes[strategy_type][info_set].get_average_strategy())
                    else:
                        targets.append(np.ones(8) / 8)

                targets = torch.FloatTensor(targets)

                # Compute loss
                predictions = self.strategy_networks[strategy_type](features)
                loss = F.kl_div(predictions.log(), targets, reduction='batchmean')

                self.strategy_optimizers[strategy_type].zero_grad()
                loss.backward()
                self.strategy_optimizers[strategy_type].step()

    def _extract_context(self, game_state: Dict) -> np.ndarray:
        """Extract context features for strategy selection"""
        features = []

        # Pot size (normalized)
        pot = game_state.get('pot', 0)
        features.append(min(pot / 1000, 1.0))

        # Stack sizes
        stacks = game_state.get('stacks', [1000, 1000])
        features.extend([min(s / 1000, 2.0) for s in stacks])

        # Betting round
        round_idx = len(game_state.get('community_cards', [])) // 2
        features.append(round_idx / 3)

        # Hand strength estimate
        if 'hole_cards' in game_state and game_state['hole_cards']:
            strength = self._estimate_hand_strength(game_state['hole_cards'][0],
                                                   game_state.get('community_cards', []))
            features.append(strength)
        else:
            features.append(0.5)

        # Aggression level (betting history)
        history = game_state.get('betting_history', '')
        num_bets = history.count('b') + history.count('r')
        num_checks = history.count('c')
        aggression = num_bets / max(num_bets + num_checks, 1)
        features.append(aggression)

        # Position
        position = game_state.get('position', 0)
        features.append(position)

        # Pad to fixed size
        while len(features) < 20:
            features.append(0)

        return np.array(features[:20])

    def _extract_features(self, game_state: Dict, player: int) -> np.ndarray:
        """Extract features for strategy networks"""
        features = []

        # Basic game state features
        pot = game_state.get('pot', 0)
        features.append(min(pot / 1000, 2.0))

        stacks = game_state.get('stacks', [1000, 1000])
        features.append(min(stacks[player] / 1000, 2.0))
        features.append(min(stacks[1-player] / 1000, 2.0))

        # Cards
        hole_cards = game_state.get('hole_cards', [[], []])
        community_cards = game_state.get('community_cards', [])

        # Encode hole cards (simplified)
        if hole_cards[player]:
            for card in hole_cards[player][:2]:
                # Convert rank to numeric value
                rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                              '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                rank_val = rank_values.get(card.rank, 7)
                features.append(rank_val / 14)
                # Convert suit to numeric
                suit_values = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
                suit_val = suit_values.get(card.suit, 0)
                features.append(suit_val / 4)
        else:
            features.extend([0, 0, 0, 0])

        # Encode community cards
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                      '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_values = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
        for i in range(5):
            if i < len(community_cards):
                rank_val = rank_values.get(community_cards[i].rank, 7)
                features.append(rank_val / 14)
                suit_val = suit_values.get(community_cards[i].suit, 0)
                features.append(suit_val / 4)
            else:
                features.extend([0, 0])

        # Betting history encoding
        history = game_state.get('betting_history', '')
        for i in range(10):
            if i < len(history):
                action_map = {'f': 0.1, 'c': 0.3, 'b': 0.6, 'r': 0.8, 'a': 1.0}
                features.append(action_map.get(history[i], 0))
            else:
                features.append(0)

        # Pad or truncate to 50 features
        while len(features) < 50:
            features.append(0)

        return np.array(features[:50])

    def _sample_game_state(self) -> Dict:
        """Sample a random game state for training"""
        deck = Deck()
        random.shuffle(deck.cards)

        # Deal hole cards
        hole_cards = [
            [deck.draw(), deck.draw()],
            [deck.draw(), deck.draw()]
        ]

        # Random community cards (0-5)
        num_community = np.random.choice([0, 3, 4, 5])
        community_cards = [deck.draw() for _ in range(num_community)]

        # Random pot and stacks
        pot = np.random.randint(20, 500)
        stacks = [np.random.randint(500, 1500), np.random.randint(500, 1500)]

        return {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot': pot,
            'stacks': stacks,
            'current_player': 0,
            'betting_history': '',
            'to_call': 0,
            'position': np.random.choice([0, 1])
        }

    def _get_info_set(self, game_state: Dict, player: int) -> str:
        """Create information set string"""
        hole_cards = game_state.get('hole_cards', [[], []])[player]
        community_cards = game_state.get('community_cards', [])
        history = game_state.get('betting_history', '')

        # Simplified info set
        cards_str = ''.join([f"{c.rank}{c.suit}" for c in hole_cards])
        comm_str = ''.join([f"{c.rank}{c.suit}" for c in community_cards])

        return f"{cards_str}|{comm_str}|{history}"

    def _get_valid_actions(self, game_state: Dict) -> List[int]:
        """Get valid actions for current state"""
        to_call = game_state.get('to_call', 0)
        player = game_state['current_player']
        stack = game_state['stacks'][player]

        valid = []

        # Always can fold
        valid.append(StrategyAction.FOLD.value)

        # Check if no bet to call
        if to_call == 0:
            valid.append(StrategyAction.CHECK.value)

        # Call if there's a bet and we have chips
        if to_call > 0 and stack > to_call:
            valid.append(StrategyAction.CALL.value)

        # Betting/raising if we have chips
        if stack > to_call:
            valid.extend([
                StrategyAction.BET_SMALL.value,
                StrategyAction.BET_MEDIUM.value,
                StrategyAction.BET_LARGE.value
            ])

        # All-in always available if we have chips
        if stack > 0:
            valid.append(StrategyAction.ALL_IN.value)

        return valid

    def _apply_action(self, game_state: Dict, action: int) -> Dict:
        """Apply action to game state"""
        new_state = game_state.copy()
        player = new_state['current_player']

        new_state['last_action'] = action
        if 'action_history' not in new_state:
            new_state['action_history'] = []
        new_state['action_history'].append(action)

        if action == StrategyAction.FOLD.value:
            new_state['is_terminal'] = True
            new_state['winner'] = 1 - player
            new_state['betting_history'] += 'f'

        elif action == StrategyAction.CHECK.value:
            new_state['betting_history'] += 'c'
            new_state['current_player'] = 1 - player

        elif action == StrategyAction.CALL.value:
            call_amount = min(new_state['to_call'], new_state['stacks'][player])
            new_state['stacks'][player] -= call_amount
            new_state['pot'] += call_amount
            new_state['betting_history'] += 'c'
            new_state['current_player'] = 1 - player

        elif action in [StrategyAction.BET_SMALL.value, StrategyAction.BET_MEDIUM.value,
                       StrategyAction.BET_LARGE.value, StrategyAction.BET_OVERBET.value]:
            # Bet sizing
            bet_sizes = {
                StrategyAction.BET_SMALL.value: 0.33,
                StrategyAction.BET_MEDIUM.value: 0.66,
                StrategyAction.BET_LARGE.value: 1.0,
                StrategyAction.BET_OVERBET.value: 1.75
            }

            bet_fraction = bet_sizes[action]
            bet_amount = int(new_state['pot'] * bet_fraction)
            bet_amount = min(bet_amount, new_state['stacks'][player])

            new_state['stacks'][player] -= bet_amount
            new_state['pot'] += bet_amount
            new_state['betting_history'] += 'b'
            new_state['current_player'] = 1 - player
            new_state['to_call'] = bet_amount

        elif action == StrategyAction.ALL_IN.value:
            all_in_amount = new_state['stacks'][player]
            new_state['stacks'][player] = 0
            new_state['pot'] += all_in_amount
            new_state['betting_history'] += 'a'
            new_state['current_player'] = 1 - player
            new_state['to_call'] = all_in_amount

        return new_state

    def _is_terminal(self, game_state: Dict) -> bool:
        """Check if game state is terminal"""
        return game_state.get('is_terminal', False) or any(s == 0 for s in game_state['stacks'])

    def _evaluate_terminal(self, game_state: Dict, player: int) -> float:
        """Evaluate terminal node"""
        if game_state.get('winner') is not None:
            winner = game_state['winner']
            pot = game_state['pot']
            starting_stack = 1000
            our_investment = starting_stack - game_state['stacks'][player]

            if winner == player:
                return pot - our_investment
            else:
                return -our_investment
        else:
            # Showdown - simplified
            return 0

    def _estimate_hand_strength(self, hole_cards: List[Card], community_cards: List[Card]) -> float:
        """Estimate hand strength (0-1)"""
        if not hole_cards:
            return 0.5

        # Simplified hand strength calculation
        all_cards = hole_cards + community_cards

        if len(all_cards) >= 5:
            hand_value = evaluate_hand(all_cards)
            # Normalize to 0-1 range (very simplified)
            return min(hand_value / 10000, 1.0)
        else:
            # Pre-flop hand strength based on hole cards
            rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                          '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            high_card = max(rank_values.get(card.rank, 7) for card in hole_cards)
            return high_card / 14

    def _print_progress(self, iteration: int):
        """Print training progress"""
        elapsed = time.time() - self.training_start_time
        win_rate = self.games_won / max(self.games_played, 1) * 100

        print(f"Iteration {iteration}/{self.iterations}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Show strategy performance
        print("  Strategy performance:")
        for strategy in StrategyType:
            stats = self.strategy_performance[strategy]
            if stats['games'] > 0:
                strategy_win_rate = stats['wins'] / stats['games'] * 100
                avg_reward = stats['total_reward'] / stats['games']
                print(f"    {strategy.name}: {strategy_win_rate:.1f}% win rate, {avg_reward:.2f} avg reward")
        print()

    def _print_final_results(self):
        """Print final training results"""
        elapsed = time.time() - self.training_start_time

        print("=" * 60)
        print("STRATEGY SELECTOR CFR TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Final win rate: {self.games_won/max(self.games_played,1)*100:.1f}%")

        print("\nStrategy Usage Statistics:")
        for strategy in StrategyType:
            stats = self.strategy_performance[strategy]
            if stats['games'] > 0:
                usage_rate = stats['games'] / self.games_played * 100
                win_rate = stats['wins'] / stats['games'] * 100
                avg_reward = stats['total_reward'] / stats['games']
                print(f"  {strategy.name}:")
                print(f"    Usage: {usage_rate:.1f}%")
                print(f"    Win rate: {win_rate:.1f}%")
                print(f"    Avg reward: {avg_reward:.2f}")

        print("\nTraining Summary:")
        print("  - 5 specialized strategy networks trained")
        print("  - Strategy selector learned to choose optimal strategies")
        print("  - System can adapt playing style based on game context")
        print("=" * 60)

    def get_action(self, hole_cards: List[Card], community_cards: List[Card],
                   betting_history: str, pot_size: int, to_call: int,
                   stack_size: int, position: int, opponent_stack: int = 1000,
                   num_players: int = 2) -> Action:
        """Get action for gameplay"""
        # Create game state
        game_state = {
            'hole_cards': [hole_cards, []],
            'community_cards': community_cards,
            'pot': pot_size,
            'stacks': [stack_size, opponent_stack],
            'current_player': 0,
            'betting_history': betting_history,
            'to_call': to_call,
            'position': position
        }

        # Extract context and select strategy (not training mode)
        context = self._extract_context(game_state)
        strategy_type = self._select_strategy(context, training=False)

        # Track strategy usage for statistics
        self.strategy_performance[strategy_type]['games'] += 1
        self.games_played += 1

        # Store last selected strategy for reference
        self.last_selected_strategy = strategy_type

        # Debug: Uncomment to see strategy selection
        # print(f"Selected {strategy_type.name} strategy (total: {self.games_played})")

        # Get features and compute action probabilities
        features = self._extract_features(game_state, 0)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.strategy_networks[strategy_type](features_tensor).squeeze(0).numpy()

        # Get valid actions and sample
        valid_actions = self._get_valid_actions(game_state)
        valid_probs = action_probs[valid_actions]

        if np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
        else:
            valid_probs = np.ones(len(valid_actions)) / len(valid_actions)

        chosen_action = np.random.choice(valid_actions, p=valid_probs)

        # Convert to game action
        action_map = {
            StrategyAction.FOLD.value: Action.FOLD,
            StrategyAction.CHECK.value: Action.CHECK,
            StrategyAction.CALL.value: Action.CALL,
            StrategyAction.BET_SMALL.value: Action.RAISE,
            StrategyAction.BET_MEDIUM.value: Action.RAISE,
            StrategyAction.BET_LARGE.value: Action.RAISE,
            StrategyAction.BET_OVERBET.value: Action.RAISE,
            StrategyAction.ALL_IN.value: Action.ALL_IN
        }

        return action_map.get(chosen_action, Action.CALL)

    def get_raise_size(self, pot_size: int, current_bet: int = 0,
                       player_chips: int = 1000, min_raise: int = 20) -> int:
        """Get raise size based on last selected strategy"""
        if pot_size <= 0:
            return min_raise

        # Use probabilities from different strategies to determine sizing
        sizes = [
            max(min_raise, int(pot_size * 0.33)),  # Small
            max(min_raise, int(pot_size * 0.66)),  # Medium
            max(min_raise, int(pot_size * 1.0))    # Large
        ]

        # Weight by strategy performance or use last selected strategy
        if hasattr(self, 'last_selected_strategy') and self.last_selected_strategy:
            # Use sizing based on last selected strategy
            if self.last_selected_strategy == StrategyType.AGGRESSIVE:
                # Aggressive sizing - prefer larger bets
                weights = np.array([0.1, 0.3, 0.6])
            elif self.last_selected_strategy == StrategyType.CONSERVATIVE:
                # Conservative sizing - prefer smaller bets
                weights = np.array([0.6, 0.3, 0.1])
            else:
                # Balanced sizing
                weights = np.array([0.33, 0.34, 0.33])
        else:
            # Fallback to equal weights
            weights = np.array([1.0/3, 1.0/3, 1.0/3])

        # Ensure weights are valid
        if np.any(np.isnan(weights)) or np.sum(weights) <= 0:
            weights = np.array([1.0/3, 1.0/3, 1.0/3])

        # Normalize weights
        weights = weights / np.sum(weights)

        return np.random.choice(sizes, p=weights)

    def save(self, filename: str):
        """Save the model"""
        data = {
            'model_type': 'StrategySelectorCFR',
            'model_version': '1.0',
            'strategy_nodes': self.strategy_nodes,
            'strategy_performance': self.strategy_performance,
            'iterations': self.iteration_count,
            'games_played': self.games_played,
            'games_won': self.games_won
        }

        # Save neural network states
        for strategy_type, network in self.strategy_networks.items():
            data[f'strategy_network_{strategy_type.name}'] = network.state_dict()

        data['selector_network'] = self.selector_network.state_dict()

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Strategy Selector CFR saved to {filename}")

    def load(self, filename: str):
        """Load the model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.strategy_nodes = data.get('strategy_nodes', {})
        self.strategy_performance = data.get('strategy_performance', {})
        self.iteration_count = data.get('iterations', 0)
        self.games_played = data.get('games_played', 0)
        self.games_won = data.get('games_won', 0)

        # Load neural network states
        for strategy_type in StrategyType:
            key = f'strategy_network_{strategy_type.name}'
            if key in data:
                self.strategy_networks[strategy_type].load_state_dict(data[key])

        if 'selector_network' in data:
            self.selector_network.load_state_dict(data['selector_network'])

        print(f"Strategy Selector CFR loaded from {filename}")

    def reset_strategy_stats(self):
        """Reset strategy usage statistics (for evaluation compatibility)"""
        # Reset performance tracking
        self.strategy_performance = {
            strategy: {'wins': 0, 'games': 0, 'total_reward': 0}
            for strategy in StrategyType
        }
        self.games_won = 0
        self.games_played = 0

    def get_strategy_usage_stats(self) -> Dict:
        """Get strategy usage statistics (for evaluation compatibility)"""
        total_games = sum(s['games'] for s in self.strategy_performance.values())

        # Calculate which strategy is used most
        if total_games > 0:
            most_used = max(self.strategy_performance.items(),
                           key=lambda x: x[1]['games'])
            most_used_name = most_used[0].name
        else:
            most_used_name = "NONE"

        return {
            'total_used': total_games,
            'total_learned': total_games,  # All decisions use learned networks
            'learned_percentage': 100.0 if total_games > 0 else 0,  # Always using neural networks
            'unique_situations': sum(len(nodes) for nodes in self.strategy_nodes.values()),
            'most_used_strategy': most_used_name,
            'strategy_distribution': {
                strategy.name: (stats['games'] / total_games * 100) if total_games > 0 else 0
                for strategy, stats in self.strategy_performance.items()
            }
        }

    def print_strategy_usage(self):
        """Print strategy usage statistics (for evaluation compatibility)"""
        stats = self.get_strategy_usage_stats()
        print(f"\nStrategy Usage Statistics:")
        print(f"  Total decisions: {stats['total_used']}")
        print(f"  Unique situations: {stats['unique_situations']}")
        print(f"  Most used strategy: {stats['most_used_strategy']}")
        print(f"  Strategy distribution:")
        for strategy, usage in stats['strategy_distribution'].items():
            print(f"    {strategy}: {usage:.1f}%")