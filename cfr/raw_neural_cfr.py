"""
Raw Neural CFR for 2-Player Texas Hold'em
Learns directly from raw game state with proper CFR and exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from core.game_constants import Action
from core.card_deck import Card, Deck, evaluate_hand


class CFRAction(Enum):
    """Actions for Raw Neural CFR"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_SMALL = 3    # 0.25-0.5 pot
    BET_MEDIUM = 4   # 0.5-0.75 pot
    BET_LARGE = 5    # 0.75-1.0 pot
    BET_POT = 6      # 1.0 pot
    ALL_IN = 7


class CardEmbedding(nn.Module):
    """Learnable card embeddings"""

    def __init__(self, embedding_dim: int = 32):
        super(CardEmbedding, self).__init__()
        # 52 cards + 1 for no card (padding)
        self.rank_embedding = nn.Embedding(15, embedding_dim // 2)  # 2-A + padding
        self.suit_embedding = nn.Embedding(5, embedding_dim // 2)   # 4 suits + padding

    def forward(self, cards: List[Optional[Card]]) -> torch.Tensor:
        """Embed a list of cards"""
        rank_indices = []
        suit_indices = []

        for card in cards:
            if card is None:
                rank_indices.append(0)  # Padding index
                suit_indices.append(0)
            else:
                # Convert rank to index (2=1, 3=2, ..., A=13)
                rank_map = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
                           '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
                rank_idx = rank_map.get(card.rank, 0)
                rank_indices.append(rank_idx)

                # Convert suit to index
                suit_map = {'♠': 1, '♥': 2, '♦': 3, '♣': 4}
                suit_idx = suit_map.get(card.suit, 0)
                suit_indices.append(suit_idx)

        rank_tensor = torch.LongTensor(rank_indices)
        suit_tensor = torch.LongTensor(suit_indices)

        rank_emb = self.rank_embedding(rank_tensor)
        suit_emb = self.suit_embedding(suit_tensor)

        # Concatenate rank and suit embeddings
        return torch.cat([rank_emb, suit_emb], dim=-1)


class ActionEncoder(nn.Module):
    """Encode action sequences with attention"""

    def __init__(self, action_dim: int = 8, hidden_dim: int = 128):
        super(ActionEncoder, self).__init__()
        self.action_embedding = nn.Embedding(action_dim + 1, hidden_dim)  # +1 for padding

        # Self-attention for action sequences
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, action_sequence: List[int], max_len: int = 20) -> torch.Tensor:
        """Encode a sequence of actions"""
        # Pad or truncate to fixed length
        if len(action_sequence) > max_len:
            action_sequence = action_sequence[-max_len:]
        else:
            action_sequence = action_sequence + [0] * (max_len - len(action_sequence))

        action_tensor = torch.LongTensor(action_sequence).unsqueeze(0)  # Add batch dim
        action_emb = self.action_embedding(action_tensor)

        # Self-attention
        attn_out, _ = self.attention(action_emb, action_emb, action_emb)
        output = self.norm(attn_out + action_emb)  # Residual connection

        # Global pooling
        return output.mean(dim=1).squeeze(0)


class RawNeuralCFRNetwork(nn.Module):
    """Main network for Raw Neural CFR"""

    def __init__(self, num_actions: int = 8, hidden_dim: int = 512):
        super(RawNeuralCFRNetwork, self).__init__()

        # Card encoders
        self.card_embedding = CardEmbedding(embedding_dim=32)
        self.action_encoder = ActionEncoder(action_dim=num_actions, hidden_dim=128)

        # Feature dimensions
        card_features = 32 * 7  # 2 hole cards + 5 community cards
        game_features = 20  # Pot, stacks, positions, etc.
        action_features = 128  # From action encoder
        total_features = card_features + game_features + action_features

        # Deep network with residual connections
        self.input_projection = nn.Linear(total_features, hidden_dim)

        # Residual blocks
        self.res_block1 = self._make_residual_block(hidden_dim)
        self.res_block2 = self._make_residual_block(hidden_dim)
        self.res_block3 = self._make_residual_block(hidden_dim)

        # Output heads
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _make_residual_block(self, hidden_dim: int) -> nn.Module:
        """Create a residual block"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def _init_weights(self):
        """Initialize weights to prevent folding bias"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Initialize bias to zero (no action preference)
                    nn.init.constant_(module.bias, 0)

    def forward(self, game_state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        # Extract and encode cards
        hole_cards = game_state.get('hole_cards', [None, None])
        community_cards = game_state.get('community_cards', [])

        # Pad community cards to 5
        while len(community_cards) < 5:
            community_cards.append(None)

        all_cards = hole_cards + community_cards[:5]
        card_features = self.card_embedding(all_cards).flatten()

        # Encode action history
        action_history = game_state.get('action_history', [])
        action_features = self.action_encoder(action_history)

        # Extract game features
        game_features = self._extract_game_features(game_state)

        # Combine all features
        combined = torch.cat([card_features, action_features, game_features])

        # Pass through network
        x = self.input_projection(combined)
        x = F.relu(x)

        # Residual blocks
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)

        residual = x
        x = self.res_block2(x)
        x = F.relu(x + residual)

        residual = x
        x = self.res_block3(x)
        x = F.relu(x + residual)

        # Output heads
        strategy = self.strategy_head(x)
        value = self.value_head(x)

        return strategy, value

    def _extract_game_features(self, game_state: Dict) -> torch.Tensor:
        """Extract numerical game features"""
        features = []

        # Pot size (normalized)
        pot = game_state.get('pot', 0)
        features.append(pot / 2000.0)  # Normalize by typical max pot

        # Stack sizes (both our stack and opponent's)
        stacks = game_state.get('stacks', [1000, 1000])
        for stack in stacks[:2]:
            features.append(stack / 1000.0)

        # CRITICAL: Amount to call (normalized)
        to_call = game_state.get('to_call', 0)
        features.append(to_call / 1000.0)  # Normalized by starting stack

        # Pot odds (call_amount / (pot + call_amount))
        if to_call > 0 and pot > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        features.append(pot_odds)

        # Current bets (shows betting in current round)
        bets = game_state.get('current_bets', [0, 0])
        for bet in bets[:2]:
            features.append(bet / 1000.0)

        # Position information (important for strategy)
        button = game_state.get('button', 0)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 2)

        # Are we in position? (acting after opponent)
        in_position = float((position - button) % num_players > (num_players // 2))
        features.append(in_position)

        # Relative position (0 = button, normalized)
        relative_position = (position - button) % num_players
        features.append(relative_position / max(1, num_players - 1))

        # Game stage (one-hot)
        stage = game_state.get('stage', 0)  # 0=preflop, 1=flop, 2=turn, 3=river
        stage_onehot = [0.0] * 4
        if 0 <= stage < 4:
            stage_onehot[stage] = 1.0
        features.extend(stage_onehot)

        # Stack to pot ratio (SPR) - important for decision making
        if pot > 0:
            spr = stacks[0] / pot if stacks[0] > 0 else 0
        else:
            spr = 10.0  # High SPR when pot is 0
        features.append(min(spr / 10.0, 1.0))  # Cap at 1.0

        # Can afford to call? (binary)
        can_call = 1.0 if stacks[0] >= to_call else 0.0
        features.append(can_call)

        # Legal actions mask (first 4 important ones)
        legal_actions = game_state.get('legal_actions', [1] * 8)
        # Include FOLD, CHECK, CALL, and whether any RAISE is legal
        features.append(float(legal_actions[CFRAction.FOLD.value]))
        features.append(float(legal_actions[CFRAction.CHECK.value]))
        features.append(float(legal_actions[CFRAction.CALL.value]))
        # Any raise action legal?
        can_raise = any(legal_actions[i] for i in [3, 4, 5, 6])
        features.append(float(can_raise))

        # Pad to fixed size (still 20)
        while len(features) < 20:
            features.append(0.0)

        return torch.FloatTensor(features[:20])


class RawNeuralCFR:
    """Raw Neural CFR with proper exploration and CFR algorithm"""

    def __init__(self, iterations: int = 50000, learning_rate: float = 0.001,
                 batch_size: int = 32, buffer_size: int = 100000,
                 initial_epsilon: float = 0.3, final_epsilon: float = 0.05,
                 initial_temperature: float = 1.0, final_temperature: float = 0.1,
                 hidden_dim: int = 512, use_improved_selfplay: bool = False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.use_improved_selfplay = use_improved_selfplay

        # Exploration parameters
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = initial_epsilon

        # Temperature for action sampling
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature

        # Neural network with configurable hidden dimension
        self.network = RawNeuralCFRNetwork(hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        if use_improved_selfplay:
            # For improved self-play, create a second network as opponent
            self.opponent_network = RawNeuralCFRNetwork(hidden_dim=hidden_dim)
            self.opponent_optimizer = optim.Adam(self.opponent_network.parameters(), lr=learning_rate)
            # No target network needed in improved self-play
            self.target_network = None
            self.update_target_every = None
        else:
            # Normal self-play: use target network
            self.target_network = RawNeuralCFRNetwork(hidden_dim=hidden_dim)
            self.target_network.load_state_dict(self.network.state_dict())
            self.update_target_every = 500  # Back to higher value for normal mode

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=buffer_size)

        # CFR tracking
        self.regret_sum = defaultdict(lambda: np.zeros(8))
        self.strategy_sum = defaultdict(lambda: np.zeros(8))
        self.iteration_count = 0

        # Training statistics
        self.training_stats = {
            'iterations': [],
            'avg_regret': [],
            'avg_utility': [],
            'loss': [],
            'epsilon': [],
            'temperature': []
        }

    def train(self, verbose: bool = True):
        """Train the Raw Neural CFR"""
        if self.use_improved_selfplay:
            return self.train_improved_selfplay(verbose)

        if verbose:
            print("Starting Raw Neural CFR Training (Normal Self-Play)")
            print(f"Iterations: {self.iterations}")
            print(f"Initial epsilon: {self.initial_epsilon} -> {self.final_epsilon}")
            print(f"Initial temperature: {self.initial_temperature} -> {self.final_temperature}")
            print(f"Target network update frequency: {self.update_target_every}")
            print("=" * 60)

        start_time = time.time()

        for iteration in range(self.iterations):
            self.iteration_count = iteration

            # Update exploration parameters
            progress = iteration / self.iterations
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * progress

            # Exponential decay for temperature: T(t) = T_final + (T_init - T_final) * exp(-decay_rate * progress)
            # We want temperature to decay from initial to final over the training duration
            # Using decay_rate = 5 means we reach ~0.7% of the initial range at progress=1
            decay_rate = 5.0
            temp_range = self.initial_temperature - self.final_temperature
            self.temperature = self.final_temperature + temp_range * math.exp(-decay_rate * progress)

            # Sample game trajectory with exploration
            trajectory = self._sample_game_trajectory()

            # Calculate advantages and update regrets
            self._update_regrets(trajectory)

            # Store experiences
            for experience in trajectory:
                self.experience_buffer.append(experience)

            # Train on batch when buffer is large enough
            if len(self.experience_buffer) >= self.batch_size * 2:
                loss = self._train_on_batch()
                self.training_stats['loss'].append(loss)

            # Update target network periodically
            if iteration % self.update_target_every == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            # Progress update
            if verbose and iteration % max(1, self.iterations // 20) == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(self.training_stats['loss'][-100:]) if self.training_stats['loss'] else 0
                print(f"Iteration {iteration}/{self.iterations} | "
                      f"Time: {elapsed:.1f}s | Loss: {avg_loss:.4f} | "
                      f"eps: {self.epsilon:.3f} | T: {self.temperature:.2f}")

        if verbose:
            print(f"\nTraining complete in {time.time() - start_time:.1f} seconds")

    def train_improved_selfplay(self, verbose: bool = True):
        """Train using improved self-play with two separate networks"""
        if verbose:
            print("Starting Raw Neural CFR Training (Improved Self-Play)")
            print(f"Mode: Two separate networks playing against each other")
            print(f"Iterations: {self.iterations}")
            print(f"Initial epsilon: {self.initial_epsilon} -> {self.final_epsilon}")
            print(f"Initial temperature: {self.initial_temperature} -> {self.final_temperature}")
            print("=" * 60)

        start_time = time.time()

        # Track separate losses for each network
        main_losses = []
        opponent_losses = []

        for iteration in range(self.iterations):
            self.iteration_count = iteration

            # Update exploration parameters
            progress = iteration / self.iterations
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * progress

            # Exponential decay for temperature
            decay_rate = 5.0
            temp_range = self.initial_temperature - self.final_temperature
            self.temperature = self.final_temperature + temp_range * math.exp(-decay_rate * progress)

            # Sample game trajectory with both networks
            trajectory = self._sample_game_trajectory_improved()

            # Update regrets
            self._update_regrets(trajectory)

            # Separate experiences by network
            main_experiences = []
            opponent_experiences = []

            for exp in trajectory:
                if exp.get('network') == 'main':
                    main_experiences.append(exp)
                else:
                    opponent_experiences.append(exp)

            # Store experiences
            for exp in trajectory:
                self.experience_buffer.append(exp)

            # Train both networks when buffer is large enough
            if len(self.experience_buffer) >= self.batch_size * 2:
                # Train main network on main player experiences
                if main_experiences:
                    main_loss = self._train_on_batch_for_network(self.network, self.optimizer, 'main')
                    main_losses.append(main_loss)

                # Train opponent network on opponent player experiences
                if opponent_experiences:
                    opponent_loss = self._train_on_batch_for_network(self.opponent_network, self.opponent_optimizer, 'opponent')
                    opponent_losses.append(opponent_loss)

            # Periodically swap networks to ensure diversity
            if iteration % 1000 == 999:
                # 50% chance to swap which network plays as Player 1
                if random.random() < 0.5:
                    self.network, self.opponent_network = self.opponent_network, self.network
                    self.optimizer, self.opponent_optimizer = self.opponent_optimizer, self.optimizer
                    if verbose:
                        print(f"  Swapped networks at iteration {iteration}")

            # Progress update
            if verbose and iteration % max(1, self.iterations // 20) == 0:
                elapsed = time.time() - start_time
                avg_main_loss = np.mean(main_losses[-100:]) if main_losses else 0
                avg_opp_loss = np.mean(opponent_losses[-100:]) if opponent_losses else 0
                print(f"Iteration {iteration}/{self.iterations} | "
                      f"Time: {elapsed:.1f}s | Main Loss: {avg_main_loss:.4f} | "
                      f"Opp Loss: {avg_opp_loss:.4f} | "
                      f"eps: {self.epsilon:.3f} | T: {self.temperature:.2f}")

        if verbose:
            print(f"\nImproved self-play training complete in {time.time() - start_time:.1f} seconds")

    def _train_on_batch_for_network(self, network: nn.Module, optimizer: optim.Optimizer, network_type: str) -> float:
        """Train a specific network on a batch of experiences"""
        # Sample experiences for this network
        network_experiences = [exp for exp in self.experience_buffer if exp.get('network') == network_type]

        if len(network_experiences) < self.batch_size:
            # Not enough experiences for this network yet
            return 0.0

        batch = random.sample(network_experiences, min(self.batch_size, len(network_experiences)))

        total_loss = 0

        for experience in batch:
            game_state = experience['state']
            action = experience['action']
            utility = experience['utility']

            # Forward pass
            strategy, value = network(game_state)

            # Calculate advantage
            with torch.no_grad():
                baseline = value.item()
                advantage = utility - baseline
                advantage = np.clip(advantage, -10, 10)

            # Policy gradient loss
            log_probs = F.log_softmax(strategy, dim=0)
            log_probs = torch.clamp(log_probs, min=-10)
            policy_loss = -log_probs[action] * advantage

            # Value loss
            value_target = torch.FloatTensor([utility])
            value_loss = F.smooth_l1_loss(value, value_target)

            # Combined loss
            loss = 0.1 * policy_loss + 0.5 * value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(batch)

    def _sample_game_trajectory(self) -> List[Dict]:
        """Sample a complete game trajectory with exploration"""
        trajectory = []

        # Initialize game state
        deck = Deck()  # Deck is automatically shuffled on creation

        # Deal hole cards
        p1_cards = [deck.draw(), deck.draw()]
        p2_cards = [deck.draw(), deck.draw()]

        # Track starting chips for utility calculation
        starting_chips = [1000, 1000]

        game_state = {
            'hole_cards': p1_cards,  # Current player's cards
            'community_cards': [],
            'pot': 3,  # Small blind + big blind
            'stacks': [999, 998],  # After blinds
            'current_player': 0,
            'button': 0,
            'current_bets': [1, 2],  # Small blind, big blind
            'active_players': [True, True],
            'stage': 0,  # Preflop
            'action_history': [],
            'legal_actions': [1, 0, 1, 1, 1, 1, 1, 1]  # All except check (can't check facing bet)
        }

        # Play through the hand
        for stage in range(4):  # Preflop, flop, turn, river
            if stage == 1:  # Flop
                game_state['community_cards'] = deck.draw(3)  # Draw 3 cards as a list
            elif stage == 2:  # Turn
                turn_card = deck.draw()  # Draw single card
                if turn_card:
                    game_state['community_cards'].append(turn_card)
            elif stage == 3:  # River
                river_card = deck.draw()  # Draw single card
                if river_card:
                    game_state['community_cards'].append(river_card)

            game_state['stage'] = stage

            # Betting round
            betting_complete = False
            num_actions = 0

            while not betting_complete and num_actions < 10:  # Max 10 actions per round
                # Get action with exploration
                action, action_probs = self._get_action_with_exploration(game_state)

                # Store experience
                experience = {
                    'state': game_state.copy(),
                    'action': action,
                    'action_probs': action_probs,
                    'player': game_state['current_player']
                }
                trajectory.append(experience)

                # Update game state based on action
                game_state = self._apply_action(game_state, action)

                # Check if betting round is complete
                if action == CFRAction.FOLD.value:
                    betting_complete = True
                elif all(game_state['current_bets'][i] == max(game_state['current_bets'])
                        for i in range(2) if game_state['active_players'][i]):
                    betting_complete = True

                num_actions += 1

            # Check if hand is over
            active_count = sum(game_state['active_players'])
            if active_count <= 1:
                break

        # Calculate utilities based on actual money won/lost
        winner = self._determine_winner(game_state, p1_cards, p2_cards)
        final_stacks = game_state['stacks']

        # Calculate utility for each player (money won/lost)
        # Normalize by big blind to keep values in reasonable range
        BIG_BLIND = 20
        final_utilities = [
            (final_stacks[0] - starting_chips[0]) / BIG_BLIND,  # Convert to BB units
            (final_stacks[1] - starting_chips[1]) / BIG_BLIND
        ]


        # Calculate returns for each action (credit assignment from that point forward)
        # Later actions get full credit, earlier actions get discounted credit
        discount = 0.95  # Slight discount for temporal difference

        # Process trajectory in reverse to calculate returns
        # Track the last seen return for each player
        player_returns = [None, None]

        for i in range(len(trajectory) - 1, -1, -1):
            exp = trajectory[i]
            player = exp['player']

            if player_returns[player] is None:
                # First time seeing this player (from the end), use final utility
                exp['utility'] = final_utilities[player]
                player_returns[player] = final_utilities[player]
            else:
                # Use discounted return from this player's future actions
                exp['utility'] = player_returns[player] * discount
                player_returns[player] = exp['utility']

            exp['baseline'] = 0  # Will be replaced with actual value prediction

        return trajectory

    def _sample_game_trajectory_improved(self) -> List[Dict]:
        """Sample a game trajectory using two different networks for improved self-play"""
        trajectory = []

        # Initialize game state
        deck = Deck()

        # Deal hole cards
        p1_cards = [deck.draw(), deck.draw()]
        p2_cards = [deck.draw(), deck.draw()]

        # Track starting chips
        starting_chips = [1000, 1000]

        # Initial game state for player 1
        game_state = {
            'hole_cards': p1_cards,
            'community_cards': [],
            'pot': 3,
            'stacks': [999, 998],
            'current_player': 0,
            'button': 0,
            'current_bets': [1, 2],
            'active_players': [True, True],
            'stage': 0,
            'action_history': [],
            'legal_actions': [1, 0, 1, 1, 1, 1, 1, 1]
        }

        # Play through the hand
        for stage in range(4):
            if stage == 1:  # Flop
                game_state['community_cards'] = deck.draw(3)
            elif stage == 2:  # Turn
                turn_card = deck.draw()
                if turn_card:
                    game_state['community_cards'].append(turn_card)
            elif stage == 3:  # River
                river_card = deck.draw()
                if river_card:
                    game_state['community_cards'].append(river_card)

            game_state['stage'] = stage

            # Betting round
            betting_complete = False
            num_actions = 0

            while not betting_complete and num_actions < 10:
                current_player = game_state['current_player']

                # Create state from current player's perspective
                if current_player == 0:
                    player_state = game_state.copy()
                    player_state['hole_cards'] = p1_cards
                    # Use main network for player 1
                    action, action_probs = self._get_action_with_exploration_for_network(
                        player_state, self.network
                    )
                else:
                    player_state = game_state.copy()
                    player_state['hole_cards'] = p2_cards
                    # Swap stacks and bets for player 2's perspective
                    player_state['stacks'] = [game_state['stacks'][1], game_state['stacks'][0]]
                    player_state['current_bets'] = [game_state['current_bets'][1], game_state['current_bets'][0]]
                    # Use opponent network for player 2
                    action, action_probs = self._get_action_with_exploration_for_network(
                        player_state, self.opponent_network
                    )

                # Store experience
                experience = {
                    'state': player_state.copy(),
                    'action': action,
                    'action_probs': action_probs,
                    'player': current_player,
                    'network': 'main' if current_player == 0 else 'opponent'
                }
                trajectory.append(experience)

                # Update game state
                game_state = self._apply_action(game_state, action)

                # Check if betting round is complete
                if action == CFRAction.FOLD.value:
                    betting_complete = True
                elif all(game_state['current_bets'][i] == max(game_state['current_bets'])
                        for i in range(2) if game_state['active_players'][i]):
                    betting_complete = True

                num_actions += 1

            # Check if hand is over
            if sum(game_state['active_players']) <= 1:
                break

        # Calculate utilities
        winner = self._determine_winner(game_state, p1_cards, p2_cards)
        final_stacks = game_state['stacks']

        BIG_BLIND = 20
        final_utilities = [
            (final_stacks[0] - starting_chips[0]) / BIG_BLIND,
            (final_stacks[1] - starting_chips[1]) / BIG_BLIND
        ]

        # Calculate returns with discounting
        discount = 0.95
        player_returns = [None, None]

        for i in range(len(trajectory) - 1, -1, -1):
            exp = trajectory[i]
            player = exp['player']

            if player_returns[player] is None:
                exp['utility'] = final_utilities[player]
                player_returns[player] = final_utilities[player]
            else:
                exp['utility'] = player_returns[player] * discount
                player_returns[player] = exp['utility']

            exp['baseline'] = 0

        return trajectory

    def _get_action_with_exploration_for_network(self, game_state: Dict, network: nn.Module) -> Tuple[int, np.ndarray]:
        """Get action using a specific network"""
        with torch.no_grad():
            strategy, value = network(game_state)

            # Apply temperature to logits
            strategy = strategy / self.temperature

            # Convert to probabilities
            action_probs = F.softmax(strategy, dim=0).numpy()

            # Apply legal action mask
            legal_actions = np.array(game_state['legal_actions'])
            action_probs = action_probs * legal_actions

            # Renormalize
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                if legal_actions.sum() > 0:
                    action_probs = legal_actions / legal_actions.sum()
                else:
                    action_probs = np.zeros(8)
                    action_probs[0] = 1.0  # FOLD

            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                legal_indices = np.where(legal_actions > 0)[0]
                if len(legal_indices) > 0:
                    action = np.random.choice(legal_indices)
                else:
                    action = 0  # FOLD
            else:
                action = np.random.choice(len(action_probs), p=action_probs)

        return action, action_probs

    def _get_action_with_exploration(self, game_state: Dict) -> Tuple[int, np.ndarray]:
        """Get action using epsilon-greedy and temperature sampling"""
        with torch.no_grad():
            strategy, value = self.network(game_state)

            # Apply temperature to logits
            strategy = strategy / self.temperature

            # Convert to probabilities
            action_probs = F.softmax(strategy, dim=0).numpy()

            # Apply legal action mask
            legal_actions = np.array(game_state['legal_actions'])
            action_probs = action_probs * legal_actions

            # Renormalize
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                # If no legal actions (shouldn't happen), uniform over legal
                if legal_actions.sum() > 0:
                    action_probs = legal_actions / legal_actions.sum()
                else:
                    # Fallback: force fold action
                    print("WARNING: No legal actions available, forcing FOLD")
                    print(f"  Game state details:")
                    print(f"    Stage: {game_state.get('stage', 'unknown')}")
                    print(f"    Current player: {game_state.get('current_player', 'unknown')}")
                    print(f"    Active players: {game_state.get('active_players', 'unknown')}")
                    print(f"    Current bets: {game_state.get('current_bets', 'unknown')}")
                    print(f"    Pot: {game_state.get('pot', 'unknown')}")
                    print(f"    Stacks: {game_state.get('stacks', 'unknown')}")
                    print(f"    Legal actions array: {legal_actions}")
                    print(f"    Action history: {game_state.get('action_history', [])}")
                    action_probs = np.zeros(8)
                    action_probs[0] = 1.0  # FOLD

            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                # Random action from legal actions
                legal_indices = np.where(legal_actions > 0)[0]
                if len(legal_indices) > 0:
                    action = np.random.choice(legal_indices)
                else:
                    print("ERROR: No legal actions for epsilon-greedy!")
                    print(f"  Legal actions: {legal_actions}")
                    print(f"  Forcing FOLD action")
                    action = 0  # FOLD
            else:
                # Sample from distribution
                action = np.random.choice(len(action_probs), p=action_probs)

        return action, action_probs

    def _update_regrets(self, trajectory: List[Dict]):
        """Update regrets using CFR algorithm"""
        # Calculate advantages
        for exp in trajectory:
            state = exp['state']
            action = exp['action']
            utility = exp['utility']

            # Create state key for regret tracking
            state_key = self._get_state_key(state)

            # Update regret for this action
            self.regret_sum[state_key][action] += utility

            # CFR+ : Set negative regrets to 0
            self.regret_sum[state_key] = np.maximum(self.regret_sum[state_key], 0)

            # Update strategy sum for averaging
            if self.regret_sum[state_key].sum() > 0:
                strategy = self.regret_sum[state_key] / self.regret_sum[state_key].sum()
            else:
                strategy = np.ones(8) / 8  # Uniform

            self.strategy_sum[state_key] += strategy

    def _get_state_key(self, state: Dict) -> str:
        """Create a key for the state (information set)"""
        # Simplified state key based on cards and stage
        hole_cards = state.get('hole_cards', [])
        community_cards = state.get('community_cards', [])
        stage = state.get('stage', 0)

        # Convert cards to string representation
        hole_str = ''.join([f"{c.rank}{c.suit}" if c else "XX" for c in hole_cards])
        comm_str = ''.join([f"{c.rank}{c.suit}" if c else "XX" for c in community_cards])

        return f"{hole_str}|{comm_str}|{stage}"

    def _apply_action(self, game_state: Dict, action: int) -> Dict:
        """Apply an action to the game state"""
        new_state = game_state.copy()
        player = game_state['current_player']

        if action == CFRAction.FOLD.value:
            new_state['active_players'][player] = False
            # Opponent wins the pot
            opponent = 1 - player
            new_state['stacks'][opponent] += new_state['pot']
            new_state['pot'] = 0
        elif action == CFRAction.CHECK.value:
            pass  # No change
        elif action == CFRAction.CALL.value:
            call_amount = max(game_state['current_bets']) - game_state['current_bets'][player]
            call_amount = min(call_amount, new_state['stacks'][player])  # Can't bet more than stack
            new_state['current_bets'][player] += call_amount
            new_state['stacks'][player] -= call_amount
            new_state['pot'] += call_amount
        else:  # Betting actions
            bet_size = self._get_bet_size(action, game_state)
            bet_amount = min(bet_size, new_state['stacks'][player])  # Can't bet more than stack
            raise_amount = bet_amount - game_state['current_bets'][player]
            new_state['current_bets'][player] = bet_amount
            new_state['stacks'][player] -= raise_amount
            new_state['pot'] += raise_amount

        # Update action history
        new_state['action_history'].append(action)

        # Switch player
        new_state['current_player'] = 1 - player

        # Update legal actions for next player
        new_state['legal_actions'] = self._get_legal_actions(new_state)

        return new_state

    def _get_bet_size(self, action: int, game_state: Dict) -> int:
        """Get bet size based on action"""
        pot = game_state['pot']

        if action == CFRAction.BET_SMALL.value:
            return int(pot * 0.33)
        elif action == CFRAction.BET_MEDIUM.value:
            return int(pot * 0.5)
        elif action == CFRAction.BET_LARGE.value:
            return int(pot * 0.75)
        elif action == CFRAction.BET_POT.value:
            return pot
        elif action == CFRAction.ALL_IN.value:
            return game_state['stacks'][game_state['current_player']]

        return 0

    def _get_legal_actions(self, game_state: Dict) -> List[int]:
        """Get legal actions for current game state"""
        legal = [0] * 8
        player = game_state['current_player']
        stack = game_state['stacks'][player]
        current_bet = game_state['current_bets'][player]
        max_bet = max(game_state['current_bets'])

        # Fold is ALWAYS legal, even with no chips
        legal[CFRAction.FOLD.value] = 1

        # If player has no chips, they can only fold or check (if no bet to match)
        if stack <= 0:
            if current_bet == max_bet:
                legal[CFRAction.CHECK.value] = 1  # Can check if no bet to call
            return legal

        # Check if no bet to call
        if current_bet == max_bet:
            legal[CFRAction.CHECK.value] = 1

        # Call if there's a bet to call and we have chips
        if current_bet < max_bet and stack > 0:
            legal[CFRAction.CALL.value] = 1

        # Betting actions if we have enough chips
        if stack > max_bet - current_bet:
            legal[CFRAction.BET_SMALL.value] = 1
            legal[CFRAction.BET_MEDIUM.value] = 1
            legal[CFRAction.BET_LARGE.value] = 1
            legal[CFRAction.BET_POT.value] = 1

        # All-in is legal if we have chips
        if stack > 0:
            legal[CFRAction.ALL_IN.value] = 1

        return legal

    def _determine_winner(self, game_state: Dict, p1_cards: List[Card],
                         p2_cards: List[Card]) -> int:
        """Determine the winner of a hand"""
        if not game_state['active_players'][0]:
            return 1
        if not game_state['active_players'][1]:
            return 0

        # Evaluate hands
        community = game_state.get('community_cards', [])

        # Filter out None values and ensure we have valid cards
        community = [c for c in community if c is not None]
        p1_cards = [c for c in p1_cards if c is not None]
        p2_cards = [c for c in p2_cards if c is not None]

        # Need at least 2 hole cards to evaluate
        if len(p1_cards) < 2 or len(p2_cards) < 2:
            return 0  # Default winner if cards are invalid

        # Evaluate hands with available cards
        p1_hand = evaluate_hand(p1_cards + community)
        p2_hand = evaluate_hand(p2_cards + community)

        winner = 0 if p1_hand > p2_hand else 1

        # Distribute pot to winner
        if game_state['pot'] > 0:
            game_state['stacks'][winner] += game_state['pot']
            game_state['pot'] = 0

        return winner

    def _train_on_batch(self) -> float:
        """Train network on a batch using policy gradient"""
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        total_loss = 0

        # Utilities are already normalized to BB units (typically -50 to +50)
        # No need for further normalization

        for experience in batch:
            game_state = experience['state']
            action = experience['action']
            utility = experience['utility']  # Already in BB units

            # Forward pass
            strategy, value = self.network(game_state)

            # Calculate advantage using value network's prediction (detached to prevent manipulation)
            with torch.no_grad():
                # Use the current network's value prediction as baseline
                baseline = value.item()
                # Both utility and baseline are in BB units
                advantage = utility - baseline
                # Clip advantage for stability (in BB units)
                advantage = np.clip(advantage, -10, 10)  # ±10 BB is a reasonable range

            # Policy gradient loss (REINFORCE with baseline)
            log_probs = F.log_softmax(strategy, dim=0)
            # Clamp log probabilities to prevent extreme values when prob is near 0
            log_probs = torch.clamp(log_probs, min=-10)  # e^-10 ≈ 0.000045
            policy_loss = -log_probs[action] * advantage

            # Value loss - train to predict actual returns accurately
            # This is pure supervised learning - no advantage manipulation
            # Utility is already normalized to BB units (-50 to +50 typical range)
            value_target = torch.FloatTensor([utility])
            value_loss = F.smooth_l1_loss(value, value_target)  # Huber loss for robustness
            
            # DEPRECATED: Entropy regularization for exploration. Just produces way too aggressive play (50%+ raise).
            #probs = F.softmax(strategy, dim=0)
            #entropy = -(probs * log_probs).sum()
            #entropy_bonus = -0.01 * entropy  # Negative because we want to maximize entropy

            # Combined loss with balanced weights
            # Scale down to prevent large losses
            loss = 0.1 * policy_loss + 0.5 * value_loss# + entropy_bonus


            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.batch_size

    def get_action(self, **kwargs) -> Action:
        """Get action for gameplay (uses average strategy)"""
        # Get valid actions from kwargs (if provided)
        valid_actions = kwargs.get('valid_actions', None)

        # Get betting information
        to_call = kwargs.get('to_call', 0)
        pot_size = kwargs.get('pot_size', 0)
        stack_size = kwargs.get('stack_size', 1000)
        opponent_stack = kwargs.get('opponent_stack', 1000)  # Get actual opponent stack
        position = kwargs.get('position', 0)
        num_players = kwargs.get('num_players', 2)

        # Calculate current bets based on pot and to_call
        # If there's an amount to call, someone has bet
        my_current_bet = 0  # Assume we haven't bet yet this round
        opponent_current_bet = to_call  # Opponent's bet is what we need to call

        # Convert valid_actions to legal actions mask for the 8 CFR actions
        legal_actions_mask = [0] * 8
        if valid_actions:
            for action in valid_actions:
                if action == Action.FOLD:
                    legal_actions_mask[CFRAction.FOLD.value] = 1
                elif action == Action.CHECK:
                    legal_actions_mask[CFRAction.CHECK.value] = 1
                elif action == Action.CALL:
                    legal_actions_mask[CFRAction.CALL.value] = 1
                elif action == Action.RAISE:
                    # Multiple bet sizes are legal if RAISE is allowed
                    legal_actions_mask[CFRAction.BET_SMALL.value] = 1
                    legal_actions_mask[CFRAction.BET_MEDIUM.value] = 1
                    legal_actions_mask[CFRAction.BET_LARGE.value] = 1
                    legal_actions_mask[CFRAction.BET_POT.value] = 1
                elif action == Action.ALL_IN:
                    legal_actions_mask[CFRAction.ALL_IN.value] = 1
        else:
            # If no valid actions specified, assume all are legal (backward compatibility)
            legal_actions_mask = [1] * 8

        # Create game state from kwargs
        game_state = {
            'hole_cards': kwargs.get('hole_cards', []),
            'community_cards': kwargs.get('community_cards', []),
            'pot': pot_size,
            'stacks': [stack_size, opponent_stack],  # Use actual opponent stack
            'current_player': 0,
            'button': position % num_players,  # Button position
            'current_bets': [my_current_bet, opponent_current_bet],
            'active_players': [True, True],
            'stage': len(kwargs.get('community_cards', [])) // 2,
            'action_history': kwargs.get('action_history', []),
            'legal_actions': legal_actions_mask,  # Use actual legal actions
            'to_call': to_call,  # Add this for feature extraction
            'position': position,  # Add raw position
            'num_players': num_players  # Add number of players
        }

        # Get state key
        state_key = self._get_state_key(game_state)

        # Create a mapping of valid action indices
        # Note: RAISE maps to multiple bet sizes internally
        action_to_idx = {
            Action.FOLD: CFRAction.FOLD.value,
            Action.CHECK: CFRAction.CHECK.value,
            Action.CALL: CFRAction.CALL.value,
            Action.RAISE: [CFRAction.BET_SMALL.value, CFRAction.BET_MEDIUM.value,
                          CFRAction.BET_LARGE.value, CFRAction.BET_POT.value],
            Action.ALL_IN: CFRAction.ALL_IN.value
        }

        # If valid actions are provided, filter the strategy
        if valid_actions:
            valid_indices = []
            for act in valid_actions:
                if act in action_to_idx:
                    idx = action_to_idx[act]
                    if isinstance(idx, list):
                        valid_indices.extend(idx)  # Add all bet sizes for RAISE
                    else:
                        valid_indices.append(idx)
        else:
            valid_indices = list(range(8))  # All actions if not specified

        # Use average strategy if available
        if state_key in self.strategy_sum and self.strategy_sum[state_key].sum() > 0:
            strategy = self.strategy_sum[state_key] / self.strategy_sum[state_key].sum()

            # Filter strategy to only valid actions
            if valid_actions:
                masked_strategy = np.zeros(len(strategy))
                for idx in valid_indices:
                    if 0 <= idx < len(strategy):
                        masked_strategy[idx] = strategy[idx]
                # Renormalize
                if masked_strategy.sum() > 0:
                    masked_strategy = masked_strategy / masked_strategy.sum()
                else:
                    # Fallback: uniform over valid actions
                    for idx in valid_indices:
                        if 0 <= idx < len(masked_strategy):
                            masked_strategy[idx] = 1.0 / len(valid_indices)
                action_idx = np.random.choice(len(masked_strategy), p=masked_strategy)
            else:
                action_idx = np.random.choice(len(strategy), p=strategy)
        else:
            # Fall back to network prediction
            with torch.no_grad():
                strategy, _ = self.network(game_state)
                action_probs = F.softmax(strategy, dim=0)

                # Filter to valid actions
                if valid_actions and valid_indices:
                    masked_probs = torch.zeros_like(action_probs)
                    for idx in valid_indices:
                        if 0 <= idx < len(action_probs):
                            masked_probs[idx] = action_probs[idx]
                    # Renormalize
                    if masked_probs.sum() > 0:
                        masked_probs = masked_probs / masked_probs.sum()
                    else:
                        # Fallback: uniform over valid actions
                        for idx in valid_indices:
                            if 0 <= idx < len(masked_probs):
                                masked_probs[idx] = 1.0 / len(valid_indices)
                    action_idx = torch.multinomial(masked_probs, 1).item()
                else:
                    action_idx = torch.multinomial(action_probs, 1).item()

        # Convert to game action, ensuring it's valid
        if action_idx == CFRAction.FOLD.value:
            selected_action = Action.FOLD
        elif action_idx == CFRAction.CHECK.value:
            selected_action = Action.CHECK
        elif action_idx == CFRAction.CALL.value:
            selected_action = Action.CALL
        else:
            selected_action = Action.RAISE

        # Final validation: ensure selected action is in valid_actions
        if valid_actions and selected_action not in valid_actions:
            # Default to first valid action
            return valid_actions[0] if valid_actions else Action.FOLD

        return selected_action

    def save(self, filename: str):
        """Save the model"""
        checkpoint = {
            'type': 'raw_neural_cfr',  # Add type flag for model identification
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'use_improved_selfplay': self.use_improved_selfplay,
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum),
            'training_stats': self.training_stats,
            'iterations': self.iterations,
            'iteration_count': self.iteration_count,
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'initial_epsilon': self.initial_epsilon,
            'final_epsilon': self.final_epsilon,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'hidden_dim': self.hidden_dim,  # Save the hidden dimension
            # Save last 1000 experiences for continuity
            'recent_experiences': list(self.experience_buffer)[-1000:] if self.experience_buffer else []
        }

        if self.use_improved_selfplay:
            # Save both networks for improved self-play
            checkpoint['opponent_network_state'] = self.opponent_network.state_dict()
            checkpoint['opponent_optimizer_state'] = self.opponent_optimizer.state_dict()
        else:
            # Save target network for normal self-play
            checkpoint['target_network_state'] = self.target_network.state_dict()

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, filename: str):
        """Load a saved model"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        # Verify model type if present
        if 'type' in checkpoint and checkpoint['type'] != 'raw_neural_cfr':
            print(f"Warning: Loading model with type '{checkpoint['type']}' as Raw Neural CFR")

        # Check if model was trained with improved self-play
        self.use_improved_selfplay = checkpoint.get('use_improved_selfplay', False)

        # Check if the model has a different hidden dimension
        if 'hidden_dim' in checkpoint and checkpoint['hidden_dim'] != self.hidden_dim:
            print(f"Note: Model was trained with hidden_dim={checkpoint['hidden_dim']}, but current is {self.hidden_dim}")
            # Recreate networks with the correct hidden dimension
            self.hidden_dim = checkpoint['hidden_dim']
            self.network = RawNeuralCFRNetwork(hidden_dim=self.hidden_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

            if self.use_improved_selfplay:
                self.opponent_network = RawNeuralCFRNetwork(hidden_dim=self.hidden_dim)
                self.opponent_optimizer = optim.Adam(self.opponent_network.parameters(), lr=self.learning_rate)
            else:
                self.target_network = RawNeuralCFRNetwork(hidden_dim=self.hidden_dim)

        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.use_improved_selfplay:
            # Load opponent network for improved self-play
            if 'opponent_network_state' in checkpoint:
                self.opponent_network.load_state_dict(checkpoint['opponent_network_state'])
                if 'opponent_optimizer_state' in checkpoint:
                    self.opponent_optimizer.load_state_dict(checkpoint['opponent_optimizer_state'])
            else:
                print("Warning: No opponent network found in checkpoint, creating a copy of main network")
                self.opponent_network.load_state_dict(self.network.state_dict())
        else:
            # Load target network for normal self-play
            if 'target_network_state' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network_state'])
            else:
                # If no target network saved, copy from main network
                self.target_network.load_state_dict(self.network.state_dict())

        self.regret_sum = defaultdict(lambda: np.zeros(8), checkpoint['regret_sum'])
        self.strategy_sum = defaultdict(lambda: np.zeros(8), checkpoint['strategy_sum'])
        self.training_stats = checkpoint['training_stats']
        self.iterations = checkpoint['iterations']
        self.iteration_count = checkpoint.get('iteration_count', 0)

        # Load exploration parameters if available
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.temperature = checkpoint.get('temperature', self.temperature)
        self.initial_epsilon = checkpoint.get('initial_epsilon', self.initial_epsilon)
        self.final_epsilon = checkpoint.get('final_epsilon', self.final_epsilon)
        self.initial_temperature = checkpoint.get('initial_temperature', self.initial_temperature)
        self.final_temperature = checkpoint.get('final_temperature', self.final_temperature)

        # Load recent experiences if available
        if 'recent_experiences' in checkpoint and checkpoint['recent_experiences']:
            for exp in checkpoint['recent_experiences']:
                self.experience_buffer.append(exp)

    def resume_training(self, additional_iterations: int, verbose: bool = True):
        """Resume training from a loaded model

        Args:
            additional_iterations: Number of additional iterations to train
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Resuming Raw Neural CFR Training")
            print(f"Previous iterations: {self.iteration_count}")
            print(f"Additional iterations: {additional_iterations}")
            print(f"Total iterations: {self.iteration_count + additional_iterations}")
            print(f"Current epsilon: {self.epsilon:.3f}")
            print(f"Current temperature: {self.temperature:.3f}")
            print("=" * 60)

        # Update total iterations for proper exploration decay
        old_total = self.iterations
        self.iterations = self.iteration_count + additional_iterations
        start_iteration = self.iteration_count
        start_time = time.time()

        for iteration in range(start_iteration, self.iterations):
            self.iteration_count = iteration

            # Update exploration parameters based on total progress
            progress = iteration / self.iterations
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * progress

            # Exponential decay for temperature: T(t) = T_final + (T_init - T_final) * exp(-decay_rate * progress)
            # We want temperature to decay from initial to final over the training duration
            # Using decay_rate = 5 means we reach ~0.7% of the initial range at progress=1
            decay_rate = 5.0
            temp_range = self.initial_temperature - self.final_temperature
            self.temperature = self.final_temperature + temp_range * math.exp(-decay_rate * progress)

            # Sample game trajectory with exploration
            trajectory = self._sample_game_trajectory()

            # Calculate advantages and update regrets
            self._update_regrets(trajectory)

            # Store experiences
            for experience in trajectory:
                self.experience_buffer.append(experience)

            # Train on batch when buffer is large enough
            if len(self.experience_buffer) >= self.batch_size * 2:
                loss = self._train_on_batch()
                self.training_stats['loss'].append(loss)

            # Update target network periodically
            if iteration % self.update_target_every == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            # Progress update
            if verbose and (iteration - start_iteration) % max(1, additional_iterations // 20) == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(self.training_stats['loss'][-100:]) if self.training_stats['loss'] else 0
                print(f"Iteration {iteration}/{self.iterations} | "
                      f"Time: {elapsed:.1f}s | Loss: {avg_loss:.4f} | "
                      f"eps: {self.epsilon:.3f} | T: {self.temperature:.2f}")

        if verbose:
            total_time = time.time() - start_time
            print(f"\nResumed training complete in {total_time:.1f} seconds")
            print(f"Total iterations completed: {self.iteration_count}")