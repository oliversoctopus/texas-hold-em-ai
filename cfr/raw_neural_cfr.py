"""
Raw Neural CFR for 2-Player Texas Hold'em
Learns directly from raw game state without manual feature engineering
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

        # Stack sizes
        stacks = game_state.get('stacks', [1000, 1000])
        for stack in stacks[:2]:
            features.append(stack / 1000.0)

        # Current player
        current_player = game_state.get('current_player', 0)
        features.append(float(current_player))

        # Button position
        button = game_state.get('button', 0)
        features.append(float(button))

        # Player positions relative to button
        num_players = 2
        for i in range(num_players):
            position = (i - button) % num_players
            features.append(position / float(num_players - 1))

        # Current bets
        bets = game_state.get('current_bets', [0, 0])
        for bet in bets[:2]:
            features.append(bet / 1000.0)

        # Active status
        active = game_state.get('active_players', [True, True])
        for is_active in active[:2]:
            features.append(float(is_active))

        # Game stage (one-hot)
        stage = game_state.get('stage', 0)  # 0=preflop, 1=flop, 2=turn, 3=river
        stage_onehot = [0.0] * 4
        if 0 <= stage < 4:
            stage_onehot[stage] = 1.0
        features.extend(stage_onehot)

        # Legal actions mask
        legal_actions = game_state.get('legal_actions', [1] * 8)
        for is_legal in legal_actions[:8]:
            features.append(float(is_legal))

        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)

        return torch.FloatTensor(features[:20])


class RawNeuralCFR:
    """Raw Neural CFR trainer and player"""

    def __init__(self, iterations: int = 50000, learning_rate: float = 0.001,
                 batch_size: int = 32, buffer_size: int = 100000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Neural network
        self.network = RawNeuralCFRNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=iterations, eta_min=learning_rate * 0.01
        )

        # Target network for stability
        self.target_network = RawNeuralCFRNetwork()
        self.target_network.load_state_dict(self.network.state_dict())
        self.update_target_every = 1000

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=buffer_size)

        # CFR tracking
        self.regret_sum = defaultdict(lambda: np.zeros(8))
        self.strategy_sum = defaultdict(lambda: np.zeros(8))

        # Training statistics
        self.training_stats = {
            'iterations': [],
            'avg_regret': [],
            'avg_utility': [],
            'loss': []
        }

    def train(self, verbose: bool = True):
        """Train the Raw Neural CFR"""
        if verbose:
            print("Starting Raw Neural CFR Training")
            print(f"Iterations: {self.iterations}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Batch size: {self.batch_size}")
            print("=" * 60)

        start_time = time.time()

        for iteration in range(self.iterations):
            # Sample game trajectory
            trajectory = self._sample_game_trajectory()

            # Store experiences
            for experience in trajectory:
                self.experience_buffer.append(experience)

            # Train on batch when buffer is large enough
            if len(self.experience_buffer) >= self.batch_size:
                loss = self._train_on_batch()
                self.training_stats['loss'].append(loss)

                # Update learning rate after optimizer.step() (called in _train_on_batch)
                self.scheduler.step()

            # Update target network periodically
            if iteration % self.update_target_every == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            # Progress update
            if verbose and iteration % (self.iterations // 20) == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(self.training_stats['loss'][-100:]) if self.training_stats['loss'] else 0
                print(f"Iteration {iteration}/{self.iterations} | "
                      f"Time: {elapsed:.1f}s | Loss: {avg_loss:.4f}")

        if verbose:
            print(f"\nTraining complete in {time.time() - start_time:.1f} seconds")

    def _sample_game_trajectory(self) -> List[Dict]:
        """Sample a complete game trajectory"""
        trajectory = []

        # Initialize game state
        deck = Deck()  # Deck is automatically shuffled on creation

        # Deal hole cards
        p1_cards = [deck.draw(), deck.draw()]
        p2_cards = [deck.draw(), deck.draw()]

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
                # Get action from network
                with torch.no_grad():
                    strategy, value = self.network(game_state)
                    action_probs = F.softmax(strategy, dim=0)

                # Sample action
                valid_actions = game_state['legal_actions']
                masked_probs = action_probs * torch.FloatTensor(valid_actions)
                masked_probs = masked_probs / masked_probs.sum()

                action = torch.multinomial(masked_probs, 1).item()

                # Store experience
                experience = {
                    'state': game_state.copy(),
                    'action': action,
                    'value': value.item(),
                    'action_probs': action_probs.numpy()
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

        # Calculate utilities
        winner = self._determine_winner(game_state, p1_cards, p2_cards)
        pot = game_state['pot']

        for i, exp in enumerate(trajectory):
            if exp['state']['current_player'] == winner:
                exp['utility'] = pot / 2  # Simplified utility
            else:
                exp['utility'] = -pot / 2

        return trajectory

    def _apply_action(self, game_state: Dict, action: int) -> Dict:
        """Apply an action to the game state"""
        new_state = game_state.copy()
        player = game_state['current_player']

        if action == CFRAction.FOLD.value:
            new_state['active_players'][player] = False
        elif action == CFRAction.CHECK.value:
            pass  # No change
        elif action == CFRAction.CALL.value:
            call_amount = max(game_state['current_bets']) - game_state['current_bets'][player]
            new_state['current_bets'][player] += call_amount
            new_state['stacks'][player] -= call_amount
            new_state['pot'] += call_amount
        else:  # Betting actions
            bet_size = self._get_bet_size(action, game_state)
            new_state['current_bets'][player] = bet_size
            new_state['stacks'][player] -= bet_size - game_state['current_bets'][player]
            new_state['pot'] += bet_size - game_state['current_bets'][player]

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

        if stack <= 0:
            return legal  # No actions if no chips

        # Fold is always legal
        legal[CFRAction.FOLD.value] = 1

        # Check if no bet to call
        if current_bet == max_bet:
            legal[CFRAction.CHECK.value] = 1

        # Call if there's a bet to call
        if current_bet < max_bet and stack > 0:
            legal[CFRAction.CALL.value] = 1

        # Betting actions if we have chips
        if stack > max_bet - current_bet:
            legal[CFRAction.BET_SMALL.value] = 1
            legal[CFRAction.BET_MEDIUM.value] = 1
            legal[CFRAction.BET_LARGE.value] = 1
            legal[CFRAction.BET_POT.value] = 1

        # All-in is always legal if we have chips
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

        return 0 if p1_hand > p2_hand else 1

    def _train_on_batch(self) -> float:
        """Train network on a batch of experiences"""
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        total_loss = 0

        for experience in batch:
            game_state = experience['state']
            action = experience['action']
            utility = experience['utility']

            # Forward pass
            strategy, value = self.network(game_state)

            # Calculate losses
            strategy_loss = F.cross_entropy(strategy.unsqueeze(0),
                                           torch.LongTensor([action]))
            value_loss = F.mse_loss(value, torch.FloatTensor([utility]))

            # Combined loss
            loss = strategy_loss + 0.5 * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.batch_size

    def get_action(self, **kwargs) -> Action:
        """Get action for gameplay"""
        # Create game state from kwargs
        game_state = {
            'hole_cards': kwargs.get('hole_cards', []),
            'community_cards': kwargs.get('community_cards', []),
            'pot': kwargs.get('pot_size', 0),
            'stacks': [kwargs.get('stack_size', 1000), 1000],  # Simplified
            'current_player': 0,
            'button': kwargs.get('position', 0),
            'current_bets': [0, 0],
            'active_players': [True, True],
            'stage': len(kwargs.get('community_cards', [])) // 2,
            'action_history': [],
            'legal_actions': [1] * 8
        }

        # Get strategy from network
        with torch.no_grad():
            strategy, _ = self.network(game_state)
            action_probs = F.softmax(strategy, dim=0)

        # Sample action
        action_idx = torch.multinomial(action_probs, 1).item()

        # Convert to game action
        if action_idx == CFRAction.FOLD.value:
            return Action.FOLD
        elif action_idx == CFRAction.CHECK.value:
            return Action.CHECK
        elif action_idx == CFRAction.CALL.value:
            return Action.CALL
        else:
            return Action.RAISE

    def save(self, filename: str):
        """Save the model"""
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'iterations': self.iterations
        }
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, filename: str):
        """Load a saved model"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_stats = checkpoint['training_stats']
        self.iterations = checkpoint['iterations']