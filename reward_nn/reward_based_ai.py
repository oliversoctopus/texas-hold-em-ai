"""
Reward-Based Neural Network AI for Texas Hold'em
Uses per-hand rewards measured in big blinds to learn optimal play
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
from core.game_constants import Action
from core.card_deck import Card

Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'action_probs'])


class CardEmbedding(nn.Module):
    """Learnable card embeddings"""

    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.rank_embedding = nn.Embedding(14, embedding_dim // 2)  # 2-A
        self.suit_embedding = nn.Embedding(5, embedding_dim // 2)   # 4 suits + none

    def forward(self, cards: List[Optional[Card]]) -> torch.Tensor:
        """Embed cards into feature vectors"""
        rank_indices = []
        suit_indices = []

        for card in cards:
            if card is None:
                rank_indices.append(0)
                suit_indices.append(0)
            else:
                # Map card values to indices
                rank_idx = card.value - 1  # 2=1, 3=2, ..., A=13
                suit_map = {'♠': 1, '♥': 2, '♦': 3, '♣': 4}
                suit_idx = suit_map.get(card.suit, 0)
                rank_indices.append(rank_idx)
                suit_indices.append(suit_idx)

        rank_tensor = torch.LongTensor(rank_indices)
        suit_tensor = torch.LongTensor(suit_indices)

        rank_emb = self.rank_embedding(rank_tensor)
        suit_emb = self.suit_embedding(suit_tensor)

        return torch.cat([rank_emb, suit_emb], dim=-1)


class PositionEncoding(nn.Module):
    """Encode position at table"""

    def __init__(self, max_players: int = 6, encoding_dim: int = 16):
        super().__init__()
        self.position_embedding = nn.Embedding(max_players, encoding_dim)

    def forward(self, position: int, num_players: int) -> torch.Tensor:
        """Encode position relative to button"""
        # Normalize position relative to number of players
        relative_pos = position % num_players
        pos_tensor = torch.LongTensor([relative_pos])
        return self.position_embedding(pos_tensor).squeeze(0)


class ActionHistoryEncoder(nn.Module):
    """Encode action history using attention mechanism"""

    def __init__(self, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim + 1, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, action_history: List[int], max_len: int = 20) -> torch.Tensor:
        """Encode sequence of actions"""
        if len(action_history) > max_len:
            action_history = action_history[-max_len:]
        elif len(action_history) < max_len:
            action_history = action_history + [0] * (max_len - len(action_history))

        action_tensor = torch.LongTensor(action_history).unsqueeze(0)
        action_emb = self.action_embedding(action_tensor)

        # Self-attention
        attn_out, _ = self.attention(action_emb, action_emb, action_emb)
        output = self.norm(attn_out + action_emb)

        # Global average pooling
        return output.mean(dim=1).squeeze(0)


class RewardBasedNN(nn.Module):
    """Actor-Critic neural network for reward-based learning"""

    def __init__(self, hidden_dim: int = 256, num_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim  # Store for architecture detection

        # Feature encoders
        self.card_embedding = CardEmbedding(embedding_dim=16)
        self.position_encoding = PositionEncoding(max_players=6, encoding_dim=16)
        self.action_encoder = ActionHistoryEncoder(action_dim=5, hidden_dim=64)

        # Feature dimensions
        card_features = 16 * 7  # 2 hole + 5 community = 112
        position_features = 16
        action_features = 64
        game_features = 10  # pot odds, stack sizes, etc.
        opponent_features = 20  # Space for opponent stack/bet info (max 5 opponents, 4 features each)
        total_features = card_features + position_features + action_features + game_features + opponent_features  # 112 + 16 + 64 + 10 + 20 = 222

        # Shared layers
        self.input_projection = nn.Linear(total_features, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Raise sizing head (3 options: half pot, 2/3 pot, full pot)
        self.raise_sizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # 3 raise size options
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits, value estimate, and raise size logits"""
        # Extract features
        features = self._extract_features(state)

        # Shared representation
        x = self.input_projection(features)
        x = F.relu(x)
        x = self.dropout(x)

        # Actor, critic, and raise sizing outputs
        policy_logits = self.actor(x)
        value = self.critic(x)
        raise_size_logits = self.raise_sizer(x)

        return policy_logits, value, raise_size_logits

    def _extract_features(self, state: Dict) -> torch.Tensor:
        """Extract features from game state"""
        # Card features
        hole_cards = state.get('hole_cards', [])
        community_cards = state.get('community_cards', [])

        # Ensure we have exactly 2 hole cards (pad with None if needed)
        # Make a copy to avoid modifying original
        hole_cards = list(hole_cards) if hole_cards else []
        while len(hole_cards) < 2:
            hole_cards.append(None)
        hole_cards = hole_cards[:2]

        # Ensure we have exactly 5 community cards (pad with None if needed)
        # Make a copy to avoid modifying original
        community_cards = list(community_cards) if community_cards else []
        while len(community_cards) < 5:
            community_cards.append(None)
        community_cards = community_cards[:5]

        all_cards = hole_cards + community_cards  # Exactly 7 cards
        card_features = self.card_embedding(all_cards).flatten()  # Should be 16 * 7 = 112

        # Position features
        position = state.get('position', 0)
        num_players = max(state.get('num_players', 2), 2)  # Ensure at least 2 players
        position_features = self.position_encoding(position, num_players)  # Should be 16

        # Action history features
        action_history = state.get('action_history', [])
        action_features = self.action_encoder(action_history)  # Should be 64

        # Game state features (exactly 10 features)
        pot_size = state.get('pot_size', 0)
        stack_size = state.get('stack_size', 1000)
        to_call = state.get('to_call', 0)
        big_blind = state.get('big_blind', 20)

        # Normalize numeric features
        pot_odds = to_call / (pot_size + to_call + 1e-8)
        stack_to_pot = stack_size / (pot_size + 1e-8)
        stack_to_bb = stack_size / (big_blind + 1e-8)
        pot_to_bb = pot_size / (big_blind + 1e-8)
        call_to_stack = to_call / (stack_size + 1e-8)

        # Players in hand
        players_in_hand = state.get('players_in_hand', 2)
        players_folded = num_players - players_in_hand

        game_features = torch.FloatTensor([
            pot_odds,
            stack_to_pot,
            stack_to_bb / 100,  # Normalize to reasonable range
            pot_to_bb / 10,
            call_to_stack,
            players_in_hand / max(num_players, 1),  # Normalize by actual number of players
            players_folded / max(num_players, 1),  # Normalize by actual number of players
            position / max(num_players - 1, 1),  # Avoid division by zero
            state.get('hand_phase', 0) / 3,  # 0=preflop, 1=flop, 2=turn, 3=river
            state.get('is_preflop_aggressor', 0)
        ])  # Exactly 10 features

        # Opponent features (max 5 opponents, 4 features each = 20 total)
        opponent_features_list = []
        opponent_info = state.get('opponent_info', [])

        for i in range(5):  # Max 5 opponents
            if i < len(opponent_info):
                opp = opponent_info[i]
                opp_chips = opp.get('chips', 0)
                opp_bet = opp.get('current_bet', 0)
                opp_all_in = float(opp.get('all_in', False))

                # Normalize opponent features
                opp_stack_bb = opp_chips / (big_blind + 1e-8)
                opp_bet_to_pot = opp_bet / (pot_size + 1e-8)
                opp_stack_to_our_stack = opp_chips / (stack_size + 1e-8)

                opponent_features_list.extend([
                    opp_stack_bb / 100,  # Normalized stack in BBs
                    opp_bet_to_pot,  # Their bet relative to pot
                    opp_stack_to_our_stack,  # Stack ratio vs us
                    opp_all_in  # Whether they're all-in
                ])
            else:
                # Pad with zeros if fewer opponents
                opponent_features_list.extend([0.0, 0.0, 0.0, 0.0])

        opponent_features = torch.FloatTensor(opponent_features_list)  # Exactly 20 features

        # Combine all features (112 + 16 + 64 + 10 + 20 = 222)
        return torch.cat([card_features, position_features, action_features, game_features, opponent_features])


class RewardBasedAI:
    """Main AI class using reward-based learning"""

    def __init__(self, learning_rate: float = 1e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.1, value_coef: float = 0.5,
                 entropy_coef: float = 0.1, hidden_dim: int = 256):
        """Initialize the reward-based AI"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.network = RewardBasedNN(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # PPO parameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Experience buffer
        self.memory = deque(maxlen=10000)

        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.loss_history = []

        # For compatibility with game engine
        self.epsilon = 0  # No epsilon-greedy, we use stochastic policy

        # Store last raise size decision for training
        self.last_raise_size_probs = None
        self.last_raise_size_idx = None

    def get_state_features(self, hand, community_cards, pot, current_bet, player_chips,
                          player_bet, num_players, players_in_hand, position,
                          action_history=None, opponent_info=None, hand_phase=0):
        """Extract features from game state (compatibility method)"""
        return {
            'hole_cards': hand if hand else [],
            'community_cards': community_cards if community_cards else [],
            'pot_size': pot,
            'to_call': max(0, current_bet - player_bet),
            'stack_size': player_chips,
            'position': position,
            'num_players': num_players,
            'players_in_hand': players_in_hand,
            'action_history': [a.value if hasattr(a, 'value') else a for a in (action_history or [])],
            'hand_phase': hand_phase,
            'big_blind': 20,  # Default big blind
            'is_preflop_aggressor': 0,
            'opponent_info': opponent_info if opponent_info else []
        }

    def choose_action(self, state, valid_actions, training=False, **kwargs):
        """Choose action using the policy network"""
        if not isinstance(state, dict):
            # Convert state to dict if needed
            state = {'state_tensor': state}

        with torch.no_grad():
            policy_logits, value, raise_size_logits = self.network(state)

            # Mask invalid actions
            action_mask = torch.zeros(5, dtype=torch.bool)
            for action in valid_actions:
                if action.value < 5:  # Only consider first 5 actions
                    action_mask[action.value] = True

            # Apply mask to logits
            masked_logits = policy_logits.clone()
            masked_logits[~action_mask] = -float('inf')

            # Get action probabilities
            action_probs = F.softmax(masked_logits, dim=-1)

            if training:
                # Sample action during training
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample().item()

                # Store for training
                self.last_action_probs = action_probs.cpu().numpy()

                # If raising, also decide raise size
                if action_idx == Action.RAISE.value:
                    raise_size_probs = F.softmax(raise_size_logits, dim=-1)
                    raise_size_dist = torch.distributions.Categorical(raise_size_probs)
                    self.last_raise_size_idx = raise_size_dist.sample().item()
                    self.last_raise_size_probs = raise_size_probs.cpu().numpy()
            else:
                # During evaluation, use stochastic policy with temperature
                # to balance exploration and exploitation
                temperature = 0.75  # Slightly reduce randomness
                scaled_logits = masked_logits / temperature
                scaled_probs = F.softmax(scaled_logits, dim=-1)
                action_dist = torch.distributions.Categorical(scaled_probs)
                action_idx = action_dist.sample().item()

                # If raising, also decide raise size
                if action_idx == Action.RAISE.value:
                    scaled_raise_logits = raise_size_logits / temperature
                    raise_size_probs = F.softmax(scaled_raise_logits, dim=-1)
                    raise_size_dist = torch.distributions.Categorical(raise_size_probs)
                    self.last_raise_size_idx = raise_size_dist.sample().item()

        # Map to Action enum
        return Action(action_idx)

    def remember(self, state, action, reward, next_state, done, **kwargs):
        """Store experience for training"""
        # Get action probabilities if available
        action_probs = getattr(self, 'last_action_probs', None)

        experience = Experience(
            state=state,
            action=action.value if hasattr(action, 'value') else action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_probs=action_probs
        )
        self.memory.append(experience)

    def train_on_batch(self, batch_size: int = 32, epochs: int = 4):
        """Train using PPO on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)

        # Prepare batch tensors
        states = [exp.state for exp in batch]
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = [exp.next_state for exp in batch]
        dones = torch.FloatTensor([exp.done for exp in batch]).to(self.device)

        # Get old action probabilities
        old_probs = []
        for exp in batch:
            if exp.action_probs is not None:
                old_probs.append(exp.action_probs[exp.action])
            else:
                old_probs.append(1.0)  # Default probability
        old_probs = torch.FloatTensor(old_probs).to(self.device)

        # Compute returns and advantages
        with torch.no_grad():
            values = []
            next_values = []

            for state, next_state in zip(states, next_states):
                _, value, _ = self.network(state)  # Now returns 3 values
                values.append(value.item())

                if next_state is not None:
                    _, next_value, _ = self.network(next_state)  # Now returns 3 values
                    next_values.append(next_value.item())
                else:
                    next_values.append(0)

            values = torch.FloatTensor(values).to(self.device)
            next_values = torch.FloatTensor(next_values).to(self.device)

            # Compute advantages
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO training loop
        for _ in range(epochs):
            total_loss = 0

            for i, state in enumerate(states):
                policy_logits, value, raise_size_logits = self.network(state)  # Now returns 3 values

                # Get action probabilities
                action_probs = F.softmax(policy_logits, dim=-1)
                action_prob = action_probs[actions[i]]

                # Compute ratio for PPO
                ratio = action_prob / (old_probs[i] + 1e-8)

                # Clipped surrogate loss
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(value.squeeze(), returns[i])

                # Entropy bonus for exploration
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                total_loss += loss

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            self.loss_history.append(total_loss.item())

        self.training_step += 1

    def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000,
                      player_current_bet=0, min_raise=20):
        """Determine raise size using learned network output"""
        # Calculate pot after calling
        call_amount = current_bet - player_current_bet
        pot_after_call = pot + call_amount

        # Use the last raise size decision from the network
        if hasattr(self, 'last_raise_size_idx') and self.last_raise_size_idx is not None:
            # Map network output to raise sizes
            # 0 = half pot, 1 = 2/3 pot, 2 = full pot
            size_multipliers = [0.5, 0.67, 1.0]
            multiplier = size_multipliers[self.last_raise_size_idx]
        else:
            # Default to 2/3 pot if no decision was made
            multiplier = 0.67

        # Calculate desired raise based on selected size
        desired_raise = int(round(pot_after_call * multiplier))

        # Ensure minimum raise requirement
        actual_raise = max(min_raise, desired_raise)

        # Cap at what we can afford after calling
        max_raise_amount = player_chips - call_amount

        # Return integer amount to avoid fractional chips
        return min(actual_raise, max_raise_amount)

    def save(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_type': 'RewardBasedAI',
            'model_version': '1.0',
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'hidden_dim': self.network.hidden_dim,  # Save architecture info
            'config': {
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }, filepath)

    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Detect architecture from checkpoint
        hidden_dim = checkpoint.get('hidden_dim', None)
        if hidden_dim is None:
            # Try to infer from layer shapes
            state_dict = checkpoint['network_state']
            if 'input_projection.weight' in state_dict:
                hidden_dim = state_dict['input_projection.weight'].shape[0]
            else:
                hidden_dim = 256  # Default fallback

        # Recreate network with correct architecture if needed
        if hidden_dim != self.network.hidden_dim:
            self.network = RewardBasedNN(hidden_dim=hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])

        # Load config
        config = checkpoint.get('config', {})
        self.gamma = config.get('gamma', 0.99)
        self.clip_epsilon = config.get('clip_epsilon', 0.1)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.1)