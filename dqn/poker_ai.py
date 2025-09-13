import torch
import torch.optim as optim
import numpy as np
import random
from core.game_constants import Action
from .neural_network import PokerNet, PrioritizedReplayBuffer

class PokerAI:
    def __init__(self, config=None):
        """Initialize with configuration dictionary"""
        if config is None:
            config = {
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon': 0.3,
                'hidden_sizes': [512, 512, 256],
                'dropout_rate': 0.3,
                'batch_size': 64,
                'update_target_every': 100,
                'min_epsilon': 0.01,
                'epsilon_decay': 0.995
            }
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks will be initialized when we know input size
        self.input_size = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        self.memory = PrioritizedReplayBuffer()
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.min_epsilon = config['min_epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.update_target_every = config['update_target_every']
        self.updates = 0
        
        # Track performance metrics
        self.loss_history = []
        self.reward_history = []
    
    def _init_networks(self, input_size):
        """Initialize networks once we know the input size"""
        if self.q_network is None:
            self.input_size = input_size
            self.q_network = PokerNet(
                input_size, 
                hidden_sizes=self.config['hidden_sizes'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
            self.target_network = PokerNet(
                input_size,
                hidden_sizes=self.config['hidden_sizes'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
    
    def get_state_features(self, hand, community_cards, pot, current_bet, player_chips, 
                          player_bet, num_players, players_in_hand, position, 
                          action_history=None, opponent_bets=None, hand_phase=0):
        """Extract comprehensive features from game state"""
        features = []
        
        # Hand strength estimation (without hardcoding specific values)
        if hand and len(hand) >= 2:
            # Basic hand features
            values = sorted([c.value for c in hand], reverse=True)
            suited = 1.0 if hand[0].suit == hand[1].suit else 0.0
            pair = 1.0 if hand[0].value == hand[1].value else 0.0
            high_card = values[0] / 14
            low_card = values[1] / 14
            gap = (values[0] - values[1]) / 12
            
            features.extend([high_card, low_card, suited, pair, gap])
            
            # Community cards analysis
            if community_cards:
                all_cards = hand + community_cards
                all_values = [c.value for c in all_cards]
                all_suits = [c.suit for c in all_cards]
                
                # Potential detection
                value_counts = {}
                for v in all_values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                suit_counts = {}
                for s in all_suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                
                # Features for hand potential
                max_same_value = max(value_counts.values()) if value_counts else 0
                max_same_suit = max(suit_counts.values()) if suit_counts else 0
                
                features.extend([
                    max_same_value / 4,  # Pair/trips/quads potential
                    max_same_suit / 7,   # Flush potential
                    len(set(all_values)) / len(all_values),  # Uniqueness (straight potential)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Betting features
        call_amount = max(0, current_bet - player_bet)
        pot_odds = call_amount / (pot + call_amount + 1e-6)
        
        features.extend([
            pot / 1000,  # Normalized pot
            current_bet / 1000,  # Normalized current bet
            player_chips / 1000,  # Normalized stack
            player_bet / 1000,  # Player's investment
            call_amount / 1000,  # Call amount
            pot_odds,  # Pot odds
            player_chips / (pot + 1e-6),  # Stack to pot ratio
            player_bet / (player_bet + player_chips + 1e-6),  # Pot commitment
        ])
        
        # Game state features
        features.extend([
            position / max(num_players - 1, 1),  # Position
            players_in_hand / num_players,  # Players remaining ratio
            1.0 / max(players_in_hand, 1),  # Heads up indicator
            hand_phase / 3,  # Game phase (0=preflop, 1=flop, 2=turn, 3=river)
        ])
        
        # Action history (last 5 actions as one-hot)
        if action_history:
            recent_actions = action_history[-5:]
            for action in recent_actions:
                action_vec = [0] * 5
                if isinstance(action, Action):
                    action_vec[action.value] = 1
                features.extend(action_vec)
            # Pad if we have less than 5 actions
            padding_needed = 5 - len(recent_actions)
            for _ in range(padding_needed):
                features.extend([0] * 5)
        else:
            features.extend([0] * 25)  # No action history - add 25 zeros (5 actions * 5 values)
        
        # Opponent aggression metrics
        if opponent_bets and len(opponent_bets) > 0:
            recent_bets = opponent_bets[-5:]
            avg_bet = np.mean(recent_bets)
            max_bet = max(recent_bets)
            aggression_factor = len([b for b in recent_bets if b > 20]) / len(recent_bets)
            
            features.extend([
                avg_bet / 1000,
                max_bet / 1000,
                aggression_factor,
                len(recent_bets) / 5,  # Betting frequency
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Add some computed features
        features.extend([
            1.0 if call_amount == 0 else 0.0,  # Can check
            1.0 if player_chips > call_amount else 0.0,  # Can call
            1.0 if player_chips > current_bet * 2 else 0.0,  # Can raise significantly
            min(1.0, current_bet / (player_chips + 1e-6)),  # Bet size pressure
        ])
        
        state = torch.FloatTensor(features).to(self.device)
        
        # Initialize networks if needed
        if self.q_network is None:
            self._init_networks(len(features))
        
        return state
    
    def choose_action(self, state, valid_actions, training=False):
        """Choose action using epsilon-greedy policy"""
        # Convert valid_actions to mask
        action_mask = torch.zeros(5).to(self.device)
        for action in valid_actions:
            action_mask[action.value] = 1
        
        if training and random.random() < self.epsilon:
            valid_indices = [a.value for a in valid_actions]
            return Action(random.choice(valid_indices))
        
        with torch.no_grad():
            self.q_network.eval()
            q_values = self.q_network(state.unsqueeze(0))
            self.q_network.train()
            
            # Mask invalid actions
            q_values = q_values.squeeze()
            q_values[action_mask == 0] = float('-inf')
            
            # Add small noise to break ties
            if training:
                q_values += torch.randn_like(q_values) * 0.01
            
            return Action(q_values.argmax().item())
    
    def remember(self, state, action, reward, next_state, done, priority=None):
        """Store experience in replay buffer"""
        self.memory.push(state, action.value, reward, next_state, done, priority=priority)
        self.reward_history.append(reward)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size or self.q_network is None:
            return
        
        # Sample with prioritization
        result = self.memory.sample(self.batch_size)
        if len(result) == 3:
            batch, weights, indices = result
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = result
            weights = torch.ones(len(batch)).to(self.device)
            indices = None
        
        states = torch.stack([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN - use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calculate loss with importance sampling weights
        losses = (current_q_values.squeeze() - target_q_values) ** 2
        loss = (losses * weights).mean()
        
        # Update priorities if using prioritized replay
        if indices is not None:
            priorities = losses.detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        # Update target network
        self.updates += 1
        if self.updates % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save the model"""
        if self.q_network is None:
            print("No model to save")
            return
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'updates': self.updates,
            'input_size': self.input_size
        }, filepath)
    
    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.config = checkpoint.get('config', self.config)
        self.input_size = checkpoint['input_size']
        self._init_networks(self.input_size)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0)
        self.updates = checkpoint.get('updates', 0)
    
    def get_raise_size(self, state, pot, current_bet, player_chips, player_bet, min_raise):
        """Strategically determine raise size based on game state"""
        with torch.no_grad():
            self.q_network.eval()
            q_values = self.q_network(state.unsqueeze(0)).squeeze()
            self.q_network.train()
            
            # Get Q-value for raise action
            raise_confidence = q_values[Action.RAISE.value].item()
        
        call_amount = current_bet - player_bet
        pot_after_call = pot + call_amount
        max_raise = player_chips - call_amount
        
        # Build list of valid raise sizes
        valid_sizes = []
        
        # Always include minimum raise
        if min_raise <= max_raise:
            valid_sizes.append(('min', min_raise, 0.15))
        
        # Calculate strategic sizes
        one_third = int(pot_after_call * 0.33)
        if one_third > min_raise and one_third <= max_raise:
            valid_sizes.append(('third', one_third, 0.20))
        
        half_pot = int(pot_after_call * 0.5)
        if half_pot > min_raise and half_pot <= max_raise:
            valid_sizes.append(('half', half_pot, 0.25))
        
        three_quarters = int(pot_after_call * 0.75)
        if three_quarters > min_raise and three_quarters <= max_raise:
            valid_sizes.append(('three_quarters', three_quarters, 0.20))
        
        full_pot = pot_after_call
        if full_pot > min_raise and full_pot <= max_raise:
            valid_sizes.append(('pot', full_pot, 0.15))
        
        overbet = int(pot_after_call * 1.5)
        if overbet > min_raise and overbet <= max_raise:
            valid_sizes.append(('overbet', overbet, 0.05))
        
        if not valid_sizes:
            return min_raise
        
        # Adjust weights based on confidence
        # High confidence = larger bets, low confidence = smaller bets
        adjusted_weights = []
        for name, size, base_weight in valid_sizes:
            if raise_confidence > 0.5:  # Confident
                # Prefer larger sizes
                if name in ['pot', 'overbet', 'three_quarters']:
                    weight = base_weight * 1.5
                elif name in ['min']:
                    weight = base_weight * 0.5
                else:
                    weight = base_weight
            elif raise_confidence < -0.5:  # Less confident
                # Prefer smaller sizes
                if name in ['min', 'third']:
                    weight = base_weight * 1.5
                elif name in ['pot', 'overbet']:
                    weight = base_weight * 0.3
                else:
                    weight = base_weight
            else:  # Neutral
                weight = base_weight
            
            adjusted_weights.append(weight)
        
        # Normalize weights
        total = sum(adjusted_weights)
        adjusted_weights = [w/total for w in adjusted_weights]
        
        # Select size based on weights
        import random
        selected = random.choices([s[1] for s in valid_sizes], weights=adjusted_weights)[0]
        
        return selected