"""
Train and save strategy bots with specific play styles
These are used as training opponents to create more diverse AI strategies
"""

import os
import random
import numpy as np
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from game_constants import Action
from neural_network import Experience  # Import Experience from neural_network

def train_conservative_bot(episodes=500):
    """Train a bot that plays tight and rarely goes all-in"""
    print("\nTraining Conservative Bot...")
    
    config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 0.7,
        'hidden_sizes': [256, 128],
        'dropout_rate': 0.25,
        'batch_size': 32,
        'update_target_every': 100,
        'min_epsilon': 0.05,
        'epsilon_decay': 0.995
    }
    
    bot = PokerAI(config=config)
    training_game = TexasHoldEmTraining(num_players=4)
    
    for episode in range(episodes):
        # Create random opponents
        opponents = []
        for _ in range(3):
            random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                       'gamma': 0, 'hidden_sizes': [64],
                                       'dropout_rate': 0, 'batch_size': 1,
                                       'update_target_every': 1000000,
                                       'min_epsilon': 1.0, 'epsilon_decay': 1.0})
            opponents.append(random_ai)
        
        # Play training hands
        for hand in range(5):
            training_game.reset_game()
            all_models = [bot] + opponents
            random.shuffle(all_models)
            
            winners = training_game.simulate_hand(all_models)
            
            # Custom reward shaping for conservative play
            for i, model in enumerate(all_models):
                if model == bot:
                    player = training_game.players[i]
                    
                    # Calculate conservative reward
                    reward = 0
                    if player in winners:
                        reward = 1.0  # Win bonus
                    
                    # Heavily penalize all-ins
                    if hasattr(training_game, 'action_history'):
                        player_actions = []
                        # Approximate which actions were this player's
                        for j, action in enumerate(training_game.action_history):
                            if j % 4 == i:  # Rough approximation
                                player_actions.append(action)
                        
                        for action in player_actions:
                            if action == Action.ALL_IN:
                                reward -= 2.0  # Heavy penalty
                            elif action == Action.FOLD:
                                reward += 0.1  # Small bonus for folding (conservative)
                            elif action == Action.CHECK:
                                reward += 0.05
                            elif action == Action.CALL:
                                reward += 0.05
                            elif action == Action.RAISE:
                                reward -= 0.1  # Small penalty for aggression
                    
                    # Store experience with shaped reward
                    if len(bot.memory.buffer) > 0:
                        last_exp = bot.memory.buffer[-1]
                        bot.memory.buffer[-1] = Experience(
                            last_exp.state, last_exp.action, reward,
                            last_exp.next_state, last_exp.done
                        )
            
            # Train
            if len(bot.memory) > bot.batch_size:
                for _ in range(3):
                    bot.replay()
        
        bot.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"  Episode {episode}/{episodes}, Epsilon: {bot.epsilon:.3f}")
    
    return bot


def train_aggressive_bot(episodes=500):
    """Train a bot that plays aggressively but avoids excessive all-ins"""
    print("\nTraining Aggressive Bot...")
    
    config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 0.7,
        'hidden_sizes': [256, 128],
        'dropout_rate': 0.25,
        'batch_size': 32,
        'update_target_every': 100,
        'min_epsilon': 0.05,
        'epsilon_decay': 0.995
    }
    
    bot = PokerAI(config=config)
    training_game = TexasHoldEmTraining(num_players=4)
    
    for episode in range(episodes):
        # Create random opponents
        opponents = []
        for _ in range(3):
            random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                       'gamma': 0, 'hidden_sizes': [64],
                                       'dropout_rate': 0, 'batch_size': 1,
                                       'update_target_every': 1000000,
                                       'min_epsilon': 1.0, 'epsilon_decay': 1.0})
            opponents.append(random_ai)
        
        # Play training hands
        for hand in range(5):
            training_game.reset_game()
            all_models = [bot] + opponents
            random.shuffle(all_models)
            
            winners = training_game.simulate_hand(all_models)
            
            # Custom reward shaping for aggressive play
            for i, model in enumerate(all_models):
                if model == bot:
                    player = training_game.players[i]
                    
                    # Calculate aggressive reward
                    reward = 0
                    if player in winners:
                        reward = 1.0  # Win bonus
                    
                    # Shape rewards for controlled aggression
                    if hasattr(training_game, 'action_history'):
                        player_actions = []
                        for j, action in enumerate(training_game.action_history):
                            if j % 4 == i:  # Rough approximation
                                player_actions.append(action)
                        
                        all_in_count = 0
                        for action in player_actions:
                            if action == Action.ALL_IN:
                                all_in_count += 1
                                if all_in_count == 1:
                                    reward -= 0.5  # Mild penalty for first all-in
                                else:
                                    reward -= 1.5  # Heavy penalty for multiple all-ins
                            elif action == Action.RAISE:
                                reward += 0.3  # Bonus for raising
                            elif action == Action.CALL:
                                reward += 0.1
                            elif action == Action.FOLD:
                                reward -= 0.2  # Penalty for folding (be aggressive)
                    
                    # Store experience with shaped reward
                    if len(bot.memory.buffer) > 0:
                        last_exp = bot.memory.buffer[-1]
                        bot.memory.buffer[-1] = Experience(
                            last_exp.state, last_exp.action, reward,
                            last_exp.next_state, last_exp.done
                        )
            
            # Train
            if len(bot.memory) > bot.batch_size:
                for _ in range(3):
                    bot.replay()
        
        bot.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"  Episode {episode}/{episodes}, Epsilon: {bot.epsilon:.3f}")
    
    return bot


def train_balanced_bot(episodes=500):
    """Train a bot that plays a balanced strategy"""
    print("\nTraining Balanced Bot...")
    
    config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 0.7,
        'hidden_sizes': [256, 128],
        'dropout_rate': 0.25,
        'batch_size': 32,
        'update_target_every': 100,
        'min_epsilon': 0.05,
        'epsilon_decay': 0.995
    }
    
    bot = PokerAI(config=config)
    training_game = TexasHoldEmTraining(num_players=4)
    
    for episode in range(episodes):
        # Create random opponents
        opponents = []
        for _ in range(3):
            random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                       'gamma': 0, 'hidden_sizes': [64],
                                       'dropout_rate': 0, 'batch_size': 1,
                                       'update_target_every': 1000000,
                                       'min_epsilon': 1.0, 'epsilon_decay': 1.0})
            opponents.append(random_ai)
        
        # Play training hands
        for hand in range(5):
            training_game.reset_game()
            all_models = [bot] + opponents
            random.shuffle(all_models)
            
            winners = training_game.simulate_hand(all_models)
            
            # Custom reward shaping for balanced play
            for i, model in enumerate(all_models):
                if model == bot:
                    player = training_game.players[i]
                    
                    # Calculate balanced reward
                    reward = 0
                    if player in winners:
                        reward = 1.0  # Win bonus
                    
                    # Shape rewards for balanced play
                    if hasattr(training_game, 'action_history'):
                        player_actions = []
                        for j, action in enumerate(training_game.action_history):
                            if j % 4 == i:  # Rough approximation
                                player_actions.append(action)
                        
                        action_counts = {a: 0 for a in Action}
                        for action in player_actions:
                            action_counts[action] += 1
                        
                        total_actions = sum(action_counts.values())
                        if total_actions > 0:
                            # Reward balanced distribution
                            if action_counts[Action.ALL_IN] > 0:
                                reward -= 1.0 * action_counts[Action.ALL_IN]  # Penalize all-ins
                            
                            # Ideal distribution rewards
                            fold_ratio = action_counts[Action.FOLD] / total_actions
                            raise_ratio = action_counts[Action.RAISE] / total_actions
                            call_ratio = action_counts[Action.CALL] / total_actions
                            
                            # Reward balanced play (not too passive, not too aggressive)
                            if 0.1 <= fold_ratio <= 0.3:
                                reward += 0.2
                            if 0.2 <= raise_ratio <= 0.4:
                                reward += 0.2
                            if 0.2 <= call_ratio <= 0.4:
                                reward += 0.1
                    
                    # Store experience with shaped reward
                    if len(bot.memory.buffer) > 0:
                        last_exp = bot.memory.buffer[-1]
                        bot.memory.buffer[-1] = Experience(
                            last_exp.state, last_exp.action, reward,
                            last_exp.next_state, last_exp.done
                        )
            
            # Train
            if len(bot.memory) > bot.batch_size:
                for _ in range(3):
                    bot.replay()
        
        bot.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"  Episode {episode}/{episodes}, Epsilon: {bot.epsilon:.3f}")
    
    return bot


def create_all_strategy_bots(save_dir='strategy_bots', episodes=500):
    """Create and save all strategy bots"""
    
    print("=" * 60)
    print("CREATING STRATEGY BOTS")
    print("=" * 60)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    bots = {}
    
    # Train conservative bot
    conservative = train_conservative_bot(episodes)
    conservative.save(os.path.join(save_dir, 'conservative_bot.pth'))
    bots['conservative'] = conservative
    print("✓ Saved conservative_bot.pth")
    
    # Train aggressive bot
    aggressive = train_aggressive_bot(episodes)
    aggressive.save(os.path.join(save_dir, 'aggressive_bot.pth'))
    bots['aggressive'] = aggressive
    print("✓ Saved aggressive_bot.pth")
    
    # Train balanced bot
    balanced = train_balanced_bot(episodes)
    balanced.save(os.path.join(save_dir, 'balanced_bot.pth'))
    bots['balanced'] = balanced
    print("✓ Saved balanced_bot.pth")
    
    print("\nAll strategy bots created successfully!")
    print(f"Location: {save_dir}/")
    
    return bots


def load_strategy_bots(save_dir='strategy_bots'):
    """Load pre-trained strategy bots"""
    bots = {}
    
    bot_files = {
        'conservative': 'conservative_bot.pth',
        'aggressive': 'aggressive_bot.pth',
        'balanced': 'balanced_bot.pth'
    }
    
    for name, filename in bot_files.items():
        path = os.path.join(save_dir, filename)
        if os.path.exists(path):
            bot = PokerAI()
            bot.load(path)
            bot.epsilon = 0  # No exploration during use
            bots[name] = bot
        else:
            print(f"Warning: {path} not found")
    
    return bots


if __name__ == "__main__":
    # Create all bots when run directly
    create_all_strategy_bots(episodes=300)
    
    # Test loading them
    print("\nTesting load...")
    loaded_bots = load_strategy_bots()
    print(f"Loaded {len(loaded_bots)} bots: {list(loaded_bots.keys())}")