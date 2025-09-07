import random
import numpy as np
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from player import Player
from card_deck import Card

def hyperparameter_tuning(num_configs=5, episodes_per_config=200, eval_games=50):
    """Automatically tune hyperparameters and return the best trained model"""
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Define hyperparameter search space
    configs = []
    for i in range(num_configs):
        config = {
            'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.002]),
            'gamma': random.choice([0.9, 0.95, 0.99]),
            'epsilon': random.choice([0.3, 0.5, 0.7]),
            'hidden_sizes': random.choice([
                [256, 256], 
                [512, 256], 
                [512, 512, 256],
                [1024, 512, 256]
            ]),
            'dropout_rate': random.choice([0.1, 0.2, 0.3]),
            'batch_size': random.choice([32, 64, 128]),
            'update_target_every': random.choice([50, 100, 200]),
            'min_epsilon': random.choice([0.01, 0.05]),
            'epsilon_decay': random.choice([0.99, 0.995, 0.999])
        }
        configs.append(config)
    
    best_config = None
    best_score = -float('inf')
    best_model = None
    results = []
    
    for idx, config in enumerate(configs):
        print(f"\nTesting configuration {idx + 1}/{num_configs}")
        print(f"Config: LR={config['learning_rate']}, Gamma={config['gamma']}, "
              f"Hidden={config['hidden_sizes']}, Batch={config['batch_size']}")
        
        # Train with this configuration
        ai_model = train_ai_advanced(
            num_episodes=episodes_per_config,
            config=config,
            verbose=False
        )
        
        # Use real evaluation instead of quick evaluation
        print(f"Evaluating configuration {idx + 1}...")
        win_rate, avg_earnings = evaluate_ai_full(
            ai_model, 
            num_games=eval_games, 
            num_players=6
        )
        
        # Calculate combined score
        score = win_rate + avg_earnings / 100  # Combined metric
        
        results.append({
            'config': config,
            'win_rate': win_rate,
            'earnings': avg_earnings,
            'score': score
        })
        
        print(f"Results: Win rate={win_rate:.1f}%, Earnings=${avg_earnings:.0f}, Score={score:.2f}")
        
        if score > best_score:
            best_score = score
            best_config = config
            best_model = ai_model  # Keep the best model
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best configuration found:")
    print(f"  Learning rate: {best_config['learning_rate']}")
    print(f"  Gamma: {best_config['gamma']}")
    print(f"  Hidden layers: {best_config['hidden_sizes']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Best score: {best_score:.2f}")
    
    return best_config, results, best_model  # Return the trained model too

def train_ai_advanced(num_episodes=1000, config=None, verbose=True):
    """Advanced training with better reward shaping"""
    if verbose:
        print(f"Training AI with {num_episodes} episodes...")
    
    # Create AI with configuration
    main_ai = PokerAI(config=config)
    
    # Create opponent AI for diversity (simpler, more random)
    opponent_ai = PokerAI(config={'learning_rate': 0.001, 'gamma': 0.9, 
                                  'epsilon': 0.8, 'hidden_sizes': [128], 
                                  'dropout_rate': 0.1, 'batch_size': 32,
                                  'update_target_every': 100, 'min_epsilon': 0.5,
                                  'epsilon_decay': 0.999})
    
    training_game = TexasHoldEmTraining(num_players=4)
    
    wins = {'main': 0, 'opponent': 0}
    
    for episode in range(num_episodes):
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Epsilon: {main_ai.epsilon:.3f}, "
                  f"Buffer: {len(main_ai.memory)}, Avg reward: {np.mean(main_ai.reward_history[-100:]) if main_ai.reward_history else 0:.3f}")
        
        # Mix of AI models for diversity
        if episode < num_episodes // 3:
            # Early training: mostly against random/weak opponents
            ai_models = [main_ai] + [opponent_ai] * 3
        elif episode < 2 * num_episodes // 3:
            # Mid training: mix of opponents
            ai_models = [main_ai, main_ai, opponent_ai, opponent_ai]
        else:
            # Late training: mostly self-play
            ai_models = [main_ai] * 3 + [opponent_ai]
        
        random.shuffle(ai_models)
        
        # Play multiple hands per episode
        episode_rewards = []
        for hand in range(10):
            training_game.reset_game()
            winners = training_game.simulate_hand(ai_models)
            
            # Track wins
            for winner in winners:
                if winner.ai_model == main_ai:
                    wins['main'] += 1
                else:
                    wins['opponent'] += 1
        
        # Train multiple times per episode
        if len(main_ai.memory) > main_ai.batch_size * 2:
            for _ in range(5):
                main_ai.replay()
        
        # Decay epsilon
        main_ai.decay_epsilon()
        
        # Also train opponent occasionally for diversity
        if episode % 10 == 0 and len(opponent_ai.memory) > opponent_ai.batch_size:
            opponent_ai.replay()
            opponent_ai.decay_epsilon()
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"Main AI wins: {wins['main']}, Opponent wins: {wins['opponent']}")
        if main_ai.loss_history:
            print(f"Final loss: {np.mean(main_ai.loss_history[-100:]):.4f}")
    
    return main_ai

def evaluate_ai_full(ai_model, num_games=100, num_players=4):
    """Full evaluation against random players"""
    print(f"Evaluating AI over {num_games} games...")
    
    original_epsilon = ai_model.epsilon
    ai_model.epsilon = 0
    
    wins = 0
    total_earnings = 0
    games_survived = 0
    
    for game_num in range(num_games):
        if (game_num + 1) % 20 == 0:
            print(f"Progress: {game_num + 1}/{num_games}")
        
        training_game = TexasHoldEmTraining(num_players=num_players)
        
        # Track chips
        player_chips = {f"Player_{i}": 1000 for i in range(num_players)}
        player_chips["Trained"] = 1000
        
        # Create AI models
        ai_models = []
        trained_idx = 0
        
        for i in range(num_players):
            if i == trained_idx:
                ai_models.append(ai_model)
            else:
                # Random AI (high epsilon, no learning)
                random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                           'gamma': 0, 'hidden_sizes': [64],
                                           'dropout_rate': 0, 'batch_size': 1,
                                           'update_target_every': 1000000,
                                           'min_epsilon': 1.0, 'epsilon_decay': 1.0})
                ai_models.append(random_ai)
        
        # Play game
        for hand_num in range(50):
            active_count = sum(1 for chips in player_chips.values() if chips > 0)
            if active_count <= 1:
                break
            
            # Update player chips
            for i, player in enumerate(training_game.players if hasattr(training_game, 'players') else []):
                if i == trained_idx:
                    player.chips = player_chips["Trained"]
                else:
                    player.chips = player_chips[f"Player_{i}"]
            
            # Simulate hand
            training_game.reset_game()
            winners = training_game.simulate_hand(ai_models)
            
            # Update chips based on winners
            if winners:
                pot_per_winner = 100  # Simplified
                for i, player in enumerate(training_game.players):
                    if player in winners:
                        if i == trained_idx:
                            player_chips["Trained"] += pot_per_winner
                        else:
                            player_chips[f"Player_{i}"] += pot_per_winner
                    else:
                        if i == trained_idx:
                            player_chips["Trained"] -= 25
                        else:
                            player_chips[f"Player_{i}"] -= 25
                    
                    # Ensure non-negative
                    if i == trained_idx:
                        player_chips["Trained"] = max(0, player_chips["Trained"])
                    else:
                        player_chips[f"Player_{i}"] = max(0, player_chips[f"Player_{i}"])
        
        # Check results
        final_chips = [(name, chips) for name, chips in player_chips.items()]
        final_chips.sort(key=lambda x: x[1], reverse=True)
        
        if final_chips[0][0] == "Trained":
            wins += 1
        
        trained_final = player_chips["Trained"]
        total_earnings += trained_final - 1000
        
        if trained_final > 0:
            games_survived += 1
    
    ai_model.epsilon = original_epsilon
    
    win_rate = (wins / num_games) * 100
    survival_rate = (games_survived / num_games) * 100
    avg_earnings = total_earnings / num_games
    
    print(f"\nEvaluation Results:")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Survival rate: {survival_rate:.1f}%")
    print(f"Average earnings: ${avg_earnings:+.0f}")
    print(f"Expected random win rate: {100/num_players:.1f}%")
    
    performance_ratio = win_rate / (100/num_players)
    if performance_ratio > 1.5:
        print(f"✓ AI is performing {performance_ratio:.1f}x better than random!")
    elif performance_ratio > 1.2:
        print(f"✓ AI is performing {performance_ratio:.1f}x better than random")
    else:
        print(f"✗ AI needs improvement (only {performance_ratio:.1f}x random performance)")
    
    return win_rate, avg_earnings