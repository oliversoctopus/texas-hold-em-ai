import random
import numpy as np
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from player import Player
from card_deck import Card

def hyperparameter_tuning(num_configs=5, episodes_per_config=200, eval_games=50, num_players=4):
    """Automatically tune hyperparameters with early stopping for bad configs"""
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print(f"Testing {num_configs} configurations with {num_players} players")
    print("=" * 60)
    
    # Define hyperparameter search space with more conservative options
    configs = []
    for i in range(num_configs):
        config = {
            'learning_rate': random.choice([0.0005, 0.001, 0.0015, 0.002]),
            'gamma': random.choice([0.92, 0.95, 0.97, 0.99]),
            'epsilon': random.choice([0.4, 0.5, 0.6]),  # Start with more exploration
            'hidden_sizes': random.choice([
                [256, 256], 
                [512, 256], 
                [512, 512, 256],
                [512, 512, 512],  # Added symmetric option
            ]),
            'dropout_rate': random.choice([0.15, 0.2, 0.25, 0.3]),
            'batch_size': random.choice([32, 64, 96]),  # Removed very large batch sizes
            'update_target_every': random.choice([75, 100, 150]),
            'min_epsilon': random.choice([0.01, 0.02, 0.05]),
            'epsilon_decay': random.choice([0.994, 0.995, 0.996, 0.997])  # Slower decay
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
        
        # Train with this configuration and specified number of players
        ai_model = train_ai_advanced(
            num_episodes=episodes_per_config,
            config=config,
            verbose=False,
            num_players=num_players
        )
        
        # Quick pre-evaluation to check if model learned anything
        print(f"Pre-evaluating configuration {idx + 1}...")
        quick_win_rate, quick_earnings = evaluate_ai_full(
            ai_model, 
            num_games=20,  # Quick evaluation
            num_players=num_players
        )
        
        # Skip full evaluation if performance is terrible
        if quick_win_rate < 10:  # Less than 10% win rate in quick eval
            print(f"Configuration {idx + 1} performing poorly (win rate: {quick_win_rate:.1f}%), skipping full evaluation")
            results.append({
                'config': config,
                'win_rate': quick_win_rate,
                'earnings': quick_earnings,
                'score': quick_win_rate + quick_earnings / 100
            })
            continue
        
        # Full evaluation for promising configs
        print(f"Full evaluation of configuration {idx + 1}...")
        win_rate, avg_earnings = evaluate_ai_full(
            ai_model, 
            num_games=eval_games, 
            num_players=num_players
        )
        
        # Calculate combined score with emphasis on consistency
        score = win_rate + avg_earnings / 100
        
        # Bonus for high win rate (consistency)
        if win_rate > 50:
            score += (win_rate - 50) * 0.5  # Bonus for win rates above 50%
        
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
            best_model = ai_model
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best configuration found:")
    print(f"  Learning rate: {best_config['learning_rate']}")
    print(f"  Gamma: {best_config['gamma']}")
    print(f"  Hidden layers: {best_config['hidden_sizes']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Epsilon decay: {best_config['epsilon_decay']}")
    print(f"  Best score: {best_score:.2f}")
    
    # Show all results sorted by score
    print("\nAll configurations ranked:")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. Win rate: {result['win_rate']:.1f}%, "
              f"Earnings: ${result['earnings']:.0f}, Score: {result['score']:.2f}")
    
    return best_config, results, best_model

def warmup_training(ai_model, num_episodes=100):
    """Warm-up training against simple opponents to establish basic strategies"""
    print("Warming up model with basic training...")
    
    # Create very simple opponent
    simple_opponent = PokerAI(config={
        'learning_rate': 0, 'gamma': 0, 
        'epsilon': 1.0, 'hidden_sizes': [64], 
        'dropout_rate': 0, 'batch_size': 1,
        'update_target_every': 1000000, 'min_epsilon': 1.0,
        'epsilon_decay': 1.0
    })
    
    training_game = TexasHoldEmTraining(num_players=2)  # Start with heads-up
    
    for episode in range(num_episodes):
        ai_models = [ai_model, simple_opponent]
        training_game.reset_game()
        training_game.num_players = 2
        winners = training_game.simulate_hand(ai_models)
        
        # Train frequently during warm-up
        if len(ai_model.memory) > ai_model.batch_size:
            for _ in range(2):
                ai_model.replay()
    
    print("Warm-up complete!")

def train_ai_advanced(num_episodes=1000, config=None, verbose=True, num_players=4):
    """Advanced training with warm-up and better reward shaping"""
    if verbose:
        print(f"Training AI with {num_episodes} episodes using {num_players} players...")
    
    # Create AI with configuration
    main_ai = PokerAI(config=config)
    
    # Warm-up phase for new models
    if len(main_ai.memory) == 0:  # New model
        warmup_training(main_ai, num_episodes=min(100, num_episodes // 10))
    
    # Create multiple opponent AIs with different skill levels for diversity
    opponents = []
    
    # Weak opponent (high exploration, simple network)
    weak_config = {
        'learning_rate': 0.001, 'gamma': 0.9, 
        'epsilon': 0.9, 'hidden_sizes': [128], 
        'dropout_rate': 0.1, 'batch_size': 32,
        'update_target_every': 100, 'min_epsilon': 0.7,
        'epsilon_decay': 0.999
    }
    opponents.append(PokerAI(config=weak_config))
    
    # Medium opponent (moderate exploration, decent network)
    medium_config = {
        'learning_rate': 0.001, 'gamma': 0.95, 
        'epsilon': 0.5, 'hidden_sizes': [256, 128], 
        'dropout_rate': 0.2, 'batch_size': 64,
        'update_target_every': 100, 'min_epsilon': 0.3,
        'epsilon_decay': 0.997
    }
    opponents.append(PokerAI(config=medium_config))
    
    # Strong opponent (low exploration, complex network) - clone of main config
    strong_config = config.copy() if config else {
        'learning_rate': 0.001, 'gamma': 0.99, 
        'epsilon': 0.3, 'hidden_sizes': [512, 256], 
        'dropout_rate': 0.3, 'batch_size': 64,
        'update_target_every': 100, 'min_epsilon': 0.05,
        'epsilon_decay': 0.995
    }
    strong_config['epsilon'] = 0.3  # Ensure some exploration
    opponents.append(PokerAI(config=strong_config))
    
    # Create additional opponents if needed for more players
    while len(opponents) < num_players - 1:
        # Create variations of existing configs
        base_config = random.choice([weak_config, medium_config, strong_config])
        variant_config = base_config.copy()
        # Slightly modify the config
        variant_config['learning_rate'] *= random.choice([0.5, 1.0, 2.0])
        variant_config['epsilon'] = min(1.0, variant_config['epsilon'] * random.uniform(0.8, 1.2))
        opponents.append(PokerAI(config=variant_config))
    
    training_game = TexasHoldEmTraining(num_players=num_players)
    
    wins = {'main': 0, 'opponents': 0}
    recent_performance = []
    
    for episode in range(num_episodes):
        # Curriculum learning - gradually increase difficulty
        if episode < num_episodes // 4:
            # Early: mostly weak opponents
            opponent_pool = [opponents[0]] * (num_players // 2) + opponents[1:]
        elif episode < num_episodes // 2:
            # Early-mid: mix of weak and medium
            opponent_pool = opponents[:2] * ((num_players - 1) // 2) + opponents[2:]
        elif episode < 3 * num_episodes // 4:
            # Late-mid: mostly medium with some strong
            opponent_pool = [opponents[1]] * ((num_players - 1) // 2) + opponents[2:]
        else:
            # Late: mix of all, including self-play
            opponent_pool = opponents + [main_ai]
        
        # Randomly select opponents for this episode (need exactly num_players - 1 opponents)
        selected_opponents = random.sample(opponent_pool * 2, num_players - 1)[:num_players - 1]
        ai_models = [main_ai] + selected_opponents
        random.shuffle(ai_models)
        
        # Play multiple hands per episode
        episode_wins = 0
        hands_per_episode = 5 if episode < num_episodes // 2 else 10
        
        for hand in range(hands_per_episode):
            training_game.reset_game()
            winners = training_game.simulate_hand(ai_models)
            
            # Track wins
            for winner in winners:
                if winner.ai_model == main_ai:
                    wins['main'] += 1
                    episode_wins += 1
                else:
                    wins['opponents'] += 1
        
        recent_performance.append(episode_wins / hands_per_episode)
        if len(recent_performance) > 100:
            recent_performance.pop(0)
        
        # Adaptive training - train more if performance is poor
        if len(main_ai.memory) > main_ai.batch_size * 2:
            # Base training
            train_iterations = 5
            
            # Additional training if struggling
            if len(recent_performance) >= 20:
                recent_avg = np.mean(recent_performance[-20:])
                if recent_avg < 0.25:  # Winning less than 25% of hands
                    train_iterations = 10
            
            for _ in range(train_iterations):
                main_ai.replay()
        
        # Also train opponents occasionally for diversity
        if episode % 20 == 0:
            for opponent in opponents[:3]:  # Train only the base 3 opponents
                if len(opponent.memory) > opponent.batch_size:
                    for _ in range(3):
                        opponent.replay()
                    opponent.decay_epsilon()
        
        # Decay epsilon with adaptive rate
        if len(recent_performance) >= 50:
            performance_trend = np.mean(recent_performance[-10:]) - np.mean(recent_performance[-50:-40])
            if performance_trend < 0:  # Performance declining
                # Slower decay to maintain exploration
                main_ai.epsilon = max(main_ai.min_epsilon, 
                                     main_ai.epsilon * (main_ai.epsilon_decay ** 0.5))
            else:
                # Normal decay
                main_ai.decay_epsilon()
        else:
            main_ai.decay_epsilon()
        
        if verbose and episode % 100 == 0:
            recent_win_rate = np.mean(recent_performance) * 100 if recent_performance else 0
            avg_reward = np.mean(main_ai.reward_history[-100:]) if main_ai.reward_history else 0
            print(f"Episode {episode}/{num_episodes}, Epsilon: {main_ai.epsilon:.3f}, "
                  f"Recent win rate: {recent_win_rate:.1f}%, Avg reward: {avg_reward:.3f}")
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"Main AI wins: {wins['main']}, Opponent wins: {wins['opponents']}")
        win_percentage = (wins['main'] / (wins['main'] + wins['opponents']) * 100) if (wins['main'] + wins['opponents']) > 0 else 0
        print(f"Overall win rate: {win_percentage:.1f}%")
        if main_ai.loss_history:
            print(f"Final loss: {np.mean(main_ai.loss_history[-100:]):.4f}")
        
        # Evaluate all models and return the best one
        print("\n" + "=" * 60)
        print("EVALUATING ALL TRAINED MODELS")
        print("=" * 60)
        
        all_models = [('Main AI', main_ai)] + [(f'Opponent {i+1}', opp) for i, opp in enumerate(opponents[:3])]
        best_model = None
        best_score = -float('inf')
        best_name = None
        
        for name, model in all_models:
            print(f"\nEvaluating {name}...")
            model.epsilon = 0  # No exploration during evaluation
            win_rate, avg_earnings = evaluate_ai_full(model, num_games=20, num_players=num_players)
            score = win_rate + avg_earnings / 100
            
            print(f"{name}: Win rate={win_rate:.1f}%, Earnings=${avg_earnings:.0f}, Score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_name} with score {best_score:.2f}")
        print("=" * 60)
        
        # Restore original epsilon for continued training
        main_ai.epsilon = main_ai.config['min_epsilon']
        for opp in opponents:
            opp.epsilon = opp.config['min_epsilon']
        
        return best_model
    
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