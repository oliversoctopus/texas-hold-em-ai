import random
import numpy as np
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from player import Player
from card_deck import Card

def hyperparameter_tuning(num_configs=5, episodes_per_config=200, eval_games=50, num_players=4):
    """Automatically tune hyperparameters using self-play training"""
    print("=" * 60)
    print("HYPERPARAMETER TUNING WITH SELF-PLAY")
    print(f"Testing {num_configs} configurations with {num_players} players")
    print("=" * 60)
    
    # Define hyperparameter search space including both simple and complex models
    configs = []
    for i in range(num_configs):
        # Randomly choose between simple, medium, and complex configurations
        complexity = random.choice(['simple', 'medium', 'complex'])
        
        if complexity == 'simple':
            # Simple models with high exploration
            config = {
                'learning_rate': random.choice([0.001, 0.0015, 0.002, 0.003]),
                'gamma': random.choice([0.85, 0.9, 0.95]),
                'epsilon': random.choice([0.7, 0.8, 0.9]),  # High exploration
                'hidden_sizes': random.choice([
                    [64],
                    [128], 
                    [128, 64],
                    [256],
                ]),
                'dropout_rate': random.choice([0.05, 0.1, 0.15]),  # Lower dropout for simple models
                'batch_size': random.choice([16, 32, 48]),  # Smaller batches
                'update_target_every': random.choice([50, 75, 100]),
                'min_epsilon': random.choice([0.2, 0.3, 0.4]),  # Higher minimum epsilon
                'epsilon_decay': random.choice([0.997, 0.998, 0.999])  # Slower decay
            }
        elif complexity == 'medium':
            # Medium complexity models
            config = {
                'learning_rate': random.choice([0.0005, 0.001, 0.0015, 0.002]),
                'gamma': random.choice([0.92, 0.95, 0.97]),
                'epsilon': random.choice([0.5, 0.6, 0.7]),  # Moderate exploration
                'hidden_sizes': random.choice([
                    [256, 128],
                    [256, 256],
                    [512, 256],
                    [384, 192],
                ]),
                'dropout_rate': random.choice([0.15, 0.2, 0.25]),
                'batch_size': random.choice([32, 64, 96]),
                'update_target_every': random.choice([75, 100, 125]),
                'min_epsilon': random.choice([0.05, 0.1, 0.15]),  # Moderate minimum
                'epsilon_decay': random.choice([0.995, 0.996, 0.997])
            }
        else:  # complex
            # Complex models with lower exploration
            config = {
                'learning_rate': random.choice([0.0001, 0.0005, 0.001]),
                'gamma': random.choice([0.95, 0.97, 0.99]),
                'epsilon': random.choice([0.3, 0.4, 0.5]),  # Lower initial exploration
                'hidden_sizes': random.choice([
                    [512, 512, 256],
                    [512, 512, 512],
                    [1024, 512, 256],
                    [768, 384, 192],
                ]),
                'dropout_rate': random.choice([0.2, 0.25, 0.3]),
                'batch_size': random.choice([64, 96, 128]),
                'update_target_every': random.choice([100, 150, 200]),
                'min_epsilon': random.choice([0.01, 0.02, 0.05]),  # Low minimum
                'epsilon_decay': random.choice([0.993, 0.994, 0.995])  # Faster decay
            }
        
        configs.append(config)
    
    # Ensure we have at least one of each complexity type if we have enough configs
    if num_configs >= 3:
        # Override first three configs with one of each type
        # Simple config
        configs[0] = {
            'learning_rate': random.choice([0.002, 0.003]),
            'gamma': random.choice([0.9, 0.95]),
            'epsilon': 0.9,  # High exploration like you mentioned
            'hidden_sizes': [128],  # Simple network like you mentioned
            'dropout_rate': 0.1,
            'batch_size': 32,
            'update_target_every': 75,
            'min_epsilon': 0.3,  # High minimum like you mentioned
            'epsilon_decay': 0.999  # Very slow decay like you mentioned
        }
        
        # Medium config
        configs[1] = {
            'learning_rate': random.choice([0.001, 0.0015]),
            'gamma': 0.95,
            'epsilon': 0.6,
            'hidden_sizes': [256, 128],
            'dropout_rate': 0.2,
            'batch_size': 64,
            'update_target_every': 100,
            'min_epsilon': 0.1,
            'epsilon_decay': 0.996
        }
        
        # Complex config
        configs[2] = {
            'learning_rate': random.choice([0.0005, 0.001]),
            'gamma': 0.97,
            'epsilon': 0.4,
            'hidden_sizes': [512, 512, 256],
            'dropout_rate': 0.25,
            'batch_size': 96,
            'update_target_every': 150,
            'min_epsilon': 0.02,
            'epsilon_decay': 0.994
        }
    
    best_config = None
    best_score = -float('inf')
    best_model = None
    results = []
    
    for idx, config in enumerate(configs):
        # Determine complexity for display
        if len(config['hidden_sizes']) == 1 and config['hidden_sizes'][0] <= 256:
            complexity = "Simple"
        elif len(config['hidden_sizes']) <= 2 and max(config['hidden_sizes']) <= 512:
            complexity = "Medium"
        else:
            complexity = "Complex"
        
        print(f"\nConfiguration {idx + 1}/{num_configs} ({complexity})")
        print(f"Testing: Hidden={config['hidden_sizes']}, LR={config['learning_rate']}, "
              f"Epsilon={config['epsilon']}->{config['min_epsilon']}, "
              f"Decay={config['epsilon_decay']}")
        
        # Train using self-play with this configuration
        ai_model = train_ai_advanced(
            num_episodes=episodes_per_config,
            config=config,
            verbose=False,
            num_players=num_players
        )
        
        # Quick evaluation
        print(f"Evaluating configuration {idx + 1}...")
        win_rate, avg_earnings = evaluate_ai_full(
            ai_model, 
            num_games=eval_games, 
            num_players=num_players
        )
        
        # Calculate score
        score = win_rate + avg_earnings / 100
        
        # Bonus for high win rate (consistency)
        if win_rate > 50:
            score += (win_rate - 50) * 0.5
        
        results.append({
            'config': config,
            'complexity': complexity,
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
    print(f"  Hidden layers: {best_config['hidden_sizes']}")
    print(f"  Learning rate: {best_config['learning_rate']}")
    print(f"  Gamma: {best_config['gamma']}")
    print(f"  Epsilon: {best_config['epsilon']} -> {best_config['min_epsilon']}")
    print(f"  Epsilon decay: {best_config['epsilon_decay']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Best score: {best_score:.2f}")
    
    # Show all results sorted by score
    print("\nAll configurations ranked:")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. {result['complexity']}: Hidden={result['config']['hidden_sizes']}, "
              f"Win rate={result['win_rate']:.1f}%, "
              f"Earnings=${result['earnings']:.0f}, Score={result['score']:.2f}")
    
    # Show complexity analysis
    simple_scores = [r['score'] for r in results if r['complexity'] == 'Simple']
    medium_scores = [r['score'] for r in results if r['complexity'] == 'Medium']
    complex_scores = [r['score'] for r in results if r['complexity'] == 'Complex']
    
    if simple_scores or medium_scores or complex_scores:
        print("\nComplexity Analysis:")
        if simple_scores:
            print(f"  Simple models: Avg score = {np.mean(simple_scores):.2f}")
        if medium_scores:
            print(f"  Medium models: Avg score = {np.mean(medium_scores):.2f}")
        if complex_scores:
            print(f"  Complex models: Avg score = {np.mean(complex_scores):.2f}")
    
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
    """Self-play training where multiple AIs with same config train against each other"""
    if verbose:
        print(f"Training {num_players} AIs using self-play for {num_episodes} episodes...")
    
    # Use default config if none provided
    if config is None:
        config = {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon': 0.5,
            'hidden_sizes': [512, 512, 256],
            'dropout_rate': 0.3,
            'batch_size': 64,
            'update_target_every': 100,
            'min_epsilon': 0.01,
            'epsilon_decay': 0.995
        }
    
    # Create multiple AIs with the same configuration
    ai_models = []
    for i in range(num_players):
        # Each AI gets the same config but different random initialization
        ai = PokerAI(config=config.copy())
        
        # Slightly vary exploration rates to encourage diversity
        ai.epsilon = config['epsilon'] * random.uniform(0.9, 1.1)
        ai.epsilon = min(1.0, max(0.1, ai.epsilon))  # Keep in valid range
        
        ai_models.append(ai)
    
    # Warm-up phase with random play to establish basic strategies
    if verbose:
        print("Warm-up phase...")
    for ai in ai_models:
        warmup_training(ai, num_episodes=min(50, num_episodes // 20))
    
    training_game = TexasHoldEmTraining(num_players=num_players)
    
    # Track wins for each AI
    wins = {f'AI_{i}': 0 for i in range(num_players)}
    recent_performances = {f'AI_{i}': [] for i in range(num_players)}
    
    for episode in range(num_episodes):
        # Rotate starting positions to ensure fairness
        rotation = episode % num_players
        rotated_models = ai_models[rotation:] + ai_models[:rotation]
        
        # Play multiple hands per episode
        hands_per_episode = 5 if episode < num_episodes // 2 else 10
        episode_wins = {f'AI_{i}': 0 for i in range(num_players)}
        
        for hand in range(hands_per_episode):
            training_game.reset_game()
            winners = training_game.simulate_hand(rotated_models)
            
            # Track wins
            for winner in winners:
                for i, model in enumerate(rotated_models):
                    if winner.ai_model == model:
                        # Account for rotation when tracking wins
                        original_idx = (i - rotation) % num_players
                        wins[f'AI_{original_idx}'] += 1
                        episode_wins[f'AI_{original_idx}'] += 1
        
        # Update recent performance tracking
        for i in range(num_players):
            recent_performances[f'AI_{i}'].append(episode_wins[f'AI_{i}'] / hands_per_episode)
            if len(recent_performances[f'AI_{i}']) > 100:
                recent_performances[f'AI_{i}'].pop(0)
        
        # Train all models
        for i, ai in enumerate(ai_models):
            if len(ai.memory) > ai.batch_size * 2:
                # Adaptive training based on performance
                train_iterations = 5
                
                # Train more if this AI is struggling
                if len(recent_performances[f'AI_{i}']) >= 20:
                    recent_avg = np.mean(recent_performances[f'AI_{i}'][-20:])
                    if recent_avg < 1.0 / num_players:  # Below expected win rate
                        train_iterations = 8
                    elif recent_avg < 0.5 / num_players:  # Way below expected
                        train_iterations = 10
                
                for _ in range(train_iterations):
                    ai.replay()
            
            # Decay epsilon
            ai.decay_epsilon()
        
        # Periodic status update
        if verbose and episode % 100 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            for i in range(num_players):
                if recent_performances[f'AI_{i}']:
                    recent_win_rate = np.mean(recent_performances[f'AI_{i}'][-20:]) * 100
                    print(f"  AI_{i}: Recent win rate: {recent_win_rate:.1f}%, "
                          f"Epsilon: {ai_models[i].epsilon:.3f}")
    
    if verbose:
        print(f"\nSelf-play training complete!")
        print("Win distribution:")
        total_wins = sum(wins.values())
        for i in range(num_players):
            win_pct = (wins[f'AI_{i}'] / total_wins * 100) if total_wins > 0 else 0
            print(f"  AI_{i}: {wins[f'AI_{i}']} wins ({win_pct:.1f}%)")
    
    # Evaluate all models against each other and random opponents
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATING ALL SELF-PLAY MODELS")
        print("=" * 60)
    
    best_model = None
    best_score = -float('inf')
    best_idx = -1
    evaluation_results = []
    
    for i, model in enumerate(ai_models):
        if verbose:
            print(f"\nEvaluating AI_{i}...")
        
        # Save original epsilon
        original_epsilon = model.epsilon
        model.epsilon = 0  # No exploration during evaluation
        
        # Test against random opponents
        if verbose:
            print(f"  Testing against random opponents...")
        random_win_rate, random_earnings = evaluate_ai_full(
            model, 
            num_games=30, 
            num_players=num_players
        )
        
        # Test against other self-play models (round-robin)
        if verbose:
            print(f"  Testing against other trained models...")
        self_play_wins = 0
        self_play_games = 20
        
        for game_num in range(self_play_games):
            # Create a game with this model and other trained models
            test_game = TexasHoldEmTraining(num_players=num_players)
            
            # Mix of this model and other trained models
            test_models = [model] + random.sample([m for j, m in enumerate(ai_models) if j != i], 
                                                  min(num_players - 1, len(ai_models) - 1))
            # Fill remaining slots with random models if needed
            while len(test_models) < num_players:
                test_models.append(model)  # Add more copies of the model being tested
            
            random.shuffle(test_models)
            
            test_game.reset_game()
            winners = test_game.simulate_hand(test_models)
            
            for winner in winners:
                if winner.ai_model == model:
                    self_play_wins += 1
        
        self_play_win_rate = (self_play_wins / self_play_games) * 100
        
        # Combined score: weight both random and self-play performance
        # Higher weight on random performance as it shows generalization
        score = (random_win_rate * 0.6) + (self_play_win_rate * 0.4) + (random_earnings / 100)
        
        evaluation_results.append({
            'model_idx': i,
            'random_win_rate': random_win_rate,
            'random_earnings': random_earnings,
            'self_play_win_rate': self_play_win_rate,
            'score': score
        })
        
        if verbose:
            print(f"  AI_{i} Results:")
            print(f"    vs Random: Win rate={random_win_rate:.1f}%, Earnings=${random_earnings:.0f}")
            print(f"    vs Self-play: Win rate={self_play_win_rate:.1f}%")
            print(f"    Combined Score: {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_idx = i
        
        # Restore original epsilon
        model.epsilon = original_epsilon
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"BEST MODEL: AI_{best_idx} with score {best_score:.2f}")
        print("=" * 60)
        
        # Show ranking
        print("\nAll models ranked by score:")
        sorted_results = sorted(evaluation_results, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            print(f"  {rank}. AI_{result['model_idx']}: Score={result['score']:.2f}, "
                  f"Random WR={result['random_win_rate']:.1f}%, "
                  f"Self-play WR={result['self_play_win_rate']:.1f}%")
    
    # Ensure the best model has the correct config
    best_model.config = config
    
    return best_model

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