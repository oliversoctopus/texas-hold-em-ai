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
        # This will now return the best model among main and opponents
        ai_model = train_ai_advanced(
            num_episodes=episodes_per_config,
            config=config,
            verbose=False,
            num_players=num_players
        )
        
        # The returned model now has the correct config (might be from an opponent)
        actual_config = ai_model.config
        
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
                'original_config': config,  # Config we tried to train with
                'actual_config': actual_config,  # Config of the best model
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
            'original_config': config,
            'actual_config': actual_config,
            'win_rate': win_rate,
            'earnings': avg_earnings,
            'score': score
        })
        
        print(f"Results: Win rate={win_rate:.1f}%, Earnings=${avg_earnings:.0f}, Score={score:.2f}")
        
        # Check if actual config differs from original
        if actual_config != config:
            print(f"  Note: Best model was an opponent, not the main AI")
            print(f"  Actual config: LR={actual_config['learning_rate']}, "
                  f"Hidden={actual_config['hidden_sizes']}")
        
        if score > best_score:
            best_score = score
            best_config = actual_config  # Use the actual config of the best model
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
        config_note = ""
        if result['actual_config'] != result['original_config']:
            config_note = " (from opponent)"
        print(f"  {i}. Win rate: {result['win_rate']:.1f}%, "
              f"Earnings: ${result['earnings']:.0f}, Score: {result['score']:.2f}{config_note}")
    
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
    
    # Create multiple opponent AIs with different skill levels for diversity
    opponents = []
    opponent_configs = []  # Track configs for all opponents
    
    # Weak opponent (high exploration, simple network)
    weak_config = {
        'learning_rate': 0.001, 'gamma': 0.9, 
        'epsilon': 0.9, 'hidden_sizes': [128], 
        'dropout_rate': 0.1, 'batch_size': 32,
        'update_target_every': 100, 'min_epsilon': 0.7,
        'epsilon_decay': 0.999
    }
    opponents.append(PokerAI(config=weak_config))
    opponent_configs.append(weak_config)
    
    # Medium opponent (moderate exploration, decent network)
    medium_config = {
        'learning_rate': 0.001, 'gamma': 0.95, 
        'epsilon': 0.5, 'hidden_sizes': [256, 128], 
        'dropout_rate': 0.2, 'batch_size': 64,
        'update_target_every': 100, 'min_epsilon': 0.3,
        'epsilon_decay': 0.997
    }
    opponents.append(PokerAI(config=medium_config))
    opponent_configs.append(medium_config)
    
    # Strong opponent (low exploration, complex network) - variation of main config
    strong_config = config.copy() if config else {
        'learning_rate': 0.001, 'gamma': 0.99, 
        'epsilon': 0.3, 'hidden_sizes': [512, 256], 
        'dropout_rate': 0.3, 'batch_size': 64,
        'update_target_every': 100, 'min_epsilon': 0.05,
        'epsilon_decay': 0.995
    }
    # Make some variations to the strong opponent
    strong_config['learning_rate'] = strong_config['learning_rate'] * 0.5  # Slower learning
    strong_config['epsilon'] = 0.4  # More exploration than main
    opponents.append(PokerAI(config=strong_config))
    opponent_configs.append(strong_config)
    
    # Create additional opponents if needed for more players
    while len(opponents) < num_players - 1:
        # Create variations of existing configs
        base_config = random.choice([weak_config, medium_config, strong_config])
        variant_config = base_config.copy()
        # Slightly modify the config
        variant_config['learning_rate'] *= random.choice([0.5, 1.0, 2.0])
        variant_config['epsilon'] = min(1.0, variant_config['epsilon'] * random.uniform(0.8, 1.2))
        opponents.append(PokerAI(config=variant_config))
        opponent_configs.append(variant_config)
    
    # Warm-up ALL models equally
    if verbose:
        print("Warming up all models...")
    
    training_game = TexasHoldEmTraining(num_players=num_players)
    all_models = [main_ai] + opponents[:min(3, len(opponents))]  # Warm up main 3 opponents
    
    for warmup_episode in range(min(50, num_episodes // 20)):
        # Rotate through models for warm-up
        selected_models = random.sample(all_models * 2, num_players)[:num_players]
        training_game.reset_game()
        training_game.simulate_hand(selected_models)
        
        # Light training during warm-up for all models
        for model in all_models:
            if len(model.memory) > model.batch_size:
                model.replay()
    
    wins = {f'model_{i}': 0 for i in range(len(all_models))}
    recent_performance = {f'model_{i}': [] for i in range(len(all_models))}
    
    for episode in range(num_episodes):
        # Select models for this episode - ensure diversity
        if episode < num_episodes // 3:
            # Early: ensure main AI plays frequently but not always
            if random.random() < 0.7:  # 70% chance main AI plays
                selected_models = [main_ai]
                remaining_slots = num_players - 1
                selected_models.extend(random.sample(opponents * 2, remaining_slots)[:remaining_slots])
            else:
                # Let opponents play against each other sometimes
                selected_models = random.sample(opponents * 2, num_players)[:num_players]
        else:
            # Later: more balanced selection
            all_available = [main_ai] + opponents
            selected_models = random.sample(all_available * 2, num_players)[:num_players]
        
        random.shuffle(selected_models)
        
        # Play multiple hands per episode
        hands_per_episode = 5 if episode < num_episodes // 2 else 10
        
        for hand in range(hands_per_episode):
            training_game.reset_game()
            winners = training_game.simulate_hand(selected_models)
            
            # Track wins for all models
            for winner in winners:
                for i, model in enumerate(all_models):
                    if winner.ai_model == model:
                        wins[f'model_{i}'] += 1
                        if i < len(recent_performance):
                            if f'model_{i}' not in recent_performance:
                                recent_performance[f'model_{i}'] = []
                            recent_performance[f'model_{i}'].append(1)
                        break
            
            # Track losses
            for model in selected_models:
                if model not in [w.ai_model for w in winners]:
                    for i, m in enumerate(all_models):
                        if model == m:
                            if f'model_{i}' not in recent_performance:
                                recent_performance[f'model_{i}'] = []
                            recent_performance[f'model_{i}'].append(0)
                            break
        
        # Trim performance history
        for key in recent_performance:
            if len(recent_performance[key]) > 100:
                recent_performance[key] = recent_performance[key][-100:]
        
        # BALANCED TRAINING - Train all models that played
        models_to_train = list(set(selected_models))
        
        for model in models_to_train:
            if len(model.memory) > model.batch_size * 2:
                # Determine training intensity based on recent performance
                model_idx = all_models.index(model) if model in all_models else -1
                
                if model_idx >= 0 and f'model_{model_idx}' in recent_performance:
                    perf_history = recent_performance[f'model_{model_idx}']
                    if len(perf_history) >= 10:
                        recent_win_rate = np.mean(perf_history[-10:])
                        
                        # Adaptive training based on performance
                        if recent_win_rate < 0.2:  # Struggling
                            train_iterations = 8
                        elif recent_win_rate < 0.4:  # Below average
                            train_iterations = 5
                        else:  # Doing well
                            train_iterations = 3
                    else:
                        train_iterations = 5
                else:
                    train_iterations = 5
                
                # Don't overtrain - add some randomness to prevent overfitting
                if random.random() < 0.8:  # 80% chance to train
                    for _ in range(train_iterations):
                        model.replay()
        
        # Decay epsilon for all models that played
        if episode % 10 == 0:  # Decay less frequently
            for model in models_to_train:
                model.decay_epsilon()
        
        if verbose and episode % 100 == 0:
            # Report performance for main AI
            main_perf = recent_performance.get('model_0', [])
            if main_perf:
                main_win_rate = np.mean(main_perf[-50:]) * 100 if len(main_perf) >= 50 else np.mean(main_perf) * 100
            else:
                main_win_rate = 0
            
            print(f"Episode {episode}/{num_episodes}, Main AI win rate: {main_win_rate:.1f}%, "
                  f"Epsilon: {main_ai.epsilon:.3f}")
    
    if verbose:
        print(f"\nTraining complete!")
        print("Win distribution:")
        total_wins = sum(wins.values())
        for i, model in enumerate(all_models[:4]):  # Show first 4 models
            model_wins = wins.get(f'model_{i}', 0)
            win_pct = (model_wins / total_wins * 100) if total_wins > 0 else 0
            model_name = "Main AI" if i == 0 else f"Opponent {i}"
            print(f"  {model_name}: {model_wins} wins ({win_pct:.1f}%)")
    
    # ALWAYS evaluate all models and return the best one
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATING ALL TRAINED MODELS")
        print("=" * 60)
    
    # Evaluate main AI and first 3 opponents (the core training partners)
    all_models_to_eval = [('Main AI', main_ai, config)] + \
                        [(f'Opponent {i+1}', opp, opponent_configs[i]) 
                         for i, opp in enumerate(opponents[:min(3, len(opponents))])]
    
    best_model = None
    best_score = -float('inf')
    best_name = None
    best_model_config = None
    
    for name, model, model_config in all_models_to_eval:
        if verbose:
            print(f"\nEvaluating {name}...")
        
        original_epsilon = model.epsilon
        model.epsilon = 0  # No exploration during evaluation
        
        # More thorough evaluation for final selection
        eval_games = 30 if not verbose else 50
        win_rate, avg_earnings = evaluate_ai_full(model, num_games=eval_games, num_players=num_players)
        score = win_rate + avg_earnings / 100
        
        # Add bonus for consistency
        if win_rate > 40:
            score += (win_rate - 40) * 0.3
        
        if verbose:
            print(f"{name}: Win rate={win_rate:.1f}%, Earnings=${avg_earnings:.0f}, Score={score:.2f}")
            if model_config:
                print(f"  Config: LR={model_config['learning_rate']}, "
                      f"Hidden={model_config['hidden_sizes']}, "
                      f"Batch={model_config['batch_size']}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
            best_model_config = model_config
        
        # Restore original epsilon
        model.epsilon = original_epsilon
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_name} with score {best_score:.2f}")
        print("=" * 60)
    
    # Set the best model's config to be accurate
    best_model.config = best_model_config
    
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