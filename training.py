import random
import numpy as np
from poker_ai import PokerAI
from game_engine import TexasHoldEmTraining
from player import Player
from card_deck import Card
import os
from create_strategy_bots import load_strategy_bots, create_all_strategy_bots
from game_constants import Action
from neural_network import Experience

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

def train_ai_with_strategy_diversity(num_episodes=1000, config=None, verbose=True, 
                                    num_players=4, use_strategy_bots=True):
    """
    Train AI using diverse strategy bots and reward shaping to prevent all-in spam
    """
    if verbose:
        print(f"Training AI with strategy diversity for {num_episodes} episodes...")
    
    # Use default config if none provided
    if config is None:
        config = {
            'learning_rate': 0.0005,  # Lower for stability
            'gamma': 0.95,
            'epsilon': 0.5,
            'hidden_sizes': [256, 128],  # Moderate complexity
            'dropout_rate': 0.2,
            'batch_size': 64,
            'update_target_every': 150,
            'min_epsilon': 0.05,
            'epsilon_decay': 0.997
        }
    
    # Load or create strategy bots
    strategy_bots = {}
    if use_strategy_bots:
        strategy_bots = load_strategy_bots('strategy_bots')
        if not strategy_bots:
            if verbose:
                print("Strategy bots not found. Creating them now...")
            create_all_strategy_bots(save_dir='strategy_bots', episodes=200)
            strategy_bots = load_strategy_bots('strategy_bots')
    
    # Create the AI to train
    ai_model = PokerAI(config=config.copy())
    
    # Warm-up phase against easier opponents
    #if verbose:
        #print("\nPhase 1: Warm-up training...")
    #warmup_training(ai_model, num_episodes=min(100, num_episodes // 10))
    
    training_game = TexasHoldEmTraining(num_players=num_players)
    
    wins = 0
    recent_all_ins = []
    recent_wins = []
    
    # Curriculum learning phases
    phases = [
        (num_episodes // 3, ['conservative', 'balanced']),  # Phase 1
        (num_episodes // 3, ['conservative', 'balanced', 'aggressive']),  # Phase 2
        (num_episodes - 2 * (num_episodes // 3), ['conservative', 'balanced', 'aggressive', 'mixed'])  # Phase 3
    ]
    
    episode_count = 0
    
    for phase_episodes, opponent_types in phases:
        phase_num = phases.index((phase_episodes, opponent_types)) + 1
        if verbose:
            print(f"\nPhase {phase_num}: Training with {opponent_types}...")
        
        for episode in range(phase_episodes):
            episode_count += 1
            
            # Build opponent pool
            ai_models = [ai_model]
            
            for i in range(num_players - 1):
                if strategy_bots and random.random() < 0.7:  # 70% chance to use strategy bot
                    if 'mixed' in opponent_types:
                        # Mix all types
                        bot_type = random.choice(list(strategy_bots.keys()))
                    else:
                        # Use specified types
                        available = [t for t in opponent_types if t in strategy_bots]
                        if available:
                            bot_type = random.choice(available)
                            ai_models.append(strategy_bots[bot_type])
                        else:
                            # Fallback to random
                            random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                                       'gamma': 0, 'hidden_sizes': [64],
                                                       'dropout_rate': 0, 'batch_size': 1,
                                                       'update_target_every': 1000000,
                                                       'min_epsilon': 1.0, 'epsilon_decay': 1.0})
                            ai_models.append(random_ai)
                else:
                    # Use random AI
                    random_ai = PokerAI(config={'epsilon': random.uniform(0.5, 1.0),
                                               'learning_rate': 0, 'gamma': 0,
                                               'hidden_sizes': [64], 'dropout_rate': 0,
                                               'batch_size': 1, 'update_target_every': 1000000,
                                               'min_epsilon': 1.0, 'epsilon_decay': 1.0})
                    ai_models.append(random_ai)
            
            # Rotate positions
            rotation = episode % num_players
            ai_models = ai_models[rotation:] + ai_models[:rotation]
            
            # Track all-ins this episode
            episode_all_ins = 0
            
            # Play multiple hands
            hands_per_episode = 5
            episode_wins = 0
            
            for hand in range(hands_per_episode):
                training_game.reset_game()
                winners = training_game.simulate_hand(ai_models)
                
                # Count all-ins
                if hasattr(training_game, 'action_history'):
                    for action in training_game.action_history:
                        if action == Action.ALL_IN:
                            episode_all_ins += 1
                
                # Apply reward shaping
                for i, model in enumerate(ai_models):
                    if model == ai_model:
                        player = training_game.players[i]
                        
                        # Calculate shaped reward
                        base_reward = 0
                        if player in winners:
                            base_reward = 1.0
                            episode_wins += 1
                        else:
                            base_reward = -0.5
                        
                        # Penalize all-ins heavily
                        if episode_all_ins > 0:
                            all_in_penalty = -0.3 * min(episode_all_ins, 5)  # Cap penalty
                            base_reward += all_in_penalty
                        # Update last experience with shaped reward
                        if len(ai_model.memory.buffer) > 0:
                            last_exp = ai_model.memory.buffer[-1]
                            # Check if last action was all-in
                            if last_exp.action == Action.ALL_IN.value:
                                base_reward -= 0.5  # Extra penalty for this AI going all-in
                            
                            ai_model.memory.buffer[-1] = Experience(  # Not ai_model.memory.Experience
                                last_exp.state, last_exp.action, base_reward,
                                last_exp.next_state, last_exp.done
                            )
            
            recent_all_ins.append(episode_all_ins)
            recent_wins.append(episode_wins / hands_per_episode)
            
            # Keep recent history bounded
            if len(recent_all_ins) > 100:
                recent_all_ins.pop(0)
                recent_wins.pop(0)
            
            # Train the model
            if len(ai_model.memory) > ai_model.batch_size * 2:
                # More training iterations if doing poorly
                train_iterations = 5
                if len(recent_wins) > 20:
                    recent_win_rate = np.mean(recent_wins[-20:])
                    if recent_win_rate < 0.2:  # Less than 20% win rate
                        train_iterations = 8
                
                for _ in range(train_iterations):
                    ai_model.replay()
            
            # Decay epsilon
            ai_model.decay_epsilon()
            
            # Progress update
            if verbose and episode_count % 100 == 0:
                avg_all_ins = np.mean(recent_all_ins[-50:]) if recent_all_ins else 0
                avg_wins = np.mean(recent_wins[-50:]) if recent_wins else 0
                print(f"  Episode {episode_count}/{num_episodes}")
                print(f"    Win rate: {avg_wins*100:.1f}%")
                print(f"    All-ins per episode: {avg_all_ins:.1f}")
                print(f"    Epsilon: {ai_model.epsilon:.3f}")
                
                # Warn if still using too many all-ins
                if avg_all_ins > 10:
                    print("    ⚠️ Still using excessive all-ins!")
    
    if verbose:
        print(f"\nTraining complete!")
        final_all_ins = np.mean(recent_all_ins[-50:]) if recent_all_ins else 0
        final_wins = np.mean(recent_wins[-50:]) if recent_wins else 0
        print(f"Final stats: Win rate={final_wins*100:.1f}%, All-ins/episode={final_all_ins:.1f}")
    
    return ai_model


# Update the original train_ai_advanced to use the new version
def train_ai_vs_strong_opponents(num_episodes=1000, config=None, verbose=True, num_players=4):
    """
    Train AI against strong pre-trained models as opponents
    """
    if verbose:
        print(f"Training AI against strong opponents for {num_episodes} episodes...")
    
    # Use default config if none provided
    if config is None:
        config = {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon': 0.6,  # Higher exploration when facing strong opponents
            'hidden_sizes': [512, 256, 128],  # Larger network to compete
            'dropout_rate': 0.2,
            'batch_size': 64,
            'update_target_every': 100,
            'min_epsilon': 0.05,
            'epsilon_decay': 0.998
        }
    
    # Load strong opponent models
    strong_opponents = []
    strong_model_paths = [
        'tuned_ai_v4.pth',
        'tuned_ai_v2.pth', 
        'poker_ai_tuned.pth',
        'standard_ai_v3.pth',
        'standard_ai_v4.pth',
        'balanced_ai_v3.pth',
        'best_final_2.pth',
        'opponent_eval_v2.pth'
    ]
    
    for path in strong_model_paths:
        if os.path.exists(path):
            try:
                opponent = PokerAI()
                opponent.load(path)
                opponent.epsilon = 0.05  # Small exploration for variety
                strong_opponents.append((opponent, path))
                if verbose:
                    print(f"  Loaded strong opponent: {path}")
            except Exception as e:
                if verbose:
                    print(f"  Could not load {path}: {e}")
    
    if not strong_opponents:
        print("No strong opponents found! Falling back to strategy diversity training.")
        return train_ai_with_strategy_diversity(num_episodes, config, verbose, num_players)
    
    if verbose:
        print(f"  Training against {len(strong_opponents)} strong opponents")
    
    # Create the AI to train
    ai_model = PokerAI(config=config.copy())
    training_game = TexasHoldEmTraining(num_players=num_players)
    
    wins = 0
    recent_wins = []
    phase_size = num_episodes // 4
    
    for episode in range(num_episodes):
        # Progressive difficulty - start with weaker opponents, add stronger ones
        current_phase = episode // phase_size
        available_opponents = strong_opponents[:min(len(strong_opponents), current_phase + 2)]
        
        # Select opponents for this episode
        opponents = []
        for i in range(num_players - 1):
            if available_opponents:
                opponent, name = random.choice(available_opponents)
                opponents.append(opponent)
            else:
                # Fallback random opponent
                random_config = {
                    'epsilon': 0.8, 'learning_rate': 0, 'gamma': 0,
                    'hidden_sizes': [128], 'dropout_rate': 0, 'batch_size': 1,
                    'update_target_every': 1000000, 'min_epsilon': 0.8, 'epsilon_decay': 1.0
                }
                opponents.append(PokerAI(config=random_config))
        
        # Play multiple hands per episode
        episode_wins = 0
        for hand in range(8):  # More hands per episode for stronger learning
            training_game.reset_game()
            all_models = [ai_model] + opponents
            random.shuffle(all_models)
            
            winners = training_game.simulate_hand(all_models)
            
            # Check if our AI won
            ai_position = all_models.index(ai_model)
            ai_player = training_game.players[ai_position]
            if ai_player in winners:
                episode_wins += 1
                wins += 1
        
        # Train the AI
        if len(ai_model.memory) > ai_model.batch_size:
            for _ in range(3):  # More training per episode
                ai_model.replay()
        
        ai_model.decay_epsilon()
        
        # Track recent performance
        recent_wins.append(episode_wins)
        if len(recent_wins) > 100:
            recent_wins.pop(0)
        
        # Progress reporting
        if verbose and episode % 100 == 0:
            recent_wr = np.mean(recent_wins) * 100 / 8 if recent_wins else 0
            print(f"  Episode {episode}/{num_episodes}, Win Rate: {recent_wr:.1f}%, Epsilon: {ai_model.epsilon:.3f}")
    
    # Final stats
    overall_win_rate = (wins / (num_episodes * 8)) * 100
    if verbose:
        print(f"\nTraining complete!")
        print(f"Overall win rate vs strong opponents: {overall_win_rate:.1f}%")
        print(f"Final epsilon: {ai_model.epsilon:.3f}")
    
    return ai_model

def train_ai_advanced(num_episodes=1000, config=None, verbose=True, num_players=4):
    """
    Wrapper for backwards compatibility - now uses strong opponent training
    """
    return train_ai_vs_strong_opponents(num_episodes, config, verbose, num_players)

def evaluate_ai_full(ai_model, num_games=100, num_players=4, use_strong_opponents=True):
    """
    Full evaluation against either random players or strong opponents
    
    Args:
        ai_model: The model to evaluate
        num_games: Number of games to play
        num_players: Number of players per game
        use_strong_opponents: If True, use strong AI opponents; if False, use random
    """
    #use_strong_opponents=False # temporary testing
    print(f"Evaluating AI over {num_games} games against {'strong' if use_strong_opponents else 'random'} opponents...")
    
    original_epsilon = ai_model.epsilon
    ai_model.epsilon = 0
    
    # Try to load strong opponent models if requested
    strong_opponents = []
    if use_strong_opponents:
        # Try to find strong models in current directory
        strong_model_paths = [
            'tuned_ai_v2.pth',
            'tuned_ai_v4.pth', 
            'poker_ai_tuned.pth',
            'standard_ai_v3.pth'
        ]
        
        for path in strong_model_paths:
            if os.path.exists(path):
                try:
                    opponent = PokerAI()
                    opponent.load(path)
                    opponent.epsilon = 0
                    strong_opponents.append(opponent)
                    print(f"  Loaded strong opponent: {path}")
                except:
                    pass
        
        if not strong_opponents:
            print("  No strong opponents found, falling back to random")
            use_strong_opponents = False
    
    wins = 0
    total_earnings = 0
    games_survived = 0
    action_distribution = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'all_in': 0}
    total_actions = 0
    
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
            elif use_strong_opponents and strong_opponents:
                # Use a strong opponent
                opponent = random.choice(strong_opponents)
                ai_models.append(opponent)
            else:
                # Random AI (high epsilon, no learning)
                random_ai = PokerAI(config={'epsilon': 1.0, 'learning_rate': 0,
                                           'gamma': 0, 'hidden_sizes': [64],
                                           'dropout_rate': 0, 'batch_size': 1,
                                           'update_target_every': 1000000,
                                           'min_epsilon': 1.0, 'epsilon_decay': 1.0})
                ai_models.append(random_ai)
        
        # Track actions for this game
        pre_game_actions = 0
        
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
            
            # Track actions before hand
            if hasattr(training_game, 'action_history'):
                pre_hand_actions = len(training_game.action_history)
            
            # Simulate hand
            training_game.reset_game()
            winners = training_game.simulate_hand(ai_models)
            
            # Analyze actions taken by test model
            if hasattr(training_game, 'action_history'):
                # Only count actions from the test model (player 0)
                for idx, action in enumerate(training_game.action_history[pre_hand_actions:]):
                    # Simple heuristic: every num_players actions, one is from test model
                    if idx % num_players == 0:  
                        action_name = action.name.lower()
                        if action_name in action_distribution:
                            action_distribution[action_name] += 1
                            total_actions += 1
            
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
    
    # Show action distribution if available
    if total_actions > 0:
        print(f"\nAction Distribution:")
        for action, count in action_distribution.items():
            pct = (count / total_actions) * 100
            print(f"  {action.capitalize()}: {pct:.1f}%")
        
        # Warn about all-in strategy
        all_in_pct = (action_distribution['all_in'] / total_actions) * 100
        if all_in_pct > 50:
            print("\n⚠️ WARNING: Model exhibits excessive all-in behavior!")
            print("  This may work against random opponents but will fail against humans.")
    
    performance_ratio = win_rate / (100/num_players)
    if use_strong_opponents:
        if performance_ratio > 1.2:
            print(f"✓ AI performs {performance_ratio:.1f}x better than expected against strong opponents!")
        elif performance_ratio > 1.0:
            print(f"✓ AI performs {performance_ratio:.1f}x expected rate against strong opponents")
        else:
            print(f"✗ AI needs improvement ({performance_ratio:.1f}x expected against strong opponents)")
    else:
        if performance_ratio > 1.5:
            print(f"✓ AI is performing {performance_ratio:.1f}x better than random!")
        elif performance_ratio > 1.2:
            print(f"✓ AI is performing {performance_ratio:.1f}x better than random")
        else:
            print(f"✗ AI needs improvement (only {performance_ratio:.1f}x random performance)")
    
    return win_rate, avg_earnings