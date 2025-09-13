import random
from game_constants import Action
from poker_ai import PokerAI
from game_engine import TexasHoldEm
from training import hyperparameter_tuning, train_ai_advanced, evaluate_ai_full
from player import Player
from advanced_evaluation import AdvancedEvaluator, evaluate_all_models
from create_strategy_bots import create_all_strategy_bots, load_strategy_bots

def main():
    """Main game loop"""
    print("=" * 60)
    print("TEXAS HOLD'EM with Advanced PyTorch AI")
    print("=" * 60)
    
    print("\n1. Train CFR AI (Counterfactual Regret Minimization)")
    print("2. Train new AI (standard DQN)")
    print("3. Train with hyperparameter tuning")
    print("4. Train with strategy diversity (anti all-in)")
    print("5. Train against strong opponents")
    print("6. Create strategy bots")
    print("7. Load existing AI")
    print("8. Play without AI")
    
    choice = input("\nChoose option (1-8): ")
    
    ai_model = None
    
    if choice == '1':
        print("\nCFR (Counterfactual Regret Minimization) Training")
        print("This uses game theory optimal strategies like real poker AIs!")
        
        iterations = int(input("CFR iterations (10000-100000 recommended): ") or "25000")
        players = int(input("Number of players for games (2-6): ") or "4")
        
        print(f"\nTraining CFR AI with {iterations} iterations...")
        from cfr_player import train_cfr_ai, evaluate_cfr_ai
        
        cfr_ai = train_cfr_ai(
            iterations=iterations,
            num_players=players,
            verbose=True
        )
        
        # Evaluate CFR AI
        eval_choice = input("\nEvaluate CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            opponent_choice = input("Use random opponents for baseline testing? (y/n): ")
            use_random = opponent_choice.lower() == 'y'
            evaluate_cfr_ai(cfr_ai, num_games=100, num_players=players, use_random_opponents=use_random)
        
        # Save CFR model
        save_choice = input("\nSave CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: cfr_poker_ai.pkl): ") or "cfr_poker_ai.pkl"
            cfr_ai.save(filename)
            print(f"CFR model saved to {filename}")
        
        # Convert CFR to compatible format for gameplay
        print("\nCreating CFR player for gameplay...")
        from cfr_player import CFRPlayer
        cfr_player = CFRPlayer(cfr_ai)
        
        # Create a wrapper that looks like the old AI for compatibility
        class CFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0  # CFR doesn't use epsilon
                
            def choose_action(self, state, valid_actions):
                # Convert state to CFR format
                game_state = {
                    'hole_cards': getattr(state, 'hole_cards', []),
                    'community_cards': getattr(state, 'community_cards', []),
                    'pot_size': getattr(state, 'pot_size', 0),
                    'to_call': getattr(state, 'to_call', 0),
                    'stack_size': getattr(state, 'stack_size', 1000),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4),
                    'action_history': getattr(state, 'action_history', [])
                }
                return self.cfr_player.choose_action(game_state)
            
            def get_raise_size(self, state):
                game_state = {
                    'pot_size': getattr(state, 'pot_size', 0),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4)
                }
                return self.cfr_player.get_raise_size(game_state)
        
        ai_model = CFRWrapper(cfr_player)
        
    elif choice == '2':
        print("\nStandard training using self-play")
        episodes = int(input("Training episodes (1000-3000 recommended): ") or "1000")
        players = int(input("Number of players for training (2-6): ") or "4")
        
        print(f"\nTraining {players} AIs using self-play...")
        ai_model = train_ai_advanced(num_episodes=episodes, num_players=players)
        
        # Evaluate
        eval_choice = input("\nEvaluate AI? (y/n): ")
        if eval_choice.lower() == 'y':
            evaluate_ai_full(ai_model, num_games=100, num_players=players)
        
        # Save
        save_choice = input("\nSave model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: poker_ai.pth): ") or "poker_ai.pth"
            ai_model.save(filename)
            print(f"Model saved to {filename}")

    elif choice == '3':
        print("\nHyperparameter tuning with self-play training")
        print("This will test multiple configurations to find the best one.")
        
        num_configs = int(input("Number of configurations to test (3-10): ") or "5")
        episodes = int(input("Training episodes per config (100-500): ") or "200")
        players = int(input("Number of players for training (2-6): ") or "4")
        
        best_config, results, best_model = hyperparameter_tuning(
            num_configs=num_configs,
            episodes_per_config=episodes,
            eval_games=30,
            num_players=players
        )
        
        # Option to save the best model from hyperparameter tuning
        print("\n" + "=" * 60)
        print("SAVE CHECKPOINT")
        print("=" * 60)
        save_checkpoint = input("Save the best model from hyperparameter tuning before additional training? (y/n): ")
        if save_checkpoint.lower() == 'y':
            checkpoint_filename = input("Checkpoint filename (default: poker_ai_checkpoint.pth): ") or "poker_ai_checkpoint.pth"
            best_model.save(checkpoint_filename)
            print(f"Checkpoint saved to {checkpoint_filename}")
        
        # Ask if user wants to continue with additional training
        continue_training = input("\nContinue with additional training? (y/n): ")
        if continue_training.lower() != 'y':
            print("Training complete. Using model from hyperparameter tuning.")
            ai_model = best_model
        else:
            # Continue training the ACTUAL best model, not starting over
            print("\nContinuing training with the best model found...")
            print(f"Best model's configuration: {best_config}")
            additional_episodes = int(input("Additional training episodes (500-3000): ") or "1000")
            
            # Adjust the best model's epsilon based on how much it has already trained
            best_model.epsilon = best_config['epsilon'] * (best_config['epsilon_decay'] ** episodes)
            print(f"Continuing from epsilon: {best_model.epsilon:.3f}")
            
            # Create copies of the best model for challenging self-play
            print(f"Creating {players - 1} copies of the best model for self-play training...")
            training_partners = []
            
            for i in range(players - 1):
                # Create a new AI with the same config
                partner = PokerAI(config=best_config.copy())
                
                # Copy the trained network weights from the best model
                partner._init_networks(best_model.input_size)
                partner.q_network.load_state_dict(best_model.q_network.state_dict())
                partner.target_network.load_state_dict(best_model.target_network.state_dict())
                
                # Copy the optimizer state for continued training
                partner.optimizer.load_state_dict(best_model.optimizer.state_dict())
                
                # Copy training progress
                partner.updates = best_model.updates
                partner.input_size = best_model.input_size
                
                # Vary epsilon slightly to encourage strategy divergence
                partner.epsilon = best_model.epsilon * random.uniform(0.8, 1.2)
                partner.epsilon = min(0.5, max(0.01, partner.epsilon))  # Keep in reasonable range
                
                # Each partner gets a slightly different learning rate to encourage divergence
                for param_group in partner.optimizer.param_groups:
                    param_group['lr'] = best_config['learning_rate'] * random.uniform(0.8, 1.2)
                
                training_partners.append(partner)
                print(f"  Partner {i+1}: epsilon={partner.epsilon:.3f}, lr={partner.optimizer.param_groups[0]['lr']:.6f}")
            
            # Continue training using self-play with these strong partners
            from game_engine import TexasHoldEmTraining
            import numpy as np
            
            training_game = TexasHoldEmTraining(num_players=players)
            
            print(f"\nContinuing self-play training for {additional_episodes} episodes...")
            print("All models start from the same strong baseline and will diverge through training.")
            
            wins = {f'Model_{i}': 0 for i in range(players)}
            recent_performances = {f'Model_{i}': [] for i in range(players)}
            all_models = [best_model] + training_partners
            
            for episode in range(additional_episodes):
                # Rotate positions for fairness
                rotation = episode % players
                rotated_models = all_models[rotation:] + all_models[:rotation]
                
                # Play multiple hands
                hands_per_episode = 10
                episode_wins = {f'Model_{i}': 0 for i in range(players)}
                
                for hand in range(hands_per_episode):
                    training_game.reset_game()
                    winners = training_game.simulate_hand(rotated_models)
                    
                    # Track wins
                    for winner in winners:
                        for i, model in enumerate(rotated_models):
                            if winner.ai_model == model:
                                # Account for rotation when tracking wins
                                original_idx = (i - rotation) % players
                                wins[f'Model_{original_idx}'] += 1
                                episode_wins[f'Model_{original_idx}'] += 1
                
                # Update performance tracking
                for i in range(players):
                    recent_performances[f'Model_{i}'].append(episode_wins[f'Model_{i}'] / hands_per_episode)
                    if len(recent_performances[f'Model_{i}']) > 100:
                        recent_performances[f'Model_{i}'].pop(0)
                
                # Train all models
                for i, model in enumerate(all_models):
                    if len(model.memory) > model.batch_size * 2:
                        # Adaptive training based on performance
                        train_iterations = 5
                        if len(recent_performances[f'Model_{i}']) >= 20:
                            recent_avg = np.mean(recent_performances[f'Model_{i}'][-20:])
                            expected_rate = 1.0 / players
                            if recent_avg < expected_rate * 0.8:  # Below 80% of expected
                                train_iterations = 8
                            elif recent_avg > expected_rate * 1.5:  # Above 150% of expected
                                train_iterations = 3  # Train less if dominating
                        
                        for _ in range(train_iterations):
                            model.replay()
                    
                    # Decay epsilon
                    model.decay_epsilon()
                
                # Status update
                if episode % 100 == 0:
                    print(f"\nEpisode {episode}/{additional_episodes}")
                    for i in range(players):
                        if recent_performances[f'Model_{i}']:
                            recent_win_rate = np.mean(recent_performances[f'Model_{i}'][-20:]) * 100
                            model_name = "Best (original)" if i == 0 else f"Partner {i}"
                            print(f"  {model_name}: Win rate: {recent_win_rate:.1f}%, Epsilon: {all_models[i].epsilon:.3f}")
            
            print(f"\nAdditional training complete!")
            
            # Evaluate all models to find the best one after continued training
            print("\n" + "=" * 60)
            print("EVALUATING ALL MODELS AFTER CONTINUED TRAINING")
            print("=" * 60)
            
            best_final_model = None
            best_final_score = -float('inf')
            best_final_idx = -1
            
            for i, model in enumerate(all_models):
                model_name = "Original best" if i == 0 else f"Partner {i}"
                print(f"\nEvaluating {model_name}...")
                
                # Save epsilon and disable exploration for evaluation
                original_epsilon = model.epsilon
                model.epsilon = 0
                
                win_rate, avg_earnings = evaluate_ai_full(model, num_games=30, num_players=players)
                score = win_rate + avg_earnings / 100
                
                print(f"  {model_name}: Win rate={win_rate:.1f}%, Earnings=${avg_earnings:.0f}, Score={score:.2f}")
                
                if score > best_final_score:
                    best_final_score = score
                    best_final_model = model
                    best_final_idx = i
                
                # Restore epsilon
                model.epsilon = original_epsilon
            
            model_name = "Original best" if best_final_idx == 0 else f"Partner {best_final_idx}"
            print(f"\nBest model after continued training: {model_name} with score {best_final_score:.2f}")
            
            ai_model = best_final_model
        
        # Full evaluation
        eval_choice = input("\nEvaluate final model? (y/n): ")
        if eval_choice.lower() == 'y':
            eval_choice = input("\nEvaluation type: (1) Random opponents, (2) Strong opponents, (3) Comprehensive: ")
            if eval_choice == '1':
                evaluate_ai_full(ai_model, num_games=100, num_players=players, use_strong_opponents=False)
            elif eval_choice == '2':
                evaluate_ai_full(ai_model, num_games=100, num_players=players, use_strong_opponents=True)
            elif eval_choice == '3':
                evaluator = AdvancedEvaluator(['tuned_ai_v2.pth', 'tuned_ai_v4.pth', 'poker_ai_tuned.pth'])
                evaluator.comprehensive_evaluation(ai_model, model_name="Current Model", num_games=200)
        
        # Save final model
        save_choice = input("\nSave final model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: poker_ai_final.pth): ") or "poker_ai_final.pth"
            ai_model.save(filename)
            print(f"Final model saved to {filename}")
    
    elif choice == '4':
        print("\nTraining with strategy diversity (prevents all-in spam)")
        
        # Check if strategy bots exist
        strategy_bots = load_strategy_bots('strategy_bots')
        if not strategy_bots:
            create_bots = input("Strategy bots not found. Create them now? (y/n): ")
            if create_bots.lower() == 'y':
                print("\nCreating strategy bots...")
                create_all_strategy_bots(save_dir='strategy_bots', episodes=300)
            else:
                print("Training without strategy bots...")
        
        episodes = int(input("Training episodes (1000-3000 recommended): ") or "1000")
        players = int(input("Number of players for training (2-6): ") or "4")
        
        # Configuration for balanced play
        config = {
            'learning_rate': 0.0005,
            'gamma': 0.95,
            'epsilon': 0.5,
            'hidden_sizes': [256, 128],
            'dropout_rate': 0.2,
            'batch_size': 64,
            'update_target_every': 150,
            'min_epsilon': 0.05,
            'epsilon_decay': 0.997
        }
        
        print(f"\nTraining with strategy diversity...")
        from training import train_ai_with_strategy_diversity
        ai_model = train_ai_with_strategy_diversity(
            num_episodes=episodes, 
            config=config,
            num_players=players,
            use_strategy_bots=True
        )
        
        # Evaluate
        eval_choice = input("\nEvaluate AI? (y/n): ")
        if eval_choice.lower() == 'y':
            # Test against strategy bots
            print("\nEvaluating against strategy bots...")
            evaluate_ai_full(ai_model, num_games=100, num_players=players, use_strong_opponents=True)
        
        # Save
        save_choice = input("\nSave model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: balanced_poker_ai.pth): ") or "balanced_poker_ai.pth"
            ai_model.save(filename)
            print(f"Model saved to {filename}")
    
    elif choice == '5':
        print("\nTraining against strong opponents")
        episodes = int(input("Training episodes (1000-3000 recommended): ") or "1500")
        players = int(input("Number of players for training (2-6): ") or "4")
        
        print(f"\nTraining against pre-trained strong opponents...")
        from training import train_ai_vs_strong_opponents
        ai_model = train_ai_vs_strong_opponents(
            num_episodes=episodes,
            num_players=players
        )
        
        # Evaluate
        eval_choice = input("\nEvaluate AI? (y/n): ")
        if eval_choice.lower() == 'y':
            evaluate_ai_full(ai_model, num_games=10000, num_players=players, use_strong_opponents=True)
        
        # Save
        save_choice = input("\nSave model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: vs_strong_ai.pth): ") or "vs_strong_ai.pth"
            ai_model.save(filename)
            print(f"Model saved to {filename}")
    
    elif choice == '6':
        print("\nCreating strategy bots...")
        episodes = int(input("Training episodes per bot (200-500 recommended): ") or "300")
        create_all_strategy_bots(save_dir='strategy_bots', episodes=episodes)
        print("\nStrategy bots created in strategy_bots/ directory")
        return
    
    elif choice == '7':
        filename = input("Model filename (.pth for DQN, .pkl for CFR, default: poker_ai.pth): ") or "poker_ai.pth"
        
        # Check if it's a CFR model (.pkl) or DQN model (.pth)
        if filename.endswith('.pkl'):
            # Load CFR model
            try:
                from cfr_poker import CFRPokerAI
                from cfr_player import CFRPlayer
                
                print(f"Loading CFR model from {filename}...")
                cfr_ai = CFRPokerAI()
                cfr_ai.load(filename)
                print(f"CFR model loaded from {filename}")
                print(f"Information sets learned: {len(cfr_ai.nodes)}")
                
                # Create CFR player wrapper
                cfr_player = CFRPlayer(cfr_ai)
                
                # Option to evaluate CFR model
                eval_choice = input("\nEvaluate CFR model? (y/n): ")
                if eval_choice.lower() == 'y':
                    players = int(input("Number of players for evaluation (2-6): ") or "6")
                    opponent_choice = input("Use random opponents for baseline testing? (y/n): ")
                    use_random = opponent_choice.lower() == 'y'
                    
                    from cfr_player import evaluate_cfr_ai
                    evaluate_cfr_ai(cfr_ai, num_games=50, num_players=players, use_random_opponents=use_random)
                
                # Create wrapper for gameplay compatibility
                class CFRWrapper:
                    def __init__(self, cfr_player):
                        self.cfr_player = cfr_player
                        self.epsilon = 0  # CFR doesn't use epsilon
                        
                    def choose_action(self, state, valid_actions):
                        # Convert state to CFR format
                        game_state = {
                            'hole_cards': getattr(state, 'hole_cards', []),
                            'community_cards': getattr(state, 'community_cards', []),
                            'pot_size': getattr(state, 'pot_size', 0),
                            'to_call': getattr(state, 'to_call', 0),
                            'stack_size': getattr(state, 'stack_size', 1000),
                            'position': getattr(state, 'position', 0),
                            'num_players': getattr(state, 'num_players', 4),
                            'action_history': getattr(state, 'action_history', [])
                        }
                        return self.cfr_player.choose_action(game_state)
                    
                    def get_raise_size(self, state):
                        game_state = {
                            'pot_size': getattr(state, 'pot_size', 0),
                            'position': getattr(state, 'position', 0),
                            'num_players': getattr(state, 'num_players', 4)
                        }
                        return self.cfr_player.get_raise_size(game_state)
                
                ai_model = CFRWrapper(cfr_player)
                print("CFR AI ready for gameplay!")
                
            except Exception as e:
                print(f"Could not load CFR model: {e}")
                print("Starting with untrained AI")
                ai_model = None
        else:
            # Load DQN model
            ai_model = PokerAI()
            try:
                ai_model.load(filename)
                print(f"DQN model loaded from {filename}")
                ai_model.epsilon = 0  # No exploration during play
                print(f"Config of loaded model: {ai_model.config}")
                
                # Option to evaluate DQN model
                eval_choice = input("\nEvaluate DQN model? (y/n): ")
                if eval_choice.lower() == 'y':
                    players = int(input("Number of players for evaluation (2-6): ") or "6")
                    evaluate_ai_full(ai_model, num_games=100, num_players=players, use_strong_opponents=True)
            except Exception as e:
                print(f"Could not load DQN model: {e}")
                print("Starting with untrained AI")
                ai_model = None
    
    elif choice == '8':
        print("Playing without AI training")
        ai_model = None
    
    else:
        print("Invalid choice, playing without AI")
        ai_model = None
    
    # Play game
    play_choice = input("\nPlay interactive game? (y/n): ")
    if play_choice.lower() != 'y':
        print("Thanks for using the poker AI trainer!")
        return
    
    print("\n--- Game Setup ---")
    num_players = int(input("Number of players (2-6): ") or "4")
    starting_chips = int(input("Starting chips (default 1000): ") or "1000")
    
    game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips)
    
    if ai_model:
        ai_model.epsilon = 0  # No exploration during play
        game.setup_players(ai_model)
    else:
        game.setup_players()
    
    # Game loop
    print("\n" + "=" * 60)
    print("STARTING GAME")
    print("=" * 60)
    
    hand_count = 0
    while True:
        # Check game over
        active = [p for p in game.players if p.chips > 0]
        if len(active) <= 1:
            winner = active[0] if active else None
            if winner:
                print(f"\n🏆 GAME OVER! {winner.name} wins!")
            else:
                print("\n🏆 GAME OVER! Draw!")
            
            # Final standings
            print("\nFinal standings:")
            sorted_players = sorted(game.players, key=lambda p: p.chips, reverse=True)
            for i, player in enumerate(sorted_players, 1):
                print(f"  {i}. {player.name}: ${player.chips}")
            break
        
        hand_count += 1
        print(f"\n{'='*60}")
        print(f"HAND #{hand_count}")
        print(f"{'='*60}")
        
        # Show chip counts
        print("\nChip counts:")
        for player in game.players:
            if player.chips > 0:
                status = " (Dealer)" if game.players.index(player) == game.dealer_position else ""
                print(f"  {player.name}: ${player.chips}{status}")
        
        # Play hand
        game.play_hand()
        
        # Check if human eliminated
        if game.players[0].chips <= 0:
            print("\n❌ You're out of chips!")
            print("\nFinal standings:")
            sorted_players = sorted(game.players, key=lambda p: p.chips, reverse=True)
            for i, player in enumerate(sorted_players, 1):
                print(f"  {i}. {player.name}: ${player.chips}")
            break
        
        # Continue?
        cont = input("\nPress Enter to continue (or 'q' to quit): ")
        if cont.lower() == 'q':
            break
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()