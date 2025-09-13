import random
from core.game_constants import Action
from dqn.poker_ai import PokerAI
from core.game_engine import TexasHoldEm
from dqn.training import hyperparameter_tuning, train_ai_advanced, evaluate_ai_full
from core.player import Player
from evaluation.advanced_evaluation import AdvancedEvaluator, evaluate_all_models
from utils.create_strategy_bots import create_all_strategy_bots, load_strategy_bots

def main():
    """Main game loop"""
    print("=" * 60)
    print("TEXAS HOLD'EM with Advanced PyTorch AI")
    print("=" * 60)
    
    print("\n1. Train CFR AI (Counterfactual Regret Minimization)")
    print("2. Train 2-Player CFR (optimized for heads-up)")
    print("3. Train Deep CFR (neural network enhanced)")
    print("4. [DEPRECATED] Train DQN AI (evaluation only)")
    print("5. [DEPRECATED] Train with hyperparameter tuning")
    print("6. [DEPRECATED] Train with strategy diversity")
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
        from cfr.cfr_player import train_cfr_ai, evaluate_cfr_ai
        
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
        from cfr.cfr_player import CFRPlayer
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
        print("\n2-Player CFR Training (Heads-Up Poker)")
        print("Optimized CFR implementation for 2-player no-limit Texas Hold'em")
        
        iterations = int(input("CFR iterations (50000-200000 recommended for 2-player): ") or "100000")
        
        print(f"\nTraining 2-Player CFR AI with {iterations} iterations...")
        from cfr.cfr_two_player import train_two_player_cfr_ai
        
        cfr_ai = train_two_player_cfr_ai(
            iterations=iterations,
            verbose=True
        )
        
        # Evaluate against random opponents using specialized 2-player evaluation
        eval_choice = input("\nEvaluate 2-Player CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            print("Running specialized 2-player CFR evaluation...")
            num_games = int(input("Number of games (default: 100): ") or "100")
            
            # Use specialized 2-player evaluation framework
            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
            results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True)
            
            # Show summary
            print(f"\n[SUMMARY] 2-Player CFR Evaluation Complete:")
            print(f"  Win rate: {results['win_rate']:.1f}%")
            print(f"  Strategy usage: {results.get('strategy_usage', 'N/A')}")
            print("2-Player CFR AI evaluation complete!")
        
        # Save CFR model
        save_choice = input("\nSave 2-Player CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: two_player_cfr.pkl): ") or "two_player_cfr.pkl" 
            cfr_ai.save(f"models/cfr/{filename}")
            print(f"2-Player CFR model saved to models/cfr/{filename}")
        
        # Create wrapper for gameplay
        print("\nCreating 2-Player CFR wrapper for gameplay...")
        class TwoPlayerCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0
                
            def choose_action(self, state, valid_actions, **kwargs):
                game_state = {
                    'hole_cards': getattr(state, 'hole_cards', []),
                    'community_cards': getattr(state, 'community_cards', []),
                    'pot_size': getattr(state, 'pot_size', 0),
                    'to_call': getattr(state, 'to_call', 0),
                    'stack_size': getattr(state, 'stack_size', 1000),
                    'position': getattr(state, 'position', 0),
                    'num_players': 2
                }
                return self.cfr_ai.get_action(**game_state)
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                return max(min_raise, pot // 2)
        
        ai_model = TwoPlayerCFRGameWrapper(cfr_ai)
        print("2-Player CFR AI ready for 2-player gameplay!")

    elif choice == '3':
        print("\nDeep CFR Training (Neural Network Enhanced)")
        print("Uses neural networks to approximate optimal strategies for 3+ players")
        
        iterations = int(input("Deep CFR iterations (5000-20000 recommended): ") or "10000")
        players = int(input("Number of players for training (3-6): ") or "4")
        
        print(f"\nTraining Deep CFR AI with {iterations} iterations for {players} players...")
        from deepcfr.deep_cfr_player import train_deep_cfr_ai, evaluate_deep_cfr_ai
        
        deep_cfr_ai = train_deep_cfr_ai(
            iterations=iterations,
            verbose=True
        )
        
        # Evaluate Deep CFR AI
        eval_choice = input("\nEvaluate Deep CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            opponent_choice = input("Use random opponents for baseline testing? (y/n): ")
            use_random = opponent_choice.lower() == 'y'
            evaluate_deep_cfr_ai(deep_cfr_ai, num_games=50, num_players=players, use_random_opponents=use_random)
        
        # Save Deep CFR model
        save_choice = input("\nSave Deep CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: deep_cfr_ai.pth): ") or "deep_cfr_ai.pth"
            deep_cfr_ai.save(f"models/cfr/{filename}")
            print(f"Deep CFR model saved to models/cfr/{filename}")
        
        # Create wrapper for gameplay
        print("\nCreating Deep CFR wrapper for gameplay...")
        from deepcfr.deep_cfr_player import DeepCFRPlayer
        
        cfr_player = DeepCFRPlayer(deep_cfr_ai)
        
        class DeepCFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0
                
            def choose_action(self, state, valid_actions, **kwargs):
                game_state = {
                    'hole_cards': getattr(state, 'hole_cards', []),
                    'community_cards': getattr(state, 'community_cards', []),
                    'pot_size': getattr(state, 'pot_size', 0),
                    'to_call': getattr(state, 'to_call', 0),
                    'stack_size': getattr(state, 'stack_size', 1000),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4)
                }
                return self.cfr_player.choose_action(game_state)
            
            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                game_state = {
                    'pot_size': getattr(state, 'pot_size', pot),
                    'position': getattr(state, 'position', 0),
                    'num_players': getattr(state, 'num_players', 4)
                }
                return self.cfr_player.get_raise_size(game_state)
        
        ai_model = DeepCFRWrapper(cfr_player)

    elif choice == '4':
        print("\n⚠️  [DEPRECATED] DQN Training with Strategy Diversity")
        print("WARNING: DQN training is deprecated. Use CFR or Deep CFR instead.")
        print("Skipping DQN training. Please use options 1-3 for modern AI training.")

    elif choice == '5':
        print("\n[DEPRECATED] Hyperparameter tuning")
        print("WARNING: DQN training is deprecated. Use CFR or Deep CFR instead.")
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
        print("\n[DEPRECATED] Training with strategy diversity (prevents all-in spam)")
        print("WARNING: DQN training is deprecated. Use CFR or Deep CFR instead.")
        
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
            
    elif choice == '7':
        filename = input("Model filename (.pth for DQN, .pkl for CFR, default: poker_ai.pth): ") or "poker_ai.pth"
        
        # Check if it's a CFR model (.pkl) or DQN model (.pth)
        if filename.endswith('.pkl'):
            # Check if it's a 2-player CFR model
            if 'two_player' in filename.lower():
                # Load 2-Player CFR model
                try:
                    from cfr.cfr_two_player import TwoPlayerCFRPokerAI
                    
                    print(f"Loading 2-Player CFR model from {filename}...")
                    cfr_ai = TwoPlayerCFRPokerAI()
                    cfr_ai.load(filename)
                    print(f"2-Player CFR model loaded from {filename}")
                    print(f"Information sets learned: {len(cfr_ai.nodes)}")
                    
                    # Option to evaluate 2-player CFR model
                    eval_choice = input("\nEvaluate 2-Player CFR model? (y/n): ")
                    if eval_choice.lower() == 'y':
                        num_games = int(input("Number of games (default: 100): ") or "100")
                        
                        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
                        results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True)
                        
                        print(f"\n[SUMMARY] Evaluation Complete:")
                        print(f"  Win rate: {results['win_rate']:.1f}%")
                        print(f"  Strategy usage: {results.get('strategy_usage', 'N/A')}")
                    
                    # Create wrapper for 2-player gameplay
                    class TwoPlayerCFRGameWrapper:
                        def __init__(self, cfr_ai):
                            self.cfr_ai = cfr_ai
                            self.epsilon = 0
                            
                        def choose_action(self, state, valid_actions, **kwargs):
                            game_state = {
                                'hole_cards': getattr(state, 'hole_cards', []),
                                'community_cards': getattr(state, 'community_cards', []),
                                'pot_size': getattr(state, 'pot_size', 0),
                                'to_call': getattr(state, 'to_call', 0),
                                'stack_size': getattr(state, 'stack_size', 1000),
                                'position': getattr(state, 'position', 0),
                                'num_players': 2
                            }
                            return self.cfr_ai.get_action(**game_state)
                        
                        def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                            return max(min_raise, pot // 2)
                    
                    ai_model = TwoPlayerCFRGameWrapper(cfr_ai)
                    print("2-Player CFR AI ready for gameplay!")
                    
                except Exception as e:
                    print(f"Error loading 2-Player CFR model: {e}")
                    return
            else:
                # Load regular multi-player CFR model
                try:
                    from cfr.cfr_poker import CFRPokerAI
                    from cfr.cfr_player import CFRPlayer
                    
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
                        
                        from cfr.cfr_player import evaluate_cfr_ai
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