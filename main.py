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
    print("2. Train ADVANCED 2-Player CFR (Neural-Enhanced CFR with DQN-competitive performance)")
    print("3. Train Deep CFR (neural network enhanced for 3+ players)")
    print("4. Load existing AI")
    print("5. Play without AI")

    choice = input("\nChoose option (1-5): ")
    
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
        print("\nADVANCED 2-Player CFR Training (Neural-Enhanced CFR)")
        print("State-of-the-art CFR with neural networks and DQN-competitive performance")
        print("Features: Advanced poker abstractions, progressive training, neural strategy approximation")

        # Training configuration
        print("\nTraining Options:")
        print("1. Quick training (10K iterations, ~30 minutes)")
        print("2. Standard training (50K iterations, ~2 hours)")
        print("3. Professional training (100K iterations, ~4 hours)")
        print("4. Custom training")

        training_choice = input("Choice (1-4, default: 2): ") or "2"

        if training_choice == '1':
            iterations = 10000
            description = "Quick"
        elif training_choice == '2':
            iterations = 50000
            description = "Standard"
        elif training_choice == '3':
            iterations = 100000
            description = "Professional"
        else:
            iterations = int(input("Custom iterations (10000-200000): ") or "50000")
            description = "Custom"

        print(f"\nStarting {description} Neural-Enhanced CFR training with {iterations:,} iterations...")

        from cfr.neural_enhanced_cfr import NeuralEnhancedTwoPlayerCFR

        cfr_ai = NeuralEnhancedTwoPlayerCFR(
            iterations=iterations,
            use_neural_networks=True,
            progressive_training=True
        )

        cfr_ai.train(verbose=True)

        # Evaluate against both random and DQN opponents
        eval_choice = input("\nEvaluate Neural-Enhanced CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            num_games = int(input("Number of games per opponent type (default: 100): ") or "100")

            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai

            print("Testing vs random opponents (baseline)...")
            random_results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True, use_strong_opponents=False)

            print("\nTesting vs DQN benchmark models (challenge)...")
            dqn_results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True, use_strong_opponents=True)

            print(f"\n[SUMMARY] Neural-Enhanced CFR Evaluation:")
            print(f"  vs Random opponents: {random_results['win_rate']:.1f}%")
            print(f"  vs DQN benchmarks: {dqn_results['win_rate']:.1f}%")
            print(f"  Strategy usage: {dqn_results.get('strategy_usage', {}).get('learned_percentage', 'N/A')}%")

        # Save model
        save_choice = input("\nSave Neural-Enhanced CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: neural_enhanced_cfr.pkl): ") or "neural_enhanced_cfr.pkl"
            cfr_ai.save(f"models/cfr/{filename}")
            print(f"Neural-Enhanced CFR model saved to models/cfr/{filename}")

        # Create wrapper for gameplay
        from utils.model_loader import create_game_wrapper_for_model
        model_info = {'model_type': 'NeuralEnhancedTwoPlayerCFR'}
        ai_model = create_game_wrapper_for_model(cfr_ai, model_info)
        print("Neural-Enhanced CFR AI ready for gameplay!")

    elif choice == '3':
        print("\nProper 2-Player CFR Training (Standard CFR Methodology)")
        print("Uses standard CFR with equity-based abstractions, diverse scenarios, and game tree traversal")

        iterations = int(input("CFR iterations (5000-50000 recommended): ") or "10000")

        print(f"\nTraining Proper 2-Player CFR AI with {iterations} iterations...")
        from cfr.proper_cfr_two_player import train_proper_two_player_cfr_ai

        cfr_ai = train_proper_two_player_cfr_ai(
            iterations=iterations,
            verbose=True
        )

        # Evaluate the model
        eval_choice = input("\nEvaluate Proper 2-Player CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            print("Running evaluation against random opponents...")
            num_games = int(input("Number of games (default: 100): ") or "100")

            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
            results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True)

            print(f"\n[SUMMARY] Proper CFR Evaluation Complete:")
            print(f"  Win rate: {results['win_rate']:.1f}%")
            print(f"  Strategy coverage: {results.get('strategy_usage', {}).get('learned_percentage', 'N/A')}%")

        # Save model
        save_choice = input("\nSave Proper 2-Player CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: proper_two_player_cfr.pkl): ") or "proper_two_player_cfr.pkl"
            cfr_ai.save(f"models/cfr/{filename}")
            print(f"Proper 2-Player CFR model saved to models/cfr/{filename}")

        # Create wrapper for gameplay
        print("\nCreating Proper 2-Player CFR wrapper for gameplay...")
        class ProperTwoPlayerCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0

            def choose_action(self, state, valid_actions, **kwargs):
                return self.cfr_ai.get_action(
                    hole_cards=getattr(state, 'hole_cards', []),
                    community_cards=getattr(state, 'community_cards', []),
                    betting_history=getattr(state, 'betting_history', ''),
                    pot_size=getattr(state, 'pot_size', 0),
                    to_call=getattr(state, 'to_call', 0),
                    stack_size=getattr(state, 'stack_size', 1000),
                    position=getattr(state, 'position', 0),
                    opponent_stack=kwargs.get('opponent_stack', 1000)
                )

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                return max(min_raise, max(50, pot // 2))

        ai_model = ProperTwoPlayerCFRGameWrapper(cfr_ai)
        print("Proper 2-Player CFR AI ready for 2-player gameplay!")

    elif choice == '4':
        print("\nSimple 2-Player CFR Training (Fixed Action Set)")
        print("Uses equity-based hand evaluation and fixed action abstractions for stability")

        iterations = int(input("CFR iterations (5000-50000 recommended): ") or "10000")

        print(f"\nTraining Simple 2-Player CFR AI with {iterations} iterations...")
        from cfr.simple_cfr_two_player import train_simple_two_player_cfr_ai

        cfr_ai = train_simple_two_player_cfr_ai(
            iterations=iterations,
            verbose=True
        )

        # Evaluate the model
        eval_choice = input("\nEvaluate Simple 2-Player CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            print("Running evaluation against random opponents...")
            num_games = int(input("Number of games (default: 100): ") or "100")

            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
            results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True)

            print(f"\n[SUMMARY] Simple CFR Evaluation Complete:")
            print(f"  Win rate: {results['win_rate']:.1f}%")
            print(f"  Strategy coverage: {results.get('strategy_usage', {}).get('learned_percentage', 'N/A')}%")

        # Save model
        save_choice = input("\nSave Simple 2-Player CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: simple_two_player_cfr.pkl): ") or "simple_two_player_cfr.pkl"
            cfr_ai.save(f"models/cfr/{filename}")
            print(f"Simple 2-Player CFR model saved to models/cfr/{filename}")

        # Create wrapper for gameplay
        print("\nCreating Simple 2-Player CFR wrapper for gameplay...")
        class SimpleTwoPlayerCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0

            def choose_action(self, state, valid_actions, **kwargs):
                return self.cfr_ai.get_action(
                    hole_cards=getattr(state, 'hole_cards', []),
                    community_cards=getattr(state, 'community_cards', []),
                    betting_history=getattr(state, 'betting_history', ''),
                    pot_size=getattr(state, 'pot_size', 0),
                    to_call=getattr(state, 'to_call', 0),
                    stack_size=getattr(state, 'stack_size', 1000),
                    position=getattr(state, 'position', 0)
                )

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                return max(min_raise, max(50, pot // 2))

        ai_model = SimpleTwoPlayerCFRGameWrapper(cfr_ai)
        print("Simple 2-Player CFR AI ready for 2-player gameplay!")

    elif choice == '5':
        print("\nFixed Enhanced 2-Player CFR Training (Monte Carlo CFR)")
        print("Uses proven Monte Carlo CFR with timeout protection and comprehensive information sets")

        iterations = int(input("CFR iterations (100-1000 recommended): ") or "500")

        print(f"\nTraining Fixed Enhanced 2-Player CFR AI with {iterations} iterations...")
        from cfr.fixed_enhanced_cfr_two_player import FixedEnhancedTwoPlayerCFR

        cfr_ai = FixedEnhancedTwoPlayerCFR(iterations=iterations)
        cfr_ai.train(verbose=True)

        # Evaluate the model
        eval_choice = input("\nEvaluate Fixed Enhanced 2-Player CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            print("Running evaluation against random opponents...")
            num_games = int(input("Number of games (default: 100): ") or "100")

            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
            results = evaluate_two_player_cfr_ai(cfr_ai, num_games=num_games, verbose=True)

            print(f"\n[SUMMARY] Fixed Enhanced CFR Evaluation Complete:")
            print(f"  Win rate: {results['win_rate']:.1f}%")
            print(f"  Strategy coverage: {results.get('strategy_usage', {}).get('learned_percentage', 'N/A')}%")

        # Save model
        save_choice = input("\nSave Fixed Enhanced 2-Player CFR model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: fixed_enhanced_two_player_cfr.pkl): ") or "fixed_enhanced_two_player_cfr.pkl"
            cfr_ai.save(f"models/cfr/{filename}")
            print(f"Fixed Enhanced 2-Player CFR model saved to models/cfr/{filename}")

        # Create wrapper for gameplay
        print("\nCreating Fixed Enhanced 2-Player CFR wrapper for gameplay...")
        class FixedEnhancedTwoPlayerCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0

            def choose_action(self, state, valid_actions, **kwargs):
                return self.cfr_ai.get_action(
                    hole_cards=getattr(state, 'hole_cards', []),
                    community_cards=getattr(state, 'community_cards', []),
                    betting_history=getattr(state, 'betting_history', ''),
                    pot_size=getattr(state, 'pot_size', 0),
                    to_call=getattr(state, 'to_call', 0),
                    stack_size=getattr(state, 'stack_size', 1000),
                    position=getattr(state, 'position', 0),
                    opponent_stack=kwargs.get('opponent_stack', 1000)
                )

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                return max(min_raise, max(50, pot // 2))

        ai_model = FixedEnhancedTwoPlayerCFRGameWrapper(cfr_ai)
        print("Fixed Enhanced 2-Player CFR AI ready for 2-player gameplay!")

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

    elif choice == '7':
        print("\n⚠️  [DEPRECATED] DQN Training with Strategy Diversity")
        print("WARNING: DQN training is deprecated. Use CFR or Deep CFR instead.")
        print("Skipping DQN training. Please use options 1-5 for modern AI training.")

    elif choice == '8':
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

    elif choice == '9':
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
            
    elif choice == '4':
        filename = input("Model filename (.pth for DQN/Deep CFR, .pkl for CFR, default: poker_ai.pth): ") or "poker_ai.pth"

        # Check if it's a CFR model (.pkl) or DQN model (.pth)
        if filename.endswith('.pkl'):
            # Use automatic model type detection
            try:
                from utils.model_loader import load_cfr_model_by_type, create_game_wrapper_for_model, evaluate_model_by_type

                print(f"Loading CFR model from {filename}...")
                model, model_info = load_cfr_model_by_type(filename, verbose=True)

                print(f"Model loaded successfully!")
                print(f"  Type: {model_info['model_type']}")
                print(f"  Version: {model_info['model_version']}")
                print(f"  Information sets: {len(model.nodes):,}")

                # Option to evaluate the model
                eval_choice = input(f"\nEvaluate {model_info['model_type']} model? (y/n): ")
                if eval_choice.lower() == 'y':
                    if model_info['model_type'] == 'MultiPlayerCFR':
                        players = int(input("Number of players for evaluation (2-6): ") or "4")
                        num_games = int(input("Number of games (default: 50): ") or "50")
                    else:
                        num_games = int(input("Number of games (default: 100): ") or "100")

                    # Ask user to choose evaluation opponents
                    print("\nChoose evaluation opponents:")
                    print("1. Random opponents (fast, baseline)")
                    print("2. DQN benchmark models (more challenging)")
                    opponent_choice = input("Choice (1-2, default: 1): ") or "1"

                    if opponent_choice == '2':
                        print(f"Using DQN benchmark models for evaluation")
                        results = evaluate_model_by_type(model, model_info, num_games=num_games, verbose=True, use_strong_opponents=True)
                    else:
                        print("Using random opponents for evaluation")
                        results = evaluate_model_by_type(model, model_info, num_games=num_games, verbose=True, use_strong_opponents=False)

                    # Print summary based on model type
                    if model_info['model_type'] == 'ImprovedTwoPlayerCFR':
                        print(f"\n[SUMMARY] {model_info['model_type']} Evaluation:")
                        print(f"  Overall win rate: {results['overall_win_rate']:.1f}%")
                        print(f"  Strategy coverage: {results['strategy_usage']['learned_percentage']:.1f}%")
                    elif model_info['model_type'] == 'TwoPlayerCFR':
                        print(f"\n[SUMMARY] {model_info['model_type']} Evaluation:")
                        print(f"  Win rate: {results['win_rate']:.1f}%")
                        print(f"  Strategy usage: {results.get('strategy_usage', 'N/A')}")
                    else:  # MultiPlayerCFR
                        print(f"\n[SUMMARY] {model_info['model_type']} Evaluation Complete")

                # Create game wrapper
                ai_model = create_game_wrapper_for_model(model, model_info)
                print(f"{model_info['model_type']} AI ready for gameplay!")

            except Exception as e:
                print(f"Error loading CFR model: {e}")
                print("Starting with untrained AI")
                ai_model = None
        else:
            # For .pth files, check model type flag to determine if it's Deep CFR or DQN
            try:
                import torch
                checkpoint = torch.load(filename, map_location='cpu')
                model_type = checkpoint.get('model_type', 'Unknown')

                if model_type == 'DeepCFR':
                    print(f"Loading Deep CFR model: {filename}")
                    # Load Deep CFR model
                    from deepcfr.deep_cfr import DeepCFRPokerAI

                    deep_cfr_ai = DeepCFRPokerAI()
                    deep_cfr_ai.load(filename)
                    print(f"Deep CFR model loaded successfully!")
                    print(f"  Model version: {checkpoint.get('model_version', '1.0')}")
                    print(f"  Iterations trained: {deep_cfr_ai.iterations_trained:,}")
                    print(f"  Feature dimension: {deep_cfr_ai.feature_dim}")
                    print(f"  Hidden size: {deep_cfr_ai.hidden_size}")

                    # Option to evaluate Deep CFR model
                    eval_choice = input("\nEvaluate Deep CFR model? (y/n): ")
                    if eval_choice.lower() == 'y':
                        players = int(input("Number of players for evaluation (3-6): ") or "4")
                        num_games = int(input("Number of games (default: 50): ") or "50")

                        # Ask user to choose evaluation opponents
                        print("\nChoose evaluation opponents:")
                        print("1. Random opponents (fast, baseline)")
                        print("2. DQN benchmark models (more challenging)")
                        opponent_choice = input("Choice (1-2, default: 1): ") or "1"

                        use_random = opponent_choice == '1'
                        if use_random:
                            print("Using random opponents for evaluation")
                        else:
                            print("Using DQN benchmark models for evaluation")
                            print("Note: Deep CFR evaluation will use random opponents (DQN opponent support coming soon)")
                            use_random = True  # Fallback for now

                        print(f"\nEvaluating Deep CFR model with {num_games} games...")

                        from deepcfr.deep_cfr_player import evaluate_deep_cfr_ai
                        try:
                            evaluate_deep_cfr_ai(deep_cfr_ai, num_games=num_games, num_players=players, use_random_opponents=use_random)
                        except Exception as eval_e:
                            print(f"Evaluation failed: {eval_e}")
                            print("Continuing with loaded model for gameplay...")

                    # Create wrapper for gameplay
                    print("\nCreating Deep CFR wrapper for gameplay...")
                    class DeepCFRGameWrapper:
                        def __init__(self, deep_cfr_ai):
                            self.deep_cfr_ai = deep_cfr_ai
                            self.epsilon = 0

                        def choose_action(self, state, valid_actions, **kwargs):
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
                            return self.deep_cfr_ai.get_action(**game_state)

                        def get_raise_size(self, state):
                            return max(50, getattr(state, 'pot_size', 0) // 2)

                    ai_model = DeepCFRGameWrapper(deep_cfr_ai)
                    print("Deep CFR AI ready for gameplay!")

                elif model_type == 'DQN':
                    print(f"Loading DQN model: {filename}")
                    # Load DQN model
                    ai_model = PokerAI()
                    ai_model.load(filename)
                    print(f"DQN model loaded successfully!")
                    print(f"  Model version: {checkpoint.get('model_version', '1.0')}")
                    print(f"  Config: {ai_model.config}")
                    ai_model.epsilon = 0  # No exploration during play

                    # Option to evaluate DQN model
                    eval_choice = input("\nEvaluate DQN model? (y/n): ")
                    if eval_choice.lower() == 'y':
                        players = int(input("Number of players for evaluation (2-6): ") or "6")
                        num_games = int(input("Number of games (default: 100): ") or "100")

                        # Ask user to choose evaluation opponents
                        print("\nChoose evaluation opponents:")
                        print("1. Random opponents (fast, baseline)")
                        print("2. DQN benchmark models (more challenging)")
                        opponent_choice = input("Choice (1-2, default: 2): ") or "2"

                        if opponent_choice == '2':
                            print("Using DQN benchmark models for evaluation")
                            evaluate_ai_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=True)
                        else:
                            print("Using random opponents for evaluation")
                            evaluate_ai_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=False)

                else:
                    # Legacy detection for models without flags
                    print(f"No model type flag found. Attempting legacy detection...")
                    if 'regret_net_state_dict' in checkpoint and 'strategy_net_state_dict' in checkpoint:
                        print("Detected legacy Deep CFR model (no type flag)")
                        print("Loading as Deep CFR...")

                        from deepcfr.deep_cfr import DeepCFRPokerAI
                        deep_cfr_ai = DeepCFRPokerAI()
                        deep_cfr_ai.load(filename)

                        # Option to evaluate legacy Deep CFR model
                        eval_choice = input("\nEvaluate legacy Deep CFR model? (y/n): ")
                        if eval_choice.lower() == 'y':
                            players = int(input("Number of players for evaluation (3-6): ") or "4")
                            num_games = int(input("Number of games (default: 50): ") or "50")

                            # Ask user to choose evaluation opponents
                            print("\nChoose evaluation opponents:")
                            print("1. Random opponents (fast, baseline)")
                            print("2. DQN benchmark models (more challenging)")
                            opponent_choice = input("Choice (1-2, default: 1): ") or "1"

                            use_random = opponent_choice == '1'
                            if use_random:
                                print("Using random opponents for evaluation")
                            else:
                                print("Using DQN benchmark models for evaluation")
                                print("Note: Deep CFR evaluation will use random opponents (DQN opponent support coming soon)")
                                use_random = True  # Fallback for now

                            print(f"\nEvaluating legacy Deep CFR model with {num_games} games...")

                            from deepcfr.deep_cfr_player import evaluate_deep_cfr_ai
                            try:
                                evaluate_deep_cfr_ai(deep_cfr_ai, num_games=num_games, num_players=players, use_random_opponents=use_random)
                            except Exception as eval_e:
                                print(f"Evaluation failed: {eval_e}")
                                print("Continuing with loaded model for gameplay...")

                        class DeepCFRGameWrapper:
                            def __init__(self, deep_cfr_ai):
                                self.deep_cfr_ai = deep_cfr_ai
                                self.epsilon = 0

                            def choose_action(self, state, valid_actions, **kwargs):
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
                                return self.deep_cfr_ai.get_action(**game_state)

                            def get_raise_size(self, state):
                                return max(50, getattr(state, 'pot_size', 0) // 2)

                        ai_model = DeepCFRGameWrapper(deep_cfr_ai)
                        print("Legacy Deep CFR model loaded and ready for gameplay!")

                    elif 'input_size' in checkpoint or 'q_network' in checkpoint:
                        print("Detected legacy DQN model (no type flag)")
                        print("Loading as DQN...")

                        ai_model = PokerAI()
                        ai_model.load(filename)
                        ai_model.epsilon = 0
                        print(f"Legacy DQN model loaded with config: {ai_model.config}")

                        # Option to evaluate legacy DQN model
                        eval_choice = input("\nEvaluate legacy DQN model? (y/n): ")
                        if eval_choice.lower() == 'y':
                            players = int(input("Number of players for evaluation (2-6): ") or "6")
                            num_games = int(input("Number of games (default: 100): ") or "100")

                            # Ask user to choose evaluation opponents
                            print("\nChoose evaluation opponents:")
                            print("1. Random opponents (fast, baseline)")
                            print("2. DQN benchmark models (more challenging)")
                            opponent_choice = input("Choice (1-2, default: 2): ") or "2"

                            if opponent_choice == '2':
                                print("Using DQN benchmark models for evaluation")
                                evaluate_ai_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=True)
                            else:
                                print("Using random opponents for evaluation")
                                evaluate_ai_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=False)

                    else:
                        print(f"Unknown .pth model format in {filename}")
                        print(f"Available keys: {list(checkpoint.keys())}")
                        print("Starting with untrained AI")
                        ai_model = None

            except Exception as e:
                print(f"Could not load .pth model: {e}")
                print("Starting with untrained AI")
                ai_model = None
    
    elif choice == '5':
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