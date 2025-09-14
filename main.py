import random
from core.game_constants import Action
from dqn.poker_ai import PokerAI
from core.game_engine import TexasHoldEm
from dqn.training import hyperparameter_tuning, train_ai_advanced
from evaluation.unified_evaluation import evaluate_dqn_full, evaluate_cfr_full, evaluate_deep_cfr_full
from core.player import Player
from evaluation.advanced_evaluation import AdvancedEvaluator, evaluate_all_models
from utils.create_strategy_bots import create_all_strategy_bots, load_strategy_bots

def main():
    """Main game loop"""
    print("=" * 60)
    print("TEXAS HOLD'EM with Advanced PyTorch AI")
    print("=" * 60)
    
    print("\n1. Train CFR AI (Multi-player Counterfactual Regret)")
    print("2. Train BASIC 2-Player CFR (Original Neural CFR)")
    print("3. Train CONSERVATIVE 2-Player CFR (Tight Hand-Aware CFR)")
    print("4. Train BALANCED 2-Player CFR (Aggressive Hand-Aware CFR)")
    print("5. Train RAW NEURAL 2-Player CFR (End-to-End Learning)")
    print("6. Train Deep CFR (neural network enhanced for 3+ players)")
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
        from cfr.cfr_player import train_cfr_ai
        
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
            evaluate_cfr_full(cfr_ai, num_games=100, num_players=players, use_random_opponents=use_random)
        
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
        print("\nBASIC 2-Player CFR Training (Original Neural CFR)")
        print("Original Neural-Enhanced CFR with basic strategy learning")
        print("Features: Neural networks, progressive training, but may bet wildly")

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

            print("Testing vs random opponents (baseline)...")
            random_results = evaluate_cfr_full(cfr_ai, num_games=num_games, num_players=2, use_random_opponents=True, verbose=True)

            print("\nTesting vs DQN benchmark models (challenge)...")
            dqn_results = evaluate_cfr_full(cfr_ai, num_games=num_games, num_players=2, use_random_opponents=False, verbose=True)

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
        print("\nCONSERVATIVE 2-Player CFR Training (Tight Play Style)")
        print("Hand-strength-aware CFR that plays tight and conservative")
        print("Features: Strong hand evaluation, careful betting, minimal bluffing")
        print("-" * 60)

        # Training configuration
        print("\nSelect training configuration:")
        print("1. Quick (1,000 iterations - 2-5 minutes)")
        print("2. Standard (10,000 iterations - 15-30 minutes)")
        print("3. Professional (50,000 iterations - 1-2 hours)")
        print("4. Custom")

        config_choice = input("Choose configuration (1-4): ")

        if config_choice == '1':
            iterations = 1000
            config_name = "Quick"
        elif config_choice == '2':
            iterations = 10000
            config_name = "Standard"
        elif config_choice == '3':
            iterations = 50000
            config_name = "Professional"
        else:
            iterations = int(input("Number of iterations: ") or "10000")
            config_name = "Custom"

        print(f"\nTraining Improved Neural-Enhanced CFR ({config_name} - {iterations:,} iterations)...")
        print("This version understands hand strength and won't bet wildly with weak hands.")
        print("Starting training...\n")

        from cfr.improved_neural_cfr import ImprovedNeuralEnhancedCFR

        # Create and train
        cfr_ai = ImprovedNeuralEnhancedCFR(
            iterations=iterations,
            use_neural_networks=True,
            use_hand_strength=True
        )

        cfr_ai.train(verbose=True)

        # Save model
        save_choice = input("\nSave trained model? (y/n): ")
        if save_choice.lower() == 'y':
            model_name = f"improved_cfr_{config_name.lower()}.pkl"
            filename = input(f"Filename (default: models/cfr/{model_name}): ") or f"models/cfr/{model_name}"

            # Ensure directory exists
            import os
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            cfr_ai.save(filename)
            print(f"Model saved to {filename}")

        # Create wrapper for gameplay
        from utils.model_loader import create_game_wrapper_for_model
        model_info = {'model_type': 'ImprovedNeuralEnhancedCFR'}
        ai_model = create_game_wrapper_for_model(cfr_ai, model_info)
        print("Conservative CFR AI ready for gameplay!")

    elif choice == '4':
        print("\nBALANCED 2-Player CFR Training (Aggressive Play Style)")
        print("Hand-strength-aware CFR with balanced aggressive play")
        print("Features: Wider ranges, more bluffing, position-aware aggression")
        print("-" * 60)

        # Training configuration
        print("\nSelect training configuration:")
        print("1. Quick (1,000 iterations - 2-5 minutes)")
        print("2. Standard (10,000 iterations - 15-30 minutes)")
        print("3. Professional (50,000 iterations - 1-2 hours)")
        print("4. Custom")

        config_choice = input("Choose configuration (1-4): ")

        if config_choice == '1':
            iterations = 1000
            config_name = "Quick"
        elif config_choice == '2':
            iterations = 10000
            config_name = "Standard"
        elif config_choice == '3':
            iterations = 50000
            config_name = "Professional"
        else:
            iterations = int(input("Number of iterations: ") or "10000")
            config_name = "Custom"

        print(f"\nTraining Balanced Neural CFR ({config_name} - {iterations:,} iterations)...")
        print("This version plays more aggressively with wider ranges and more bluffs.")
        print("Starting training...\n")

        from cfr.balanced_neural_cfr import BalancedNeuralCFR

        # Create and train
        cfr_ai = BalancedNeuralCFR(
            iterations=iterations,
            use_neural_networks=True,
            use_hand_strength=True
        )

        cfr_ai.train(verbose=True)

        # Save model
        save_choice = input("\nSave trained model? (y/n): ")
        if save_choice.lower() == 'y':
            model_name = f"balanced_cfr_{config_name.lower()}.pkl"
            filename = input(f"Filename (default: models/cfr/{model_name}): ") or f"models/cfr/{model_name}"

            # Ensure directory exists
            import os
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            cfr_ai.save(filename)
            print(f"Model saved to {filename}")

        # Create wrapper for gameplay
        from utils.model_loader import create_game_wrapper_for_model
        model_info = {'model_type': 'BalancedNeuralCFR'}
        ai_model = create_game_wrapper_for_model(cfr_ai, model_info)
        print("Balanced CFR AI ready for gameplay!")

    elif choice == '5':
        print("\nRAW NEURAL 2-Player CFR Training (End-to-End Learning)")
        print("Learns directly from raw game state without manual feature engineering")
        print("Features: Deep neural networks, attention mechanisms, residual connections")
        print("-" * 60)

        # Training configuration
        print("\nSelect training configuration:")
        print("1. Quick (5,000 iterations - 5-10 minutes)")
        print("2. Standard (25,000 iterations - 30-45 minutes)")
        print("3. Professional (50,000 iterations - 1-2 hours)")
        print("4. Custom")

        config_choice = input("Choose configuration (1-4): ")

        if config_choice == '1':
            iterations = 5000
            config_name = "Quick"
        elif config_choice == '2':
            iterations = 25000
            config_name = "Standard"
        elif config_choice == '3':
            iterations = 50000
            config_name = "Professional"
        else:
            iterations = int(input("Number of iterations: ") or "25000")
            config_name = "Custom"

        # Advanced settings
        learning_rate = 0.001
        batch_size = 32

        advanced = input("\nConfigure advanced settings? (y/n): ")
        if advanced.lower() == 'y':
            lr = input(f"Learning rate (default: {learning_rate}): ")
            if lr:
                learning_rate = float(lr)
            bs = input(f"Batch size (default: {batch_size}): ")
            if bs:
                batch_size = int(bs)

        print(f"\nTraining Raw Neural CFR ({config_name} - {iterations:,} iterations)...")
        print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
        print("This version learns strategies directly from raw game data.")
        print("Starting training...\n")

        from cfr.raw_neural_cfr import RawNeuralCFR

        # Create and train
        cfr_ai = RawNeuralCFR(
            iterations=iterations,
            learning_rate=learning_rate,
            batch_size=batch_size
        )

        cfr_ai.train(verbose=True)

        # Evaluate the model
        eval_choice = input("\nEvaluate Raw Neural CFR? (y/n): ")
        if eval_choice.lower() == 'y':
            num_games = int(input("Number of games per opponent type (default: 100): ") or "100")

            print("Testing vs random opponents (baseline)...")
            random_results = evaluate_cfr_full(cfr_ai, num_games=num_games, num_players=2,
                                              use_random_opponents=True, verbose=True)

            print("\nTesting vs DQN benchmark models (challenge)...")
            dqn_results = evaluate_cfr_full(cfr_ai, num_games=num_games, num_players=2,
                                           use_random_opponents=False, verbose=True)

            print(f"\n[SUMMARY] Raw Neural CFR Evaluation:")
            print(f"  vs Random opponents: {random_results['win_rate']:.1f}%")
            print(f"  vs DQN benchmarks: {dqn_results['win_rate']:.1f}%")

        # Save model
        save_choice = input("\nSave trained model? (y/n): ")
        if save_choice.lower() == 'y':
            model_name = f"raw_neural_{config_name.lower()}.pkl"
            filename = input(f"Filename (default: models/cfr/{model_name}): ") or f"models/cfr/{model_name}"

            # Ensure directory exists
            import os
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            cfr_ai.save(filename)
            print(f"Model saved to {filename}")

        # Create wrapper for gameplay
        from utils.model_loader import create_game_wrapper_for_model
        model_info = {'model_type': 'RawNeuralCFR'}
        ai_model = create_game_wrapper_for_model(cfr_ai, model_info)
        print("Raw Neural CFR AI ready for gameplay!")

    elif choice == '6':
        print("\nDeep CFR Training (Neural Network Enhanced)")
        print("Uses neural networks to approximate optimal strategies for 3+ players")

        iterations = int(input("Deep CFR iterations (5000-20000 recommended): ") or "10000")
        players = int(input("Number of players for training (3-6): ") or "4")
        
        print(f"\nTraining Deep CFR AI with {iterations} iterations for {players} players...")
        from deepcfr.deep_cfr_player import train_deep_cfr_ai
        
        deep_cfr_ai = train_deep_cfr_ai(
            iterations=iterations,
            verbose=True
        )
        
        # Evaluate Deep CFR AI
        eval_choice = input("\nEvaluate Deep CFR AI? (y/n): ")
        if eval_choice.lower() == 'y':
            opponent_choice = input("Use random opponents for baseline testing? (y/n): ")
            use_random = opponent_choice.lower() == 'y'
            evaluate_deep_cfr_full(deep_cfr_ai, num_games=50, num_players=players, use_random_opponents=use_random)
        
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

                # Handle different model types for node count
                if model_info['model_type'] == 'StrategySelectorCFR':
                    # Count total nodes across all strategies
                    total_nodes = sum(len(nodes) for nodes in model.strategy_nodes.values())
                    print(f"  Information sets: {total_nodes:,} (across 5 strategies)")
                elif hasattr(model, 'nodes'):
                    print(f"  Information sets: {len(model.nodes):,}")
                else:
                    print(f"  Model loaded")

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

                        try:
                            evaluate_deep_cfr_full(deep_cfr_ai, num_games=num_games, num_players=players, use_random_opponents=use_random)
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
                            evaluate_dqn_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=True)
                        else:
                            print("Using random opponents for evaluation")
                            evaluate_dqn_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=False)

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

                            try:
                                evaluate_deep_cfr_full(deep_cfr_ai, num_games=num_games, num_players=players, use_random_opponents=use_random)
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
                                evaluate_dqn_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=True)
                            else:
                                print("Using random opponents for evaluation")
                                evaluate_dqn_full(ai_model, num_games=num_games, num_players=players, use_strong_opponents=False)

                    else:
                        print(f"Unknown .pth model format in {filename}")
                        print(f"Available keys: {list(checkpoint.keys())}")
                        print("Starting with untrained AI")
                        ai_model = None

            except Exception as e:
                print(f"Could not load .pth model: {e}")
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