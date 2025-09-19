"""
Utility functions for loading CFR models with automatic type detection
"""

import pickle
from typing import Dict, Any, Optional


def detect_model_type(filename: str) -> Dict[str, Any]:
    """
    Detect the type of CFR model from the file

    Args:
        filename: Path to the model file

    Returns:
        Dictionary with model information including type and version
    """
    import torch

    # First check if it's a PyTorch model (RewardBasedAI uses .pth files)
    if filename.endswith('.pth'):
        try:
            data = torch.load(filename, map_location='cpu')
            if isinstance(data, dict) and 'model_type' in data:
                return {
                    'model_type': data['model_type'],
                    'model_version': data.get('model_version', '1.0'),
                    'has_flag': True,
                    'data': data
                }
        except:
            pass  # Not a valid PyTorch model, try pickle

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Check if model has type flag (new format)
        if isinstance(data, dict) and 'model_type' in data:
            return {
                'model_type': data['model_type'],
                'model_version': data.get('model_version', '1.0'),
                'has_flag': True,
                'data': data
            }

        # Also check for 'type' field (used by RawNeuralCFR)
        if isinstance(data, dict) and 'type' in data:
            return {
                'model_type': data['type'],
                'model_version': data.get('version', '1.0'),
                'has_flag': True,
                'data': data
            }

        # Fallback: try to infer from content structure (old format)
        if isinstance(data, dict):
            # Check for specific keys that indicate model type
            if 'nodes' in data:
                # Try to distinguish between regular and two-player CFR by examining node keys
                nodes = data.get('nodes', {})
                if nodes:
                    sample_key = next(iter(nodes.keys()))
                    if sample_key.startswith('2p_'):
                        return {
                            'model_type': 'TwoPlayerCFR',
                            'model_version': '1.0',
                            'has_flag': False,
                            'data': data
                        }
                    else:
                        return {
                            'model_type': 'MultiPlayerCFR',
                            'model_version': '1.0',
                            'has_flag': False,
                            'data': data
                        }

        # Unknown format
        return {
            'model_type': 'Unknown',
            'model_version': '1.0',
            'has_flag': False,
            'data': data
        }

    except Exception as e:
        raise ValueError(f"Could not detect model type from {filename}: {e}")


def load_cfr_model_by_type(filename: str, verbose: bool = True):
    """
    Load CFR model with automatic type detection

    Args:
        filename: Path to the model file
        verbose: Whether to print loading information

    Returns:
        Tuple of (model_instance, model_info)
    """
    # Detect model type
    model_info = detect_model_type(filename)
    model_type = model_info['model_type']

    if verbose:
        print(f"Detected model type: {model_type} v{model_info['model_version']}")
        if not model_info['has_flag']:
            print("  [Note] Model uses legacy format (no type flag)")

    # Load the appropriate model
    if model_type == 'ProperTwoPlayerCFR':
        from cfr.proper_cfr_two_player import ProperTwoPlayerCFR
        model = ProperTwoPlayerCFR()
        model.load(filename)
        return model, model_info


    elif model_type == 'SimpleTwoPlayerCFR':
        from cfr.simple_cfr_two_player import SimpleTwoPlayerCFR
        model = SimpleTwoPlayerCFR()
        model.load(filename)
        return model, model_info

    elif model_type == 'EnhancedTwoPlayerCFR':
        from cfr.enhanced_cfr_two_player import EnhancedTwoPlayerCFR
        model = EnhancedTwoPlayerCFR()
        model.load(filename)
        return model, model_info

    elif model_type == 'FixedEnhancedTwoPlayerCFR':
        from cfr.fixed_enhanced_cfr_two_player import FixedEnhancedTwoPlayerCFR
        model = FixedEnhancedTwoPlayerCFR()
        model.load(filename)
        return model, model_info

    elif model_type == 'NeuralEnhancedTwoPlayerCFR':
        from cfr.neural_enhanced_cfr import NeuralEnhancedTwoPlayerCFR
        model = NeuralEnhancedTwoPlayerCFR()
        model.load(filename)
        return model, model_info


    elif model_type == 'StrategySelectorCFR':
        from cfr.strategy_selector_cfr import StrategySelectorCFR
        model = StrategySelectorCFR()
        model.load(filename)
        return model, model_info

    elif model_type == 'TwoPlayerCFR':
        from cfr.cfr_two_player import TwoPlayerCFRPokerAI
        model = TwoPlayerCFRPokerAI()
        model.load(filename)
        return model, model_info

    elif model_type == 'MultiPlayerCFR':
        from cfr.cfr_poker import CFRPokerAI
        model = CFRPokerAI()
        model.load(filename)
        return model, model_info

    elif model_type == 'raw_neural_cfr':
        from cfr.raw_neural_cfr import RawNeuralCFR
        model = RawNeuralCFR()
        model.load(filename)
        return model, model_info

    elif model_type == 'RewardBasedAI':
        from reward_nn.reward_based_ai import RewardBasedAI
        model = RewardBasedAI()
        model.load(filename)
        return model, model_info

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_game_wrapper_for_model(model, model_info: Dict[str, Any]):
    """
    Create appropriate game wrapper for the model type

    Args:
        model: The loaded model instance
        model_info: Model information from detection

    Returns:
        Game wrapper compatible with the game engine
    """
    model_type = model_info['model_type']

    if model_type == 'ProperTwoPlayerCFR':
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

        return ProperTwoPlayerCFRGameWrapper(model)

    elif model_type == 'FixedEnhancedTwoPlayerCFR':
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

        return FixedEnhancedTwoPlayerCFRGameWrapper(model)

    elif model_type == 'NeuralEnhancedTwoPlayerCFR':
        class NeuralEnhancedTwoPlayerCFRGameWrapper:
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
                return self.cfr_ai.get_raise_size(pot, current_bet, player_chips, min_raise)

            def get_state_features(self, hand, community_cards, pot, current_bet, player_chips,
                                 player_bet, num_players, players_in_hand, position=0,
                                 action_history=None, opponent_bets=None, hand_phase=0):
                """Create state object for Neural-Enhanced CFR"""
                class CFRState:
                    def __init__(self, hand, community_cards, pot, current_bet, player_chips,
                               player_bet, num_players, action_history, position):
                        self.hole_cards = hand
                        self.community_cards = community_cards
                        self.pot_size = pot
                        self.to_call = max(0, current_bet - player_bet)
                        self.stack_size = player_chips
                        self.position = position
                        self.num_players = num_players
                        self.action_history = action_history[-10:] if action_history else []
                        self.betting_history = ''  # CFR uses string format

                return CFRState(hand, community_cards, pot, current_bet, player_chips,
                              player_bet, num_players, action_history, position)

        return NeuralEnhancedTwoPlayerCFRGameWrapper(model)


    elif model_type == 'RawNeuralCFR':
        class RawNeuralCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0

            def choose_action(self, state, valid_actions, **kwargs):
                # Try to get opponent stack from state or kwargs
                if hasattr(state, 'opponent_stack'):
                    # DummyState from game engine now includes this
                    opponent_stack = state.opponent_stack
                else:
                    # Fallback to kwargs or default
                    opponent_stack = kwargs.get('opponent_stack', 1000)

                return self.cfr_ai.get_action(
                    hole_cards=getattr(state, 'hole_cards', []),
                    community_cards=getattr(state, 'community_cards', []),
                    pot_size=getattr(state, 'pot_size', 0),
                    to_call=getattr(state, 'to_call', 0),
                    stack_size=getattr(state, 'stack_size', 1000),
                    position=getattr(state, 'position', 0),
                    num_players=getattr(state, 'num_players', 2),
                    action_history=getattr(state, 'action_history', []),
                    opponent_stack=opponent_stack,
                    valid_actions=valid_actions  # Pass valid actions to the model
                )

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                # Raw Neural CFR determines its own bet sizing
                pot_size = getattr(state, 'pot_size', pot)
                return min(player_chips, int(pot_size * 0.75))  # Default to 75% pot

            def get_state_features(self, hand, community_cards, pot, current_bet, player_chips,
                                 player_bet, num_players, players_in_hand, position=0,
                                 action_history=None, opponent_bets=None, hand_phase=0):
                """Create state object for Raw Neural CFR"""
                class CFRState:
                    def __init__(self, hand, community_cards, pot, current_bet, player_chips,
                               player_bet, num_players, action_history, position):
                        self.hole_cards = hand
                        self.community_cards = community_cards
                        self.pot_size = pot
                        self.to_call = max(0, current_bet - player_bet)
                        self.stack_size = player_chips
                        self.position = position
                        self.num_players = num_players
                        self.action_history = action_history if action_history else []

                return CFRState(hand, community_cards, pot, current_bet, player_chips,
                              player_bet, num_players, action_history, position)

        return RawNeuralCFRGameWrapper(model)

    elif model_type == 'raw_neural_cfr':
        # Same as RawNeuralCFR but lowercase (from saved models)
        class RawNeuralCFRGameWrapper:
            def __init__(self, cfr_ai):
                self.cfr_ai = cfr_ai
                self.epsilon = 0

            def choose_action(self, state, valid_actions, **kwargs):
                # Try to get opponent stack from state or kwargs
                if hasattr(state, 'opponent_stack'):
                    # DummyState from game engine now includes this
                    opponent_stack = state.opponent_stack
                else:
                    # Fallback to kwargs or default
                    opponent_stack = kwargs.get('opponent_stack', 1000)

                return self.cfr_ai.get_action(
                    hole_cards=getattr(state, 'hole_cards', []),
                    community_cards=getattr(state, 'community_cards', []),
                    pot_size=getattr(state, 'pot_size', 0),
                    to_call=getattr(state, 'to_call', 0),
                    stack_size=getattr(state, 'stack_size', 1000),
                    position=getattr(state, 'position', 0),
                    num_players=getattr(state, 'num_players', 2),
                    action_history=getattr(state, 'action_history', []),
                    opponent_stack=opponent_stack,
                    valid_actions=valid_actions  # Pass valid actions to the model
                )

            def get_raise_size(self, state, pot=0, current_bet=0, player_chips=1000, player_current_bet=0, min_raise=20):
                # Raw Neural CFR determines its own bet sizing
                pot_size = getattr(state, 'pot_size', pot)
                return min(player_chips, int(pot_size * 0.75))  # Default to 75% pot

            def get_state_features(self, hand, community_cards, pot, current_bet, player_chips,
                                 player_bet, num_players, players_in_hand, position=0,
                                 action_history=None, opponent_bets=None, hand_phase=0):
                """Create state object for Raw Neural CFR"""
                class CFRState:
                    def __init__(self, hand, community_cards, pot, current_bet, player_chips,
                                player_bet, num_players, action_history, position):
                        self.hole_cards = hand
                        self.community_cards = community_cards
                        self.pot_size = pot
                        self.to_call = current_bet - player_bet
                        self.stack_size = player_chips
                        self.num_players = num_players
                        self.action_history = action_history or []
                        self.position = position

                return CFRState(hand, community_cards, pot, current_bet, player_chips,
                              player_bet, num_players, action_history, position)

        return RawNeuralCFRGameWrapper(model)

    elif model_type == 'StrategySelectorCFR':
        class StrategySelectorCFRGameWrapper:
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
                return self.cfr_ai.get_raise_size(pot, current_bet, player_chips, min_raise)

            def get_state_features(self, hand, community_cards, pot, current_bet, player_chips,
                                 player_bet, num_players, players_in_hand, position=0,
                                 action_history=None, opponent_bets=None, hand_phase=0):
                """Create state object for Strategy Selector CFR"""
                class CFRState:
                    def __init__(self, hand, community_cards, pot, current_bet, player_chips,
                               player_bet, num_players, action_history, position):
                        self.hole_cards = hand
                        self.community_cards = community_cards
                        self.pot_size = pot
                        self.to_call = max(0, current_bet - player_bet)
                        self.stack_size = player_chips
                        self.position = position
                        self.num_players = num_players
                        self.action_history = action_history[-10:] if action_history else []
                        self.betting_history = ''

                return CFRState(hand, community_cards, pot, current_bet, player_chips,
                              player_bet, num_players, action_history, position)

        return StrategySelectorCFRGameWrapper(model)

    elif model_type == 'TwoPlayerCFR':
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

        return TwoPlayerCFRGameWrapper(model)

    elif model_type == 'MultiPlayerCFR':
        from cfr.cfr_player import CFRPlayer

        cfr_player = CFRPlayer(model)

        class CFRWrapper:
            def __init__(self, cfr_player):
                self.cfr_player = cfr_player
                self.epsilon = 0

            def choose_action(self, state, valid_actions):
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

        return CFRWrapper(cfr_player)

    elif model_type == 'RewardBasedAI':
        # RewardBasedAI already has the right interface
        return model

    else:
        raise ValueError(f"Cannot create wrapper for unknown model type: {model_type}")


def evaluate_model_by_type(model, model_info: Dict[str, Any], num_games: int = 100, verbose: bool = True, use_strong_opponents: bool = False):
    """
    Evaluate model using appropriate evaluation function

    Args:
        model: The loaded model instance
        model_info: Model information from detection
        num_games: Number of games to evaluate
        verbose: Whether to print evaluation progress
        use_strong_opponents: Whether to use DQN benchmark models instead of random opponents

    Returns:
        Evaluation results dictionary
    """
    model_type = model_info['model_type']

    if model_type == 'ProperTwoPlayerCFR':
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
        return evaluate_two_player_cfr_ai(model, num_games=num_games, verbose=verbose, use_strong_opponents=use_strong_opponents)

    elif model_type == 'FixedEnhancedTwoPlayerCFR':
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
        return evaluate_two_player_cfr_ai(model, num_games=num_games, verbose=verbose, use_strong_opponents=use_strong_opponents)

    elif model_type == 'NeuralEnhancedTwoPlayerCFR':
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
        return evaluate_two_player_cfr_ai(model, num_games=num_games, verbose=verbose, use_strong_opponents=use_strong_opponents)


    elif model_type == 'StrategySelectorCFR':
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
        return evaluate_two_player_cfr_ai(model, num_games=num_games, verbose=verbose, use_strong_opponents=use_strong_opponents)

    elif model_type == 'TwoPlayerCFR':
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
        return evaluate_two_player_cfr_ai(model, num_games=num_games, verbose=verbose, use_strong_opponents=use_strong_opponents)

    elif model_type == 'MultiPlayerCFR':
        from cfr.cfr_player import evaluate_cfr_ai
        players = 2  # Default for multi-player
        return evaluate_cfr_ai(model, num_games=num_games, num_players=players,
                             use_random_opponents=True, verbose=verbose)

    elif model_type == 'raw_neural_cfr':
        # Raw Neural CFR uses a different evaluation method (neural network-based)
        from evaluation.unified_evaluation import evaluate_cfr_full

        # Use DQN benchmarks if requested, otherwise use random opponents
        use_random = not use_strong_opponents

        if verbose:
            opponent_type = "random opponents" if use_random else "DQN benchmark models"
            print(f"Evaluating Raw Neural CFR against {opponent_type}...")

        return evaluate_cfr_full(
            model,
            num_games=num_games,
            num_players=2,
            use_random_opponents=use_random,
            verbose=verbose
        )

    elif model_type == 'RewardBasedAI':
        from evaluation.unified_evaluation import evaluate_reward_based_ai
        # Default to testing vs random opponents
        return evaluate_reward_based_ai(
            model,
            num_games=num_games,
            use_random_opponents=not use_strong_opponents,
            verbose=verbose
        )

    else:
        raise ValueError(f"Cannot evaluate unknown model type: {model_type}")