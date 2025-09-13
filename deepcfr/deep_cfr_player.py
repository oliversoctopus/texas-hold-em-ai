"""
Deep CFR Player wrapper for integration with the game engine
"""

import numpy as np
from typing import List, Optional
from core.game_constants import Action
from core.card_deck import Card
from .deep_cfr import DeepCFRPokerAI


class DeepCFRPlayer:
    """
    Player class that uses Deep CFR AI for decision making
    Compatible with the existing game engine
    """
    
    def __init__(self, deep_cfr_ai: Optional[DeepCFRPokerAI] = None):
        self.deep_cfr_ai = deep_cfr_ai or DeepCFRPokerAI()
        self.name = "DeepCFR_AI"
        
    def choose_action(self, game_state: dict) -> Action:
        """
        Choose action using Deep CFR strategy
        
        Args:
            game_state: Dictionary containing game state information
        """
        # Extract necessary information
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        pot_size = game_state.get('pot_size', 0)
        to_call = game_state.get('to_call', 0)
        stack_size = game_state.get('stack_size', 1000)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 4)
        
        # Create simplified betting history for Deep CFR
        betting_history = self._create_betting_history(game_state.get('action_history', []))
        
        # Get action from Deep CFR AI
        return self.deep_cfr_ai.get_action(
            hole_cards=hole_cards,
            community_cards=community_cards,
            betting_history=betting_history,
            pot_size=pot_size,
            to_call=to_call,
            stack_size=stack_size,
            position=position,
            num_players=num_players
        )
    
    def _create_betting_history(self, action_history: List) -> str:
        """Convert action history to simplified betting history string"""
        history = ""
        for action in action_history:
            if hasattr(action, 'action'):
                if action.action == Action.FOLD:
                    history += "F"
                elif action.action == Action.CHECK:
                    history += "K"
                elif action.action == Action.CALL:
                    history += "C"
                elif action.action == Action.RAISE:
                    history += "R"
                elif action.action == Action.ALL_IN:
                    history += "A"
        return history
    
    def get_raise_size(self, game_state: dict) -> int:
        """
        Get raise size based on Deep CFR action
        """
        # Use neural network's learned sizing or default to pot-based sizing
        pot_size = game_state.get('pot_size', 0)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 4)
        
        # Adaptive raise sizing based on position and pot
        base_raise = max(50, pot_size // 3)
        
        # Adjust based on position (later position = more aggressive)
        position_factor = 1.0 + (position / num_players) * 0.5
        
        return int(base_raise * position_factor)


def train_deep_cfr_ai(iterations: int = 10000, save_filename: Optional[str] = None, 
                     verbose: bool = True) -> DeepCFRPokerAI:
    """
    Convenience function to train a Deep CFR AI
    
    Args:
        iterations: Number of training iterations
        save_filename: Optional filename to save the model
        verbose: Whether to print training progress
        
    Returns:
        Trained DeepCFRPokerAI instance
    """
    from .deep_cfr import train_deep_cfr_ai as _train_deep_cfr_ai
    return _train_deep_cfr_ai(iterations, save_filename, verbose)


def evaluate_deep_cfr_ai(deep_cfr_ai: DeepCFRPokerAI, num_games: int = 100, 
                        num_players: int = 4, verbose: bool = True, 
                        use_random_opponents: bool = False) -> dict:
    """
    Evaluate Deep CFR AI performance using real poker games
    
    Args:
        deep_cfr_ai: Trained Deep CFR AI
        num_games: Number of games to evaluate
        num_players: Number of players per game
        verbose: Whether to print evaluation progress
        use_random_opponents: If True, use random AIs; if False, use strong benchmark models
        
    Returns:
        Dictionary with evaluation results
    """
    # Import here to avoid circular imports
    from cfr.cfr_evaluation import evaluate_cfr_ai_real
    
    if verbose:
        print(f"Evaluating Deep CFR AI over {num_games} real poker games...")
    
    # Create a wrapper that converts Deep CFR to CFR interface
    class DeepCFRWrapper:
        def __init__(self, deep_cfr_ai):
            self.deep_cfr_ai = deep_cfr_ai
            self.nodes = {}  # Dummy for compatibility
            
        def get_action(self, hole_cards, community_cards, betting_history, 
                      pot_size, to_call, stack_size, position, num_players):
            return self.deep_cfr_ai.get_action(hole_cards, community_cards, betting_history,
                                             pot_size, to_call, stack_size, position, num_players)
        
        def get_strategy_usage_stats(self):
            return self.deep_cfr_ai.get_strategy_usage_stats()
        
        def reset_strategy_stats(self):
            self.deep_cfr_ai.reset_strategy_stats()
        
        def print_strategy_usage(self):
            self.deep_cfr_ai.print_strategy_usage()
    
    wrapper = DeepCFRWrapper(deep_cfr_ai)
    return evaluate_cfr_ai_real(wrapper, num_games, num_players, verbose, use_random_opponents)