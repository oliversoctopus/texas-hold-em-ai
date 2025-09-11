"""
CFR-based Poker Player that integrates with the existing game engine
"""

import numpy as np
from typing import List, Optional
from game_constants import Action
from card_deck import Card
from cfr_poker import CFRPokerAI


class CFRPlayer:
    """
    Player class that uses CFR AI for decision making
    Compatible with the existing game engine
    """
    
    def __init__(self, cfr_ai: Optional[CFRPokerAI] = None):
        self.cfr_ai = cfr_ai or CFRPokerAI()
        self.name = "CFR_AI"
        
    def choose_action(self, game_state: dict) -> Action:
        """
        Choose action using CFR strategy
        
        Args:
            game_state: Dictionary containing:
                - hole_cards: List[Card] - player's hole cards
                - community_cards: List[Card] - community cards
                - pot_size: int - current pot size
                - to_call: int - amount to call
                - current_bet: int - current bet amount
                - stack_size: int - player's remaining chips
                - position: int - player's position
                - num_players: int - number of players
                - betting_history: str - betting history for this hand
                - action_history: List - full action history
        """
        # Extract necessary information
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        pot_size = game_state.get('pot_size', 0)
        to_call = game_state.get('to_call', 0)
        stack_size = game_state.get('stack_size', 1000)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 4)
        
        # Create simplified betting history for CFR
        betting_history = self._create_betting_history(game_state.get('action_history', []))
        
        # Get action from CFR AI
        return self.cfr_ai.get_action(
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
        Get raise size based on CFR action
        This method provides compatibility with existing game engine
        """
        # Simple raise sizing based on pot and position
        pot_size = game_state.get('pot_size', 0)
        position = game_state.get('position', 0)
        num_players = game_state.get('num_players', 4)
        
        # Conservative raise sizing
        base_raise = max(20, pot_size // 4)
        
        # Adjust based on position
        if position < num_players // 3:  # Early position
            return base_raise
        elif position < 2 * num_players // 3:  # Middle position
            return int(base_raise * 1.2)
        else:  # Late position
            return int(base_raise * 1.5)


class CFRTrainingEngine:
    """
    Training engine for CFR that integrates with existing game components
    """
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.cfr_ai = CFRPokerAI()
        
    def train_cfr(self, iterations: int = 10000, verbose: bool = True) -> CFRPokerAI:
        """
        Train CFR AI using the specified number of iterations
        """
        if verbose:
            print(f"Training CFR AI with {iterations} iterations...")
            print("This will create a strong poker AI using game theory optimal strategies.")
        
        # Train the CFR algorithm
        self.cfr_ai = CFRPokerAI(iterations=iterations)
        self.cfr_ai.train(verbose=verbose)
        
        if verbose:
            print(f"CFR training completed!")
            print(f"Generated {len(self.cfr_ai.nodes)} unique information sets")
            
        return self.cfr_ai
    
    def create_cfr_player(self, cfr_ai: Optional[CFRPokerAI] = None) -> CFRPlayer:
        """Create a CFR player instance"""
        return CFRPlayer(cfr_ai or self.cfr_ai)
    
    def save_model(self, filename: str):
        """Save the trained CFR model"""
        self.cfr_ai.save(filename)
    
    def load_model(self, filename: str) -> CFRPokerAI:
        """Load a trained CFR model"""
        self.cfr_ai = CFRPokerAI()
        self.cfr_ai.load(filename)
        return self.cfr_ai


def train_cfr_ai(iterations: int = 10000, num_players: int = 4, 
                 save_filename: Optional[str] = None, verbose: bool = True) -> CFRPokerAI:
    """
    Convenience function to train a CFR AI
    
    Args:
        iterations: Number of CFR iterations to run
        num_players: Number of players (affects information set abstraction)
        save_filename: Optional filename to save the trained model
        verbose: Whether to print training progress
        
    Returns:
        Trained CFRPokerAI instance
    """
    trainer = CFRTrainingEngine(num_players=num_players)
    cfr_ai = trainer.train_cfr(iterations=iterations, verbose=verbose)
    
    if save_filename:
        trainer.save_model(save_filename)
        
    return cfr_ai


def evaluate_cfr_ai(cfr_ai: CFRPokerAI, num_games: int = 100, 
                    num_players: int = 4, verbose: bool = True) -> dict:
    """
    Evaluate CFR AI performance using real poker games
    
    Args:
        cfr_ai: Trained CFR AI
        num_games: Number of games to evaluate
        num_players: Number of players per game
        verbose: Whether to print evaluation progress
        
    Returns:
        Dictionary with evaluation results
    """
    # Import here to avoid circular imports
    from cfr_evaluation import evaluate_cfr_ai_real
    
    if verbose:
        print(f"Evaluating CFR AI over {num_games} real poker games...")
    
    # Use real game evaluation
    return evaluate_cfr_ai_real(cfr_ai, num_games, num_players, verbose)