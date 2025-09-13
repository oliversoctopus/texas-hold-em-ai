"""
Deep CFR implementation for 3+ player poker using neural networks
"""

from .deep_cfr import DeepCFRPokerAI
from .deep_cfr_player import DeepCFRPlayer, train_deep_cfr_ai, evaluate_deep_cfr_ai

__all__ = ['DeepCFRPokerAI', 'DeepCFRPlayer', 'train_deep_cfr_ai', 'evaluate_deep_cfr_ai']