#!/usr/bin/env python3
"""
Quick test of improved CFR with minimal iterations
"""

from cfr_player import train_cfr_ai, evaluate_cfr_ai

def quick_cfr_test():
    print("Quick CFR Test - Improved Algorithm")
    print("=" * 50)
    
    # Very small training for speed
    print("Training CFR with 1000 iterations...")
    cfr_ai = train_cfr_ai(iterations=1000, verbose=True)
    
    print(f"Generated {len(cfr_ai.nodes)} information sets")
    
    # Quick evaluation
    print("\nEvaluating with 20 games against random opponents...")
    results = evaluate_cfr_ai(cfr_ai, num_games=20, use_random_opponents=True, verbose=True)
    
    return cfr_ai, results

if __name__ == "__main__":
    quick_cfr_test()