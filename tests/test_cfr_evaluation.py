#!/usr/bin/env python3
"""
Test improved CFR with actual evaluation
"""

from cfr_player import train_cfr_ai, evaluate_cfr_ai

def test_cfr_evaluation():
    print("CFR Evaluation Test - Improved Algorithm")
    print("=" * 50)
    
    # Train with moderate iterations to test improvements
    print("Training CFR with 2000 iterations...")
    cfr_ai = train_cfr_ai(iterations=2000, verbose=True)
    
    print(f"\nTraining complete! Generated {len(cfr_ai.nodes)} information sets")
    
    # Evaluate quickly against random opponents
    print("\nEvaluating with 20 games against random opponents...")
    results = evaluate_cfr_ai(
        cfr_ai, 
        num_games=20, 
        use_random_opponents=True, 
        verbose=True
    )
    
    # Show improvement summary
    print(f"\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Information sets generated: {len(cfr_ai.nodes)}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Expected random rate: {results['expected_random_rate']:.1f}%")
    print(f"Performance vs random: {results['performance_vs_random']:.2f}x")
    
    # Get strategy usage from the CFR AI's stats
    if hasattr(cfr_ai, 'strategy_stats'):
        total = cfr_ai.strategy_stats['total_decisions']
        learned = cfr_ai.strategy_stats['learned_strategy_count']
        if total > 0:
            learned_pct = (learned / total) * 100
            print(f"Learned strategy usage: {learned_pct:.1f}%")
    
    if results['performance_vs_random'] > 1.3:
        print("✓ EXCELLENT: CFR AI significantly outperforms random!")
    elif results['performance_vs_random'] > 1.1:
        print("✓ GOOD: CFR AI clearly beats random opponents")
    elif results['performance_vs_random'] > 1.0:
        print("~ OK: CFR AI slightly better than random")
    else:
        print("✗ NEEDS MORE TRAINING: Performance not yet above random")
    
    return cfr_ai, results

if __name__ == "__main__":
    test_cfr_evaluation()