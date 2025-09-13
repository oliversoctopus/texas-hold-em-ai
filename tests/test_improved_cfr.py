#!/usr/bin/env python3
"""
Test the improved CFR training to verify strategy quality improvements
"""

from cfr_player import train_cfr_ai, evaluate_cfr_ai

def test_improved_cfr():
    """Test the improved CFR training"""
    print("=" * 60)
    print("TESTING IMPROVED CFR TRAINING")
    print("=" * 60)
    
    # Train with improved parameters (higher iterations, better exploration, deeper trees)
    print("\n1. Training CFR AI with improved parameters...")
    print("   - Default iterations increased from 10,000 to 50,000")
    print("   - Depth limit increased from 6 to 12")
    print("   - Better exploration strategy")
    print("   - Improved hand evaluation and payoff calculation")
    
    # Start with smaller number for quick testing
    cfr_ai = train_cfr_ai(
        iterations=3000,  # Further reduced for testing the improvements
        num_players=4,
        verbose=True
    )
    
    print(f"\n2. Training completed! Generated {len(cfr_ai.nodes)} information sets")
    
    # Test against random opponents first
    print("\n3. Evaluating against random opponents (baseline)...")
    results_random = evaluate_cfr_ai(
        cfr_ai, 
        num_games=30,  # Smaller for quick test
        num_players=4, 
        verbose=True,
        use_random_opponents=True
    )
    
    print(f"\nResults against random opponents:")
    print(f"  Strategy usage: {(results_random.get('strategy_usage', {}).get('learned_percentage', 0)):.1f}% learned")
    print(f"  Win rate: {results_random['win_rate']:.1f}%")
    print(f"  Expected random: {results_random['expected_random_rate']:.1f}%")
    print(f"  Performance vs random: {results_random['performance_vs_random']:.2f}x")
    
    # Performance assessment
    if results_random['performance_vs_random'] > 1.5:
        print("  [EXCELLENT] CFR AI significantly outperforms random!")
    elif results_random['performance_vs_random'] > 1.2:
        print("  [GOOD] CFR AI clearly outperforms random")
    elif results_random['performance_vs_random'] > 1.0:
        print("  [OK] CFR AI slightly outperforms random")
    else:
        print("  [NEEDS WORK] CFR AI not yet better than random")
    
    return cfr_ai, results_random

if __name__ == "__main__":
    test_improved_cfr()