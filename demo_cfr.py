"""
Demonstration of CFR training and performance
"""

from cfr_player import train_cfr_ai, evaluate_cfr_ai
import time

def demo_cfr_training():
    """Demonstrate CFR training with different iteration counts"""
    
    print("=" * 60)
    print("CFR (Counterfactual Regret Minimization) Demo")
    print("=" * 60)
    print()
    print("CFR is the algorithm used by professional poker AIs like:")
    print("- Libratus (defeated top human players in 2017)")
    print("- Pluribus (defeated 5 top human players in 2019)")
    print()
    print("Testing different iteration counts...")
    print()
    
    iteration_counts = [1000, 5000, 10000]
    
    for iterations in iteration_counts:
        print(f"Training CFR AI with {iterations} iterations...")
        start_time = time.time()
        
        cfr_ai = train_cfr_ai(
            iterations=iterations,
            num_players=4,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Evaluate the AI
        results = evaluate_cfr_ai(cfr_ai, num_games=50, verbose=False)
        
        print(f"Results after {iterations} iterations:")
        print(f"  Training time: {training_time:.1f} seconds")
        print(f"  Information sets: {results['information_sets']}")
        print(f"  Strategy diversity: {results['strategy_diversity']:.3f}")
        print(f"  Estimated win rate: {results['estimated_win_rate']:.1f}%")
        print()
    
    print("=" * 60)
    print("CFR Training Complete!")
    print("=" * 60)
    print()
    print("Key advantages of CFR over Deep Q-Learning:")
    print("1. Game theory optimal - converges to Nash equilibrium")
    print("2. No need for neural networks or hyperparameter tuning")
    print("3. Proven to work in real professional poker")
    print("4. Handles multi-player games naturally")
    print("5. More consistent and predictable results")
    print()
    print("To use CFR in the main program, select option 1!")

if __name__ == "__main__":
    demo_cfr_training()