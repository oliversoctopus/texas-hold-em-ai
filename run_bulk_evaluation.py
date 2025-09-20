"""
Runner script for bulk evaluation of reward-based AI models
Handles failures gracefully and continues evaluation
"""

import os
import sys
import glob
import traceback
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.bulk_reward_evaluation import BulkRewardEvaluator


def run_evaluation():
    """Run the bulk evaluation with error handling"""

    print("\n" + "="*80)
    print("REWARD-BASED AI MODEL BULK EVALUATION RUNNER")
    print("="*80)

    # Setup
    models_dir = "models/reward_nn"

    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"\nError: Models directory '{models_dir}' does not exist!")
        print("Please train some reward-based AI models first.")
        return

    # Count available models
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    model_files.extend(glob.glob(os.path.join(models_dir, "*.pkl")))

    if not model_files:
        print(f"\nNo models found in {models_dir}")
        print("Please train some reward-based AI models first.")
        return

    print(f"\nFound {len(model_files)} models to evaluate")
    print("\nModel files:")
    for i, f in enumerate(model_files, 1):
        print(f"  {i:2d}. {os.path.basename(f)}")

    # Get evaluation parameters
    print("\n" + "-"*40)
    print("EVALUATION SETTINGS")
    print("-"*40)

    try:
        num_games = input("Number of games per model (default: 50): ").strip()
        num_games = int(num_games) if num_games else 50
    except:
        num_games = 50

    print(f"\nWill evaluate each model with {num_games} games")
    print(f"Estimated time: {len(model_files) * num_games * 0.1:.1f} seconds")

    proceed = input("\nProceed with evaluation? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Evaluation cancelled.")
        return

    # Create evaluator
    evaluator = BulkRewardEvaluator(models_dir)

    # Evaluate all models with individual error handling
    print(f"\n{'='*80}")
    print("STARTING EVALUATION")
    print(f"{'='*80}\n")

    all_results = []
    successful = 0
    failed = 0

    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path)
        print(f"[{i}/{len(model_files)}] Evaluating {model_name}...")

        try:
            # Evaluate this model
            result = evaluator.evaluate_model(model_path, num_games)

            if result['status'] == 'success':
                successful += 1
                print(f"  ✓ Success: Win rate = {result.get('win_rate', 0)*100:.1f}%, "
                      f"All-in rate = {result.get('action_distribution', {}).get('ALL_IN', 0)*100:.1f}%")
            else:
                failed += 1
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

            all_results.append(result)

        except Exception as e:
            failed += 1
            print(f"  ✗ Error evaluating {model_name}: {str(e)}")
            # Add error result
            all_results.append({
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            })
            # Continue to next model
            continue

        print()

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total models: {len(model_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    # Print detailed report if we have results
    if any(r['status'] == 'success' for r in all_results):
        evaluator.print_summary_report(all_results)

    # Save results
    print("\n" + "-"*40)
    save_choice = input("Save results to file? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_results(all_results)


def save_results(results):
    """Save evaluation results to JSON file"""
    import json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reward_evaluation_{timestamp}.json"

    # Clean up results for JSON serialization
    clean_results = []
    for r in results:
        clean_r = r.copy()

        # Convert any Action enums to strings
        if 'action_counts' in clean_r:
            clean_r['action_counts'] = {
                str(k): v for k, v in clean_r['action_counts'].items()
            }
        if 'preflop_actions' in clean_r:
            clean_r['preflop_actions'] = {
                str(k): v for k, v in clean_r['preflop_actions'].items()
            }
        if 'postflop_actions' in clean_r:
            clean_r['postflop_actions'] = {
                str(k): v for k, v in clean_r['postflop_actions'].items()
            }

        clean_results.append(clean_r)

    try:
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main entry point"""
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()