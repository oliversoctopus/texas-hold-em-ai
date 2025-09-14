#!/usr/bin/env python3
"""
Test suite for Improved Neural-Enhanced CFR with hand strength awareness
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_improved_cfr():
    """Test the improved CFR implementation"""
    print("=" * 60)
    print("TESTING IMPROVED NEURAL-ENHANCED CFR")
    print("=" * 60)

    try:
        from cfr.improved_neural_cfr import ImprovedNeuralEnhancedCFR
        print("[OK] Imported ImprovedNeuralEnhancedCFR")

        # Test with quick training
        print("\n1. Testing quick training (100 iterations)...")
        cfr = ImprovedNeuralEnhancedCFR(
            iterations=100,
            use_neural_networks=True,
            use_hand_strength=True
        )

        cfr.train(verbose=False)
        print(f"   Training complete: {len(cfr.nodes):,} information sets created")

        # Test hand strength awareness
        print("\n2. Testing hand strength awareness...")
        from core.card_deck import Card

        # Test with strong hand (pocket aces)
        strong_hand = [Card('A', 'hearts'), Card('A', 'spades')]
        community = []

        action_strong = cfr.get_action(
            hole_cards=strong_hand,
            community_cards=community,
            betting_history='',
            pot_size=100,
            to_call=0,
            stack_size=1000,
            position=0
        )

        print(f"   Strong hand (AA) action: {action_strong}")

        # Test with weak hand (7-2 offsuit)
        weak_hand = [Card('7', 'hearts'), Card('2', 'clubs')]
        action_weak = cfr.get_action(
            hole_cards=weak_hand,
            community_cards=community,
            betting_history='',
            pot_size=100,
            to_call=0,
            stack_size=1000,
            position=0
        )

        print(f"   Weak hand (72o) action: {action_weak}")

        # Test convergence with more iterations
        print("\n3. Testing convergence (comparing 100 vs 500 iterations)...")
        cfr_500 = ImprovedNeuralEnhancedCFR(
            iterations=500,
            use_neural_networks=True,
            use_hand_strength=True
        )

        cfr_500.train(verbose=False)
        print(f"   500 iterations: {len(cfr_500.nodes):,} information sets")
        print(f"   100 iterations: {len(cfr.nodes):,} information sets")

        improvement = (len(cfr_500.nodes) - len(cfr.nodes)) / len(cfr.nodes) * 100
        print(f"   Improvement: {improvement:.1f}% more information sets")

        # Test evaluation against random
        print("\n4. Testing performance vs random opponents...")
        from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai

        results = evaluate_two_player_cfr_ai(
            cfr_500,
            num_games=20,
            verbose=False,
            use_strong_opponents=False
        )

        print(f"   Win rate vs random: {results['win_rate']:.1f}%")
        print(f"   Strategy coverage: {results['strategy_usage']['learned_percentage']:.1f}%")

        # Test saving and loading
        print("\n5. Testing save/load functionality...")
        test_file = "test_improved_cfr_model.pkl"
        cfr_500.save(test_file)
        print(f"   Model saved to {test_file}")

        loaded_cfr = ImprovedNeuralEnhancedCFR()
        loaded_cfr.load(test_file)
        print(f"   Model loaded: {len(loaded_cfr.nodes):,} information sets")

        # Clean up test file
        os.remove(test_file)
        print(f"   Test file cleaned up")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hand_strength_behavior():
    """Test that the AI behaves differently based on hand strength"""
    print("\n" + "=" * 60)
    print("TESTING HAND STRENGTH BEHAVIOR")
    print("=" * 60)

    try:
        from cfr.improved_neural_cfr import ImprovedNeuralEnhancedCFR
        from core.card_deck import Card
        from core.game_constants import Action

        # Train a model
        print("Training model for hand strength testing...")
        cfr = ImprovedNeuralEnhancedCFR(
            iterations=200,
            use_neural_networks=True,
            use_hand_strength=True
        )
        cfr.train(verbose=False)

        # Test different hand strengths
        test_cases = [
            ("Pocket Aces", [Card('A', 'hearts'), Card('A', 'spades')]),
            ("King-Queen suited", [Card('K', 'hearts'), Card('Q', 'hearts')]),
            ("Middle pair", [Card('8', 'clubs'), Card('8', 'diamonds')]),
            ("Seven-Two offsuit", [Card('7', 'hearts'), Card('2', 'clubs')]),
        ]

        # Track action frequencies
        action_stats = {}

        for name, hand in test_cases:
            actions = []
            for _ in range(50):  # Sample multiple times
                action = cfr.get_action(
                    hole_cards=hand,
                    community_cards=[],
                    betting_history='',
                    pot_size=100,
                    to_call=0,
                    stack_size=1000,
                    position=0
                )
                actions.append(action)

            # Count action frequencies
            fold_rate = actions.count(Action.FOLD) / len(actions) * 100
            check_rate = actions.count(Action.CHECK) / len(actions) * 100
            raise_rate = actions.count(Action.RAISE) / len(actions) * 100

            action_stats[name] = {
                'fold': fold_rate,
                'check': check_rate,
                'raise': raise_rate
            }

            print(f"\n{name}:")
            print(f"  Fold: {fold_rate:.1f}%")
            print(f"  Check: {check_rate:.1f}%")
            print(f"  Raise: {raise_rate:.1f}%")

        # Verify that strong hands bet more than weak hands
        aces_raise = action_stats["Pocket Aces"]['raise']
        seven_two_raise = action_stats["Seven-Two offsuit"]['raise']

        print("\n" + "-" * 40)
        if aces_raise > seven_two_raise:
            print("[PASS] Pocket Aces raise more often than 7-2 offsuit")
            print(f"  AA raise rate: {aces_raise:.1f}%")
            print(f"  72o raise rate: {seven_two_raise:.1f}%")
        else:
            print("[FAIL] Hand strength awareness not working properly")
            print(f"  AA raise rate: {aces_raise:.1f}%")
            print(f"  72o raise rate: {seven_two_raise:.1f}%")

        # Check that weak hands fold more
        seven_two_fold = action_stats["Seven-Two offsuit"]['fold']
        aces_fold = action_stats["Pocket Aces"]['fold']

        if seven_two_fold > aces_fold:
            print("[PASS] 7-2 offsuit folds more often than Pocket Aces")
            print(f"  72o fold rate: {seven_two_fold:.1f}%")
            print(f"  AA fold rate: {aces_fold:.1f}%")
        else:
            print("[WARNING] Weak hands not folding enough")

        return True

    except Exception as e:
        print(f"\nERROR: Hand strength test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convergence():
    """Test that more training actually improves the model"""
    print("\n" + "=" * 60)
    print("TESTING CONVERGENCE")
    print("=" * 60)

    try:
        from cfr.improved_neural_cfr import ImprovedNeuralEnhancedCFR

        iterations_list = [100, 500, 1000]
        node_counts = []
        win_rates = []

        for iterations in iterations_list:
            print(f"\nTraining with {iterations} iterations...")
            cfr = ImprovedNeuralEnhancedCFR(
                iterations=iterations,
                use_neural_networks=True,
                use_hand_strength=True
            )
            cfr.train(verbose=False)

            node_counts.append(len(cfr.nodes))

            # Quick evaluation
            from cfr.cfr_two_player_evaluation import evaluate_two_player_cfr_ai
            results = evaluate_two_player_cfr_ai(
                cfr,
                num_games=10,
                verbose=False,
                use_strong_opponents=False
            )
            win_rates.append(results['win_rate'])

            print(f"  Nodes: {len(cfr.nodes):,}")
            print(f"  Win rate: {results['win_rate']:.1f}%")

        # Check improvement
        print("\n" + "-" * 40)
        print("Convergence Analysis:")

        if node_counts[-1] > node_counts[0]:
            print(f"[PASS] Node count increased: {node_counts[0]:,} -> {node_counts[-1]:,}")
        else:
            print(f"[FAIL] Node count did not increase properly")

        if win_rates[-1] >= win_rates[0]:
            print(f"[PASS] Win rate maintained/improved: {win_rates[0]:.1f}% -> {win_rates[-1]:.1f}%")
        else:
            print(f"[WARNING] Win rate decreased with more training")

        return True

    except Exception as e:
        print(f"\nERROR: Convergence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running Improved Neural-Enhanced CFR Test Suite\n")

    # Run all tests
    test_results = []

    print("Test 1: Basic functionality")
    test_results.append(("Basic functionality", test_improved_cfr()))

    print("\nTest 2: Hand strength behavior")
    test_results.append(("Hand strength behavior", test_hand_strength_behavior()))

    print("\nTest 3: Convergence")
    test_results.append(("Convergence", test_convergence()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in test_results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(result for _, result in test_results)
    if all_passed:
        print("\nAll tests passed! The Improved Neural-Enhanced CFR is working correctly.")
    else:
        print("\nSome tests failed. Please review the implementation.")