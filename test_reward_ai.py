"""
Test script for Reward-Based Neural Network AI
"""

import torch
from reward_nn.reward_based_ai import RewardBasedAI
from reward_nn.training import RewardBasedTrainer
from core.game_constants import Action

def test_network_initialization():
    """Test that the neural network initializes correctly"""
    print("Testing network initialization...")

    ai = RewardBasedAI(hidden_dim=256)

    # Test that network exists
    assert ai.network is not None
    assert ai.optimizer is not None

    # Test forward pass with dummy state
    dummy_state = {
        'hole_cards': [],
        'community_cards': [],
        'pot_size': 100,
        'to_call': 20,
        'stack_size': 980,
        'position': 0,
        'num_players': 2,
        'players_in_hand': 2,
        'action_history': [],
        'hand_phase': 0,
        'big_blind': 20,
        'is_preflop_aggressor': 0
    }

    policy_logits, value = ai.network(dummy_state)

    assert policy_logits.shape == (5,)  # 5 actions
    assert value.shape == (1,)

    print("[OK] Network initialization successful")


def test_action_selection():
    """Test that the AI can select valid actions"""
    print("\nTesting action selection...")

    ai = RewardBasedAI(hidden_dim=256)

    # Create dummy state
    state = {
        'hole_cards': [],
        'community_cards': [],
        'pot_size': 100,
        'to_call': 20,
        'stack_size': 980,
        'position': 0,
        'num_players': 2,
        'players_in_hand': 2,
        'action_history': [],
        'hand_phase': 0,
        'big_blind': 20,
        'is_preflop_aggressor': 0
    }

    # Test with all valid actions
    valid_actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE, Action.ALL_IN]

    for _ in range(10):
        action = ai.choose_action(state, valid_actions, training=True)
        assert action in valid_actions

    print("[OK] Action selection working")


def test_training_loop():
    """Test a minimal training loop"""
    print("\nTesting training loop...")

    ai = RewardBasedAI(hidden_dim=256, learning_rate=1e-3)
    trainer = RewardBasedTrainer(ai, num_players=2)

    # Train for just a few hands to test functionality
    print("Training for 10 hands...")
    trainer.train(num_hands=10, batch_size=4, update_every=5, verbose=False)

    # Check that some training happened
    assert trainer.hand_count == 10

    print("[OK] Training loop successful")


def test_memory_and_learning():
    """Test that experiences are stored and used for learning"""
    print("\nTesting memory and learning...")

    ai = RewardBasedAI(hidden_dim=256)

    # Create dummy experiences
    state = ai.get_state_features(
        hand=[], community_cards=[], pot=100, current_bet=20,
        player_chips=980, player_bet=0, num_players=2,
        players_in_hand=2, position=0
    )

    # Store some experiences
    for i in range(10):
        reward = (i - 5) / 10  # Mix of positive and negative rewards
        ai.remember(state, Action.CALL, reward, state, False)

    assert len(ai.memory) == 10

    # Try training on the batch
    initial_loss_count = len(ai.loss_history)
    ai.train_on_batch(batch_size=5, epochs=1)

    # Check that training occurred
    assert len(ai.loss_history) > initial_loss_count

    print("[OK] Memory and learning working")


def test_save_and_load():
    """Test model saving and loading"""
    print("\nTesting save and load...")

    import os
    import tempfile

    ai1 = RewardBasedAI(hidden_dim=256)

    # Train briefly to modify weights
    trainer = RewardBasedTrainer(ai1, num_players=2)
    trainer.train(num_hands=5, verbose=False)

    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        filepath = tmp.name

    ai1.save(filepath)

    # Load into new model
    ai2 = RewardBasedAI(hidden_dim=256)
    ai2.load(filepath)

    # Check that parameters match
    for p1, p2 in zip(ai1.network.parameters(), ai2.network.parameters()):
        assert torch.allclose(p1, p2)

    # Clean up
    os.remove(filepath)

    print("[OK] Save and load working")


def test_reward_calculation():
    """Test that rewards are properly calculated in BB"""
    print("\nTesting reward calculation...")

    ai = RewardBasedAI(hidden_dim=256)
    trainer = RewardBasedTrainer(ai, num_players=2, big_blind=20)

    # Simulate a hand and check reward calculation
    reward_bb, won = trainer.simulate_hand(0)

    # Reward should be in reasonable range (typically -50 to +50 BB per hand)
    assert -100 <= reward_bb <= 100
    assert isinstance(won, bool)

    print(f"[OK] Reward calculation working (sample reward: {reward_bb:+.2f} BB)")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Reward-Based AI Tests")
    print("="*60)

    tests = [
        test_network_initialization,
        test_action_selection,
        test_training_loop,
        test_memory_and_learning,
        test_save_and_load,
        test_reward_calculation
    ]

    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"X {test_func.__name__} failed: {e}")
            failed.append(test_func.__name__)

    print("\n" + "="*60)
    if failed:
        print(f"Tests failed: {', '.join(failed)}")
    else:
        print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()