"""Test script to verify different model sizes work correctly"""

import torch
from cfr.raw_neural_cfr import RawNeuralCFR, RawNeuralCFRNetwork
import time

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_sizes():
    print("Testing Raw Neural CFR with different model sizes")
    print("=" * 60)

    # Test different model sizes
    sizes = [
        (128, "Tiny"),
        (256, "Small"),
        (512, "Standard"),
        (768, "Large"),
    ]

    for hidden_dim, name in sizes:
        print(f"\nTesting {name} model (hidden_dim={hidden_dim})...")

        # Create the network
        network = RawNeuralCFRNetwork(hidden_dim=hidden_dim)

        # Count parameters
        param_count = count_parameters(network)
        print(f"  Total parameters: {param_count:,}")

        # Test forward pass
        test_state = {
            'hole_cards': [None, None],
            'community_cards': [],
            'pot': 100,
            'stacks': [900, 950],
            'current_player': 0,
            'button': 0,
            'current_bets': [10, 20],
            'active_players': [True, True],
            'stage': 0,
            'action_history': [],
            'legal_actions': [1, 1, 1, 1, 1, 1, 1, 1]
        }

        # Time the forward pass
        start_time = time.time()
        with torch.no_grad():
            strategy, value = network(test_state)
        forward_time = time.time() - start_time

        print(f"  Strategy output shape: {strategy.shape}")
        print(f"  Value output shape: {value.shape}")
        print(f"  Forward pass time: {forward_time*1000:.2f}ms")

        # Memory estimate (rough)
        memory_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"  Estimated memory: {memory_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Testing full RawNeuralCFR with different sizes...")

    # Test creating full CFR models
    for hidden_dim, name in [(256, "Small"), (512, "Standard")]:
        print(f"\n{name} model (hidden_dim={hidden_dim}):")

        # Create CFR model
        cfr_model = RawNeuralCFR(
            iterations=10,  # Very few iterations just for testing
            hidden_dim=hidden_dim
        )

        # Skip training test (takes too long)
        print("  Model created successfully")

        # Skip get_action test (requires proper initialization)
        print("  Skipping action test (model not trained)")

        # Test save/load
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_filename = tmp.name

        print(f"  Testing save/load...")
        cfr_model.save(temp_filename)
        file_size = os.path.getsize(temp_filename) / (1024 * 1024)
        print(f"  Saved model size: {file_size:.2f} MB")

        # Test loading
        loaded_model = RawNeuralCFR()
        loaded_model.load(temp_filename)
        print(f"  Successfully loaded model with hidden_dim={loaded_model.hidden_dim}")

        # Cleanup
        os.remove(temp_filename)

    print("\n" + "=" * 60)
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_model_sizes()