"""
Script to add type flags to existing Raw Neural CFR models
"""

import pickle
import os

def add_type_flag_to_model(filepath, model_type='raw_neural_cfr'):
    """Add type flag to an existing model file"""

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    try:
        # Load the existing model
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check if type flag already exists
        if 'type' in checkpoint:
            print(f"Model {filepath} already has type flag: {checkpoint['type']}")
            return True

        # Add the type flag
        checkpoint['type'] = model_type

        # Save the updated model
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Successfully added type flag '{model_type}' to {filepath}")
        return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix the two Raw Neural CFR models"""

    print("Adding type flags to Raw Neural CFR models...")
    print("=" * 60)

    models_dir = "models/cfr"

    # List of models to fix
    models_to_fix = [
        "raw_neural_quick.pkl",
        "raw_neural_tiny.pkl"
    ]

    success_count = 0

    for model_file in models_to_fix:
        filepath = os.path.join(models_dir, model_file)
        print(f"\nProcessing: {model_file}")

        if add_type_flag_to_model(filepath, 'raw_neural_cfr'):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{len(models_to_fix)} models fixed")

    # Also check for any other raw_neural models
    print("\nChecking for other raw_neural models...")
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.startswith("raw_neural") and file.endswith(".pkl"):
                if file not in models_to_fix:
                    filepath = os.path.join(models_dir, file)
                    print(f"Found additional model: {file}")
                    add_type_flag_to_model(filepath, 'raw_neural_cfr')

if __name__ == "__main__":
    main()