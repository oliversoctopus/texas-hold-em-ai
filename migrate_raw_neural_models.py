"""
One-time migration script to add hidden_dim parameter to existing Raw Neural CFR models
"""

import pickle
import os
import glob

def detect_hidden_dim(network_state):
    """Detect the hidden dimension from network weights"""
    # The input_projection layer maps from 372 inputs to hidden_dim
    if 'input_projection.weight' in network_state:
        input_projection_weight = network_state['input_projection.weight']
        return input_projection_weight.shape[0]  # Output dimension is the hidden_dim
    else:
        # Default to 512 if we can't detect it
        return 512

def migrate_model(filepath):
    """Migrate a single model file to include hidden_dim parameter"""
    print(f"\nProcessing: {filepath}")

    try:
        # Load the model
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check if already migrated
        if 'hidden_dim' in checkpoint:
            print(f"  [OK] Already has hidden_dim={checkpoint['hidden_dim']}, skipping")
            return True

        # Detect hidden dimension from weights
        network_state = checkpoint.get('network_state', {})
        hidden_dim = detect_hidden_dim(network_state)
        print(f"  Detected hidden_dim={hidden_dim}")

        # Add hidden_dim to checkpoint
        checkpoint['hidden_dim'] = hidden_dim

        # Create backup
        backup_path = filepath + '.backup'
        if not os.path.exists(backup_path):
            os.rename(filepath, backup_path)
            print(f"  Created backup: {backup_path}")
        else:
            print(f"  Backup already exists: {backup_path}")

        # Save updated model
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  [SUCCESS] Successfully migrated with hidden_dim={hidden_dim}")

        return True

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        return False

def main():
    print("Raw Neural CFR Model Migration Script")
    print("=" * 60)
    print("This script will add the hidden_dim parameter to existing models")
    print("Backups will be created with .backup extension")
    print()

    # Find all raw neural models
    model_dir = "models/cfr"
    patterns = ["raw_neural*.pkl", "rn*.pkl"]

    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(os.path.join(model_dir, pattern)))

    if not model_files:
        print("No raw neural models found to migrate!")
        return

    print(f"Found {len(model_files)} model(s) to check:")
    for f in model_files:
        print(f"  - {os.path.basename(f)}")

    # Ask for confirmation
    confirm = input("\nProceed with migration? (y/n): ")
    if confirm.lower() != 'y':
        print("Migration cancelled.")
        return

    # Migrate each model
    success_count = 0
    skip_count = 0
    fail_count = 0

    for filepath in model_files:
        result = migrate_model(filepath)
        if result is True:
            if 'Already has hidden_dim' in open(filepath + '.log', 'w').read() if os.path.exists(filepath + '.log') else '':
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1

    # Actually count properly
    success_count = 0
    skip_count = 0
    fail_count = 0
    for filepath in model_files:
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            if 'hidden_dim' in checkpoint:
                success_count += 1
            else:
                fail_count += 1
        except:
            fail_count += 1

    print("\n" + "=" * 60)
    print("Migration Summary:")
    print(f"  Successfully migrated: {success_count}")
    print(f"  Failed: {fail_count}")
    print()

    # Test one of the migrated models
    if success_count > 0 and model_files:
        print("Testing migrated model...")
        test_file = model_files[0]

        try:
            from cfr.raw_neural_cfr import RawNeuralCFR

            # Try loading with the new code
            model = RawNeuralCFR()
            model.load(test_file)
            print(f"  [OK] Successfully loaded {os.path.basename(test_file)} with hidden_dim={model.hidden_dim}")
        except Exception as e:
            print(f"  [ERROR] Error loading migrated model: {e}")

    print("\nMigration complete!")
    print("Backup files have been created with .backup extension")
    print("If you need to restore, rename .backup files back to .pkl")

if __name__ == "__main__":
    main()