#!/usr/bin/env python3
"""
Update training configuration with optimized values for stratified data
"""

import json

NOTEBOOK_PATH = 'MARL_STL_BFT_Training_v1_T4_Optimized [FIXED].ipynb'
CONFIG_CELL_INDEX = 22

def update_config():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Get the config cell
    cell = nb['cells'][CONFIG_CELL_INDEX]
    source = ''.join(cell.get('source', []))

    print("=" * 80)
    print("UPDATING TRAINING CONFIGURATION FOR STRATIFIED DATA")
    print("=" * 80)
    print()

    # Print current values
    print("Current configuration:")
    for line in source.split('\n'):
        if "'lr':" in line or "'entropy_coef':" in line or "'value_coef':" in line:
            print(f"  {line.strip()}")
    print()

    # Update values
    lines = source.split('\n')
    new_lines = []
    updated_count = 0

    for line in lines:
        # Update lr
        if "'lr':" in line and 'FIXED' in line:
            new_line = line.split('#')[0].rstrip()  # Remove old comment
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + "'lr': 3e-4,  # UPDATED: Reduced from 1e-3 for stable learning with stratified data"
            new_lines.append(new_line)
            print("✓ Updated lr: 1e-3 → 3e-4")
            updated_count += 1

        # Update entropy_coef
        elif "'entropy_coef':" in line and 'FIXED' in line:
            new_line = line.split('#')[0].rstrip()  # Remove old comment
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + "'entropy_coef': 0.05,  # UPDATED: Reduced from 0.15 for better exploitation with stratified data"
            new_lines.append(new_line)
            print("✓ Updated entropy_coef: 0.15 → 0.05")
            updated_count += 1

        # Update value_coef
        elif "'value_coef':" in line and 'FIXED' in line:
            new_line = line.split('#')[0].rstrip()  # Remove old comment
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + "'value_coef': 1.5,  # UPDATED: Increased from 1.0 for stronger value learning"
            new_lines.append(new_line)
            print("✓ Updated value_coef: 1.0 → 1.5")
            updated_count += 1

        else:
            new_lines.append(line)

    if updated_count != 3:
        print(f"\n⚠️  Warning: Only updated {updated_count}/3 values")
        return False

    # Update cell source (preserve line endings)
    new_source = '\n'.join(new_lines)
    cell['source'] = [line + '\n' if i < len(new_lines) - 1 else line
                     for i, line in enumerate(new_lines)]

    # Write updated notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=2)

    print()
    print("=" * 80)
    print("✅ CONFIGURATION UPDATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("New configuration for stratified data:")
    print("  lr: 3e-4 (was 1e-3)")
    print("  entropy_coef: 0.05 (was 0.15)")
    print("  value_coef: 1.5 (was 1.0)")
    print()
    print("These values are optimized for the properly stratified dataset where")
    print("all splits have consistent 10.93% threat distribution.")
    print()

    return True

if __name__ == "__main__":
    success = update_config()
    if not success:
        exit(1)
