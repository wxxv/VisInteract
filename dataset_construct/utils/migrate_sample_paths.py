"""
Script to migrate database paths in existing sample files from absolute to relative paths.

Converts:
  FROM: 'D:\\...\\dev_databases\\{db_id}\\{db_id}.sqlite'
  TO:   './databases/{db_id}.sqlite'

Target files:
  - output/samples.json
  - output/samples_full.json

Target fields:
  - data_processing_code
  - altair_code
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List


def migrate_code_paths(code: str) -> tuple[str, int]:
    """
    Replace absolute database paths with relative paths in Python code.
    
    Args:
        code: Python code string
        
    Returns:
        tuple of (modified_code, num_replacements)
    """
    # Pattern to match absolute Windows paths to databases
    # Matches patterns like:
    #   'D:\\...\\dev_databases\\{db_id}\\{db_id}.sqlite'
    #   "D:\\...\\dev_databases\\{db_id}\\{db_id}.sqlite"
    #   'D:/...dev_databases/{db_id}/{db_id}.sqlite'
    
    pattern = r'["\']([A-Za-z]:[\\\/].*?[\\\/]dev_databases[\\\/](\w+)[\\\/]\2\.sqlite)["\']'
    
    def replace_func(match):
        db_id = match.group(2)
        return f"'./databases/{db_id}.sqlite'"
    
    modified_code, count = re.subn(pattern, replace_func, code)
    
    return modified_code, count


def migrate_sample(sample: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    """
    Migrate database paths in a single sample.
    
    Args:
        sample: Sample dictionary
        
    Returns:
        tuple of (modified_sample, total_replacements)
    """
    total_replacements = 0
    modified_sample = sample.copy()
    
    # Migrate data_processing_code
    if 'data_processing_code' in sample and sample['data_processing_code']:
        modified_code, count = migrate_code_paths(sample['data_processing_code'])
        modified_sample['data_processing_code'] = modified_code
        total_replacements += count
    
    # Migrate altair_code
    if 'altair_code' in sample and sample['altair_code']:
        modified_code, count = migrate_code_paths(sample['altair_code'])
        modified_sample['altair_code'] = modified_code
        total_replacements += count
    
    return modified_sample, total_replacements


def migrate_samples_file(file_path: Path) -> tuple[bool, str]:
    """
    Migrate database paths in a samples JSON file.
    
    Args:
        file_path: Path to samples JSON file
        
    Returns:
        tuple of (success, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    print(f"\n{'='*60}")
    print(f"Migrating: {file_path}")
    print(f"{'='*60}")
    
    # Read samples
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    except Exception as e:
        return False, f"Failed to read file: {e}"
    
    if not isinstance(samples, list):
        return False, f"Invalid format: expected list, got {type(samples)}"
    
    print(f"Total samples: {len(samples)}")
    
    # Migrate each sample
    migrated_samples = []
    total_replacements = 0
    samples_modified = 0
    
    for i, sample in enumerate(samples):
        modified_sample, replacements = migrate_sample(sample)
        migrated_samples.append(modified_sample)
        
        if replacements > 0:
            samples_modified += 1
            total_replacements += replacements
            sample_id = sample.get('sample_id', f'index_{i}')
            print(f"  [{sample_id}]: {replacements} path(s) replaced")
    
    print(f"\nSummary:")
    print(f"  - Samples modified: {samples_modified}/{len(samples)}")
    print(f"  - Total replacements: {total_replacements}")
    
    # Write migrated samples
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(migrated_samples, f, ensure_ascii=False, indent=2)
        print(f"✓ Successfully migrated: {file_path}")
        return True, f"Migrated {samples_modified} samples with {total_replacements} replacements"
    except Exception as e:
        return False, f"Failed to write file: {e}"


def main():
    """Main migration script."""
    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "dataset_construct" / "output"
    
    print("\n" + "="*60)
    print("Database Path Migration Script")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    
    # Target files
    target_files = [
        output_dir / "samples.json",
        output_dir / "samples_full.json"
    ]
    
    # Check if files exist
    existing_files = [f for f in target_files if f.exists()]
    
    if not existing_files:
        print("\n⚠ No sample files found to migrate.")
        print(f"  Expected locations:")
        for f in target_files:
            print(f"    - {f}")
        return 0
    
    print(f"\nFound {len(existing_files)} file(s) to migrate:")
    for f in existing_files:
        print(f"  - {f.name}")
    
    # Confirm migration
    print("\n" + "="*60)
    print("⚠ WARNING: This will directly modify the files.")
    print("  Make sure you have backed up your files!")
    print("="*60)
    response = input("\nContinue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nMigration cancelled.")
        return 0
    
    # Migrate each file
    results = []
    for file_path in existing_files:
        success, message = migrate_samples_file(file_path)
        results.append((file_path.name, success, message))
    
    # Print final summary
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    
    for filename, success, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {filename}: {message}")
    
    # Return exit code
    all_success = all(success for _, success, _ in results)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
