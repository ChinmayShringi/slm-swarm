#!/usr/bin/env python3
"""
Script to download and prepare SAMSum dataset for evaluation.

Usage:
    python prepare_samsum.py [--output dataset/samsum.jsonl] [--split test]
"""

import json
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    exit(1)


def prepare_samsum(output_path: str, split: str = "test"):
    """
    Download SAMSum dataset from HuggingFace and convert to JSONL format
    
    Args:
        output_path: Path to output JSONL file
        split: Dataset split to use (train, validation, or test)
    """
    print(f"Loading SAMSum dataset (split: {split})...")
    
    try:
        # Try different possible names for SAMSum
        dataset = None
        for name in ["samsum", "Samsung/samsum", "knkarthick/samsum"]:
            try:
                print(f"Trying dataset name: {name}")
                dataset = load_dataset(name, split=split)
                print(f"✓ Successfully loaded from: {name}")
                break
            except:
                continue
        
        if dataset is None:
            raise Exception("Could not find SAMSum dataset")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAlternatively, you can download from: https://huggingface.co/datasets/samsum")
        return
    
    print(f"Loaded {len(dataset)} examples")
    
    # Convert to our format
    output_data = []
    for idx, example in enumerate(dataset):
        entry = {
            "id": f"samsum_{split}_{idx:04d}",
            "message": example["dialogue"],
            "reference": {
                "tldr": example["summary"]
            }
        }
        output_data.append(entry)
    
    # Save to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nSuccessfully saved {len(output_data)} examples to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Show sample
    if output_data:
        print("\nSample entry:")
        print(json.dumps(output_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare SAMSum dataset for evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="../dataset/samsum.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )
    
    args = parser.parse_args()
    
    prepare_samsum(args.output, args.split)


if __name__ == "__main__":
    main()

