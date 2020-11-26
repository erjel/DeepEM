import argparse
import json
import os
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--spec',
        type=str,
        required=True,
        help='local path to the JSON batch spec')
    parser.add_argument(
        '--params', 
        type=str,
        required=True,
        help='inference parameters')
    parser.add_argument(
        '--dry_run',
        action='store_true')
    args = parser.parse_args()

    # JSON batch spec
    with open(args.spec, "r") as f:
        batch = json.load(f)

    # Run inference
    for i, b in enumerate(batch):
        print(f"Batch run {i+1}")
        cmd = ("python deepem/test/run.py " + args.params).format(**b)
        if args.dry_run:
            print(cmd)
        else:
            subprocess.run(cmd)