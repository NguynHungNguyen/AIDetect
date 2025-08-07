#!/usr/bin/env python
import os, json, glob, argparse
from pathlib import Path

def summarize(root):
    pattern = os.path.join(root, "*.jsonl")
    for fp in sorted(glob.glob(pattern)):
        seen_keys = set()
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_keys |= set(rec.keys())
        name = Path(fp).name
        print(f"{name}: {sorted(seen_keys)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarize JSONL keys per file")
    p.add_argument("root", help="folder containing .jsonl files")
    args = p.parse_args()
    summarize(args.root)
