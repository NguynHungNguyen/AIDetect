#!/usr/bin/env python
# peek_m4.py
# --------------------------------------------------
# Quick viewer for MBZUAI-NLP M4 dataset *.jsonl files
# --------------------------------------------------

import os, json, textwrap, argparse, glob
from pathlib import Path

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


def find_files(root):
    """Return { basename : full_path } for every *.jsonl in root."""
    files = {}
    for fp in glob.glob(os.path.join(root, "*.jsonl")):
        files[Path(fp).name] = fp
    return files


def print_samples(file_path, n=3, width=120):
    print(f"\n📂 Opening: {file_path}")
    with open(file_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rec = json.loads(line)
            print(f"\n── sample #{i+1} ──")
            # Show whichever of human_text / machine_text exists
            text_key = "human_text" if "human_text" in rec else "machine_text"
            for k in ("prompt", "human_text", "machine_text", "model", "source", "source_ID"):
                if k not in rec:
                    continue
                val = rec[k]
                if isinstance(val, str):
                    val = textwrap.shorten(val.replace("\n", " "), width)
                print(f"{k:<12}: {val}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="M4/data",
                    help="Folder containing the jsonl files (default: M4/data)")
    ap.add_argument("--file", default="arxiv_chatGPT.jsonl",
                    help="Filename to inspect (default: arxiv_chatGPT.jsonl)")
    ap.add_argument("--n", type=int, default=3, help="#samples to print")
    ap.add_argument("--pandas", action="store_true",
                    help="Load first 2 000 lines into a DataFrame (needs pandas)")
    args = ap.parse_args()

    root = args.path
    files = find_files(root)
    if args.file not in files:
        print("⚠️  File not found. Available choices:")
        for name in sorted(files):
            print("  •", name)
        return

    fp = files[args.file]
    print_samples(fp, args.n)

    # Optional DataFrame preview
    if args.pandas:
        if not PANDAS_OK:
            print("\nInstall pandas to enable DataFrame preview.")
            return
        rows = []
        with open(fp, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 2000:      # safety cap
                    break
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        print("\n🗒  DataFrame (first five rows):")
        print(df.head())


if __name__ == "__main__":
    main()
