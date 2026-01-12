# src/cli/index_traces.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from src.traces.store import TraceStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="path to traces/<run_id>")
    ap.add_argument("--validate", action="store_true", help="print array shapes & meta")
    args = ap.parse_args()
    
    store = TraceStore(args.run_dir)
    idx = store.build_index(persist=True)
    print(f"[index] {len(idx)} example_ids, row indices :  {store.index_path}")

    if args.validate:
        print("[meta]", store.meta)
        print("[arrays]", store.arrays())

if __name__ == "__main__":
    main()
