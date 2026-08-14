"""
HF-008 — Model Cache
====================

Everything you download from the Hub lands in a local *cache*. This lesson
opens the cache, explains its layout, measures disk usage, and shows how to
remove models you no longer need (and how to force re-downloads).

Usage:

    python 08_Model_Cache.py --list          # all cached models + sizes
    python 08_Model_Cache.py --details       # cache location + layout
    python 08_Model_Cache.py --du            # total disk usage snapshot
    python 08_Model_Cache.py --remove distilgpt2   # delete a cached model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import huggingface_hub


def human(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def cache_root() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def fmt_size(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def demo_list(max_items: int) -> None:
    from huggingface_hub import scan_cache_dir

    print("=" * 60)
    print("1) what is in the cache")
    print("=" * 60)
    info = scan_cache_dir(cache_dir=cache_root())
    print(f"  cache location : {cache_root()}")
    print(f"  models cached  : {len(info.repos)}")
    print(f"  total size     : {getattr(info, 'size_on_disk_str', human(info.size_on_disk))}\n")

    for repo in sorted(info.repos, key=lambda r: r.size_on_disk, reverse=True)[:max_items]:
        refs = sorted(repo.refs.keys()) if hasattr(repo, "refs") else []
        revs = " | ".join(
            f"{rev.commit_hash[:8]} {getattr(rev, 'size_on_disk_str', fmt_size(rev.size_on_disk))}"
            for rev in repo.revisions
        )
        print(f"  {repo.repo_id:<50} {getattr(repo, 'size_on_disk_str', fmt_size(repo.size_on_disk))}")
        if refs:
            print(f"    refs ({', '.join(refs)}) — revisions: {revs}")
        else:
            print(f"    revisions: {revs}")
    print()


def demo_details() -> None:
    print("=" * 60)
    print("2) cache layout")
    print("=" * 60)
    root = cache_root()
    print(f"  default location: {root}")
    print(f"  env variables   : HF_HOME, HF_HUB_CACHE (HF_HOME wins)")
    print(f"  models live in  : {root}")
    if root.exists():
        for p in sorted(root.iterdir())[:8]:
            kind = "dir" if p.is_dir() else "file"
            print(f"    - {p.name:<60} {kind}")
    print()


def demo_du() -> None:
    from huggingface_hub import scan_cache_dir

    print("=" * 60)
    print("3) disk usage snapshot")
    print("=" * 60)
    info = scan_cache_dir(cache_dir=cache_root())
    print(f"  models cached  : {len(info.repos)}")
    print(f"  revisions      : {sum(len(r.revisions) for r in info.repos)}")
    print(f"  total on disk  : {getattr(info, 'size_on_disk_str', human(info.size_on_disk))}")
    print()


def demo_remove(model_id: str) -> None:
    from huggingface_hub import delete_revisions, scan_cache_dir

    print("=" * 60)
    print("4) removing a cached model")
    print("=" * 60)
    info = scan_cache_dir(cache_dir=cache_root())
    repo = next((r for r in info.repos if r.repo_id == model_id), None)
    if repo is None:
        print(f"  '{model_id}' is not in the cache — nothing to remove.")
        return

    print(f"  found '{model_id}' with revisions:")
    for rev in repo.revisions:
        print(f"    - {rev.commit_hash[:12]}  {getattr(rev, 'size_on_disk_str', fmt_size(rev.size_on_disk))}")
    print()

    print("  deleting ... (re-download later with any from_pretrained call)")
    delete_revisions(repo_id=model_id, revisions=[r.commit_hash for r in repo.revisions])
    info = scan_cache_dir(cache_dir=cache_root())
    remaining = next((r for r in info.repos if r.repo_id == model_id), None)
    print(f"  after delete: cached={remaining is None}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and manage the Hugging Face model cache.")
    parser.add_argument("--list", action="store_true", help="list all cached models")
    parser.add_argument("--details", action="store_true", help="show cache location and layout")
    parser.add_argument("--du", action="store_true", help="disk usage snapshot")
    parser.add_argument("--remove", metavar="MODEL_ID", help="delete a cached model")
    parser.add_argument("--max", type=int, default=25, help="max models to list (default 25)")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-008 — Model Cache")
    print("=" * 60)
    print(f"huggingface_hub version: {huggingface_hub.__version__}")
    print(f"HF cache root: {cache_root()}\n")

    if not (args.list or args.details or args.du or args.remove):
        args.list = True  # sensible default: show the cache

    if args.details:
        demo_details()
    if args.list:
        demo_list(args.max)
    if args.du:
        demo_du()
    if args.remove:
        demo_remove(args.remove)

    print("=" * 60)
    print("Cache = your disk, your control. Delete freely; downloads re-run.")


if __name__ == "__main__":
    main()