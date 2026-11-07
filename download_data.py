#!/usr/bin/env python3
"""Restore the separately hosted data files and verify their SHA-256 hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(path, record):
    if not path.is_file():
        return False
    if path.stat().st_size != record["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["sha256"]


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "data_manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=manifest["repo_id"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            parser.error("Install the downloader first: pip install huggingface_hub")
    failed = []
    for record in manifest["files"]:
        destination = root / record["path"]
        if verify(destination, record):
            print("Verified:", record["path"])
            continue
        if not args.verify_only:
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type=manifest["repo_type"],
                revision=args.revision,
                filename=record["path"],
                local_dir=str(root),
            )
        if not verify(destination, record):
            failed.append(record["path"])
        else:
            print("Verified:", record["path"])
    if failed:
        raise SystemExit("Missing or invalid data files:\n" + "\n".join(failed))
    print("All data files verified. Only load serialized files from trusted sources.")


if __name__ == "__main__":
    main()
