"""Sync generated digit images from a cloud-mounted folder into local data splits.

Expected filename format: digit_X_NNNNNN.png
Expected source structure:
    <source>/train/*.png
    <source>/val/*.png
    <source>/test/*.png
"""

import argparse
import os
import re
import shutil
from pathlib import Path


FILENAME_PATTERN = re.compile(r"^digit_[0-9]_[0-9]{6}\.png$")
SPLITS = ("train", "val", "test")


def collect_png_files(split_dir: Path):
    return [p for p in split_dir.rglob("*.png") if p.is_file()]


def validate_filename(path: Path) -> bool:
    return bool(FILENAME_PATTERN.match(path.name))


def sync_split(source_split_dir: Path, target_split_dir: Path, overwrite: bool):
    target_split_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_existing = 0
    invalid = []

    for src in collect_png_files(source_split_dir):
        if not validate_filename(src):
            invalid.append(src)
            continue

        dst = target_split_dir / src.name
        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied, skipped_existing, invalid


def main():
    parser = argparse.ArgumentParser(
        description="Sync cloud-generated PNG images into local data/train|val|test."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source root directory (cloud-mounted/local synced path containing train/val/test).",
    )
    parser.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Target data root directory (default: <project>/data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in target splits.",
    )
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    target_root = Path(args.target).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    total_copied = 0
    total_skipped_existing = 0
    total_invalid = []

    for split in SPLITS:
        source_split = source_root / split
        if not source_split.exists():
            raise FileNotFoundError(
                f"Source split directory not found: {source_split}. "
                "Ensure the source contains train/, val/, and test/ subdirectories."
            )

        target_split = target_root / split
        copied, skipped_existing, invalid = sync_split(source_split, target_split, args.overwrite)
        total_copied += copied
        total_skipped_existing += skipped_existing
        total_invalid.extend(invalid)
        print(
            f"[{split}] copied {copied} files, skipped existing {skipped_existing} -> {target_split}"
        )

    print(f"Done. Total copied: {total_copied}")
    print(f"Total skipped existing: {total_skipped_existing}")
    if total_invalid:
        print(
            "\nFound invalid filenames "
            "(expected like digit_X_123456.png, where X is 0-9 and 123456 is 6 digits):"
        )
        for path in total_invalid:
            print(f" - {path}")
        raise ValueError(f"Found {len(total_invalid)} invalid filename(s).")


if __name__ == "__main__":
    main()
