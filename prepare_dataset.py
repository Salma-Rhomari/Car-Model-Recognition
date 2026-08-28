import argparse
import os
import random
import shutil

import scipy.io


def sanitize_class_name(name: str) -> str:
    """Turn 'BMW 3 Series Sedan 2012' into 'bmw_3_series_sedan_2012'."""
    return (
        name.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_annotations(devkit_dir: str):
    """Return list of (filename, class_name) and sorted list of class names."""
    meta = scipy.io.loadmat(os.path.join(devkit_dir, "cars_meta.mat"))
    class_names = [c[0] for c in meta["class_names"][0]]

    annos = scipy.io.loadmat(os.path.join(devkit_dir, "cars_train_annos.mat"))
    annotations = annos["annotations"][0]

    records = []
    for entry in annotations:
        fname = entry["fname"][0]
        class_idx = int(entry["class"][0][0]) - 1  # MATLAB is 1-indexed
        class_name = class_names[class_idx]
        records.append((fname, class_name))

    return records, class_names


def split_records(records, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Group by class, then split each class's images into train/val/test."""
    by_class = {}
    for fname, class_name in records:
        by_class.setdefault(class_name, []).append(fname)

    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name, filenames in by_class.items():
        random.shuffle(filenames)
        n = len(filenames)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_files = filenames[:n_train]
        val_files = filenames[n_train:n_train + n_val]
        test_files = filenames[n_train + n_val:]

        # Guard against empty test split on very small classes
        if not test_files and len(filenames) > n_train + n_val:
            test_files = filenames[n_train + n_val:]

        splits["train"].extend((f, class_name) for f in train_files)
        splits["val"].extend((f, class_name) for f in val_files)
        splits["test"].extend((f, class_name) for f in test_files)

    return splits


def copy_files(splits, images_dir, output_dir):
    for split_name, items in splits.items():
        print(f"\n{split_name}: {len(items)} images")
        for fname, class_name in items:
            class_slug = sanitize_class_name(class_name)
            dest_dir = os.path.join(output_dir, split_name, class_slug)
            os.makedirs(dest_dir, exist_ok=True)

            src_path = os.path.join(images_dir, fname)
            dest_path = os.path.join(dest_dir, fname)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"  WARNING: missing file {src_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devkit-dir", required=True, help="Path to devkit/ folder (contains .mat files)")
    parser.add_argument("--images-dir", required=True, help="Path to cars_train/ folder")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    print("Loading annotations...")
    records, class_names = load_annotations(args.devkit_dir)
    print(f"Found {len(records)} labeled images across {len(class_names)} classes")

    print("\nSplitting into train/val/test...")
    splits = split_records(records, args.train_ratio, args.val_ratio)

    print("\nCopying files (this may take a few minutes)...")
    copy_files(splits, args.images_dir, args.output_dir)

    print("\nDone! Dataset organized at:", args.output_dir)


if __name__ == "__main__":
    main()