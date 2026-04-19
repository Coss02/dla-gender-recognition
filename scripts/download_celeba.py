"""Download and prepare the CelebA dataset for the gender recognition pipeline.

Tries multiple download strategies in order of reliability:
  1. Kaggle API (requires kaggle credentials)
  2. gdown from Google Drive (may hit rate limits)
  3. torchvision built-in downloader (wraps gdown, same limits)

After download, validates that all required files are present and
prints dataset statistics.

Usage:
    python scripts/download_celeba.py
    python scripts/download_celeba.py --method kaggle
    python scripts/download_celeba.py --data-dir data/celeba
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_FILES = [
    "img_align_celeba",          # directory with ~200K images
    "list_attr_celeba.txt",      # attribute annotations
    "list_eval_partition.txt",   # official train/val/test split
]

GDRIVE_FILES = {
    "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download CelebA dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data/celeba",
        help="Target directory for CelebA files (default: data/celeba)",
    )
    parser.add_argument(
        "--method", type=str, default="auto",
        choices=["auto", "kaggle", "gdrive", "torchvision"],
        help="Download method (default: auto — tries all in order)",
    )
    return parser.parse_args()


def validate_dataset(data_dir: Path) -> bool:
    """Check that all required CelebA files are present."""
    missing = []
    for name in REQUIRED_FILES:
        path = data_dir / name
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False

    img_dir = data_dir / "img_align_celeba"
    n_images = len(list(img_dir.glob("*.jpg")))
    if n_images < 100:
        print(f"Only {n_images} images found, expected ~200K")
        return False

    print(f"Dataset validated: {n_images:,} images found")
    return True


def print_dataset_stats(data_dir: Path) -> None:
    """Print basic dataset statistics after successful download."""
    import pandas as pd

    attrs = pd.read_csv(data_dir / "list_attr_celeba.txt", sep=r"\s+", header=1, index_col=0)
    parts = pd.read_csv(
        data_dir / "list_eval_partition.txt",
        sep=r"\s+", header=None, names=["filename", "partition"], index_col=0,
    )

    n_male = (attrs["Male"] == 1).sum()
    n_female = (attrs["Male"] == -1).sum()
    n_total = len(attrs)

    split_names = {0: "train", 1: "val", 2: "test"}

    print(f"\n{'='*45}")
    print(f"  CelebA Dataset Summary")
    print(f"{'='*45}")
    print(f"  Total images:  {n_total:>8,}")
    print(f"  Male:          {n_male:>8,} ({100*n_male/n_total:.1f}%)")
    print(f"  Female:        {n_female:>8,} ({100*n_female/n_total:.1f}%)")
    print(f"{'─'*45}")
    for code, name in split_names.items():
        count = (parts["partition"] == code).sum()
        print(f"  {name:>5} split:  {count:>8,}")
    print(f"{'='*45}\n")


def download_kaggle(data_dir: Path) -> bool:
    """Download CelebA from Kaggle (most reliable, requires API key)."""
    print("\n[1/3] Trying Kaggle download...")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset",
             "-p", str(data_dir), "--unzip"],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            if "403" in result.stderr or "Could not find kaggle.json" in result.stderr:
                print("  Kaggle credentials not configured.")
                print("  To set up: https://www.kaggle.com/docs/api#authentication")
            else:
                print(f"  Kaggle download failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  kaggle CLI not found. Install with: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("  Kaggle download timed out (30 min limit)")
        return False

    # Kaggle unpacks into a nested structure, fix it
    _fix_kaggle_layout(data_dir)
    return validate_dataset(data_dir)


def _fix_kaggle_layout(data_dir: Path) -> None:
    """Kaggle unpacks CelebA into nested folders. Flatten to expected layout."""
    # Kaggle structure: data_dir/img_align_celeba/img_align_celeba/*.jpg
    nested_img_dir = data_dir / "img_align_celeba" / "img_align_celeba"
    if nested_img_dir.is_dir():
        target = data_dir / "img_align_celeba_flat"
        shutil.move(str(nested_img_dir), str(target))
        shutil.rmtree(str(data_dir / "img_align_celeba"))
        target.rename(data_dir / "img_align_celeba")

    # Kaggle may put annotations in a subfolder
    for name in ["list_attr_celeba.csv", "list_attr_celeba.txt",
                 "list_eval_partition.csv", "list_eval_partition.txt"]:
        # Check nested locations
        for subdir in [data_dir, data_dir / "celeba-dataset"]:
            src = subdir / name
            if src.exists() and src.parent != data_dir:
                shutil.move(str(src), str(data_dir / name))

    # Kaggle sometimes provides CSV instead of TXT; our pipeline reads TXT
    _convert_kaggle_csv_to_txt(data_dir)


def _convert_kaggle_csv_to_txt(data_dir: Path) -> None:
    """Convert Kaggle CSV annotations to CelebA TXT format if needed."""
    import pandas as pd

    attr_csv = data_dir / "list_attr_celeba.csv"
    attr_txt = data_dir / "list_attr_celeba.txt"
    if attr_csv.exists() and not attr_txt.exists():
        print("  Converting list_attr_celeba.csv -> .txt ...")
        df = pd.read_csv(attr_csv, index_col=0)
        with open(attr_txt, "w") as f:
            f.write(f"{len(df)}\n")
            f.write(" ".join(df.columns) + "\n")
            for idx, row in df.iterrows():
                values = " ".join(str(v) for v in row.values)
                f.write(f"{idx} {values}\n")

    part_csv = data_dir / "list_eval_partition.csv"
    part_txt = data_dir / "list_eval_partition.txt"
    if part_csv.exists() and not part_txt.exists():
        print("  Converting list_eval_partition.csv -> .txt ...")
        df = pd.read_csv(part_csv, header=0)
        with open(part_txt, "w") as f:
            for _, row in df.iterrows():
                f.write(f"{row.iloc[0]} {row.iloc[1]}\n")


def download_gdrive(data_dir: Path) -> bool:
    """Download CelebA files directly from Google Drive via gdown."""
    print("\n[2/3] Trying Google Drive download...")

    try:
        import gdown
    except ImportError:
        print("  gdown not installed. Install with: pip install gdown")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in GDRIVE_FILES.items():
        target = data_dir / filename
        if target.exists():
            print(f"  {filename} already exists, skipping")
            continue

        print(f"  Downloading {filename} ...")
        try:
            gdown.download(id=file_id, output=str(target), quiet=False)
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            return False

    # Unzip images
    zip_path = data_dir / "img_align_celeba.zip"
    img_dir = data_dir / "img_align_celeba"
    if zip_path.exists() and not img_dir.exists():
        print("  Extracting img_align_celeba.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
        print("  Extraction complete, zip removed")

    return validate_dataset(data_dir)


def download_torchvision(data_dir: Path) -> bool:
    """Download CelebA using torchvision's built-in downloader."""
    print("\n[3/3] Trying torchvision download...")

    try:
        import torchvision
        # torchvision expects root to be the parent of 'celeba/'
        root = data_dir.parent
        torchvision.datasets.CelebA(root=root, split="all", download=True)
    except Exception as e:
        print(f"  torchvision download failed: {e}")
        return False

    return validate_dataset(data_dir)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Skip download if already valid
    if validate_dataset(data_dir):
        print("Dataset already present and valid. Nothing to do.")
        print_dataset_stats(data_dir)
        return

    methods = {
        "kaggle": download_kaggle,
        "gdrive": download_gdrive,
        "torchvision": download_torchvision,
    }

    if args.method != "auto":
        success = methods[args.method](data_dir)
    else:
        success = False
        for name, fn in methods.items():
            success = fn(data_dir)
            if success:
                break

    if success:
        print("\nDownload successful!")
        print_dataset_stats(data_dir)
    else:
        print("\n" + "="*60)
        print("  ALL AUTOMATIC DOWNLOAD METHODS FAILED")
        print("="*60)
        print()
        print("Please download CelebA manually:")
        print()
        print("  Option A — Kaggle (recommended):")
        print("    1. Create account at https://www.kaggle.com")
        print("    2. Go to Account > API > Create New Token")
        print("    3. Place kaggle.json in ~/.kaggle/")
        print("    4. Run: python scripts/download_celeba.py --method kaggle")
        print()
        print("  Option B — Manual download:")
        print("    1. Go to https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("    2. Download: img_align_celeba.zip, list_attr_celeba.txt,")
        print("       list_eval_partition.txt")
        print(f"    3. Place files in: {data_dir.resolve()}/")
        print(f"    4. Unzip img_align_celeba.zip inside {data_dir.resolve()}/")
        print()
        print("  Expected layout:")
        print(f"    {data_dir}/")
        print(f"    ├── img_align_celeba/   (~200K .jpg files)")
        print(f"    ├── list_attr_celeba.txt")
        print(f"    └── list_eval_partition.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
