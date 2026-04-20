"""CelebA dataset wrapper for binary gender classification.

Reads the official CelebA annotation and partition files to produce
a PyTorch Dataset that returns (image_tensor, label) pairs where
label is 0 (female) or 1 (male).
"""

from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

SPLIT_MAP: dict[str, int] = {"train": 0, "val": 1, "test": 2}


class CelebAGenderDataset(Dataset):
    """CelebA dataset filtered to the binary gender attribute.

    Expected directory layout under ``root_dir``:
        img_align_celeba/       — cropped face images
        list_attr_celeba.txt    — attribute annotations
        list_eval_partition.txt — official train/val/test split
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "img_align_celeba"
        self.transform = transform

        if split not in SPLIT_MAP:
            raise ValueError(f"split must be one of {list(SPLIT_MAP.keys())}, got '{split}'")

        attrs = self._load_attributes()
        partitions = self._load_partitions()

        merged = attrs.join(partitions, how="inner")
        mask = merged["partition"] == SPLIT_MAP[split]
        self.data = merged.loc[mask].rename_axis("filename").reset_index()

    def _load_attributes(self) -> pd.DataFrame:
        """Parse list_attr_celeba.txt into a DataFrame with a binary 'male' column."""
        attr_path = self.root_dir / "list_attr_celeba.txt"
        if not attr_path.exists():
            raise FileNotFoundError(
                f"Attribute file not found: {attr_path}. "
                "Download CelebA and place it under the root_dir."
            )

        df = pd.read_csv(attr_path, sep=r"\s+", header=1, index_col=0)
        # CelebA encodes attributes as -1/1; convert Male to 0/1
        df["male"] = (df["Male"] == 1).astype(int)
        return df[["male"]]

    def _load_partitions(self) -> pd.DataFrame:
        """Parse list_eval_partition.txt into a DataFrame."""
        part_path = self.root_dir / "list_eval_partition.txt"
        if not part_path.exists():
            raise FileNotFoundError(
                f"Partition file not found: {part_path}. "
                "Download CelebA and place it under the root_dir."
            )

        df = pd.read_csv(
            part_path,
            sep=r"\s+",
            header=None,
            names=["filename", "partition"],
            index_col=0,
        )
        return df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        row = self.data.iloc[idx]
        img_path = self.image_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = row["male"]
        return image, label
