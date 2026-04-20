"""Tests for the CelebA data pipeline.

Uses small synthetic files so no real dataset download is needed.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import CelebAGenderDataset
from src.data.transforms import get_eval_transforms, get_train_transforms


@pytest.fixture()
def fake_celeba(tmp_path: Path):
    """Create a minimal fake CelebA directory structure."""
    img_dir = tmp_path / "img_align_celeba"
    img_dir.mkdir()

    filenames = [f"{i:06d}.jpg" for i in range(1, 7)]
    for fname in filenames:
        img = Image.new("RGB", (178, 218), color="red")
        img.save(img_dir / fname)

    # Attribute file: header count + header row + data rows
    # CelebA uses -1/1 encoding for attributes
    attr_lines = [
        "6",
        "filename Male",
        "000001.jpg  1",
        "000002.jpg -1",
        "000003.jpg  1",
        "000004.jpg -1",
        "000005.jpg  1",
        "000006.jpg -1",
    ]
    (tmp_path / "list_attr_celeba.txt").write_text("\n".join(attr_lines) + "\n")

    # Partition file: 0=train, 1=val, 2=test
    part_lines = [
        "000001.jpg 0",
        "000002.jpg 0",
        "000003.jpg 0",
        "000004.jpg 1",
        "000005.jpg 2",
        "000006.jpg 2",
    ]
    (tmp_path / "list_eval_partition.txt").write_text("\n".join(part_lines) + "\n")

    return tmp_path


@pytest.fixture()
def fake_celeba_official_header(tmp_path: Path):
    """Create a minimal CelebA layout matching the official header format."""
    img_dir = tmp_path / "img_align_celeba"
    img_dir.mkdir()

    filenames = [f"{i:06d}.jpg" for i in range(1, 4)]
    for fname in filenames:
        img = Image.new("RGB", (178, 218), color="green")
        img.save(img_dir / fname)

    attr_lines = [
        "3",
        "Male",
        "000001.jpg  1",
        "000002.jpg -1",
        "000003.jpg  1",
    ]
    (tmp_path / "list_attr_celeba.txt").write_text("\n".join(attr_lines) + "\n")

    part_lines = [
        "000001.jpg 0",
        "000002.jpg 1",
        "000003.jpg 2",
    ]
    (tmp_path / "list_eval_partition.txt").write_text("\n".join(part_lines) + "\n")

    return tmp_path


class TestCelebAGenderDataset:
    def test_train_split_length(self, fake_celeba):
        ds = CelebAGenderDataset(fake_celeba, split="train")
        assert len(ds) == 3

    def test_val_split_length(self, fake_celeba):
        ds = CelebAGenderDataset(fake_celeba, split="val")
        assert len(ds) == 1

    def test_test_split_length(self, fake_celeba):
        ds = CelebAGenderDataset(fake_celeba, split="test")
        assert len(ds) == 2

    def test_labels_are_binary(self, fake_celeba):
        ds = CelebAGenderDataset(fake_celeba, split="train")
        for _, label in ds:
            assert label in (0, 1)

    def test_output_shape_with_transform(self, fake_celeba):
        transform = get_eval_transforms(image_size=224)
        ds = CelebAGenderDataset(fake_celeba, split="train", transform=transform)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)

    def test_splits_do_not_overlap(self, fake_celeba):
        train_ds = CelebAGenderDataset(fake_celeba, split="train")
        val_ds = CelebAGenderDataset(fake_celeba, split="val")
        test_ds = CelebAGenderDataset(fake_celeba, split="test")

        train_files = set(train_ds.data["filename"])
        val_files = set(val_ds.data["filename"])
        test_files = set(test_ds.data["filename"])

        assert train_files.isdisjoint(val_files)
        assert train_files.isdisjoint(test_files)
        assert val_files.isdisjoint(test_files)

    def test_official_header_still_exposes_filename_column(self, fake_celeba_official_header):
        ds = CelebAGenderDataset(fake_celeba_official_header, split="train")
        assert "filename" in ds.data.columns
        _, label = ds[0]
        assert label in (0, 1)

    def test_invalid_split_raises(self, fake_celeba):
        with pytest.raises(ValueError, match="split must be one of"):
            CelebAGenderDataset(fake_celeba, split="invalid")

    def test_missing_attr_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Attribute file"):
            CelebAGenderDataset(tmp_path, split="train")


class TestTransforms:
    def test_train_transforms_output_shape(self):
        transform = get_train_transforms(image_size=224)
        img = Image.new("RGB", (178, 218), color="blue")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_output_shape(self):
        transform = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (178, 218), color="blue")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_deterministic(self):
        transform = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (178, 218), color="green")
        out1 = transform(img)
        out2 = transform(img)
        assert torch.equal(out1, out2)
