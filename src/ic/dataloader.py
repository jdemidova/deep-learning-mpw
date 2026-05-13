from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import CaptionTokenizer, CaptionTokenizerProtocol

SplitName = Literal["train", "test"]
CaptionSampling = Literal["all", "first", "random"]
CAPTION_SAMPLING_MODES: tuple[CaptionSampling, ...] = ("all", "first", "random")

def _convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

def _normalize_split_name(split: str) -> SplitName:
    return split.lower().strip()

def _get_images_dir(data_dir: str):
    return Path(data_dir) / "images"

def _normalize_caption_sampling(caption_sampling: str | None) -> CaptionSampling:
    normalized = "all" if caption_sampling is None else caption_sampling.lower().strip()
    if normalized not in CAPTION_SAMPLING_MODES:
        raise ValueError(
            "caption_sampling must be one of "
            f"{sorted(CAPTION_SAMPLING_MODES)}, found '{caption_sampling}'."
        )
    return normalized

def _filter_captions_dataframe(captions_df: pd.DataFrame, image_ids: Sequence[str] | None) -> pd.DataFrame:
    if image_ids is None:
        return captions_df.reset_index(drop=True)

    image_order = {image_id: index for index, image_id in enumerate(image_ids)}
    filtered = captions_df[captions_df["image"].isin(image_order)].copy()
    filtered["_image_order"] = filtered["image"].map(image_order)
    filtered = filtered.sort_values("_image_order", kind="stable")
    return filtered.drop(columns="_image_order").reset_index(drop=True)

def load_captions_dataframe(captions_file: str | Path) -> pd.DataFrame:
    captions_df = pd.read_csv(captions_file)
    required_columns = {"image", "caption"}
    missing_columns = required_columns.difference(captions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns {sorted(missing_columns)} in {Path(captions_file)}."
        )
    captions_df = captions_df.loc[:, ["image", "caption"]].dropna()
    captions_df["image"] = captions_df["image"].astype(str)
    captions_df["caption"] = captions_df["caption"].astype(str)
    return captions_df.reset_index(drop=True)

def load_image_ids(image_list_file: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(image_list_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    
def load_split_dataframe(split: str, data_dir: str | Path) -> pd.DataFrame:
    split = _normalize_split_name(split)
    split_dir = Path(data_dir) / "splits"
    return load_captions_dataframe(Path(split_dir) / f"{split}_captions.csv")

def load_split_image_ids(split: str, data_dir: str | Path) -> list[str]:
    split = _normalize_split_name(split)
    split_dir = Path(data_dir) / "splits"
    return load_image_ids(Path(split_dir) / f"{split}_images.txt")

def build_captions_map(captions_source: pd.DataFrame | str | Path) -> dict[str, list[str]]:
    captions_df = (
        captions_source.copy()
        if isinstance(captions_source, pd.DataFrame)
        else load_captions_dataframe(captions_source)
    )
    return (
        captions_df.groupby("image", sort=False)["caption"]
        .agg(list)
        .to_dict()
    )

def build_tokenizer_from_split(
    split: str,
    data_dir: str | Path,
    min_freq: int = 5,
    tokenizer: CaptionTokenizerProtocol | None = None,
) -> CaptionTokenizerProtocol:
    """
    Build a tokenizer from raw captions in one split.

    Pass ``tokenizer=...`` to use a notebook-defined tokenizer. The object only
    needs a ``build(captions)`` method for construction and the dataset expects
    ``encode(caption, max_len=...)`` plus ``pad_idx`` when samples are read.
    ``min_freq`` is used only when the default ``CaptionTokenizer`` is created.
    """
    captions_df = load_split_dataframe(split=split, data_dir=data_dir)
    tokenizer = CaptionTokenizer(min_freq=min_freq) if tokenizer is None else tokenizer

    return tokenizer.build(captions_df["caption"].tolist())

class Flickr8kCaptionsDatasetBase(Dataset):
    """
    Minimal Flickr8k dataset for captioning experiments.

    Each item is returned as ``(image, caption_ids, caption_length, image_id,
    raw_caption)``. Image transforms, caption encoding, truncation, and padding
    all happen in ``__getitem__`` after the raw image and raw caption have been
    loaded/selected.

    Args:
        split: either 'train' or 'test' - for the train or test split.
        data_dir: Directory containing all the data.
        tokenizer: Tokenizer instance implementing CaptionTokenizerProtocol for
            encoding captions into token IDs.
        transform: image transformation callable.
        image_ids: Optional sequence of image IDs to filter the dataset. If None,
            all images from captions_source are included.
        max_len: Maximum length for caption token sequences. Captions longer than
            this will be truncated. Must be at least 1. Defaults to 40.
        caption_sampling: How to sample captions when multiple exist per image.
            Options: 'all' (return all captions), 'first' (return first caption),
            'random' (return random caption). Defaults to 'all'.
        pad_to_max_len: Whether to pad caption sequences to max_len. If False,
            sequences are truncated but not padded. Defaults to True.
    """

    def __init__(
        self,
        split: str,
        data_dir: str | Path,
        tokenizer: CaptionTokenizerProtocol,
        transform: Callable,
        image_ids: Sequence[str] | None = None,
        max_len: int = 40,
        caption_sampling: CaptionSampling | None = None,
        pad_to_max_len: bool = True,
    ) -> None:
        
        split = _normalize_split_name(split)
        if max_len < 1:
            raise ValueError("max_len must be at least 1.")

        captions_df = load_split_dataframe(split, data_dir)
        captions_df = _filter_captions_dataframe(captions_df, image_ids)
        if captions_df.empty:
            raise ValueError("No captions found for the requested dataset.")

        self.image_dir = Path(_get_images_dir(data_dir))
        self.tokenizer = tokenizer
        
        if transform is None:
            raise ValueError("transform must not be None")
        self.transform = transform        
        self.max_len = max_len
        self.caption_sampling = _normalize_caption_sampling(caption_sampling)
        self.pad_to_max_len = pad_to_max_len
        self.captions_map = build_captions_map(captions_df)
        self.image_ids = list(dict.fromkeys(captions_df["image"].tolist()))
        self.samples = list(captions_df.itertuples(index=False, name=None))

    def __len__(self) -> int:
        if self.caption_sampling == "all":
            return len(self.samples)
        return len(self.image_ids)

    def _select_caption(self, index: int) -> tuple[str, str]:
        if self.caption_sampling == "all":
            image_id, caption = self.samples[index]
            return image_id, caption

        image_id = self.image_ids[index]
        captions = self.captions_map[image_id]

        if self.caption_sampling == "random" and len(captions) > 1:
            caption_index = torch.randint(len(captions), size=(1,)).item()
            return image_id, captions[caption_index]
        
        return image_id, captions[0]

    def _load_image(self, image_id: str) -> Image.Image:
        image_path = self.image_dir / image_id
        with Image.open(image_path) as image_file:
            return image_file.copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        
        image_id, raw_caption = self._select_caption(index)
        image = self._load_image(image_id)

        image = self.transform(image)

        caption_ids = self.tokenizer.encode(raw_caption, max_len=self.max_len)
        caption_length = len(caption_ids)
        if self.pad_to_max_len:
            caption_ids = caption_ids + [self.tokenizer.pad_idx] * (
                self.max_len - caption_length
            )

        return (
            image,
            torch.tensor(caption_ids, dtype=torch.long),
            torch.tensor(caption_length, dtype=torch.long),
            image_id,
            raw_caption,
        )

def caption_collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    images, captions, lengths, image_ids, raw_captions = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(captions),
        torch.as_tensor(lengths, dtype=torch.long),
        list(image_ids),
        list(raw_captions),
    )


def create_caption_dataloader(
    split: str,
    data_dir: str | Path,
    tokenizer: CaptionTokenizerProtocol,
    transform: Callable,
    batch_size: int = 32,
    max_len: int = 40,
    caption_sampling: CaptionSampling | None = None,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    
    split = _normalize_split_name(split)
    if shuffle is None:
        shuffle = split == "train"
        
    dataset = Flickr8kCaptionsDatasetBase(
        split=split,
        data_dir=data_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_len=max_len,
        caption_sampling=caption_sampling,
        pad_to_max_len=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=caption_collate_fn,
    )

__all__ = [
    "Flickr8kCaptionsDatasetBase",
    "SplitName",
    "build_captions_map",
    "build_image_transform",
    "build_tokenizer_from_split",
    "caption_collate_fn",
    "create_caption_dataloader",
    "create_dataloaders",
    "load_captions_dataframe",
    "load_image_ids",
    "load_split_dataframe",
    "load_split_image_ids",
]
