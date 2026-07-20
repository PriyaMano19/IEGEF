from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.dataset import ChestXrayDataset
from utils.transforms import (
    train_transform,
    val_transform,
    mask_transform
)


CLASSES = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]


def create_dataframe(dataset_path):
    """
    Create a DataFrame containing image paths and labels.
    """

    dataset_path = Path(dataset_path)

    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASSES):

        image_folder = dataset_path / class_name / "images"

        for img_path in image_folder.glob("*.png"):
            image_paths.append(str(img_path))
            labels.append(label)

    df = pd.DataFrame({
        "image": image_paths,
        "label": labels
    })

    return df


def split_dataframe(df, random_state=42):
    """
    Split the DataFrame into train, validation, and test sets.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_state
    )

    return train_df, val_df, test_df


def create_datasets(
    train_df,
    val_df,
    test_df,
    load_masks=False
):
    """
    Create Dataset objects.
    """

    train_dataset = ChestXrayDataset(
        train_df,
        transform=train_transform,
        load_masks=load_masks,
        mask_transform=mask_transform if load_masks else None
    )

    val_dataset = ChestXrayDataset(
        val_df,
        transform=val_transform,
        load_masks=load_masks,
        mask_transform=mask_transform if load_masks else None
    )

    test_dataset = ChestXrayDataset(
        test_df,
        transform=val_transform,
        load_masks=load_masks,
        mask_transform=mask_transform if load_masks else None
    )

    return train_dataset, val_dataset, test_dataset
def create_dataloaders(
    dataset_path,
    batch_size=32,
    num_workers=0,
    load_masks=False
):
    """
    Create DataLoaders and also return the DataFrames.
    """

    df = create_dataframe(dataset_path)

    train_df, val_df, test_df = split_dataframe(df)

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df,
        val_df,
        test_df,
        load_masks=load_masks
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df
    )