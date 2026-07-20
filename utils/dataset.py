from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
        load_masks=False,
        mask_transform=None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.load_masks = load_masks
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df.loc[idx, "image"]
        label = self.df.loc[idx, "label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Default behavior (same as before)
        if not self.load_masks:
            return image, label

        # -------------------------
        # Load corresponding mask
        # -------------------------

        image_path = Path(image_path)

        mask_name = image_path.stem + ".png"

        mask_path = (
            image_path.parent.parent
            / "masks"
            / mask_name
        )

        mask = Image.open(mask_path).convert("L")

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label