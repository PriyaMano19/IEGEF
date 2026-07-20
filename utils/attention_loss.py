import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAlignmentLoss(nn.Module):
    """
    Dice-based Attention Alignment Loss.

    Encourages the model's attention map to overlap
    with the ground-truth lung mask.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, attention_map, lung_mask):

        # Resize masks to the attention map resolution
        lung_mask = F.interpolate(
            lung_mask.float(),
            size=attention_map.shape[-2:],
            mode="nearest"
        )

        # Normalize each image independently
        min_val = attention_map.amin(dim=(2, 3), keepdim=True)
        max_val = attention_map.amax(dim=(2, 3), keepdim=True)

        attention_map = (attention_map - min_val) / (
            max_val - min_val + 1e-8
        )

        batch_size = attention_map.size(0)

        attention_map = attention_map.view(batch_size, -1)
        lung_mask = lung_mask.view(batch_size, -1)

        intersection = (attention_map * lung_mask).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            attention_map.sum(dim=1)
            + lung_mask.sum(dim=1)
            + self.smooth
        )

        loss = 1.0 - dice

        return loss.mean()