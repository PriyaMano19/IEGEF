import torch
import torch.nn as nn

from utils.attention_loss import AttentionAlignmentLoss


class IEGEFLoss(nn.Module):
    """
    Iterative Explainability-Guided Enhancement Framework Loss.

    Total Loss =
        Classification Loss
        + lambda_attention × Attention Alignment Loss
    """

    def __init__(self, lambda_attention=0.3):
        super().__init__()

        self.classification_loss = nn.CrossEntropyLoss()
        self.attention_loss = AttentionAlignmentLoss()

        self.lambda_attention = lambda_attention

    def forward(
        self,
        logits,
        labels,
        attention_map,
        lung_mask
    ):

        cls_loss = self.classification_loss(
            logits,
            labels
        )

        att_loss = self.attention_loss(
            attention_map,
            lung_mask
        )

        total_loss = (
            cls_loss
            + self.lambda_attention * att_loss
        )

        return total_loss, cls_loss, att_loss