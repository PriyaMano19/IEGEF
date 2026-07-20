import torch
import torch.nn.functional as F


class FeatureMapExtractor:
    """
    Extracts feature-map attention from the final convolutional layer.
    """

    def __init__(self, model):

        self.feature_maps = None

        # Hook into the last Bottleneck block of layer4
        model.layer4[-1].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.feature_maps = output

    def get_attention_map(self):

        if self.feature_maps is None:
            raise RuntimeError(
                "Run a forward pass before requesting the attention map."
            )

        # Shape:
        # Batch × 2048 × 7 × 7

        attention = self.feature_maps.mean(dim=1, keepdim=True)

        attention = F.relu(attention)

        attention = attention - attention.amin(
            dim=(2, 3),
            keepdim=True
        )

        attention = attention / (
            attention.amax(dim=(2, 3), keepdim=True)
            + 1e-8
        )

        return attention