import cv2
import numpy as np
import torch

from captum.attr import IntegratedGradients


def generate_integrated_gradients(
    model,
    input_tensor,
    target_class=None,
    n_steps=50
):
    """
    Generate Integrated Gradients attribution map.
    """

    ig = IntegratedGradients(model)

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    attributions = ig.attribute(
        input_tensor,
        target=target_class,
        n_steps=n_steps
    )

    attribution = (
        attributions.squeeze()
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    attribution = np.mean(np.abs(attribution), axis=2)

    attribution -= attribution.min()
    attribution /= attribution.max() + 1e-8

    return attribution


def overlay_integrated_gradients(image, attribution):
    """
    Overlay Integrated Gradients attribution map on the original image.
    """

    heatmap = np.uint8(255 * attribution)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = 0.6 * image + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay