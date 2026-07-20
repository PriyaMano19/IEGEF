import numpy as np
import cv2
import torch

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, input_tensor, target_layers, target_class=None):
    """
    Generate Grad-CAM heatmap.

    Args:
        model: Trained PyTorch model
        input_tensor: Image tensor (1,C,H,W)
        target_layers: List of target layers
        target_class: Class index (optional)

    Returns:
        grayscale_cam
    """

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = None

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    return grayscale_cam[0]


def generate_gradcam_plus_plus(model, input_tensor, target_layers, target_class=None):
    """
    Generate Grad-CAM++ heatmap.

    Args:
        model: Trained PyTorch model
        input_tensor: Image tensor (1,C,H,W)
        target_layers: List of target layers
        target_class: Class index (optional)

    Returns:
        grayscale_cam
    """

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    targets = None

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    return grayscale_cam[0]


def tensor_to_image(image_tensor):
    """
    Convert normalized tensor to RGB image.
    """

    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def overlay_heatmap(image, grayscale_cam):
    """
    Overlay heatmap on image.
    """

    visualization = show_cam_on_image(
        image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


def save_image(image, filename):
    """
    Save RGB image.
    """

    cv2.imwrite(
        filename,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )