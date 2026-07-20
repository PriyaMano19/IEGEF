import cv2
import numpy as np
import torch

def normalize_map(explanation):
    """
    Normalize explanation map to [0,1].
    """
    explanation = explanation.astype(np.float32)
    explanation -= explanation.min()
    explanation /= (explanation.max() + 1e-8)
    return explanation


def threshold_map(explanation, threshold=0.5):
    """
    Convert explanation map into a binary mask.
    """
    return (explanation >= threshold).astype(np.uint8)


def resize_mask(mask, target_shape):
    """
    Resize mask to match explanation size.
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if union.sum() == 0:
        return 0.0

    return intersection.sum() / union.sum()


def pointing_game(explanation, gt_mask):
    """
    Returns 1 if the highest activation is inside the mask.
    """
    y, x = np.unravel_index(
        np.argmax(explanation),
        explanation.shape
    )

    return int(gt_mask[y, x] > 0)


def heatmap_coverage(explanation, gt_mask, threshold=0.5):
    """
    Percentage of important pixels that lie inside the lung mask.
    """

    binary = explanation >= threshold

    important_pixels = np.sum(binary)

    if important_pixels == 0:
        return 0.0

    inside = np.logical_and(binary, gt_mask)

    return inside.sum() / important_pixels


def confidence_drop(model, image, explanation, device, top_percent=20):
    """
    Faithfulness (Deletion Test)

    Remove the most important pixels and
    measure confidence drop.
    """

    model.eval()

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output, dim=1)

        confidence_before = probs.max().item()

    exp = explanation.copy()

    threshold = np.percentile(
        exp,
        100 - top_percent
    )

    important = exp >= threshold

    image_np = image.squeeze().permute(1,2,0).cpu().numpy()

    image_np[important] = 0

    image_tensor = (
        torch.tensor(image_np)
        .permute(2,0,1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(output, dim=1)

        confidence_after = probs.max().item()

    return confidence_before - confidence_after