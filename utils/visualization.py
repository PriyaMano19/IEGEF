import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(image_tensor):
    """
    Convert a normalized tensor into an RGB image in [0,1].

    Input:
        torch.Tensor (3,H,W)

    Output:
        numpy float32 (H,W,3)
    """

    image = image_tensor.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))

    image = image * IMAGENET_STD + IMAGENET_MEAN

    image = np.clip(image, 0, 1)

    return image.astype(np.float32)