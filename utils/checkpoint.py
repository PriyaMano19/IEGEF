from pathlib import Path
import torch


def save_checkpoint(model, filepath):
    """
    Save model weights.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), filepath)


def load_checkpoint(model, filepath, device):
    """
    Load model weights.
    """

    state_dict = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(state_dict)

    return model