from pathlib import Path

import pandas as pd


def save_history(history, filepath):
    """
    Save training history to CSV.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(history)

    df.index.name = "epoch"

    df.to_csv(filepath)