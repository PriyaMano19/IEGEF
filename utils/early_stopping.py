import numpy as np


class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after
    a given number of epochs.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        verbose=True
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.verbose:
                print(
                    f"EarlyStopping Counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True