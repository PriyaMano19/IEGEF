import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, dataloader, device):
    """
    Evaluate a trained model on a dataset.

    Returns:
        results (dict)
        y_true
        y_pred
        y_prob
    """

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in dataloader:

            # Supports both:
            # (images, labels)
            # (images, masks, labels)

            if len(batch) == 2:
                images, labels = batch

            elif len(batch) == 3:
                images, _, labels = batch

            else:
                raise ValueError(
                    f"Unexpected batch format. Expected 2 or 3 elements, got {len(batch)}."
                )

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(y_prob.shape[1])
    )

    auc = roc_auc_score(
        y_true_bin,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "Confusion Matrix": cm,
        "Classification Report": report
    }

    return results, y_true, y_pred, y_prob