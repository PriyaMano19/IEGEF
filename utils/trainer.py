import copy
import time

import torch
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping
from utils.history import save_history


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        running_correct += (preds == labels).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        running_correct += (preds == labels).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=20,
    patience=5
):

    history = {

        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []

    }

    best_accuracy = 0.0

    best_weights = copy.deepcopy(model.state_dict())

    early_stopping = EarlyStopping(
        patience=patience
    )

    start_time = time.time()

    for epoch in range(epochs):

        print("=" * 60)
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4%}")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4%}")

        print(f"Learning Rate : {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                model,
                "../checkpoints/best_resnet50.pth"
            )

            print("✅ Best model saved")

        early_stopping(val_loss)

        if early_stopping.early_stop:

            print("🛑 Early stopping triggered")

            break

    total_time = (time.time() - start_time) / 60

    print("=" * 60)
    print(f"Training completed in {total_time:.2f} minutes")
    print(f"Best Validation Accuracy : {best_accuracy:.4%}")

    model.load_state_dict(best_weights)

    save_history(
        history,
        "../results/training_history.csv"
    )

    return model, history