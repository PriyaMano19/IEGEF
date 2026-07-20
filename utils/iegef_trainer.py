import copy
import time

import torch
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping
from utils.history import save_history
from utils.attention_extractor import FeatureMapExtractor
from utils.iegef_optimizer import IEGEFLoss


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    extractor,
    device
):
    model.train()

   

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_att_loss = 0.0

    running_correct = 0
    total = 0

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, masks, labels in progress:

        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        attention = extractor.get_attention_map()

        total_loss, cls_loss, att_loss = criterion(
            outputs,
            labels,
            attention,
            masks
        )

        total_loss.backward()

        optimizer.step()

        running_total_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_att_loss += att_loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            att=f"{att_loss.item():.4f}"
        )

    epoch_total_loss = running_total_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_att_loss = running_att_loss / total
    epoch_acc = running_correct / total

    return (
        epoch_total_loss,
        epoch_cls_loss,
        epoch_att_loss,
        epoch_acc
    )
@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    extractor,
    device
):
    model.eval()

   

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_att_loss = 0.0

    running_correct = 0
    total = 0

    progress = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for images, masks, labels in progress:

        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        outputs = model(images)

        attention = extractor.get_attention_map()

        total_loss, cls_loss, att_loss = criterion(
            outputs,
            labels,
            attention,
            masks
        )

        running_total_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_att_loss += att_loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            att=f"{att_loss.item():.4f}"
        )

    epoch_total_loss = running_total_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_att_loss = running_att_loss / total
    epoch_acc = running_correct / total

    return (
        epoch_total_loss,
        epoch_cls_loss,
        epoch_att_loss,
        epoch_acc
    )
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
        "train_cls_loss": [],
        "train_att_loss": [],
        "train_acc": [],

        "val_loss": [],
        "val_cls_loss": [],
        "val_att_loss": [],
        "val_acc": []

    }

    best_accuracy = 0.0

    best_weights = copy.deepcopy(model.state_dict())

    early_stopping = EarlyStopping(
        patience=patience
    )

    start_time = time.time()

    extractor = FeatureMapExtractor(model)

    for epoch in range(epochs):

        print("=" * 60)
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_cls_loss, train_att_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            extractor,
            device
        )

        val_loss, val_cls_loss, val_att_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            extractor,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_att_loss"].append(train_att_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_att_loss"].append(val_att_loss)
        history["val_acc"].append(val_acc)

        print(
        f"Train Total Loss : {train_loss:.4f} | "
        f"Cls Loss : {train_cls_loss:.4f} | "
        f"Att Loss : {train_att_loss:.4f}"
     )

        print(
            f"Val Total Loss   : {val_loss:.4f} | "
            f"Cls Loss : {val_cls_loss:.4f} | "
            f"Att Loss : {val_att_loss:.4f}"
        )

        print(f"Train Accuracy : {train_acc:.4%}")
        print(f"Validation Accuracy : {val_acc:.4%}")

        print(f"Learning Rate : {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
            model,
            "../checkpoints/best_iegef_resnet50.pth"
        )
            print("✅ Best IEGEF model saved")

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
    "../results/iegef_training_history.csv"
)

    return model, history