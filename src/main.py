
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_and_dataloader import train_dataloader, train_eval_dataloader, val_dataloader,test_dataloader
from model import flower_identification_model

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, 100 * correct / total

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, 100 * correct / total


def main():

    torch.manual_seed(88)

    batch_size = 32
    learning_rate = 0.0001
    epochs = 20

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU，请检查算力平台环境。")

    device = torch.device("cuda")
    print("计算设备：", torch.cuda.get_device_name(0))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = flower_identification_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_dataloader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%",
            flush=True
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth"
            )

    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy"
        ])

        for i in range(epochs):
            writer.writerow([
                i + 1,
                train_losses[i],
                val_losses[i],
                train_accuracies[i],
                val_accuracies[i]
            ])

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, epochs + 1),
        train_accuracies,
        label="Train Accuracy"
    )

    plt.plot(
        range(1, epochs + 1),
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Flower CNN Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=150
    )

    plt.close()

    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    _, final_train_acc = evaluate(
        model, train_eval_dataloader, criterion, device
    )

    _, final_val_acc = evaluate(
        model, val_dataloader, criterion, device
    )

    _, final_test_acc = evaluate(
        model, test_dataloader, criterion, device
    )

    print("\n最终评价结果：")
    print(
        f"最佳 Epoch：{best_epoch}\n"
        f"训练集准确率：{final_train_acc:.2f}%\n"
        f"验证集准确率：{final_val_acc:.2f}%\n"
        f"测试集准确率：{final_test_acc:.2f}%\n"
    )

if __name__ == "__main__":
    main()