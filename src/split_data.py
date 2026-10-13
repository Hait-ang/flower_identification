from pathlib import Path
import random
import csv

train_root = Path("data/flower/train")
test_root = Path("data/flower/test")
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(88)

train_rows = []
val_rows = []
test_rows = []

for category in sorted(train_root.iterdir()):
    if not category.is_dir():
        continue

    origin_train_rows = []

    for path in category.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            origin_train_rows.append([str(path), category.name])

    origin_train_rows.sort()
    rng.shuffle(origin_train_rows)

    n_val = round(len(origin_train_rows) * 0.8)

    train_rows += origin_train_rows[:n_val]
    val_rows += origin_train_rows[n_val:]

for category in sorted(test_root.iterdir()):
    if not category.is_dir():
        continue

    for path in category.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            test_rows.append([str(path), category.name])

print(f"{category.name}:",
      f"训练={len(train_rows)}",
      f"验证={len(val_rows)}",
      f"测试={len(test_rows)}"
      )

for filename, rows in [
    ("train.csv", train_rows),
    ("val.csv", val_rows),
    ("test.csv", test_rows)
    ]:
    csv_path = output_dir / filename

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label"])
        writer.writerows(rows)