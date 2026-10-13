from pathlib import Path
from PIL import Image
import shutil

root = Path("data/flower")
quarantine = Path("data/quarantine")

damaged = []

for split in ["train", "test"]:
    for category in (root / split).iterdir():
        if not category.is_dir():
            continue

        for path in category.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                with Image.open(path) as image:
                    image.load()

            except OSError:
                target = quarantine / path.relative_to(root)
                damaged.append((path, target))

print(f"异常图片数量{len(damaged)}")
for source, target in damaged:
    if target.exists():
        raise FileExistsError(f"目标文件已存在：{target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, target)

print("已完成隔离！")

