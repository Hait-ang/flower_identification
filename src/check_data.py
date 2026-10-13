from pathlib import Path
from PIL import Image

root = Path("data/flower")

for split in ["train", "test"]:
    for category in sorted((root / split).iterdir()):
        if not category.is_dir():
            continue

        total = 0
        damaged = []

        for path in category.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            total += 1

            try:
                with Image.open(path) as image:
                    image.load()

            except OSError as error:
                damaged.append(path)
                print("损坏图片：", path)
                print("错误原因：", error)

        print(
            f"{split}/{category.name}: "
            f"总数={total}, 损坏={len(damaged)}"
        )