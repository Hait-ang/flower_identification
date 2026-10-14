
from pathlib import Path

import torch
from PIL import Image

from model import flower_identification_model
from data_processing_and_augmentation import eval_transform

class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def main():

    model_path = Path(
        "outputs/best_model.pth"
    )
    image_path = Path("samples/flower.jpg")

    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在：{image_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("计算设备：", device)

    model = flower_identification_model().to(device)

    model.load_state_dict(
        torch.load(
        model_path,
        map_location=device,
        weights_only=True
        )
    )

    model.eval()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = eval_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = probabilities.argmax().item()

    print("图片路径：", image_path)
    print("预测类别：", class_names[predicted_id])

    for class_id, name in enumerate(class_names):
        probability = probabilities[class_id].item()
        print(f"{name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
