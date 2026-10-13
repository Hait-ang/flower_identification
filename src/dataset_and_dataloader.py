import csv

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from data_processing_and_augmentation import train_transform, eval_transform


class FlowerDataset(Dataset):

    def __init__(self, csv_path, transform):
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))

        self.transform = transform

        self.label_to_id = {
            "daisy": 0,
            "dandelion": 1,
            "rose": 2,
            "sunflower": 3,
            "tulip": 4
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        label = self.label_to_id[row["label"]]

        return image_tensor, label

train_dataset = FlowerDataset("data/splits/train.csv", train_transform)
train_eval_dataset = FlowerDataset("data/splits/train.csv", eval_transform)
val_dataset = FlowerDataset("data/splits/val.csv", eval_transform)
test_dataset = FlowerDataset("data/splits/test.csv", eval_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True)
train_eval_dataloader = DataLoader(train_eval_dataset,batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)