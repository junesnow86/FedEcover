import os

# import torch
# from datasets.arrow_dataset import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root: str, train=True, transform=None):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.transform = transform
        self.data = []
        self.targets = []
        self.label_to_idx = {}

        self._load_label_to_idx()

        if train:
            self._load_train_data()
        else:
            self._load_val_data()

    def _load_label_to_idx(self):
        train_dir = os.path.join(self.root, "train")
        classes = sorted(os.listdir(train_dir))
        for idx, cls in enumerate(classes):
            self.label_to_idx[cls] = idx

    def _load_train_data(self):
        train_dir = os.path.join(self.root, "train")
        classes = sorted(os.listdir(train_dir))

        for _, cls in enumerate(classes):
            cls_dir = os.path.join(train_dir, cls, "images")
            images = os.listdir(cls_dir)
            for img in images:
                img_path = os.path.join(cls_dir, img)
                self.data.append((img_path, cls))
                self.targets.append(self.label_to_idx[cls])

    def _load_val_data(self):
        val_dir = os.path.join(self.root, "val")
        with open(os.path.join(val_dir, "val_annotations.txt"), "r") as f:
            for line in f:
                img, cls = line.strip().split("\t")[:2]
                img_path = os.path.join(val_dir, "images", img)
                self.data.append((img_path, cls))
                self.targets.append(self.label_to_idx[cls])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, cls = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[cls]

        return image, label_idx


# Create input-output pairs for next word prediction
# class NextWordPredictionDataset(Dataset):
#     def __init__(self, tokenized_data: HFDataset, padding_index: int = 0):
#         self.input_ids = tokenized_data["input_ids"]
#         self.attention_mask = tokenized_data["attention_mask"]
#         self.padding_index = padding_index

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         input_ids = torch.tensor(self.input_ids[idx])
#         labels = input_ids.clone()
#         labels[:-1] = input_ids[1:]
#         labels[-1] = self.padding_index
#         return {
#             "input_ids": input_ids,
#             "attention_mask": torch.tensor(self.attention_mask[idx]),
#             "labels": labels,
#         }
