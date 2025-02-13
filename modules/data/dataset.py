import json
import os

import torch

# import torch
# from datasets.arrow_dataset import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from modules.constants import NORMALIZATION_STATS


class CIFAR10:
    def __init__(self, root: str, train: bool = True, augmentation: bool = False):
        if train and augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["cifar10"]["mean"],
                        std=NORMALIZATION_STATS["cifar10"]["std"],
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["cifar10"]["mean"],
                        std=NORMALIZATION_STATS["cifar10"]["std"],
                    ),
                ]
            )

        return datasets.CIFAR10(
            root=root, train=train, transform=transform, download=False
        )


class CIFAR100:
    def __init__(self, root: str, train: bool = True, augmentation: bool = False):
        if train and augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["cifar100"]["mean"],
                        std=NORMALIZATION_STATS["cifar100"]["std"],
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["cifar100"]["mean"],
                        std=NORMALIZATION_STATS["cifar100"]["std"],
                    ),
                ]
            )

        return datasets.CIFAR100(
            root=root, train=train, transform=transform, download=False
        )


class TinyImageNet(Dataset):
    def __init__(self, root: str, train=True, augmentation=False):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.data = []
        self.targets = []
        self.label_to_idx = {}

        if train and augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(64, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["tiny-imagenet"]["mean"],
                        std=NORMALIZATION_STATS["tiny-imagenet"]["std"],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=NORMALIZATION_STATS["tiny-imagenet"]["mean"],
                        std=NORMALIZATION_STATS["tiny-imagenet"]["std"],
                    ),
                ]
            )

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


class CelebA(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        user_id: str = None,
        augmentation: bool = False,
    ):
        self.root = os.path.join(root, "celeba")
        self.data = []
        self.targets = []
        self.user_id = user_id

        if train and augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        if train:
            self._load_train_data()
        else:
            self._load_test_data()

    def _load_train_data(self):
        train_dir = os.path.join(self.root, "train", "by_user", self.user_id)
        classes = sorted(os.listdir(train_dir))

        for _, label in enumerate(classes):
            cls_dir = os.path.join(train_dir, label)
            images = os.listdir(cls_dir)
            for img in images:
                img_path = os.path.join(cls_dir, img)
                self.data.append((img_path, int(label)))
                self.targets.append(int(label))

    def _load_test_data(self):
        test_dir = os.path.join(self.root, "test")
        with open(os.path.join(test_dir, "annotations.txt"), "r") as f:
            for line in f:
                img, label = line.strip().split("\t")[:2]
                img_path = os.path.join(test_dir, "images", img)
                self.data.append((img_path, int(label)))
                self.targets.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FEMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        user_id: str = None,
        augmentation: bool = False,
    ):
        self.root = os.path.join(root, "femnist")
        self.train = train
        self.user_id = user_id
        self._load_data()
        self.transform = transforms.Normalize((0.5,), (0.5,))

    def _load_data(self):
        if self.train:
            json_file_path = os.path.join(self.root, "train", f"{self.user_id}.json")
        else:
            json_file_path = os.path.join(self.root, "test", "test_data.json")

        with open(json_file_path, "r") as f:
            self.data = json.load(f)

        x = self.data["data"]["x"]
        y = self.data["data"]["y"]

        # Transform data to torch tensors
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1, 28, 28)
        y = torch.tensor(y, dtype=torch.long)

        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.transform(x)
        return x, y


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
