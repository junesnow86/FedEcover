import torch
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset


# Create input-output pairs for next word prediction
class NextWordPredictionDataset(Dataset):
    def __init__(self, tokenized_data: HFDataset):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx])
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore the last token
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": labels,
        }
