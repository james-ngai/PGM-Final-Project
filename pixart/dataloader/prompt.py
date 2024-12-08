import json
from torch.utils.data import Dataset
import dnnlib

class PromptDataset(Dataset):
    """Dataset class for loading prompts from a .jsonl file."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        # Count the number of lines in the file for __len__
        with open(file_path, 'r') as f:
            self.num_lines = sum(1 for _ in f)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == idx:
                    entry = json.loads(line)
                    return entry["student_prompt"]
