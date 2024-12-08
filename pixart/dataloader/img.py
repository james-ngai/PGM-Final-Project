
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageTextDataset(Dataset):
    def __init__(self, img_dir, sample_size=512):
        """
        Args:
            img_dir (string): Directory with all the images and text files.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.img_dir = img_dir
        self.sample_size = sample_size
        self.img_names = [
            f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.img_names[idx]
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)

                # Load caption
                text_name = img_name.rsplit(".", 1)[0] + ".txt"
                text_path = os.path.join(self.img_dir, text_name)
                with open(text_path, "r") as f:
                    text = f.read().strip()

                return image, text
            except Exception as e:
                print(f"Error loading data at index {idx}: {e}")
                idx = np.random.randint(len(self.img_names))
                continue