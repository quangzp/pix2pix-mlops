import os
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image


class Dataset(torch.utils.data.Dataset):
   
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.sketch_dir = self.root_dir / split / "sketch"
        self.real_dir = self.root_dir / split / "photo"

        self.files = sorted(os.listdir(self.sketch_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        sketch_path = self.sketch_dir / filename
        real_path = self.real_dir / filename

        sketch = Image.open(sketch_path).convert("RGB")
        real = Image.open(real_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            real = self.transform(real)

        return {"sketch": sketch, "real": real}


def create_dataloader(root_dir, split, batch_size, transform, shuffle=True):
    dataset = Dataset(
        root_dir=root_dir,
        split=split,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
