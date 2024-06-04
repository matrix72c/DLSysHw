import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
            X = X.astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)

        self.images = X
        self.labels = y
        self.transforms = transforms if transforms is not None else []

    def __getitem__(self, index) -> object:
        image = self.images[index]
        label = self.labels[index]
        for func in self.transforms:
            image = func(image)
        return image, label

    def __len__(self) -> int:
        return len(self.labels)