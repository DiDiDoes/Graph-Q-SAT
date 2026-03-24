import os

from torch.utils.data import Dataset

class CNFDataset(Dataset):
    def __init__(self, num_variables: int, sat: bool, split: str = "test"):
        self.num_variables = num_variables
        self.sat = sat
        self.root = "data"
        prefix = f"uf{num_variables}" if sat else f"uuf{num_variables}"
        subdir = [d for d in os.listdir(self.root) if d.startswith(prefix) and not d.endswith(".tar.gz")]
        assert len(subdir) == 1, f"Expected exactly one subdirectory for prefix '{prefix}', found {len(subdir)}"
        self.root = os.path.join(self.root, subdir[0])
        if split == "test":
            indices = range(1, 101)
        elif split == "valid":
            indices = range(101, 201)
        elif split == "train":
            indices = range(201, 1001)
        else:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'valid', or 'test'.")
        self.cnf_files = [os.path.join(self.root, f"{prefix}-0{i}.cnf") for i in indices]

    def __len__(self):
        return len(self.cnf_files)

    def __getitem__(self, idx):
        return self.cnf_files[idx]
