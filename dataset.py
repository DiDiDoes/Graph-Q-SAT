from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

VALID_SPLITS = {"train", "valid", "test"}
MAS_SAT_ROOT = Path("~/data/mas_sat").expanduser()
MAS_SAT_SPLIT_SIZES = {
    "train": 80_000,
    "valid": 1_000,
    "test": 1_000,
}


class CNFDataset(Dataset):
    def __init__(self, cnf_files: list[str]):
        self.cnf_files = cnf_files

    def __len__(self):
        return len(self.cnf_files)

    def __getitem__(self, idx):
        return self.cnf_files[idx]


class LegacyCNFDataset(CNFDataset):
    def __init__(self, num_variables: int, sat: bool, split: str = "test"):
        split = validate_split(split)
        self.num_variables = num_variables
        self.sat = sat
        self.split = split
        self.root = Path("data")

        prefix = f"uf{num_variables}" if sat else f"uuf{num_variables}"
        subdirs = [
            path for path in self.root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ]
        if len(subdirs) != 1:
            raise ValueError(
                f"Expected exactly one subdirectory for prefix '{prefix}', found {len(subdirs)}"
            )

        self.dataset_root = subdirs[0]
        indices = split_indices(split)
        cnf_files = [self.dataset_root / f"{prefix}-0{i}.cnf" for i in indices]
        missing_files = [str(path) for path in cnf_files if not path.is_file()]
        if missing_files:
            missing_preview = ", ".join(missing_files[:3])
            raise FileNotFoundError(f"Missing legacy CNF files for split '{split}': {missing_preview}")

        super().__init__([str(path) for path in cnf_files])


class MasSatDataset(CNFDataset):
    def __init__(
        self,
        dataset_id: str,
        sat: bool,
        split: str = "test",
        root: Union[str, Path] = MAS_SAT_ROOT,
    ):
        split = validate_split(split)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError("mas_sat dataset id cannot be empty.")
        if dataset_id == "satcomp" or dataset_id.startswith("satcomp/"):
            raise ValueError("The mas_sat 'satcomp' dataset is not supported because it is unsplit and uses .cnf.xz files.")

        dataset_path = Path(dataset_id)
        if dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise ValueError(f"Invalid mas_sat dataset id: {dataset_id}")
        if len(dataset_path.parts) < 2:
            raise ValueError(
                "mas_sat dataset ids must look like '<family>/<name>', for example '3-sat/easy'."
            )

        self.dataset_id = dataset_id
        self.sat = sat
        self.split = split
        self.root = Path(root).expanduser()
        self.label_dir = "sat" if sat else "unsat"
        self.dataset_root = self.root / dataset_path
        self.split_root = self.dataset_root / split / self.label_dir

        if not self.split_root.is_dir():
            raise FileNotFoundError(f"mas_sat split directory not found: {self.split_root}")

        split_size = MAS_SAT_SPLIT_SIZES[split]
        cnf_files = [
            self.split_root / f"{file_idx:05d}.cnf"
            for file_idx in range(split_size)
        ]
        missing_files = [str(path) for path in cnf_files if not path.is_file()]
        if missing_files:
            missing_preview = ", ".join(missing_files[:3])
            raise FileNotFoundError(
                f"Missing mas_sat CNF files for split '{split}': {missing_preview}"
            )

        super().__init__([str(path) for path in cnf_files])


def validate_split(split: str) -> str:
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'valid', or 'test'.")
    return split


def split_indices(split: str) -> range:
    split = validate_split(split)
    if split == "test":
        return range(1, 101)
    if split == "valid":
        return range(101, 201)
    return range(201, 1001)


def parse_legacy_dataset_spec(dataset_spec: str) -> int | None:
    dataset_spec = dataset_spec.strip()
    if not dataset_spec:
        return None
    try:
        return int(dataset_spec)
    except ValueError:
        return None


def infer_dataset_source(dataset_spec: str) -> str:
    dataset_spec = dataset_spec.strip()
    if dataset_spec == "satcomp" or dataset_spec.startswith("satcomp/"):
        raise ValueError("The mas_sat 'satcomp' dataset is not supported because it is unsplit and uses .cnf.xz files.")
    if parse_legacy_dataset_spec(dataset_spec) is not None:
        return "legacy"
    if "/" in dataset_spec:
        return "mas_sat"
    raise ValueError(
        "Unsupported dataset specification. Use an integer like '50' for legacy datasets or '<family>/<name>' like '3-sat/easy' for mas_sat."
    )


def build_dataset(dataset_spec: str, sat: bool, split: str) -> CNFDataset:
    source = infer_dataset_source(dataset_spec)
    if source == "legacy":
        return LegacyCNFDataset(num_variables=int(dataset_spec), sat=sat, split=split)
    return MasSatDataset(dataset_id=dataset_spec, sat=sat, split=split)
