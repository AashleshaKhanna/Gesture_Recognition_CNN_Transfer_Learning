"""Data loading and splitting utilities for ASL gesture recognition."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

@dataclass
class DataSplit:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

def build_transform(image_size:int=224, imagenet_norm:bool=False)->transforms.Compose:
    steps=[transforms.Resize((image_size,image_size)), transforms.ToTensor()]
    if imagenet_norm:
        steps.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(steps)

def split_indices(n:int, train_frac:float=0.70, val_frac:float=0.15, seed:int=42):
    rng=np.random.default_rng(seed); idx=np.arange(n); rng.shuffle(idx)
    n_train=int(train_frac*n); n_val=int(val_frac*n)
    return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()

def class_counts(dataset:ImageFolder, indices:list[int])->dict[str,int]:
    labels=[dataset.samples[i][1] for i in indices]; counts=Counter(labels)
    return {dataset.classes[k]:counts.get(k,0) for k in range(len(dataset.classes))}

def make_loaders(data_dir:str, batch_size:int=32, image_size:int=224, imagenet_norm:bool=False, seed:int=42, num_workers:int=2)->DataSplit:
    root=Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f'Dataset directory not found: {root}. Expected ImageFolder structure.')
    dataset=ImageFolder(root, transform=build_transform(image_size, imagenet_norm))
    train_idx,val_idx,test_idx=split_indices(len(dataset), seed=seed)
    train_ds, val_ds, test_ds = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    return DataSplit(
        DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers),
        DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers),
        DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers),
        list(dataset.classes), train_idx, val_idx, test_idx)

def make_small_subset_loader(data_split:DataSplit, size:int=64, batch_size:int=64, seed:int=0)->DataLoader:
    train_subset=data_split.train_loader.dataset
    rng=np.random.default_rng(seed)
    selected=rng.choice(len(train_subset), size=min(size,len(train_subset)), replace=False)
    return DataLoader(Subset(train_subset, selected.tolist()), batch_size=batch_size, shuffle=True, num_workers=0)
