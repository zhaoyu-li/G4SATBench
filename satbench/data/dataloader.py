import torch
import numpy as np
import random
import itertools

from satbench.data.dataset import SATDataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


def collate_fn(batch):
    return Batch.from_data_list([s for s in list(itertools.chain(*batch))])


def get_dataloader(data_dir, splits, sample_size, opts, mode, use_contrastive_learning=False):
    dataset = SATDataset(data_dir, splits, sample_size, use_contrastive_learning, opts)
    batch_size = opts.batch_size // len(splits) if opts.data_fetching == 'parallel' else opts.batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )
