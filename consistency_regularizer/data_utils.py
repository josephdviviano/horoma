import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import sampler


class InfiniteSampler(sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


def get_iters(
    train_labeled_dataset,
    train_unlabeled_dataset,
    valid_dataset,
    test_dataset=None,
    l_batch_size=32,
    ul_batch_size=128,
    val_batch_size=256,
    workers=8,
):

    data_iterators = {
        "labeled": iter(
            DataLoader(
                train_labeled_dataset,
                batch_size=l_batch_size,
                num_workers=workers,
                sampler=InfiniteSampler(len(train_labeled_dataset)),
            )
        )
        if train_labeled_dataset is not None
        else None,
        "unlabeled": iter(
            DataLoader(
                train_unlabeled_dataset,
                batch_size=ul_batch_size,
                num_workers=workers,
                sampler=InfiniteSampler(len(train_unlabeled_dataset)),
            )
        )
        if train_unlabeled_dataset is not None
        else None,
        "val": DataLoader(
            valid_dataset, batch_size=val_batch_size, num_workers=workers, shuffle=False
        )
        if valid_dataset is not None
        else None,
        "test": DataLoader(
            test_dataset, batch_size=val_batch_size, num_workers=workers, shuffle=False
        )
        if test_dataset is not None
        else None,
    }

    return data_iterators
