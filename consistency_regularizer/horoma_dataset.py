import os

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional


class HoromaDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        subset=None,
        skip=0,
        flattened=False,
        return_doublon=False,
        transforms=None,
    ):
        """
        Initialize the horoma dataset.

        :param data_dir: Path to the directory containing the samples.
        :param split: Which split to use. [train, valid, test]
        :param subset: Percentage size of dataset to use. Default: all.
        :param skip: How many element to skip before taking the subset.
        :param flattened: If True return the images in a flatten format.
        :param transforms: Transforms to apply on the dataset before using it.
        """
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        # if split == "train":
        #     self.nb_examples = 150900
        # elif split == "valid":
        #     self.nb_examples = 480
        # elif split == "test":
        #     self.nb_examples = 498
        # elif split == "train_overlapped":
        #     self.nb_examples = 544749
        # elif split == "valid_overlapped":
        #     self.nb_examples = 1331
        # else:
        #     raise ("Dataset: Invalid split. "
        #            "Must be [train, valid, test, train_overlapped, valid_overlapped]")

        if split == "train":
            self.nb_examples = 152_000
        elif split == "train_labeled":
            self.nb_examples = 228
        elif split == "test":
            self.nb_examples = 498
        elif split == "valid":
            self.nb_examples = 252
        elif split == "full_labeled":
            self.nb_examples = 480
        elif split == "train_overlapped":
            self.nb_examples = 548_720
        elif split == "train_labeled_overlapped":
            self.nb_examples = 635
        elif split == "valid_overlapped":
            self.nb_examples = 696
        elif split == "full_labeled_overlapped":
            self.nb_examples = 1331
        else:
            raise (
                "Dataset: Invalid split. "
                "Must be [train, valid, test, train_overlapped, valid_overlapped]"
            )

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        filename_region_ids = os.path.join(data_dir, "{}_regions_id.txt".format(split))
        self.region_ids = np.loadtxt(filename_region_ids, dtype=object)

        self.targets = None
        if os.path.exists(filename_y) and "train_y.txt" not in filename_y:
            print(f"Using {filename_y} for labels")
            pre_targets = np.loadtxt(filename_y, "U2")

            if subset is None:
                pre_targets = pre_targets[skip:None]
            else:
                pre_targets = pre_targets[skip : skip + subset]

            self.map_labels = {'BJ': 0, 'BP': 1, 'CR': 2, 'EB': 3, 'EN': 4, 'EO': 5, 'ES': 6, 'EU': 7, 'FR': 8, 'HG': 9, 'PB': 10, 'PE': 11, 'PR': 12, 'PT': 13, 'PU': 14, 'SB': 15, 'TO': 16}
            # print(f"for {filename_y}, map_labels: \n{self.map_labels}")
            self.targets = np.asarray(
                [self.map_labels[k] for k in pre_targets]
            )
        self.labeled = (self.targets is not None)

        self.data = np.memmap(
            filename_x,
            dtype=datatype,
            mode="r",
            shape=(self.nb_examples, height, width, nb_channels),
        )

        if subset is None:
            self.data = self.data[skip:None]
            self.region_ids = self.region_ids[skip:None]
        else:
            self.data = self.data[skip : skip + subset]
            self.region_ids = self.region_ids[skip : skip + subset]

        self.flattened = flattened
        self.return_doublon = return_doublon

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        orig_img = torch.Tensor(img)
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.Tensor(img)

        if self.flattened:
            img = img.view(-1).unsqueeze(0)
            orig_img = orig_img.view(-1).unsqueeze(0)

        if self.targets is not None:
            return img, torch.Tensor([self.targets[index]]).long()

        if self.return_doublon:
            return orig_img, img

        return img


class SplitDataset:
    def __init__(self, split=0.9):
        """
        Callable class that performs a split according to the region.

        Args:
            split (float): The proportion of examples to keep in the training set.
        """

        assert 0 < split < 1

        self.split = split

    def __call__(self, dataset):
        """
        Takes a dataset and returns a split between training and validation.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to split.

        Returns:
            train_set (torch.utils.data.Dataset): The new training set.
            valid_set (torch.utils.data.Dataset): The new validation set.
        """

        n = len(dataset)

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids, return_counts=True, return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        np.random.shuffle(unique_regions)
        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class KFoldSplitDataset:
    def __init__(self, split=0.9, permutation=1):
        """
        Callable class that performs a k-split according to the region.

        :param split: The proportion of examples to keep in the training set.
        :param permutation:
        """

        assert 0 < split < 1

        self.split = split
        self.permutation = permutation

    def __call__(self, dataset):
        """
        Takes a dataset and returns a split between training and validation.

        :param dataset: The original dataset to split.
        :return:
        train_set: The new training set.
        valid_set: The new validation set.
        """
        n = len(dataset)

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids, return_counts=True, return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        unique_regions = np.concatenate(
            [unique_regions[self.permutation :], unique_regions[: self.permutation]]
        )

        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class FullDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.dataset.transforms = None

        indices = np.arange(len(self)) % len(dataset)

        self.region_ids = self.dataset.region_ids[indices]
        self.targets = self.dataset.targets[indices]

    @staticmethod
    def transform(img, transform):

        img = functional.to_pil_image(img)

        if transform >= 11:
            transform += 1

        transforms = np.zeros((2 * 2 * 4))
        transforms[transform] = 1
        transforms.reshape((2, 2, 4))

        a = transform // (2 * 2)
        transform = transform % (2 * 2)
        h = transform // 2
        v = transform % 2

        if v == 1:
            img = functional.vflip(img)
        if h == 1:
            img = functional.hflip(img)

        angle = a * 90
        img = functional.rotate(img, angle)

        return img

    def __len__(self):
        return 15 * len(self.dataset)

    def __getitem__(self, item):

        transform = item // len(self.dataset)
        i = item % len(self.dataset)

        data = self.dataset[i]

        label = None

        if isinstance(data, tuple):
            img, label = data
        else:
            img = data

        img = self.transform(img, transform)
        img = functional.to_tensor(img)

        if label is None:
            return img
        else:
            return img, label


if __name__ == "__main__":
    dataset = HoromaDataset(
        data_dir="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma",
        split="train",
        transforms=functional.to_pil_image,
    )

    # dataset = FullDataset(dataset)

    # loader = DataLoader(dataset, shuffle=False, batch_size=100)

    splitter = SplitDataset(0.9)

    train, valid = splitter(dataset)

    print(len(train), len(valid))
