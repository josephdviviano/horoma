from sklearn.metrics import f1_score
from time import time
from trainer.basetrainer import BaseTrainer
from utils.dataset import SplitDataset
import torch

import os

import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Transforms to be used when defining loaders
train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop((28, 28)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.56121268, 0.20801756, 0.2602411], std=[0.22911494, 0.10410614, 0.11500103]),
])

eval_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.56121268, 0.20801756, 0.2602411], std=[0.22911494, 0.10410614, 0.11500103]),
])

mappings = {'BJ': 0, 'BP': 1, 'CR': 2, 'EB': 3, 'EN': 4, 'EO': 5, 'ES': 6, 'EU': 7, 'FR': 8, 'HG': 9,
            'PB': 10, 'PE': 11, 'PR': 12, 'PT': 13, 'PU': 14, 'SB': 15, 'TO': 16, 'UN': -1}


class HoromaDataset(Dataset):
    def __init__(self, root='/rap/jvb-000-aa/COURS2019/etudiants/data/horoma', train=False, transform=None, target_transform=lambda x: mappings[x]):
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            # Labeled examples from Horoma (recommended split)
            self.train_data = np.memmap(os.path.join(self.root, 'train_labeled_overlapped_x.dat'),
                                        dtype='uint8', mode="r", shape=(635, 32, 32, 3))
            self.train_labels = np.loadtxt(os.path.join(self.root, 'train_labeled_overlapped_y.txt'), 'U2').tolist()

        else:
            self.test_data = np.memmap(os.path.join(self.root, 'valid_x.dat'),
                                       dtype='uint8', mode="r", shape=(252, 32, 32, 3))
            self.test_labels = np.loadtxt(os.path.join(self.root, 'valid_y.txt'), 'U2').tolist()
            # self.test_data = np.memmap(os.path.join(self.root, 'valid_overlapped_x.dat'),
            #                            dtype='uint8', mode="r", shape=(696, 32, 32, 3))
            # self.test_labels = np.loadtxt(os.path.join(self.root, 'valid_overlapped_y.txt'), 'U2').tolist()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class SupervisedTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, optimizer, resume, config,
                 labelled, helios_run, experiment_folder=None, **kwargs):
        """
        Initialize the trainer.

        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param unlabelled: unlabelled dataset to use for training the AE.
        :param helios_run: datetime helios task was started.
        :param experiment_folder: optional argument for where to log
        and save checkpoints (used for hyperparamter search).
        :param kwargs: additional arguments if necessary
        """
        super(SupervisedTrainer, self).__init__(model, optimizer, resume, config,
                                      helios_run, experiment_folder)
        self.config = config

        ############################################
        #    Splitting into training/validation    #
        ############################################

        # Splitting 9:1 by default
        split = config['data']['dataloader'].get('split', .9)

        splitter = SplitDataset(split)

        train_set, valid_set = splitter(labelled)
        #
        # ############################################
        #  Creating the corresponding dataloaders  #
        ############################################

        train_loader = DataLoader(
            dataset=train_set,
            **config['data']['dataloader']['train'],
            pin_memory=True
        )

        valid_loader = DataLoader(
            dataset=valid_set,
            **config['data']['dataloader']['valid'],
            pin_memory=True
        )

        # train_loader = DataLoader(
        #     dataset=HoromaDataset(train=True, transform=train_transformer),
        #     **config['data']['dataloader']['train'],
        #     pin_memory=True
        # )
        #
        # valid_loader = DataLoader(
        #     dataset=HoromaDataset(train=False, transform=eval_transformer),
        #     **config['data']['dataloader']['valid'],
        #     pin_memory=True
        # )

        print(
            '>> Total batch number for training: {}'.format(len(train_loader)))
        print('>> Total batch number for validation: {}'.format(
            len(valid_loader)))
        print()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_step = int(np.sqrt(len(train_loader)))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        self.model.train()
        total_loss = 0

        self.logger.info('Train Epoch: {}'.format(epoch))

        predicted = []
        labels = []

        for batch_idx, (X, y) in enumerate(self.train_loader):
            start_it = time()
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)

            loss = self.model.loss(output, y.squeeze().long())
            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.train_loader) + batch_idx
            self.tb_writer.add_scalar('train/loss', loss.item(), step)
            # self.comet_writer.log_metric('loss', loss.item(), step)

            total_loss += loss.item()

            _, pred = torch.max(output, 1)
            predicted += pred.data.cpu().numpy().tolist()
            labels += y.squeeze().data.cpu().numpy().tolist()

            end_it = time()
            time_it = end_it - start_it
            #if batch_idx % self.log_step == 0:
            #    self.logger.info(
            #        '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} '.format(
            #            batch_idx * self.train_loader.batch_size + X.size(
            #                0),
            #            len(self.train_loader.dataset),
            #            100.0 * batch_idx / len(self.train_loader),
            #            time_it * (len(self.train_loader) - batch_idx),
            #            loss.item()))

        self.logger.info('   > Total loss: {:.6f} Total F1: {:.6f}'.format(
            total_loss / len(self.train_loader),
            f1_score(labels, predicted, average='weighted')
        ))
        self.tb_writer.add_scalar('train/f1', f1_score(labels, predicted, average='weighted'), epoch)

        return total_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        self.model.eval()
        total_loss = 0
        total_f1 = 0

        self.logger.info('Valid Epoch: {}'.format(epoch))

        predicted = []
        labels = []

        for batch_idx, (X, y) in enumerate(self.valid_loader):
            start_it = time()
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)

            loss = self.model.loss(output, y.squeeze().long())
            total_loss += loss.item()

            # F1 score
            _, pred = torch.max(output, 1)
            predicted += pred.data.cpu().numpy().tolist()
            labels += y.squeeze().data.cpu().numpy().tolist()

            step = epoch * len(self.valid_loader) + batch_idx
            self.tb_writer.add_scalar('valid/loss', loss.item(), step)

            MSG = '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} F1: {:.6f}'

            end_it = time()
            time_it = end_it - start_it
            #if batch_idx % self.log_step == 0:
            #    self.logger.info(MSG.format(
            #        batch_idx * self.valid_loader.batch_size + X.size(0),
            #        len(self.valid_loader.dataset),
            #        100.0 * batch_idx / len(self.valid_loader),
            #        time_it * (len(self.valid_loader) - batch_idx),
            #        loss.item(),
            #        f1))

        self.logger.info('   > Total loss: {:.6f}, Total F1: {:.6f}'.format(
            total_loss / len(self.valid_loader),
            f1_score(labels, predicted, average='weighted')))
        self.tb_writer.add_scalar('valid/f1', f1_score(labels, predicted, average='weighted'), epoch)

        return total_loss / len(self.valid_loader)
