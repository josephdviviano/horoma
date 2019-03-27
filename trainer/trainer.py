from time import time

import numpy as np
from torch.utils.data import DataLoader

from utils.dataset import SplitDataset
from trainer.basetrainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, optimizer, resume, config,
                 unlabelled, helios_run, experiment_folder=None, **kwargs):
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
        super(Trainer, self).__init__(model, optimizer, resume, config,
                                      helios_run, experiment_folder)
        self.config = config

        ############################################
        #    Splitting into training/validation    #
        ############################################

        # Splitting 9:1 by default
        split = config['data']['dataloader'].get('split', .9)

        splitter = SplitDataset(split)

        train_set, valid_set = splitter(unlabelled)

        ############################################
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

        for batch_idx, (data) in enumerate(self.train_loader):
            start_it = time()
            data = data.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            if isinstance(output, tuple):
                loss = self.model.loss(data, *output)
            else:
                loss = self.model.loss(data, output)
            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.train_loader) + batch_idx
            self.tb_writer.add_scalar('train/loss', loss.item(), step)
            # self.comet_writer.log_metric('loss', loss.item(), step)

            total_loss += loss.item()

            end_it = time()
            time_it = end_it - start_it
            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} '.format(
                        batch_idx * self.train_loader.batch_size + data.size(
                            0),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        time_it * (len(self.train_loader) - batch_idx),
                        loss.item()))
                # grid = make_grid(data.cpu(), nrow=8, normalize=True)
                # self.tb_writer.add_image('input', grid, step)

        self.logger.info('   > Total loss: {:.6f}'.format(
            total_loss / len(self.train_loader)
        ))

        return total_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        self.model.eval()
        total_loss = 0

        self.logger.info('Valid Epoch: {}'.format(epoch))

        for batch_idx, (data) in enumerate(self.valid_loader):
            start_it = time()
            data = data.to(self.device)

            output = self.model(data)
            if isinstance(output, tuple):
                loss = self.model.loss(data, *output)
            else:
                loss = self.model.loss(data, output)

            step = epoch * len(self.valid_loader) + batch_idx
            self.tb_writer.add_scalar('valid/loss', loss.item(), step)

            total_loss += loss.item()

            end_it = time()
            time_it = end_it - start_it
            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} '.format(
                        batch_idx * self.valid_loader.batch_size + data.size(
                            0),
                        len(self.valid_loader.dataset),
                        100.0 * batch_idx / len(self.valid_loader),
                        time_it * (len(self.valid_loader) - batch_idx),
                        loss.item()))
                # grid = make_grid(data.cpu(), nrow=8, normalize=True)
                # self.tb_writer.add_image('input', grid, step)

        self.logger.info('   > Total loss: {:.6f}'.format(
            total_loss / len(self.valid_loader)
        ))

        return total_loss / len(self.valid_loader)
