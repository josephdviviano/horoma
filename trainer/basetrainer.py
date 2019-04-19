import os
import json
import logging
import datetime
from time import time
import sys

import torch
from tensorboardX import SummaryWriter

from utils.util import set_logger
from utils.EarlyStopping import EarlyStopping

logging.basicConfig(level=logging.INFO, format='')


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, resume, config, helios_run, 
                 experiment_folder=None, WithEarlyStop = False):
        """
        Initialize all the folders and logging for a training.

        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param helios_run: datetime helios task was started.
        :param experiment_folder: optional argument for where to log
        and save checkpoints (used for hyperparamter search).
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # Keep track of time...
        self.wall_time = config['wall_time']

        # setup directory for checkpoint saving
        if not experiment_folder:
            if helios_run:
                experiment_folder = helios_run
            else:
                experiment_folder = datetime.datetime.now().strftime(
                    '%m%d_%H%M%S')

        # setup visualization writer instance
        self.log_dir = os.path.join(cfg_trainer['log_dir'], config['name'],
                                    experiment_folder)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        set_logger(self.logger, os.path.join(self.log_dir, "file.log"))

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.tb_writer.add_text('model', str(self.model))
        self.tb_writer.add_text('config', json.dumps(config))

        self.optimizer = optimizer

        self.start_epoch = 0

        self.best_other_metrics = {}
        self.best_valid = sys.maxsize
        self.WithEarlyStop = WithEarlyStop 

        # Save configuration file into checkpoint directory:
        if not helios_run:
            config_save_path = os.path.join(self.log_dir, 'config.json')
            with open(config_save_path, 'w') as handle:
                json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        Setup GPU device if available, move model into configured device.

        :param n_gpu_use: number of gpu to use
        :return: the device where to send the tensors and the ids.
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, "
                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        t0 = time()
        if self.WithEarlyStop:
            EarlyStop = EarlyStopping()

        # with self.comet_writer.train():
        for epoch in range(self.start_epoch, self.epochs):

            # self.comet_writer.log_current_epoch(epoch)

            # Get the training and validation loss
            train_loss = self._train_epoch(epoch)
            valid_loss, f1_scr = self._valid_epoch(epoch)
            

            if isinstance(valid_loss, tuple):
                valid_loss, other_metrics = valid_loss
            else:
                other_metrics = None

            if self.WithEarlyStop:
                Stop = EarlyStop(f1_scr)
            else:
                Stop = False

            self.tb_writer.add_scalar("train/epoch_loss", train_loss,
                                      epoch)
            self.tb_writer.add_scalar("valid/epoch_loss", valid_loss,
                                      epoch)

            self._save_checkpoint(epoch, train_loss, valid_loss, other_metrics)

            time_elapsed = time() - t0

            # Break the loop if there is no more time left
            if time_elapsed * (1 + 1 / (
                    epoch - self.start_epoch + 1)) > .95 * \
                    self.wall_time * 3600:
                break
            if Stop:
                print("Breaking training Loop because of Early Stopping")
                break

        # Save the checkpoint if it's not already done.
        if not epoch % self.save_period == 0:
            self._save_checkpoint(epoch, train_loss, valid_loss, other_metrics)

        return valid_loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, train_loss, valid_loss,
                         other_metrics=None, cluster_collection=None):
        """
        Saving checkpoints. The regular checkpoint is save only if
        the epoch is within the save period.

        If the current model is the best performing, it is saved as the
        best_checkpoint, regardless of the save period.

        This method also checks for other losses, and saves the model as the
        best_checkpoint_{loss} if the model performs best according to this
        metric.

        :param epoch: current epoch number
        :param train_loss: current training loss
        :param valid_loss: current validation loss
        :param other_metrics: other metrics to save and create checkpoint from.
        :param cluster_collection: the clusters to checkpoint
        """
        arch = type(self.model).__name__

        if other_metrics is None:
            other_metrics = {}

        if self.best_valid > valid_loss or epoch == 0:
            self.best_valid = valid_loss

        for key, value in other_metrics.items():
            metric = key.split('-')[-1]
            if metric not in self.best_other_metrics or \
                    self.best_other_metrics[metric] < value:
                self.best_other_metrics[metric] = value

        # We keep the best validation loss so that
        # we don't have to load the best model to know this
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'other_metrics': other_metrics,
            'best_valid': self.best_valid,
            'best_other_metrics': self.best_other_metrics,
            'cluster_collection': cluster_collection
        }

        best_metrics_filename = os.path.join(self.log_dir, 'best_metrics.json')
        with open(best_metrics_filename, 'w') as f:
            json.dump(self.best_other_metrics, f, indent=4)

        if epoch % self.save_period == 0:
            filename = os.path.join(self.log_dir, 'last_checkpoint.pth')
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

        if self.best_valid == valid_loss:
            filename_best = os.path.join(self.log_dir, 'best_checkpoint.pth')
            torch.save(state, filename_best)
            self.logger.info(
                "Saving best checkpoint: {} ...".format(filename_best))

        for key, value in other_metrics.items():
            metric = key.split('-')[-1]
            if metric in {'accuracy', 'F1', 'recall'}:
                if self.best_other_metrics[metric] == value:
                    filename_best = os.path.join(self.log_dir,
                                                 'best_checkpoint_{}.pth'
                                                 .format(metric))
                    torch.save(state, filename_best)
                    self.logger.info(
                        "Saving best checkpoint: {} ...".format(filename_best))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is '
                'different from that of checkpoint. '
                'This may yield an exception while state_dict '
                'is being loaded.')

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint
        # only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != \
                self.config['optimizer']['type']:
            self.logger.warning(
                'Warning: Optimizer type given in config file is '
                'different from that of checkpoint. '
                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint '{}' (epoch {}) loaded".format(resume_path,
                                                       self.start_epoch))

        # Load the best model
        self.best_valid = checkpoint['best_valid']
        self.best_other_metrics = checkpoint['best_other_metrics']
