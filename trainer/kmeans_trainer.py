from time import time

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional

from trainer.trainer import Trainer


class KMeansTrainer(Trainer):
    """
    Trainer class that applies a kmeans penalty proportional to how far
    the clusters are.
    """

    def __init__(self, model, optimizer, resume, config, helios_run,
                 test_run=False, experiment_folder=None, n_clusters=20,
                 kmeans_interval=1, kmeans_headstart=2, kmeans_weight=.1,
                 **kwargs):
        """
        Initialize a KmeansTrainer.

        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param helios_run: datetime helios task was started.
        :param experiment_folder: optional argument for where to log
        :param n_clusters: the number of clusters for the kmeans.
        :param kmeans_interval: at which interval to execute a clustering with
        the kmeans.
        :param kmeans_headstart: how many epoch to wait because doing
        a clustering.
        :param kmeans_weight: weight for the kmeans penalty
        :param kwargs: additional arguments
        """

        super(KMeansTrainer, self).__init__(model, optimizer, resume, config,
                                            helios_run, test_run,
                                            experiment_folder)

        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans_interval = kmeans_interval
        self.kmeans_headstart = kmeans_headstart
        self.kmeans_weight = kmeans_weight

    def _fit_kmeans(self):
        """
        Train the kmeans.

        :return:
        """
        embeddings = []

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data) in enumerate(self.train_loader):
                data = data.to(self.device)

                z = self.model.encode(data).cpu()

                embeddings.append(z.detach().numpy())

            embeddings = np.concatenate(embeddings)

            self.kmeans.fit(embeddings)

    def _train_epoch_kmeans(self, epoch):
        """
        Add the distance to the centroid in the loss.

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """

        self._fit_kmeans()

        self.model.train()

        total_loss = 0
        total_kmeans_loss = 0
        total_model_loss = 0

        self.logger.info('K-Means Train Epoch: {}'.format(epoch))

        for batch_idx, (data) in enumerate(self.train_loader):
            start_it = time()
            data = data.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            z = self.model.encode(data)

            if isinstance(output, tuple):
                model_loss = self.model.loss(data, *output)
            else:
                model_loss = self.model.loss(data, output)

            centroids = torch.tensor(self.kmeans.cluster_centers_).to(
                self.device)
            clusters = torch.tensor(self.kmeans.predict(z.cpu().detach()),
                                    dtype=torch.long).to(self.device)

            closest_centroids = torch.index_select(centroids, 0, clusters)

            kmeans_loss = functional.mse_loss(z, closest_centroids)

            loss = model_loss + kmeans_loss * self.kmeans_weight

            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.train_loader) + batch_idx
            self.tb_writer.add_scalar('train/loss', loss.item(), step)

            total_loss += loss.item()
            total_kmeans_loss += kmeans_loss.item()
            total_model_loss += model_loss.item()

            end_it = time()
            time_it = end_it - start_it
            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] '
                    'Loss: {:.6f} ({:.3f} + {:.3f} x {:.1f})'.format(
                        batch_idx * self.train_loader.batch_size + data.size(
                            0),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        time_it * (len(self.train_loader) - batch_idx),
                        loss.item(),
                        model_loss.item(),
                        kmeans_loss.item(),
                        self.kmeans_weight
                    ))
                # grid = make_grid(data.cpu(), nrow=8, normalize=True)
                # self.tb_writer.add_image('input', grid, step)

        self.logger.info(
            '   > Total loss: {:.6f} ({:.3f} + {:.3f} x {:.1f})'.format(
                total_loss / len(self.train_loader),
                total_model_loss / len(self.train_loader),
                total_kmeans_loss / len(self.train_loader),
                self.kmeans_weight
            ))

        # We return the model loss for coherence
        return total_model_loss / len(self.train_loader)

    def train(self):
        """
        Full training logic for the kmeanstrainer
        """

        t0 = time()

        for epoch in range(self.start_epoch, self.epochs):

            if epoch % (
                    self.kmeans_interval + 1) == 0 \
                    and epoch >= self.kmeans_headstart:
                train_loss = self._train_epoch_kmeans(epoch)
            else:
                train_loss = self._train_epoch(epoch)

            valid_loss = self._valid_epoch(epoch)

            self.tb_writer.add_scalar("train/epoch_loss", train_loss,
                                      epoch)
            self.tb_writer.add_scalar("valid/epoch_loss", valid_loss,
                                      epoch)

            self._save_checkpoint(epoch, train_loss, valid_loss)

            time_elapsed = time() - t0

            # Break the loop if there is no more time left
            if time_elapsed * (1 + 1 / (
                    epoch - self.start_epoch + 1)) > .95 \
                    * self.wall_time * 3600:
                break

        # Save the checkpoint if it's not already done.
        if not epoch % self.save_period == 0:
            self._save_checkpoint(epoch, train_loss, valid_loss)
