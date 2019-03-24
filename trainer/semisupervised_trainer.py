from time import time

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional

from utils.clusters import ClusterModel
from trainer.cluster_kmeans_trainer import ClusterKMeansTrainer


class SemiSupervisedTrainer(ClusterKMeansTrainer):
    """
    Trainer class that uses the Clustering loss as the validation criterion.
    """

    def __init__(self, model, optimizer, resume, config, unlabelled, labelled,
                 helios_run, experiment_folder=None, n_clusters=20,
                 kmeans_interval=0,
                 kmeans_headstart=0, kmeans_weight=1, supervised_interval=.1):
        super(SemiSupervisedTrainer, self).__init__(model, optimizer, resume,
                                                    config, unlabelled,
                                                    labelled,
                                                    helios_run,
                                                    experiment_folder,
                                                    n_clusters,
                                                    kmeans_interval,
                                                    kmeans_headstart,
                                                    kmeans_weight)

        self.kmeans = ClusterModel(
            KMeans(n_clusters=n_clusters),
            cluster_helper=self.cluster_collection.cluster_helper
        )

        self.kmeans_interval = kmeans_interval
        self.kmeans_headstart = kmeans_headstart
        self.kmeans_weight = kmeans_weight

        self.supervised_interval = int(
            supervised_interval * len(self.train_loader))

    def _train_epoch_kmeans(self, epoch):
        """
        Add the distance to the centroid in the loss.

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """

        self.kmeans.fit()

        self.model.train()

        total_loss = 0
        total_kmeans_loss = 0
        total_model_loss = 0

        self.logger.info('K-Means Train Epoch: {}'.format(epoch))

        for batch_idx, (data) in enumerate(self.train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            z = self.model.encode(data)

            if isinstance(z, tuple):
                z = z[0]

            if isinstance(output, tuple):
                model_loss = self.model.loss(data, *output)
            else:
                model_loss = self.model.loss(data, output)

            centroids = torch.tensor(self.kmeans.model.cluster_centers_).to(
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

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%)] '
                    'Loss: {:.6f} ({:.3f} + {:.3f} x {:.1f})'.format(
                        batch_idx * self.train_loader.batch_size + data.size(
                            0),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
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

            if batch_idx % self.log_step == 0:
                self.logger.info('   > [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    batch_idx * self.train_loader.batch_size + data.size(0),
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader),
                    loss.item()))

            if batch_idx > 0 and batch_idx % self.supervised_interval == 0:
                self.cluster_collection.cluster_helper.build_embeddings()

                embeddings = \
                    self.cluster_collection.cluster_helper.train_embedding
                loader = self.cluster_collection.cluster_helper.train_loader

                self.kmeans.full_fit(embeddings)

                for d, l in loader:
                    d = d.to(self.device)

                    z = self.model(d)

                    signs = -1 + 2 * np.equal(
                        self.kmeans.predict(embeddings.cpu().detach()),
                        np.array(l)
                    )

                    centroids = torch.tensor(
                        self.kmeans.model.cluster_centers_).to(self.device)
                    clusters = torch.tensor(
                        self.kmeans.predict(z.cpu().detach()),
                        dtype=torch.long).to(self.device)

                    closest_centroids = torch.index_select(centroids, 0,
                                                           clusters)

                    loss = (z - closest_centroids).pow(2).mean(1) * signs

                    loss.backward()
                    self.optimizer.step()

                    pass

        self.logger.info('   > Total loss: {:.6f}'.format(
            total_loss / len(self.train_loader)
        ))

        return total_loss / len(self.train_loader)

    def train(self):
        """
        Full training logic
        """

        t0 = time()

        # We need to construct the embeddings
        self.cluster_collection.cluster_helper.build_embeddings()

        for epoch in range(self.start_epoch, self.epochs):

            if epoch % (
                    self.kmeans_interval + 1) == 0 \
                    and epoch >= self.kmeans_headstart:
                train_loss = self._train_epoch_kmeans(epoch)
            else:
                train_loss = self._train_epoch(epoch)

            valid_loss, other_metrics = self._valid_epoch(epoch)

            self.tb_writer.add_scalar("train/epoch_loss", train_loss,
                                      epoch)
            self.tb_writer.add_scalar("valid/epoch_loss", valid_loss,
                                      epoch)

            self._save_checkpoint(epoch, train_loss, valid_loss, other_metrics)

            time_elapsed = time() - t0

            # Break the loop if there is no more time left
            if time_elapsed * (1 + 1 / (
                    epoch - self.start_epoch + 1)) > .95 \
                    * self.wall_time * 3600:
                break

        # Save the checkpoint if it's not already done.
        if not epoch % self.save_period == 0:
            self._save_checkpoint(epoch, train_loss, valid_loss, other_metrics)

        return valid_loss
