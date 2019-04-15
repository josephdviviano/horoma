from time import time

from sklearn.cluster import KMeans
import torch
from torch.nn import functional

from utils.clusters import ClusterModel
from trainer.cluster_trainer import ClusterTrainer


class ClusterKMeansVadeTrainer(ClusterTrainer):
    """
    Cluster trainer that apply a kmeans penalty for the VADE model, which
    requires the user to initialize the GMM before training.
    """

    def __init__(self, model, optimizer, resume, config, unlabelled, labelled,
                 helios_run, experiment_folder=None,
                 n_clusters=20, kmeans_interval=0, kmeans_headstart=0,
                 kmeans_weight=1):
        """
        Initialize a cluster kmeans trainer for the Vade model.

        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param unlabelled: unlabelled dataset to use for training the AE.
        :param labelled: labelled dataset to use for clustering.
        :param helios_run: datetime helios task was started.
        :param experiment_folder: optional argument for where to log
        and save checkpoints (used for hyperparamter search).
        :param n_clusters: the number of clusters for the kmeans.
        :param kmeans_interval: at which interval to execute a clustering with
        the kmeans.
        :param kmeans_headstart: how many epoch to wait because doing
        a clustering.
        :param kmeans_weight: weight for the kmeans penalty
        """
        super(ClusterKMeansVadeTrainer, self).__init__(
            model, optimizer, resume, config, unlabelled, labelled,
            helios_run, experiment_folder)

        self.kmeans = ClusterModel(
            KMeans(n_clusters=n_clusters),
            cluster_helper=self.cluster_collection.cluster_helper
        )

        self.kmeans_interval = kmeans_interval
        self.kmeans_headstart = kmeans_headstart
        self.kmeans_weight = kmeans_weight

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
            start_it = time()
            data = data.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            z = self.model.encode(data)

            if isinstance(z, tuple):
                z = z[0]

            if isinstance(output, tuple):
                model_loss = self.model.loss(*output)
            else:
                # TODO: This will never be used with VaDE!
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
                loss = self.model.loss(*output)
            else:
                # TODO: This will never be used with VaDE!
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

    def _valid_epoch_model_loss(self, epoch):
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
                loss = self.model.loss(*output)
            else:
                # TODO: This will never be used with VaDE!
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

        return(total_loss / len(self.valid_loader))

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        # Get standard loss.
        loss = self._valid_epoch_model_loss(epoch)

        # Get loss from cluster centroid distances.
        self.cluster_collection.full_fit()
        other_metrics = self.cluster_collection.get_clustering_metrics()

        for key, value in other_metrics.items():
            model, metric = key.split('-')
            self.tb_writer.add_scalar('{}/{}'.format(model, metric), value,
                                      epoch)

        return(loss, other_metrics)

    def train(self):
        """
        Full training logic for the cluster trainer.
        """
        t0 = time()

        # 5 epochs of pre-training
        pretrain_criterion = torch.nn.MSELoss()

        self.model.train()
        self.model.set_mode("pretrain")
        for epoch in range(5):
            print('pretrain epoch {}'.format(epoch))
            for batch_idx, data in enumerate(self.train_loader):

                data = data.to(self.device)
                data_recon = self.model.pretrain(data)
                loss = pretrain_criterion(data, data_recon)
                loss.backward()
                self.optimizer.step()

        self.model.set_mode("train")

        # Construct the embeddings.
        self.cluster_collection.cluster_helper.build_embeddings()

        # Initialize the GMM.
        self.model.initialize_gmm(self.train_loader)

        for epoch in range(self.start_epoch, self.epochs):

            if epoch % (
                    self.kmeans_interval + 1) == 0 \
                    and epoch >= self.kmeans_headstart:
                train_loss = self._train_epoch_kmeans(epoch)
            else:
                train_loss = self._train_epoch(epoch)

            valid_loss, other_metrics = self._valid_epoch(epoch)

            self.tb_writer.add_scalar("train/epoch_loss", train_loss, epoch)
            self.tb_writer.add_scalar("valid/epoch_loss", valid_loss, epoch)
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
