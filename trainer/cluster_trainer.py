from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from utils.clusters import ClusterCollection
from trainer.trainer import Trainer
from utils.dataset import SplitDataset


class ClusterTrainer(Trainer):
    """
    Trainer class that uses the Clustering loss as the validation criterion.
    """

    def __init__(self, model, optimizer, resume, config, unlabelled, labelled,
                 helios_run, experiment_folder=None):
        """
        Initialize a cluster trainer.

        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param unlabelled: unlabelled dataset to use for training the AE.
        :param labelled: labelled dataset to use for clustering.
        :param helios_run: datetime helios task was started.
        :param experiment_folder: optional argument for where to log
        and save checkpoints (used for hyperparamter search).
        """
        super(ClusterTrainer, self).__init__(model, optimizer, resume, config,
                                             unlabelled, helios_run,
                                             experiment_folder)

        splitter = SplitDataset(.7)

        labelled_train_set, labelled_valid_set = splitter(labelled)

        labelled_train_loader = DataLoader(
            dataset=labelled_train_set,
            shuffle=False,
            batch_size=100
        )

        labelled_valid_loader = DataLoader(
            dataset=labelled_valid_set,
            shuffle=False,
            batch_size=100
        )

        models = {
            'kmeans20': KMeans(n_clusters=20, n_jobs=-1),
            'kmeans100': KMeans(n_clusters=100, n_jobs=-1),
            'kmeans300': KMeans(n_clusters=300, n_jobs=-1),
            'gmm20': GaussianMixture(n_components=20),
            'gmm100': GaussianMixture(n_components=100),
            'gmm300': GaussianMixture(n_components=300)
        }

        self.cluster_collection = ClusterCollection(
            models,
            self.model,
            self.device,
            self.valid_loader,
            labelled_train_loader,
            labelled_valid_loader
        )

    def _save_checkpoint(self, epoch, train_loss, valid_loss,
                         other_metrics=None, cluster_collection=None):
        super()._save_checkpoint(epoch, train_loss, valid_loss, other_metrics,
                                 self.cluster_collection)

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        loss = super(ClusterTrainer, self)._valid_epoch(epoch)

        self.cluster_collection.full_fit()

        other_metrics = self.cluster_collection.get_clustering_metrics()

        for key, value in other_metrics.items():
            model, metric = key.split('-')
            self.tb_writer.add_scalar('{}/{}'.format(model, metric), value,
                                      epoch)

        return loss, other_metrics
