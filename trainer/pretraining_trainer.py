import pickle

import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam

from .cluster_kmeans_trainer import ClusterKMeansTrainer


def get_pca():
    with open('baseline/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    return pca


class PreTrainer(ClusterKMeansTrainer):
    """
    Trainer class that uses the Clustering loss as the validation criterion.
    """

    def __init__(self, model, optimizer, resume, config, unlabelled, labelled,
                 helios_run,
                 experiment_folder=None, n_clusters=20, kmeans_interval=0,
                 kmeans_headstart=0,
                 kmeans_weight=1):
        super(PreTrainer, self).__init__(model, optimizer, resume, config,
                                         unlabelled,
                                         labelled, helios_run,
                                         experiment_folder,
                                         n_clusters, kmeans_interval,
                                         kmeans_headstart,
                                         kmeans_weight)

    def _pretrain_encoder_epoch(self):

        print('Encoder')

        pca = get_pca()

        self.model.encoder.train()
        self.model.decoder.eval()
        optimizer = Adam(self.model.encoder.parameters(), lr=.001)

        for data in self.train_loader:
            self.optimizer.zero_grad()

            n = data.size(0)

            target = pca.transform(np.array(data.view(n, -1)))

            data = data.to(self.device)
            target = torch.FloatTensor(target).to(self.device)

            z = self.model.encoder(data)

            loss = functional.mse_loss(z, target)

            loss.backward()

            optimizer.step()

    def _pretrain_decoder_epoch(self):

        print('Decoder')

        pca = get_pca()

        self.model.encoder.eval()
        self.model.decoder.train()

        optimizer = Adam(self.model.decoder.parameters(), lr=.001)

        for data in self.train_loader:
            n = data.size(0)

            self.optimizer.zero_grad()

            data = data.to(self.device)

            z = torch.FloatTensor(pca.transform(np.array(data.view(n, -1))))
            z = z.to(self.device)

            x_tilde = self.model.decoder(z)

            loss = functional.mse_loss(x_tilde, data)

            loss.backward()

            optimizer.step()

    def train(self):

        print('Pretraining...')

        for _ in range(4):
            self._pretrain_encoder_epoch()
            self._pretrain_decoder_epoch()

        super()._valid_epoch(-1)

        super().train()
