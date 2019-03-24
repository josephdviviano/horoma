import torch
from torch.utils.data import DataLoader
import numpy as np


class ModelWrapper:

    def __init__(self, autoencoder, model):
        """
        Wrapper around a full clustering model.

        Args:
            autoencoder (torch.nn.Module): An auto-encoder model.
            model (utils.cluster.ClusterModel): The (pretrained) clustering model.
        """

        self.autoencoder = autoencoder
        self.model = model

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)

    def get_embeddings(self, dataset):
        """
        Transforms the data to their latent-space representation

        Args:
            dataset (torch.utils.data.Dataset): The dataset to use.

        Returns:
            embeddings (np.array): The latent representation as a numpy array.
        """

        loader = DataLoader(dataset, shuffle=False, batch_size=100)

        self.autoencoder.eval()

        embeddings = []

        with torch.no_grad():

            for data in loader:

                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]

                data = data.to(self.device)

                z = self.autoencoder.encode(data).cpu()

                if isinstance(z, tuple) or isinstance(data, list):
                    z = z[0]

                embeddings.append(z.detach().numpy())

        embeddings = np.concatenate(embeddings)

        return embeddings

    def predict(self, dataset):
        """
        Returns the Nx1 numpy array of predictions.

        Args:
            dataset (utils.dataset.HoromaDataset): The dataset on which to make the prediction.

        Returns:
            prediction (np.array): Nx1 array representing the predictions.
        """

        embeddings = self.get_embeddings(dataset)

        predictions = self.model.labelled_predict(embeddings)
        predictions = predictions.reshape(-1)

        return predictions.astype(int)
