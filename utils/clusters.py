from time import time

from sklearn import metrics
import torch
import numpy as np


class ClusterModel:
    """
    Interface for cluster algorithm
    """

    def __init__(self, model, cluster_helper):
        """
        :param model: The clustering model used.
        :param cluster_helper: A ClusterHelper instance
        that avoids recomputing everything
        """

        self.cluster_helper = cluster_helper

        self.model = model

        self.mapping = None

    def fit(self, train=None):
        """
        Fit the cluster algorithm.

        :param train: training data
        """
        if train is None:
            train = self.cluster_helper.unlabelled_embeddings

        self.model.fit(train)

    def predict(self, x):
        """
        Make a prediction from x

        :param x: data to use to predict
        :return: prediction
        """
        predictions = self.model.predict(x)
        return predictions

    def labelled_predict(self, x):
        """
        Make a prediction and label it.

        :param x: data to use to predict
        :return: labeled prediction
        """
        predictions = self.predict(x)
        return np.array([self.mapping.get(p, -1) for p in predictions])

    def create_mapping(self, valid=None, labels=None):
        """
        Associate a label with each cluster from the train embeddings

        :param valid: split from the validition used for labeling
        :param labels:
        :return:
        """
        if valid is None:
            valid = self.cluster_helper.train_embedding
            labels = self.cluster_helper.train_labels

        prediction = self.model.predict(valid)

        uniques = np.unique(prediction)

        self.mapping = dict()

        for u in uniques:
            filtered_labels = labels[prediction == u]

            if len(filtered_labels) != 0:
                values, counts = np.unique(filtered_labels, return_counts=True)
                self.mapping[u] = values[counts.argmax()]

    def full_fit(self, train=None, valid=None, labels=None):
        """
        Performs the fit and labels the clusters.

        Args:
            train (np.array): Array of training examples.
            valid (np.array): Array of validation examples.
            labels (np.array): Array containing the labels of the validation examples.
        """

        self.fit(train)
        self.create_mapping(valid, labels)


class ClusterHelper:
    """
    Contains pre-computed
    """

    def __init__(self, model, device, unlabelled_loader, train_loader,
                 valid_loader):

        self.unlabelled_loader = unlabelled_loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.train_labels = self.get_labels(train_loader)
        self.valid_labels = self.get_labels(valid_loader)

        self.unlabelled_embeddings = None
        self.train_embedding = None
        self.valid_embedding = None

        self.model = model
        self.device = device

    @staticmethod
    def get_labels(dataloader):
        """
        Returns the labels from a dataloader as a numpy array.

        Args:
            dataloader (torch.utils.data.DataLoader): A labelled dataloader.

        Returns:
            labels (np.array): The labels contained in the dataloader.
        """

        labels = []

        for _, label in dataloader:
            labels.append(label)

        labels = np.concatenate(labels)

        return labels.reshape(-1)

    def get_embeddings(self, dataloader):
        """
        Returns the latent representation of the dataset loaded by the dataloader,
        as a numpy array.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader.

        Returns:
            embeddings (np.array): The embedding of the dataset.
        """

        embeddings = []

        self.model.eval()

        with torch.no_grad():
            for data in dataloader:

                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]

                data = data.to(self.device)

                z = self.model.encode(data)

                # Assume the latent representation is the first item
                # returned by self.encode(). Move it to the cpu.
                if isinstance(z, tuple) or isinstance(data, list):
                    z = z[0].cpu()
                else:
                    z = z.cpu()

                embeddings.append(z.detach().numpy())

        embeddings = np.concatenate(embeddings)

        return embeddings

    def build_unlabelled_embeddings(self):
        """Builds the embedding for the unlabelled dataset"""
        self.unlabelled_embeddings = self.get_embeddings(
            self.unlabelled_loader)

    def build_train_embeddings(self):
        """Builds the embedding for the training labelled dataset"""
        self.train_embedding = self.get_embeddings(self.train_loader)

    def build_valid_embeddings(self):
        """Builds the embedding for the validation labelled dataset"""
        self.valid_embedding = self.get_embeddings(self.valid_loader)

    def build_embeddings(self):
        """Builds all enecessary embeddings"""
        self.build_unlabelled_embeddings()
        self.build_train_embeddings()
        self.build_valid_embeddings()


class ClusterCollection:
    """
    Contains a collection of clustering algorithms,
    along with a ClusterHelper object to avoid repetition.
    """

    def __init__(self, models, model, device, unlabelled_loader, train_loader,
                 valid_loader):

        self.cluster_helper = ClusterHelper(
            model,
            device,
            unlabelled_loader,
            train_loader,
            valid_loader
        )

        self.models = {
            key: ClusterModel(value, self.cluster_helper)
            for key, value in models.items()
        }

    def full_fit(self):
        """
        Performs the full fit on every clustering model.
        """

        self.cluster_helper.build_embeddings()

        for model in self.models.values():
            t0 = time()
            print('Fitting Model: {}'.format(model.model), end='... ')
            model.full_fit()
            print('Done. Time elapsed : {:.3f}s.'.format(time() - t0))

    def get_clustering_metrics(self):
        """
        Gets the clustering metrics for each model.

        Returns:
            other_metrics (dict): A dictionary containing the metric for each model.
        """

        clustering_metrics = {
            'ARI': metrics.adjusted_rand_score,
            'mutual_info': metrics.adjusted_mutual_info_score,
            'homogeneity': metrics.homogeneity_score,
            'completeness': metrics.completeness_score,
            'F1': metrics.f1_score,
            'accuracy': metrics.accuracy_score,
            'recall': metrics.recall_score
        }

        other_metrics = {}

        for key, model in self.models.items():

            cluster_predict = model.predict(
                self.cluster_helper.valid_embedding)
            label_predict = model.labelled_predict(
                self.cluster_helper.valid_embedding)
            labels = self.cluster_helper.valid_labels

            for name, metric in clustering_metrics.items():

                if name in {'F1', 'recall'}:
                    other_metrics['{}-{}'.format(key, name)] = metric(labels,
                                                                      label_predict,
                                                                      average='weighted')
                elif name == 'accuracy':
                    other_metrics['{}-{}'.format(key, name)] = metric(labels,
                                                                      label_predict)
                elif name == 'mutual_info':
                    other_metrics['{}-{}'.format(key, name)] = metric(labels,
                                                                      cluster_predict,
                                                                      average_method='arithmetic')
                else:
                    other_metrics['{}-{}'.format(key, name)] = metric(labels,
                                                                      cluster_predict)

                print('{}-{}: {}'.format(key, name, other_metrics[
                    '{}-{}'.format(key, name)]))

        return other_metrics


class MajorityVote(object):

    def __init__(self, models):
        """
        Args:
            models (list of ClusterModels): Pre-trained ClusterModel objects.
        """

        self.models = models

        for model in self.models:
            del model.cluster_helper

    def labelled_predict(self, x):
        """
        Returns a majority vote for each row of x.

        Args:
            x (np.array): Array of embeddings.

        Returns:
            prediction (np.array): 1D array of predictions.
        """

        predictions = np.stack([
            model.labelled_predict(x)
            for model in self.models
        ], axis=1)

        def get_vote(row):
            """
            Gets the vote from a row of prediction.
            In case of ties, returns a random choice.

            Args:
                row (np.array): A row of a numpy array.

            Returns:
                vote (int): The majority vote
            """

            uniques, counts = np.unique(row, return_counts=True)

            counts = counts[uniques != -1]
            uniques = uniques[uniques != -1]

            if len(counts) == 0:
                return -1

            uniques = uniques[counts == counts.max()]

            return np.random.choice(uniques)

        prediction = np.array([
            get_vote(row)
            for row in predictions
        ])

        return prediction
