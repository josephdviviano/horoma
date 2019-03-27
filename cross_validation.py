import json
import argparse
import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

import models
from utils.dataset import HoromaDataset, SplitDataset, FullDataset
from utils.transforms import HoromaTransforms
from utils.clusters import ClusterModel, ClusterHelper, MajorityVote
from utils.wrapper import ModelWrapper

import warnings  # To mute scikit-learn warnings about f1 score.
warnings.filterwarnings("ignore")


def get_classes(split_dataset):
    labels = split_dataset.dataset.targets
    classes = np.zeros(labels.max() + 1)
    labels = labels[split_dataset.indices]

    values, counts = np.unique(labels, return_counts=True)

    for value, count in zip(values, counts):
        classes[value] = count

    return classes


def get_model(conf):
    return getattr(models, conf['model']['type'])(**conf['model']['args'])


def retrieve_config(name):
    with open(name, 'r') as f:
        conf = json.load(f)
    return conf


def main(path, data, model_name):
    print('Path: {}'.format(path))
    print('Data: {}'.format(data))
    print('Model: {}'.format(model_name))

    print()

    print('>> Beginning...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unlabelled = HoromaDataset(
        data,
        split='train_overlapped',
        transforms=HoromaTransforms()
    )

    labelled = HoromaDataset(
        data_dir=data,
        split='valid_overlapped',
        transforms=HoromaTransforms()
    )
    labelled = FullDataset(labelled)

    print('>> Dataset lengths : {}, {}'.format(len(unlabelled), len(labelled)))

    labelled_loader = DataLoader(labelled, batch_size=100, shuffle=False)

    print('>> Getting the configuration...', end=' ')

    config = retrieve_config(path + 'config.json')

    print('Done.')

    try:
        model = get_model(config).to(device)
    except TypeError:
        config['model']['args']['dropout'] = .1
        model = get_model(config).to(device)

    print('>> Getting the checkpoint...', end=' ')

    checkpoint = torch.load(path + model_name, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        kmeans = checkpoint['cluster_collection'].models['kmeans100'].model
        del checkpoint
    else:
        state_dict = checkpoint

    print('Done.')

    # torch.save(state_dict, path + 'bare_model.pth')

    model.load_state_dict(state_dict)

    model.to('cpu')
    torch.save(model, path + 'bare_model.pth')
    model.to(device)

    cluster_helper = ClusterHelper(
        model=model,
        device=device,
        unlabelled_loader=None,
        train_loader=labelled_loader,
        valid_loader=labelled_loader
    )

    best_kmeans = None
    best_score = 0.

    threshold = .4
    good_models = []

    for n_clusters in [30, 50, 80, 100, 150, 200]:

        print()
        print()

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000)
        # kmeans = KMeans(n_clusters=100)

        print(kmeans)

        print('>> Getting embeddings...', end=' ')
        data = cluster_helper.get_embeddings(
            DataLoader(unlabelled, batch_size=100, shuffle=True)
        )
        print('Done.')

        print('>> Fitting data...', end=' ')
        kmeans.fit(data)
        print('Done.')

        # n = len(np.unique(labelled.region_ids))
        n = 40

        accuracies = np.empty(n)
        f1_scores = np.empty(n)

        for i in range(n):
            print()

            kfold = SplitDataset(split=.7)
            train, valid = kfold(labelled)

            print('Train: {}'.format(get_classes(train)))
            print('Valid: {}'.format(get_classes(valid)))

            cluster_helper = ClusterHelper(
                model=model,
                device=device,
                unlabelled_loader=None,
                train_loader=DataLoader(train, shuffle=False, batch_size=100),
                valid_loader=DataLoader(valid, shuffle=False, batch_size=100)
            )

            clustering_model = ClusterModel(
                kmeans,
                cluster_helper=cluster_helper
            )

            cluster_helper.build_valid_embeddings()
            cluster_helper.build_train_embeddings()

            clustering_model.create_mapping()

            labels = cluster_helper.valid_labels
            prediction = clustering_model.labelled_predict(
                cluster_helper.valid_embedding
            )

            unlabelled_examples = (prediction == -1).sum()

            print('Unlabelled examples : {}/{} ({:.2%})'.format(
                unlabelled_examples,
                len(prediction),
                unlabelled_examples / len(prediction)
            ))

            accuracy = metrics.accuracy_score(labels, prediction)
            f1_score = metrics.f1_score(labels, prediction, average='weighted')

            accuracies[i] = accuracy
            f1_scores[i] = f1_score

            if accuracy > threshold and f1_score > threshold:
                good_models.append(clustering_model)

            print('Accuracy: {:.3%}'.format(accuracy))
            print('F1 Score: {:.3%}'.format(f1_score))

        print()
        print('  > Average accuracy: {:.3%}'.format(accuracies.mean()))
        print('  > Average F1 Score: {:.3%}'.format(f1_scores.mean()))

        if f1_scores.mean() > best_score:
            best_score = f1_scores.mean()
            best_kmeans = kmeans

    majority_vote = MajorityVote(good_models)

    # Final fit
    cluster_helper = ClusterHelper(
        model=model,
        device=device,
        unlabelled_loader=None,
        train_loader=labelled_loader,
        valid_loader=labelled_loader
    )

    print()
    print('>> Training finished')
    print('>> Best score: {:.3%}\n{}'.format(best_score, best_kmeans))

    clustering_model = ClusterModel(
        best_kmeans,
        cluster_helper=cluster_helper
    )

    cluster_helper.build_valid_embeddings()
    cluster_helper.build_train_embeddings()

    # Test the majority vote
    print('>> Majority vote : {}'.format(
        metrics.f1_score(
            cluster_helper.valid_labels,
            majority_vote.labelled_predict(cluster_helper.valid_embedding),
            average='weighted'
        )
    ))

    clustering_model.create_mapping()
    del clustering_model.cluster_helper

    model.to('cpu')

    wrapper = ModelWrapper(model, clustering_model)
    wrapper_maj = ModelWrapper(model, majority_vote)

    with open(path + 'final_model_full.pkl', 'wb') as f:
        pickle.dump(wrapper, f)

    with open(path + 'final_model_maj_full.pkl', 'wb') as f:
        pickle.dump(wrapper_maj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross validation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='path to the model experiment')
    parser.add_argument('-m', '--model', type=str, help='model to use')
    parser.add_argument('-d', '--data', default=None, type=str,
                        help='path to data_directory')
    args = parser.parse_args()

    main(args.path, args.data, args.model)
