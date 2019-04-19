import os
import json
import argparse

import torch
import numpy as np

from utils.dataset import HoromaDataset
from utils.mnist_dataset import CustomMNIST
from utils.factories import ModelFactory, OptimizerFactory, TrainerFactory
from utils.transforms import HoromaTransforms, HoromaTransformsResNet
from utils.mnist_dataset import CustomLabelledMNIST
import torch.utils.data as data_utils
from utils.dataset import SplitDataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

def main(config):
    TRANSFORMS = HoromaTransformsResNet()

    labelled_train = HoromaDataset(
        data_dir=config["data"]["dataset"]['data_dir'],
        flattened=False,
        split='train_labeled_overlapped',
        transforms=TRANSFORMS
    )

    
    labelled_test= HoromaDataset(
        data_dir=config["data"]["dataset"]['data_dir'],
        flattened=False,
        split='valid_overlapped',
        transforms=TRANSFORMS
    )

    num_models = config["model"]["args"]["num_heads"]
    model = ModelFactory.get(config)

    model.cuda()
    model.eval()

    resume = False
    helios_run = None
    trainable_params = model.parameters()
    optimizer = OptimizerFactory.get(config, trainable_params)

    features_train = torch.zeros(len(labelled_train),512)
    labels_train = torch.zeros(len(labelled_train))

    for idx, (x, label) in enumerate(labelled_train):
        features_train[idx] = model.feature_extract(x.unsqueeze(0).cuda()).cpu()
        labels_train[idx] = label.cpu()

    features_test = torch.zeros(len(labelled_test),512)
    labels_test = torch.zeros(len(labelled_test))

    for idx, (x, label) in enumerate(labelled_test):
        features_test[idx] = model.feature_extract(x.unsqueeze(0).cuda()).cpu()
        labels_test[idx] = label.cpu()
    
    """

    labelled_full = HoromaDataset(
        data_dir=config["data"]["dataset"]['data_dir'],
        flattened=False,
        split='full_labeled_overlapped',
        transforms=TRANSFORMS
    )

    features_full = torch.zeros(len(labelled_full),512)
    labels_full = torch.zeros(len(labelled_full))

    for idx, (x, label) in enumerate(labelled_full):
        features_full[idx] = model.feature_extract(x.unsqueeze(0).cuda()).cpu()
        labels_full[idx] = label.cpu()


    n = len(labelled_full)
    test_split = .5
    n_split = int(test_split * n)

    full_region_ids = np.loadtxt("{}/full_labeled_overlapped_regions_id.txt".format(config["data"]["dataset"]["data_dir"]), dtype=object)
    unique_regions, unique_region_inverse, unique_region_counts = np.unique( full_region_ids , return_counts=True, return_inverse=True)
    unique_regions = np.arange(unique_region_inverse.max() + 1)

    np.random.shuffle(unique_regions)
    cumsum = np.cumsum(unique_region_counts[unique_regions])

    last_region = np.argmax(1 * (cumsum > n_split))

    train_regions = unique_regions[:last_region]
    test_regions = unique_regions[last_region:]

    indices = np.arange(n)

    train_indices = indices[np.isin(unique_region_inverse, train_regions)]
    test_indices = indices[np.isin(unique_region_inverse, test_regions)]

    features_train = features_full[train_indices,:]
    labels_train = labels_full[train_indices]
    region_ids_train = full_region_ids[train_indices]
    dataset_train = data_utils.TensorDataset(features_train, labels_train)
    dataset_train.region_ids = region_ids_train

    features_test = features_full[test_indices,:]
    labels_test = labels_full[test_indices]
    region_ids_test = full_region_ids[test_indices]
    dataset_test = data_utils.TensorDataset(features_test, labels_test)
    dataset_test.region_ids = region_ids_test

    print(features_train.size())

    """

    dataset_train = data_utils.TensorDataset(features_train, labels_train)
    dataset_train.region_ids = labelled_train.region_ids

    dataset_test = data_utils.TensorDataset(features_test, labels_test)
    dataset_test.region_ids = labelled_test.region_ids


    for i in range(num_models):
        model.change_head(i)
        model.train()
        trainer = TrainerFactory.get(config)(
            model,
            optimizer,
            resume=resume,
            config=config,
            unlabelled=None,
            labelled=dataset_train,
            helios_run=helios_run,
            **config['trainer']['options']
        )

        trainer.train()

    test_loader = DataLoader(dataset=dataset_test, **config['data']['dataloader']['valid'], pin_memory=True)


    f1_scores = np.zeros(num_models)

    for i in range(num_models):
        model.change_head(i)
        model.eval()
        predicted = []
        labels = []

        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(trainer.device)
            y = y.to(trainer.device)
            output = model(X)
            _, pred = torch.max(output, 1)
            predicted += pred.data.cpu().numpy().tolist()
            labels += y.squeeze().data.cpu().numpy().tolist()

        curr_f1 = f1_score(labels, predicted, average='weighted')
        f1_scores[i] = curr_f1

        print("Score for head {} : {}".format(model.curr_head,curr_f1))

    best_heads = np.where(f1_scores>np.percentile(f1_scores,70))[0].tolist()


    model.eval()
    predicted = []
    labels = []
    for batch_idx, (X, y) in enumerate(test_loader):
        predicted += model.ensemble_predict(X.cuda(),"hard", best_heads).cpu().numpy().tolist()
        labels += y.squeeze().data.cpu().numpy().tolist()

    print("Score for ensemble with only best heads: {}".format(f1_score(labels, predicted, average='weighted')))

    model.eval()
    predicted = []
    labels = []
    for batch_idx, (X, y) in enumerate(test_loader):
        predicted += model.ensemble_predict(X.cuda(),"hard").cpu().numpy().tolist()
        labels += y.squeeze().data.cpu().numpy().tolist()

    print("Score for ensemble : {}".format(f1_score(labels, predicted, average='weighted')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Training')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--config', default=None, type=str,
                       help='config file path (default: None)')
    args = parser.parse_args()

    config = json.load(open(args.config))


    main(config)
