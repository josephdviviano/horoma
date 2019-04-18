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

    labelled_full = HoromaDataset(
        data_dir=config["data"]["dataset"]['data_dir'],
        flattened=False,
        split='full_labeled_overlapped',
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

    features_full = torch.zeros(len(labelled_full),512)
    labels_full = torch.zeros(len(labelled_full))

    for idx, (x, label) in enumerate(labelled_full):
        features_full[idx] = model.feature_extract(x.unsqueeze(0).cuda()).cpu()
        labels_full[idx] = label.cpu()

    dataset_full = data_utils.TensorDataset(features_full, labels_full)
    dataset_full.region_ids = np.loadtxt("data/full_labeled_overlapped_regions_id.txt", dtype=object)   
    labeled_loader = DataLoader(dataset=dataset_full , **config['data']['dataloader']['valid'], pin_memory=True)

    for i in range(num_models):
        model.change_head(i)
        model.train()
        trainer = TrainerFactory.get(config)(
            model,
            optimizer,
            resume=resume,
            config=config,
            unlabelled=None,
            labelled=dataset_full,
            helios_run=helios_run,
            **config['trainer']['options']
        )

        trainer.train()

    model.eval()

    for i in range(num_models):
        model.change_head(i)    
        model.eval()
        total_loss = 0
        predicted = []
        labels = []

        for batch_idx, (X, y) in enumerate(labeled_loader):
            X = X.to(trainer.device)
            y = y.to(trainer.device)
            output = model(X)
            _, pred = torch.max(output, 1)
            predicted += pred.data.cpu().numpy().tolist()
            labels += y.squeeze().data.cpu().numpy().tolist()

        print("Score for head {} : {}".format(model.curr_head,f1_score(labels, predicted, average='weighted')))

    predicted = []
    labels = []
    for batch_idx, (X, y) in enumerate(labeled_loader):
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