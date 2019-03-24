import os
import json
import argparse
import datetime
import pickle
import copy

from skopt.space import Real
from skopt import Optimizer
import torch
import numpy as np

from utils.dataset import HoromaDataset
from utils.mnist_dataset import CustomMNIST, CustomLabelledMNIST
from utils.factories import ModelFactory, OptimizerFactory, TrainerFactory
from utils.util import table
from utils.transforms import HoromaTransforms

hyperparameters_space = [
    Real(10 ** -5, 10 ** -3, "log-uniform", name='optimizer.lr')
]


def save_optimizer(optimizer, path):
    """
    Save an instance of the hyperparameters optimizer as a pickle.

    :param optimizer: the hyperparameters optimizer to save.
    :param path: where to save.
    """
    with open(path, 'wb+') as f:
        pickle.dump(optimizer, f)


def load_optimizer(path):
    """
    Load a pickled hyperparameters optimizer.

    :param path: where the hyperparameters optimizer was saved.
    :return: the loaded hyperparameters optimizer.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def hyperparameters_parsing(hyperparameters, hyperparameters_space, config):
    """
    Create two dictionaries of parameters for the model optimizer
    and the model.
    Also, creates a markdown table to easily log the hyperparameters.

    :param hyperparameters: an array of hyperparameters
    to separate in dictionnaries.
    :param hyperparameters_space: the initial hyperparameters space,
    used to get hyperparameters name.
    :param config: the config containing the hyperparameters default values
    if not in the hyperparameters object.
    :return: model optimizer hyperparameters, model hyperparameters,
    markdown table
    """
    optimizer_hp = {}
    model_hp = copy.deepcopy(config["model"]["args"])
    hp_list = []
    for i, hp_value in enumerate(hyperparameters):
        key_name = hyperparameters_space[i].name
        type, key = key_name.split(".")
        if "optimizer" in type:
            optimizer_hp[key] = hp_value
        elif "model" in type:
            model_hp[key] = hp_value
        hp_list.append((key, str(hp_value)))
    hp_markdown = table(hp_list, [0, 1], ["Hyperparameter", "Value"])

    return optimizer_hp, model_hp, hp_markdown


def main(config, resume, test_run=False, helios_run=None, horoma_test=False):
    """
    Run an hyperparameter search with bayesian optimization.

    :param config: configuration of the model optimizer, trainer and
    model to use.
    :param resume: path to a pickled hyperparameters optimizer.
    :param test_run: whether it's a test run or not. In case of test run,
    uses custom mnist dataset.
    :param helios_run: start datetime of a run on helios.
    :param horoma_test: whether to use the test horoma dataset or not.
    """
    np.random.seed(config["numpy_seed"])
    torch.manual_seed(config["torch_seed"])
    torch.cuda.manual_seed_all(config["torch_seed"])

    # setup data_loader instances
    if not test_run:

        unlabelled = HoromaDataset(
            **config["data"]["dataset"],
            split='train_overlapped',
            transforms=HoromaTransforms()
        )

        labelled = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='valid_overlapped',
            transforms=HoromaTransforms()
        )
    elif horoma_test:

        unlabelled = HoromaDataset(
            **config["data"]["dataset"],
            split='train_overlapped',
            transforms=HoromaTransforms(),
            subset=5000
        )

        labelled = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='valid_overlapped',
            transforms=HoromaTransforms(),
            subset=1000
        )
    else:
        unlabelled = CustomMNIST(**config["data"]["dataset"], subset=5000)
        labelled = CustomLabelledMNIST(**config["data"]["dataset"],
                                       subset=1000)

    model_hyperparameters_space = ModelFactory.getclass(
        config["model"]["type"]
    ).model_hyperparameters_space

    for h in model_hyperparameters_space:
        h.name = "model.{}".format(h.name)

    hyperparameters_space.extend(
        model_hyperparameters_space
    )

    if not helios_run:
        experiment_datetime = datetime.datetime.now().strftime('%m%d_%H%M%S')
    else:
        experiment_datetime = helios_run

    checkpoint_path = os.path.join(config["trainer"]["log_dir"],
                                   config["name"],
                                   experiment_datetime, 'optimizer.pkl')

    if not resume:
        hp_optimizer = Optimizer(hyperparameters_space)
    else:
        hp_optimizer = load_optimizer(resume)

    for experiment_number in range(len(hp_optimizer.yi), 20):
        hyperparameters = hp_optimizer.ask()

        experiment_folder = os.path.join(experiment_datetime,
                                         str(experiment_number))

        optimizer_hp, model_hp, hp_markdown = \
            hyperparameters_parsing(hyperparameters, hyperparameters_space,
                                    config)

        model = ModelFactory.getclass(
            config["model"]["type"]
        )(**model_hp)

        trainable_params = filter(lambda p: p.requires_grad,
                                  model.parameters())
        optimizer = OptimizerFactory.getclass(
            config["optimizer"]["type"]
        )(trainable_params, **optimizer_hp)

        trainer = TrainerFactory.get(config)(
            model,
            optimizer,
            resume=None,
            config=config,
            unlabelled=unlabelled,
            labelled=labelled,
            helios_run=helios_run,
            experiment_folder=experiment_folder,
            **config['trainer']['options']
        )
        trainer.logger.info(hp_markdown)
        trainer.tb_writer.add_text("hyperparameters", hp_markdown)
        score = trainer.train()

        hp_optimizer.tell(hyperparameters, score)

        save_optimizer(hp_optimizer, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoEncoder Training')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--test-run', action='store_true',
                        help='execute a test run on MNIST')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument("--helios-run", default=None, type=str,
                        help='if the train is run on helios '
                             'with the run_experiment script,'
                             'the value should be the time at '
                             'which the task was submitted')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    elif args.test_run:
        config = json.load(open("./configs/variational_mnist.json"))
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.test_run, args.helios_run)
