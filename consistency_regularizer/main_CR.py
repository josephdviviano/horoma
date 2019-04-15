import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_utils
from trainer import Trainer
import models

# import ecgdataset
import horoma_dataset
import torch.nn.functional as F
from torchvision import transforms
# from ..utils.transforms import HoromaTransforms


def entropy_classification(x):
    return (F.log_softmax(x, dim=1) * F.softmax(x, dim=1)).sum()


# 17 classes for classification task
target_out_size_dict = {"userid": 17}

# Criterion loss to use for each output
target_criterion_dict = {"userid": nn.CrossEntropyLoss()}

# Entropy regularization for the different outputs - None means N/A (it does not apply)
target_entropy_dict = {"userid": entropy_classification}


# 0 for regression (MSE), 1 for classification (Kl_Div), None for Nothing
target_vat_dict = {"userid": 1}  # 1, # None, #1,

loss_weight = None


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch ECG Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for evaluation (default: 256)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=7000,
        metavar="N",
        help="number of iterations to train (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",  # 0.001, # 0.0001
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.005,
        metavar="ALPHA",  # 0.001 (actuellement),
        help="regularization coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        metavar="EMA",  # 0.001 (actuellement),
        help="decay for exponential moving average (default: 0.999)",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=5.0,
        metavar="XI",
        help="hyperparameter of VAT (default: 5.0)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        metavar="EPS",
        help="hyperparameter of VAT (default: 1.0)",
    )
    parser.add_argument(
        "--cr_type",
        type=int,
        default=3,
        metavar="CR",
        help="Consistency Regularization (1:Stochastic Pertubation, 2:VAT, 3:MeanTeacher - default: 3)",
    )
    parser.add_argument(
        "--ip",
        type=int,
        default=1,
        metavar="IP",
        help="hyperparameter of VAT (default: 1)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="W", help="number of CPU"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["userid"],
        help="list of targets to use for training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/rap/jvb-000-aa/COURS2019/etudiants/submissions/b3phot2/data",
        help="directory where to find the data",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="directory where to checkpoints the models",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="tensorboard/",
        help="directory where to log tensorboard data",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--chkpt-freq",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before performing checkpointing",
    )
    parser.add_argument(
        "--no-entropy",
        action="store_true",
        default=False,
        help="enables Entropy based regularization",
    )
    parser.add_argument(
        "--reg-vat-var",
        type=float,
        default=0.1,
        help="Assumed variance of the predicted Gaussian for regression tasks (default: 0.1)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    entropy_flag = not args.no_entropy
    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_gpu else "cpu")

    print("device: {}".format("GPU" if use_gpu else "GPU"))
    print("Entropy Regularization: ", entropy_flag)
    print("Targets: ", args.targets)

    targets = args.targets
    targets = ",".join(targets)

    # scaler_dict = None

    # train_labeled_dataset = ecgdataset.ECGDataset(
    #     "{}/MILA_TrainLabeledData.dat".format(args.data_dir),
    #     shape=(160, 3754),
    #     use_transform=True,
    #     target=targets,
    # )
    # scaler_dict = train_labeled_dataset.normalize_labels(scaler_dict)

    # train_unlabeled_dataset = ecgdataset.ECGDataset(
    #     "{}/MILA_UnlabeledData.dat".format(args.data_dir),
    #     shape=(657233, 3750),
    #     use_transform=False,
    #     target=None,
    #     return_doublon=True,
    # )
    # valid_dataset = ecgdataset.ECGDataset(
    #     "{}/MILA_ValidationLabeledData.dat".format(args.data_dir),
    #     shape=(160, 3754),
    #     use_transform=False,
    #     target=targets,
    # )
    # scaler_dict = valid_dataset.normalize_labels(scaler_dict)

    train_unlabeled_dataset = horoma_dataset.HoromaDataset(
        data_dir=args.data_dir,
        split="train",
        transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        flattened=True,
        # subset=1,
    )

    all_labeled_dataset = horoma_dataset.HoromaDataset(
        # data_dir="/rap/jvb-000-aa/COURS2019/etudiants/submissions/b3phot2",
        data_dir=args.data_dir,
        split="full_labeled",
        transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        flattened=True,
    )
    splitter = horoma_dataset.SplitDataset(split=0.8)
    train_labeled_dataset, valid_dataset = splitter(all_labeled_dataset)

    data_iterators = data_utils.get_iters(
        train_labeled_dataset,
        train_unlabeled_dataset,
        valid_dataset,
        None,
        l_batch_size=args.batch_size,
        ul_batch_size=args.batch_size,
        val_batch_size=args.eval_batch_size,
        workers=args.workers,
    )

    input_size = 3072
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]
    if len(target_labels) == 1:
        out_size = target_out_size_dict[target_labels[0]]
    else:
        out_size = [target_out_size_dict[a] for a in target_labels]
    n_layers = 0  # hyper-parameter
    hidden_size = 256  # hyper-parameter # 128
    kernel_size = 8  # for CNN1D only
    pool_size = 4  # for CNN1D only
    dropout = 0.2
    n_heads = 8  # 4
    key_dim = 128
    val_dim = 128
    inner_dim = 128

    model = models.TransformerNet(
        1,
        out_size,
        hidden_size,
        n_layers,
        kernel_size=kernel_size,
        pool_size=pool_size,
        n_heads=n_heads,
        key_dim=key_dim,
        val_dim=val_dim,
        inner_dim=inner_dim,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_iter = 0

    trainer = Trainer(
        args,
        device,
        target_criterion_dict,
        args.targets,
        loss_weight,
        target_vat_dict,
        target_entropy_dict if entropy_flag else None,
        init_iter,
    )

    trainer.train(
        model, data_iterators, optimizer, tb_prefix="horoma", prefix="neural_network"
    )


if __name__ == "__main__":
    main()