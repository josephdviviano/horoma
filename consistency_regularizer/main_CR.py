import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_utils
from trainer import Trainer
import models

import horoma_dataset
import torch.nn.functional as F
from torchvision import transforms
from image_transforms import HoromaTransforms, HoromaTransformsCR


def entropy_classification(x):
    return (F.log_softmax(x, dim=1) * F.softmax(x, dim=1)).sum()


# 17 classes for classification task
target_out_size_dict = {"treeid": 17}

# Criterion loss to use for each output
target_criterion_dict = {"treeid": nn.CrossEntropyLoss()}

# Entropy regularization for the different outputs - None means N/A (it does not apply)
target_entropy_dict = {"treeid": entropy_classification}


# 0 for regression (MSE), 1 for classification (Kl_Div), None for Nothing
target_vat_dict = {"treeid": 1}  # 1, # None, #1,

loss_weight = None


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Horoma tree classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=635,
        metavar="N",
        help="input batch size for training (default: 635)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=252,
        metavar="N",
        help="input batch size for evaluation (default: 252)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        metavar="N",
        help="number of iterations to train (default: 2000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",  # 0.001, # 0.0001
        help="learning rate (default: 0.0001)",
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
        help="regularization coefficient (default: 0.005)",
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
        "--dropout",
        type=float,
        default=0.5,
        metavar="DROPOUT",
        help="hyperparameter of convnet (default: 0.5)",
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
        default=["treeid"],
        help="list of targets to use for training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="/rap/jvb-000-aa/COURS2019/etudiants/submissions/b3phot2/data",
        default="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/",
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
        default=1,
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
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="If not to use transformer in prediction model",
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


    train_unlabeled_dataset = horoma_dataset.HoromaDataset(
        data_dir=args.data_dir,
        split="train",
        transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        # flattened=True, # set it true for 1D model
        # subset=10000,
    )

    # all_labeled_dataset = horoma_dataset.HoromaDataset(
    #     data_dir=args.data_dir,
    #     split="full_labeled",
    #     transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
    #     # flattened=True, # set it true for 1D model
    # )
    # splitter = horoma_dataset.SplitDataset(split=0.8)
    # train_labeled_dataset, valid_dataset = splitter(all_labeled_dataset)

    train_labeled_dataset = horoma_dataset.HoromaDataset(
        data_dir=args.data_dir,
        split="train_labeled_overlapped",
        # transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        transforms=HoromaTransforms(),
        # flattened=True, # set it true for 1D model
    )

    valid_dataset = horoma_dataset.HoromaDataset(
        data_dir=args.data_dir,
        split="valid",
        # transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.56121268, 0.20801756, 0.2602411], std=[0.22911494, 0.10410614, 0.11500103])]),
        transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        # flattened=True, # set it true for 1D model
    )


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

    # -------------- model params for 1d model --------------
    # input_channels = 1
    # target_labels = targets.split(",")
    # target_labels = [s.lower().strip() for s in target_labels]
    # if len(target_labels) == 1:
    #     out_size = target_out_size_dict[target_labels[0]]
    # else:
    #     out_size = [target_out_size_dict[a] for a in target_labels]
    # n_layers = 0  # hyper-parameter
    # hidden_size = 256  # hyper-parameter # 128
    # kernel_size = 8  # for CNN1D only
    # pool_size = 4  # for CNN1D only
    # dropout = 0.2
    # n_heads = 8  # 4
    # key_dim = 128
    # val_dim = 128
    # inner_dim = 128

    # -------------- model params for 2d model --------------
    input_channels = 3
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]
    if len(target_labels) == 1:
        out_size = target_out_size_dict[target_labels[0]]
    else:
        out_size = [target_out_size_dict[a] for a in target_labels]
    n_layers = 0  # hyper-parameter
    hidden_size = 256  # hyper-parameter # 128
    kernel_size = 2  # for CNN2D only
    pool_size = 2  # for CNN2D only
    dropout = args.dropout
    n_heads = 8  # 4
    key_dim = 128
    val_dim = 128
    inner_dim = 128

    # model = models.TransformerNet( # 1d model
    model = models.TransformerNet2D( # 2d model
        input_channels,
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
        use_transformer=(not args.no_transformer)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_iter = 0

    print(model)

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
