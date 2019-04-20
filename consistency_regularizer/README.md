# ConvNet model with consistency regularization

All the files needed for running this model are lying flat in this folder.<br>
One thing to note is that most of the scripts here are taken from code for b2pomt_baseline given by TAs for block-2 of OMSignal project. Since OMSignal had multi-task problem so code is written in such a way that it can run for multi-task optimization. It would be usefule to keep this in mind when trying to understand the implementations.


## Usage

For training use the `main_CR.py` script.
Different commandline arguments for this script are given in usage below:

```
usage: main_CR.py [-h] [--batch-size N] [--eval-batch-size N] [--iters N]
                  [--lr LR] [--momentum M] [--alpha ALPHA] [--ema_decay EMA]
                  [--xi XI] [--eps EPS] [--cr_type CR] [--ip IP]
                  [--dropout DROPOUT] [--workers W] [--seed S]
                  [--targets TARGETS [TARGETS ...]] [--data_dir DATA_DIR]
                  [--checkpoint_dir CHECKPOINT_DIR]
                  [--tensorboard_dir TENSORBOARD_DIR] [--log-interval N]
                  [--chkpt-freq N] [--no-entropy] [--reg-vat-var REG_VAT_VAR]
                  [--no-transformer]

Horoma tree classification

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 635)
  --eval-batch-size N   input batch size for evaluation (default: 252)
  --iters N             number of iterations to train (default: 2000)
  --lr LR               learning rate (default: 0.0001)
  --momentum M          SGD momentum (default: 0.9)
  --alpha ALPHA         regularization coefficient (default: 0.005)
  --ema_decay EMA       decay for exponential moving average (default: 0.999)
  --xi XI               hyperparameter of VAT (default: 5.0)
  --eps EPS             hyperparameter of VAT (default: 1.0)
  --cr_type CR          Consistency Regularization (1:Stochastic Pertubation,
                        2:VAT, 3:MeanTeacher - default: 3)
  --ip IP               hyperparameter of VAT (default: 1)
  --dropout DROPOUT     hyperparameter of convnet (default: 0.5)
  --workers W           number of CPU
  --seed S              random seed (default: 1)
  --targets TARGETS [TARGETS ...]
                        list of targets to use for training
  --data_dir DATA_DIR   directory where to find the data
  --checkpoint_dir CHECKPOINT_DIR
                        directory where to checkpoints the models
  --tensorboard_dir TENSORBOARD_DIR
                        directory where to log tensorboard data
  --log-interval N      how many batches to wait before logging training
                        status
  --chkpt-freq N        how many batches to wait before performing
                        checkpointing
  --no-entropy          enables Entropy based regularization
  --reg-vat-var REG_VAT_VAR
                        Assumed variance of the predicted Gaussian for
                        regression tasks (default: 0.1)
  --no-transformer      If not to use transformer in prediction model
```

## Hyperparameter search

The details (results and settings) of hyper-parameter search are added in [hyper_param_search.md](hyper_param_search.md).
