# Horoma

Third block of the Horoma project, dedicated to sem-supervised algorithms for classification.
The code repository is built upon the code repository of b2phot1 team from previsous block of this project.

The b2phot1 team from previous worked on mutiple models and we keep them here as it is.

```
usage: train.py [-h] [-c CONFIG | -r RESUME | --test-run] [-d DEVICE]
                [--helios-run HELIOS_RUN]

AutoEncoder Training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  --test-run            execute a test run on MNIST
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --helios-run HELIOS_RUN
                        if the train is run on helios with the run_experiment
                        script,the value should be the time at which the task
                        was submitted
```

```
usage: hyperparameter_search.py [-h] [-c CONFIG] [-r RESUME] [--test-run]
                                [-d DEVICE] [--helios-run HELIOS_RUN]

AutoEncoder Training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  --test-run            execute a test run on MNIST
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --helios-run HELIOS_RUN
                        if the train is run on helios with the run_experiment
                        script,the value should be the time at which the task
                        was submitted
```

## Running an experiment

In order to run an experiment, run the script `run_experiment.sh` while specifying the configuration file to use. This will create the PBS configuration script, initialise the result folder and launch the experiment.

`sh run_experiment.sh configs/model_config.json`

## New models added for block-3

For this block, we worked on four different models. We have merged the best performing model to the master and have kept the other 3 models on unmerged branches. Below are the details specific to each model - 

## ConvNet with mean-teacher regularizer

This model was built using code from b2pomt_baseline given by TAs for block-2 of OMSignal project.
For using this model use the details in [consistency_regularizer/README.md](consistency_regularizer/README.md)

## Pretrained ResNet as feature-extractor

Code for this model lies on git branch - `resnet`. We have left it unmerged as it is not our best performing model.
This model follow same usage as given above by b2phot1.
Sample config file - `resnet_clusters_kmeans.json` - can be found on the corresponding branch.

## Variational Deep Embedding (VADE) model

Code for this model lies on git branch - `vade`. We have left it unmerged as it is not our best performing model.
This model follow same usage as given above by b2phot1.
Sample config file - `vade_clusters_kmeans.json` - can be found on the corresponding branch.

## Ladder Network model

We also experimented with a sem-supervised ladder-network model. But we are not including the code for that in this repo. If requested we can hand over the code for that model in a seprarately.

## Other experiments

We experimented with two other models on `IIC` and `resnet-deepens` git branches. But we decided to leave them unreported.