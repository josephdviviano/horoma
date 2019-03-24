# Horoma

Third block of the Horoma project, dedicated to _deep clustering_ algorithms.

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
