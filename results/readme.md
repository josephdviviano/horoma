# Results

One folder for each experiment, containing the complete configuration and the best performing model.

When running an experiment, a folder is automatically created and the configuration file is copied there. This enables us to submit multiple experiments on Helios, without the need to wait until the experiment is running (otherwise the configuration file might change between the time the experiment is submitted and actually launched).

```
results/
    |-- experiment1/
    |       |-- config.json
    |       |-- best_model.pth
    |       |-- checkpoint.pth
    |
    |-- experiment2/
```
