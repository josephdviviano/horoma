from datetime import datetime

import json
import argparse

import os
import sys


def check_priority():
    """
    Checks whether we have priority on Helios (the PBS options need to change).

    Returns:
        bool: True if we have priority
    """

    now = datetime.now()

    day = now.weekday()
    hour = now.hour

    if day == 2:
        if 15 <= hour <= 23:
            return True

    if day == 5:
        if 14 <= hour <= 22:
            return True

    return False


def generate_pbs_file(conf, log_dir, start_time):
    """
    Generate a run.pbs file, that will be run by Moab on Helios.

    Uses pbs.template file as template.

    Args:
        conf (dict): the configuration to use
    """

    wall_time = conf['wall_time']
    n_gpu = conf['n_gpu']
    files_name = "{}_{}".format(conf["name"], start_time)
    priority = check_priority()

    with open('template/pbs.template', 'r') as file:
        template = file.read()

    specific = '' if not priority else '#PBS -l advres=MILA2019\n#PBS -l feature=k80'

    pbs = template.format(
        wall_time=int(wall_time * 3600),
        n_gpu=n_gpu,
        files_name=files_name,
        prioritization=specific,
        result_dir=log_dir,
        start_time=start_time,
        PBS_O_WORKDIR='{PBS_O_WORKDIR}',
        MOAB_JOBID='{MOAB_JOBID}'
    )

    with open('run.pbs', 'w') as file:
        file.write(pbs)


def initialize_experiment_folder(configuration):
    """
    Create a result folder, and initialises it with the configuration file.

    Args:
        configuration (dict): Configuration for the experiment.

    Raises:
        FileExistsError: If the directory already exists (risk overwriting previous experiments).
    """

    name = configuration['name']
    log_dir = configuration['trainer']['log_dir']
    start_time = datetime.now().strftime('%m%d_%H%M%S')

    path = os.path.join(log_dir, name, start_time)
    print(path)
    os.makedirs(path)

    with open(os.path.join(path, 'config.json'), 'w') as file:
        json.dump(configuration, file, indent=2, separators=(',', ': '))

    return path, start_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    try:
        log_dir, start_time = initialize_experiment_folder(config)
    except OSError as e:
        # If the directory already exist, print an error message and kill the msub call.
        print(
            '>> Warning: a directory name "{name}" already exists. Stopping the script.'.format(
                name=config['name']))
        print(e)
        sys.exit(1)

    generate_pbs_file(config, log_dir, start_time)
