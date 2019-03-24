import subprocess
from template.generate_script import generate_script
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json')
    parser.add_argument('-cp', '--checkpoint', help='Path to checkpoint to load', type=str, default='None')
    parser.add_argument('-t', '--train', help='Specify if training is wanted', action='store_true')

    args = parser.parse_args()

    if args.train is None:
        assert args.checkpoint != 'None', 'You must specify a checkpoint file with -cp or train with -t'

    generate_script(args.config, args.checkpoint, args.train)

    subprocess.call(['msub', 'run.pbs'])
