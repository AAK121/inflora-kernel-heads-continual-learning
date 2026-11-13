import json
import argparse
from trainer import train
import ipdb

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    overrides = parse_overrides(args.set)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    args.update(overrides)  # --set wins over the json
    train(args)


def parse_overrides(items):
    """--set key=value ...; values are parsed as JSON when possible (10, 0.5, [0], true), else kept as strings."""
    out = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise ValueError('--set expects key=value, got {!r}'.format(item))
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override config entries, e.g. --set psi=rff psi_dim=128 seed=[0] data_path=/data/imagenet-r')

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == '__main__':
    main()
