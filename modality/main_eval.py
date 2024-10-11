from __future__ import print_function
import argparse
import yaml

from logger import Logger
from trainers.selfsupervised import selfsupervised

if __name__ == "__main__":

    # Load the config file
    parser = argparse.ArgumentParser(description="Sensor fusion model")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--notes", default="", help="run notes")
    parser.add_argument("--dev", type=bool, default=False, help="run in dev mode")
    parser.add_argument(
        "--continuation",
        type=bool,
        default=False,
        help="continue a previous run. Will continue the log file",
    )
    args = parser.parse_args()

    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)

    # Initialize the loggers
    logger = Logger(configs)

    # Initialize the trainer
    trainer = selfsupervised(configs, logger)

    just_eval = True

    if not just_eval:
        trainer.train()

    # TODO: specify configs["load"] -> to load already trained model

    # eval single demo file or multiple
    # todo check good demo for report etc.
    demo_path = "/home/rickmer/Documents/Vision_Touch_pjahr/pearl_eval_data/0.h5"
    trainer.eval_demo(demo_path)