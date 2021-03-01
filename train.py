import argparse
import yaml

from trainer import GlowTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--param_file", help="Path to parameter yaml file")

    args = parser.parse_args()

    param_file = args.param_file

    with open(param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = GlowTrainer()
    trainer = trainer.train(params)
