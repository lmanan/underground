import os
import argparse

from underground.models import get_model
from underground.datasets import get_dataset


def train(args):
    print(args)
    
    # create train dataset


    # create val dataset


    # create model


    # create loss object


    train_epoch()

def train_epoch():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Config')
    parser.add_argument('--crop_size')
    args = parser.parse_args()
    print(args.crop_size)







