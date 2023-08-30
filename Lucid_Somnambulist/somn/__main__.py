##Something
from sys import argv
from argparse import ArgumentParser

use_msg = """
Welcome to the somn command-line interface. 

[NORMAL USE] To make predictions from pre-trained models, use:
predict [project ID with pre-trained models] [model set ID of specific pre-trained models]  [new prediction ID]

[POWER USER USE] To retrain models, two steps are required: generating a particular set of partitions, then optimizing hyperparameters
partition-wise. These are separated so that custom preprocessing or features can be incorporated at the partition step,
and custom modeling changes can be incorporated or tested at the modeling step. New data for training should be 
incorporated into the dataset_yields.hdf5 file for inclusion in the retraining process.

To create new partitions, use:
partition [project ID, new or old project with no partitions] 

To train a new model set on partitions, use:
learn [project ID with partitions] [new ID for model set]
"""


def main():
    print("DEV - module")
    parser = ArgumentParser(usage=use_msg)
    parser.add_argument(nargs="?",
                        choices=['predict','partition','learn','add','calculate','visualize','help'],
                        dest="operation",
                        default='help'
                        )
    parser.add_argument(dest="options",
                        nargs='*',
                        default=False,
                        )
    args = parser.parse_args()
    print(args)

