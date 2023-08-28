##Something
from sys import argv
from argparse import ArgumentParser

use_msg = """
Welcome to the somn command-line interface. 

To create new partitions, use:
partition [project ID, new or old project with no partitions] 

To train a new model set, use:
learn [project ID with partitions] [new ID for model set]

To make predictions from pre-trained models, use:
predict [project ID with pre-trained models] [model set ID of specific pre-trained models]  [new prediction ID]
"""


def main():
    print("DEV - module")
    parser = ArgumentParser(usage=use_msg)
