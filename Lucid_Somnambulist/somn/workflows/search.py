"""
Use nn search script as base for this. Need tforg and tfdriver classes defined in modeling section + imports here. 

Set up serialization directories for graceful start/stop using package path variables

Need to perform cleanup after all of the tf scratch work - that is not necessary to save. The oracle trials are way too heavy and not usable. 

Tensorboard integration might be nice - see some graphs of what's going on with the models' gradients. 
"""

from somn.learn.learning import hypermodel_search
from sys import argv
exp = argv[1]

assert isinstance(exp,str)

if __name__ == "__main__":
    ## Will do hypermodel search with specified experiment - this will create models from an existing partition
    ## Need to test how to ensure that the partition call and the search call operate on the same folder. 
    hypermodel_search(exp)
