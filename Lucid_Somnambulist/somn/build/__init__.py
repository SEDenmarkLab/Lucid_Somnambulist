# This module uses molli-firstgen developed in the Denmark Laboratory primarily by Alex S. Shved, and its documentation/repository
# can be found here: https://github.com/SEDenmarkLab/molli_firstgen. N. Ian Rinehart, Reilly Brennan, and Blake Ocampo all contributed
# to the development of this first generation package.

# from ..build import parsing

# from parsing import InputParser


from .parsing import InputParser, DataHandler, cleanup_handles
from .assemble import assemble_descriptors_from_handles, preprocess_feature_arrays

# from somn.database import databasing, dtypes
