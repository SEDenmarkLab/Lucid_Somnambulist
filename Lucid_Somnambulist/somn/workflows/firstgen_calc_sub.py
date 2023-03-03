import somn

from somn.build.parsing import cleanup_handles
from somn.calculate.reactant_firstgen import (
    retrieve_amine_rdf_descriptors,
    retrieve_bromide_rdf_descriptors,
)
from somn.calculate.preprocess import new_mask_random_feature_arrays
from somn.build.assemble import (
    assemble_descriptors_from_handles,
    assemble_random_descriptors_from_handles,
)

# ====================================================================
# Load in data shipped with package for manipulation. Optional import + function call
# ====================================================================

from somn import data
from somn.data import load_sub_mols

data.load_sub_mols()

# DEBUG: the global variables exist within the namespace of data, and can be intuitively loaded via:
# data.{global var}
# print(len(data.ACOL.molecules))
# SIMILAR for using universal read/write directories. This is from the workflows.
# from somn.workflows import UNIQUE_
# print(UNIQUE_)

############################ Calculate reactant descriptors #############################
