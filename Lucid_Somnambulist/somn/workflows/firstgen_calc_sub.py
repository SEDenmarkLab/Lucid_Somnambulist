# import somn
# from copy import deepcopy
import pickle
from somn.calculate.reactant_firstgen import (
    retrieve_amine_rdf_descriptors,
    retrieve_bromide_rdf_descriptors,
)
from somn.calculate.preprocess import new_mask_random_feature_arrays
from somn.build.assemble import (
    assemble_descriptors_from_handles,
    assemble_random_descriptors_from_handles,
    make_randomized_features,
)

# ====================================================================
# Load in data shipped with package for manipulation. Optional import + function call
# ====================================================================
from somn import data
from somn.calculate import preprocess

data.load_sub_mols()
data.load_all_desc()

from somn.workflows import DESC_

# DEBUG: the global variables exist within the namespace of data, and can be intuitively loaded via:
# data.{global var}
# print(len(data.ACOL.molecules))
# SIMILAR for using universal read/write directories. This is from the workflows.
# from somn.workflows import UNIQUE_
# print(UNIQUE_)

############################ Calculate reactant descriptors #############################


def main(inc=0.75):
    """
    Run workflow to calculate real and random descriptors for substrates. Saves random features for ALL components,
    but only calculates substrate features. These are keyed feature sets, not assembled arrays.

    Can be called to return real desc (5 member tuple, am,br,cat,solv,base) and random desc (similar tuple)
    """
    (
        amines,
        bromides,
        dataset,
        handles,
        unique_couplings,
        a_prop,
        br_prop,
        base_desc,
        solv_desc,
        cat_desc,
    ) = preprocess.load_data()

    ### Calculate descriptors for the reactants, and store their 1D vector arrays in a dictionary-like output.
    _inc = inc
    sub_am_dict = retrieve_amine_rdf_descriptors(amines, a_prop, increment=_inc)
    sub_br_dict = retrieve_bromide_rdf_descriptors(bromides, br_prop, increment=_inc)

    rand = make_randomized_features(
        sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc
    )
    with open(DESC_ + "random_am_br_cat_solv_base.p", "wb") as k:
        pickle.dump(rand, k)
    with open(DESC_ + f"real_amine_desc_{_inc}.p", "wb") as g:
        pickle.dump(sub_am_dict, g)
    with open(DESC_ + f"real_bromide_desc_{_inc}.p", "wb") as q:
        pickle.dump(sub_br_dict, q)
    return ((sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc), rand)


if __name__ == "__main__":
    (
        (sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc),
        rand,
    ) = main()
    # print(sub_am_dict)
