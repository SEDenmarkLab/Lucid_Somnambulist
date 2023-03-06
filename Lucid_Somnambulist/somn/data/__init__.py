import pandas as pd
import pickle
import pkg_resources
import molli as ml


# =============================
# If optionally imported, then this loads things into memory for use in calculation/descriptor
# assembly workflows
# =============================

global DATA
stream_data = pkg_resources.resource_filename(__name__, "dataset.csv")
DATA = pd.read_csv(stream_data, header=None, index_col=0)


def load_all_desc():
    """
    Load descriptors for all reaction components into memory.

    Use this for feature assembly.
    """
    global BASEDESC, SOLVDESC, CATDESC, AMINES, BROMIDES
    stream_base = pkg_resources.resource_filename(__name__, "base_params.csv")
    stream_solv = pkg_resources.resource_filename(__name__, "solvent_params.csv")
    stream_amine = pkg_resources.resource_filename(__name__, "amine_pickle_dict.p")
    stream_bromide = pkg_resources.resource_filename(__name__, "bromide_pickle_dict.p")
    BASEDESC = pd.read_csv(stream_base, header=None, index_col=0)
    SOLVDESC = pd.read_csv(stream_solv, header=None, index_col=0)
    with open(stream_amine, "rb") as g:
        AMINES = pickle.load(g)
    with open(stream_bromide, "rb") as k:
        BROMIDES = pickle.load(k)
    stream_cat = pkg_resources.resource_filename(__name__, "cat_aso_aeif_combined.csv")
    CATDESC = pd.read_csv(stream_cat, header=None, index_col=0)


def load_sub_mols():
    """
    Load substrate molecules from package - use this to calculate new substrate descriptors
    """
    global ACOL, BCOL
    stream_amol = pkg_resources.resource_filename(__name__, "amines_all_f.zip")
    stream_bmol = pkg_resources.resource_filename(__name__, "bromides_all_f.zip")
    ACOL = ml.Collection.from_zip(stream_amol)
    BCOL = ml.Collection.from_zip(stream_bmol)


def load_cat_desc(test=False, embedding=False):
    """
    Standalone function to load catalyst descriptors into memory - without the rest

    Optionally print to stdout, or open embedding (10 dimensional manifold projection) of catalyst features
    for visualization or modeling purposes.
    """
    if embedding:
        stream_cat = pkg_resources.resource_filename(
            __name__, "iso_catalyst_embedding.csv"
        )
        CATDESC = pd.read_csv(stream_cat, header=None, index_col=0)
    else:
        stream_cat = pkg_resources.resource_filename(
            __name__, "cat_aso_aeif_combined.csv"
        )
        CATDESC = pd.read_csv(stream_cat, header=None, index_col=0)
    if test:
        print(CATDESC)


###
### Debugging - this is to test global variables (which aren't directly accessible by interpreter, but this function call is)
## Should only work if all descriptors are loaded + substrate molecules.
def test():
    print(AMINES.keys(), BROMIDES.keys(), ACOL.name, len(BCOL.molecules), DATA.index)


### i.e. test looks like this:
### from somn.data import load_all_desc,test,load_sub_mols
### load_all_desc()
### load_sub_mols()
### test()
