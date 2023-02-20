import molli as ml
import warnings

###         This is for active development of core functionality, classes used for debugging/checks.


def check_parsed_mols(mols: list, col: ml.Collection):
    """
    Check input molecules, return error stream if there is a problem

    Accepts list of structures, returns two lists if there are errors
    """
    mols_ = [f for f in mols if isinstance(f, ml.Molecule)]
    errs_ = [
        col.molecules[i] for i, f in enumerate(mols) if not isinstance(f, ml.Molecule)
    ]  # Get mols from input to whatever failed operation
    if len(errs_) > 0:
        warnings.warn(
            f"There are molecules which did not parse correctly: {','.join([f.name for f in errs_])}\n \
            These are automatically-generated names, but can be serialized for troubleshooting.",
            category=UserWarning,
            stacklevel=1,
        )
    return mols_, errs_
