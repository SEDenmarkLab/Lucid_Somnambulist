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


def check_reactant_role(mols: list):
    """
    Try to intuit role of reactants based on their structures.

    This is under development, and is only to try and make structure input robust to user errors.

    """
    # ======================================================================
    # Check if N-H present in structure. If not, must be electrophile.
    # ======================================================================
    roles_round_1 = []
    N_bonds_list = []
    for mol in mols:
        N_atoms = [f for f in mol.atoms if f.symbol == "N"]
        if len(N_atoms) == 0:
            N_bonds_list.append(None)
            continue
        for n in N_atoms:
            N_bonds_list.append(
                [b.__return_other__(n).symbol for b in mol.bonds if b.__contains__(n)]
            )
    print(N_bonds_list)
    for nbrs in N_bonds_list:
        if nbrs == None:
            roles_round_1.append("el")
        else:
            if "H" in nbrs:
                roles_round_1.append("maybe_nuc")
    return roles_round_1, mols
