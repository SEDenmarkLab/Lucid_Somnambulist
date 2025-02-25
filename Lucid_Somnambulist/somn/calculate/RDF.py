from math import sqrt

import molli as ml
import numpy as np
import pandas as pd
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import rdqueries
import warnings

# William M Haynes. CRC Handbook of Chemistry and Physics.
# CRC Press, London, 95th edition, 2014. ISBN 9781482208689.
vdw_dict: dict[str, float] = {
    "H": 1.1,
    "He": 1.4,
    "Li": 1.82,
    "Be": 1.53,
    "B": 1.92,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.1,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Ca": 2.31,
    "Sc": 2.15,
    "Ti": 2.11,
    "V": 2.07,
    "Cr": 2.06,
    "Mn": 2.05,
    "Fe": 2.04,
    "Co": 2.0,
    "Ni": 1.97,
    "Cu": 1.96,
    "Zn": 2.01,
    "Ga": 1.87,
    "Ge": 2.11,
    "As": 1.85,
    "Se": 1.9,
    "Br": 1.85,
    "Kr": 2.02,
    "Rb": 3.03,
    "Sr": 2.49,
    "Y": 2.32,
    "Zr": 2.23,
    "Nb": 2.18,
    "Mo": 2.17,
    "Tc": 2.16,
    "Ru": 2.13,
    "Rh": 2.1,
    "Pd": 2.1,
    "Ag": 2.11,
    "Cd": 2.18,
    "In": 1.93,
    "Sn": 2.17,
    "Sb": 2.06,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Cs": 3.43,
    "Ba": 2.68,
    "La": 2.43,
    "Ce": 2.42,
    "Pr": 2.4,
    "Nd": 2.39,
    "Pm": 2.38,
    "Sm": 2.36,
    "Eu": 2.35,
    "Gd": 2.34,
    "Tb": 2.33,
    "Dy": 2.31,
    "Ho": 2.3,
    "Er": 2.29,
    "Tm": 2.27,
    "Yb": 2.26,
    "Lu": 2.24,
    "Hf": 2.23,
    "Ta": 2.22,
    "W": 2.18,
    "Re": 2.16,
    "Os": 2.16,
    "Ir": 2.13,
    "Pt": 2.13,
    "Au": 2.14,
    "Hg": 2.23,
    "Tl": 1.96,
    "Pb": 2.02,
    "Bi": 2.07,
    "Po": 1.97,
    "At": 2.02,
    "Rn": 2.2,
    "Fr": 3.48,
    "Ra": 2.83,
    "Ac": 2.47,
    "Th": 2.45,
    "Pa": 2.43,
    "U": 2.41,
    "Np": 2.39,
    "Pu": 2.43,
    "Am": 2.44,
    "Cm": 2.45,
    "Bk": 2.44,
    "Cf": 2.45,
    "Es": 2.45,
    "Fm": 2.45,
    "Md": 2.46,
    "No": 2.46,
    "Lr": 2.46,
}
"""Atomic numbers as keys and CRC vdW radii as values."""


symbols_for_rdf = ["C", "N", "S", "O", "F"]


def select_ref_atoms(chlorides, bromides):
    """
    Choose reference atom from lists of bromides and chlorides in molecule.
    Br and Cl, choose Br
    Multiple Brs and Cls, pick one Br
    Cl only, choose cl
    Neither present, raise exception
    """
    if len(chlorides) == 0 and len(bromides) == 0:
        raise Exception(
            "Electrophiles passed which do not have Br or Cl present. \
Check inputs to ensure that (hetero)aryl bromides and chlorides are being requested."
        )
    elif len(chlorides) == 0 and len(bromides) > 0:
        if len(bromides) > 1:
            warnings.warn("Multiple bromides identified, selecting one arbitrarily.")
        return bromides[0]
    elif len(chlorides) > 0 and len(bromides) == 0:
        if len(chlorides) > 1:
            warnings.warn("Multiple chlorides identified, selecting one arbitrarily.")
        return chlorides[0]
    elif len(chlorides) > 0 and len(bromides) > 0:
        warnings.warn(
            "Multiple halides identified, selecting a bromine atom arbitrarily."
        )
        return bromides[0]


def calculate_electrophile_rdf_descriptors(
    col: ml.Collection = None,
    apd: dict = None,
    increment: float = 0.75,
    radial_scale: int = 0,
    slices: int = 10,
    ref_atoms=None,
):
    """
    Generalized RDF descriptor calculation - handles Cl or Br.

    ref_atoms should be a list that maps onto the collection with either a molli Atom class (for the Br/Cl) or a '-' symbol to autodetect
    """
    assert col is not None
    assert apd is not None
    failures = []
    molecule_rdfs = {}

    def automatically_identify_ref_atoms():
        reference = []
        for mol in col:
            chlorides = mol.get_atoms_by_symbol(symbol="Cl")
            bromides = mol.get_atoms_by_symbol(symbol="Br")
            ref_atm = select_ref_atoms(chlorides, bromides)
            reference.append(ref_atm)
        return reference

    if ref_atoms is not None:
        try:
            assert isinstance(ref_atoms, list)
            assert all(isinstance(f, ml.dtypes.Atom) or f == "-" for f in ref_atoms)
            reference = []
            for i, k in enumerate(ref_atoms):
                if k == "-":
                    mol = col.molecules[i]
                    ref_atm = select_ref_atoms(
                        mol.get_atoms_by_symbol(symbol="Cl"),
                        mol.get_atoms_by_symbol(symbol="Br"),
                    )
                    reference.append(ref_atm)
                elif isinstance(k, ml.dtypes.Atom):
                    reference.append(k)
                else:
                    ValueError()

        except:
            warnings.warn(
                "Calculating RDF descriptors failed because a ref_atom \
argument was passed which was neither an atom class nor an atom index. None is a valid case, \
and will result in a guess of the reference halide atom (Br > Cl)."
            )
    elif ref_atoms is None:
        reference = automatically_identify_ref_atoms()
    else:
        raise ValueError(
            "Passed an improper input for ref_atoms during new substrate descriptor calculation."
        )
    assert len(col.molecules) == len(reference)
    for mol, ref in zip(col, reference):
        conn = list(mol.get_connected_atoms(ref))
        try:
            assert len(conn) == 1
        except:
            if mol.name not in failures:
                failures.append(mol.name)
            continue
        ipso_atom = conn[0]
        ipso_idx = mol.atoms.index(ipso_atom)
        halide_idx = mol.atoms.index(ref)
        rdk_mol = Chem.MolFromMol2Block(mol.to_mol2(), sanitize=False)
        if rdk_mol == None:
            obconv = openbabel.OBConversion()
            obconv.SetInAndOutFormats("mol2", "smi")
            obmol = openbabel.OBMol()
            with open("buffer.mol2", "w") as g:
                g.write(mol.to_mol2())
            obconv.ReadFile(obmol, "buffer.mol2")
            obconv.Convert()
            smi = obconv.WriteString(obmol).split()[0]
            if "([N](=O)[O-])" in smi:
                smi = smi.replace("([N](=O)[O-])", "([N+](=O)[O-])")
            rdk_mol = Chem.MolFromSmiles(smi)
        leftref = get_left_reference(rdk_mol, ipso_idx, halide_idx)
        conf_rdfs = {}
        for k, conf in enumerate(mol.conformers):
            df = pd.DataFrame.from_dict(apd[mol.name][k], orient="columns")
            coords = conf.coord
            a, b, c, d = get_molplane(coords, halide_idx, ipso_idx, leftref)
            orth_out = get_orthogonal_plane(
                coords, halide_idx, ipso_idx, a, b, c, leftref
            )
            if orth_out == None:
                raise Exception(
                    f"Cannot find orthogonal plane direction for molecule {mol.name}, molecule number {col.molecules.index(mol)} in collection"
                )
            e, f, g, h = orth_out
            h1, h2 = sort_into_halves(mol, conf, e, f, g, h)
            for prop in df.index:
                rdf_ser_1 = get_rdf(
                    coords,
                    halide_idx,
                    h1,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                rdf_ser_2 = get_rdf(
                    coords,
                    halide_idx,
                    h2,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                if prop in conf_rdfs.keys():
                    conf_rdfs[prop].append([list(rdf_ser_1), list(rdf_ser_2)])
                else:
                    conf_rdfs[prop] = [[list(rdf_ser_1), list(rdf_ser_2)]]
            rdf_ser_3 = get_atom_ind_rdf(
                mol.atoms, coords, halide_idx, h1, inc_size=increment, first_int=1.80
            )
            rdf_ser_4 = get_atom_ind_rdf(
                mol.atoms, coords, halide_idx, h2, inc_size=increment, first_int=1.80
            )
        for sym, _3, _4 in zip(symbols_for_rdf, rdf_ser_3, rdf_ser_4):
            conf_rdfs[sym + "_rdf"] = [[_3, _4]]
        desc_df = pd.DataFrame()
        for prop, values in conf_rdfs.items():
            array_ = np.array(values)
            avg_array = np.mean(array_, axis=0)
            desc_df[prop] = pd.concat([pd.Series(f) for f in avg_array], axis=0)
        desc_df.index = ["slice_" + str(f + 1) for f in range(20)]
        molecule_rdfs[mol.name] = desc_df
    return molecule_rdfs


def retrieve_chloride_rdf_descriptors(
    col, apd, increment: float = 1.5, radial_scale: int = 0
):
    """
    Takes collection and json-type atom property descriptors (generated by scraping function built for xTB outputs)

    outputs dataframe for each molecule with conformer-averaged descriptor columns and spherical slices for indices

    These are put into a dictionary with molecule names from the collection as keys.

    Shape of df is 20 rows (two 10 sphere slices for each half of the molecule) with 14 columns of electronic and indicator
    RDFs

    This should be applicable to nitrogen nucleophiles with the exception that one atom list with all atoms should be passed.
    This would give an output with 10 extra rows that could be trimmed or just removed later with variance threshold.

    """
    mol_rdfs = {}  # Going to store dfs in here with name for retrieval for now
    for mol in col:
        labels = [f.symbol for f in mol.atoms]
        cl_atom = mol.get_atoms_by_symbol(symbol="Cl")[0]
        cl_idx = mol.atoms.index(cl_atom)
        conn = mol.get_connected_atoms(cl_atom)
        if len(conn) != 1:
            raise Exception(
                "More than one group found bonded to cl atom. Check structures"
            )
        elif len(conn) == 1:
            ipso_atom = list(conn)[0]
        else:
            print("foundglitch")
        ipso_idx = mol.atoms.index(ipso_atom)
        rdk_mol = Chem.MolFromMol2Block(mol.to_mol2(), sanitize=False)
        if rdk_mol == None:
            obconv = openbabel.OBConversion()
            obconv.SetInAndOutFormats("mol2", "smi")
            obmol = openbabel.OBMol()
            with open("buffer.mol2", "w") as g:
                g.write(mol.to_mol2())
            obconv.ReadFile(obmol, "buffer.mol2")
            obconv.Convert()
            smi = obconv.WriteString(obmol).split()[0]
            if "([N](=O)[O-])" in smi:
                smi = smi.replace("([N](=O)[O-])", "([N+](=O)[O-])")
            rdk_mol = Chem.MolFromSmiles(smi)
        leftref = get_left_reference(rdk_mol, ipso_idx, cl_idx)
        conf_rdfs = {}
        for k, conf in enumerate(mol.conformers):
            df = pd.DataFrame.from_dict(apd[mol.name][k], orient="columns")
            coords = conf.coord
            a, b, c, d = get_molplane(coords, cl_idx, ipso_idx, leftref)
            orth_out = get_orthogonal_plane(coords, cl_idx, ipso_idx, a, b, c, leftref)
            if orth_out == None:
                raise Exception(
                    f"Cannot find orthogonal plane direction for molecule {mol.name}, molecule number {col.molecules.index(mol)} in collection"
                )
            e, f, g, h = orth_out
            h1, h2 = sort_into_halves(mol, conf, e, f, g, h)
            for prop in df.index:
                rdf_ser_1 = get_rdf(
                    coords,
                    cl_idx,
                    h1,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                rdf_ser_2 = get_rdf(
                    coords,
                    cl_idx,
                    h2,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                if prop in conf_rdfs.keys():
                    conf_rdfs[prop].append([list(rdf_ser_1), list(rdf_ser_2)])
                else:
                    conf_rdfs[prop] = [[list(rdf_ser_1), list(rdf_ser_2)]]
            rdf_ser_3 = get_atom_ind_rdf(
                mol.atoms, coords, cl_idx, h1, inc_size=increment, first_int=1.80
            )
            rdf_ser_4 = get_atom_ind_rdf(
                mol.atoms, coords, cl_idx, h2, inc_size=increment, first_int=1.80
            )
        for sym, _3, _4 in zip(symbols_for_rdf, rdf_ser_3, rdf_ser_4):
            conf_rdfs[sym + "_rdf"] = [[_3, _4]]
        desc_df = pd.DataFrame()
        for prop, values in conf_rdfs.items():
            array_ = np.array(values)
            avg_array = np.mean(array_, axis=0)
            desc_df[prop] = pd.concat([pd.Series(f) for f in avg_array], axis=0)
        desc_df.index = ["slice_" + str(f + 1) for f in range(20)]
        mol_rdfs[mol.name] = desc_df
    return mol_rdfs


def retrieve_bromide_rdf_descriptors(
    col, apd, increment: float = 1.5, radial_scale: int = 0
):
    """
    Takes collection and json-type atom property descriptors (generated by scraping function built for xTB outputs)

    outputs dataframe for each molecule with conformer-averaged descriptor columns and spherical slices for indices

    These are put into a dictionary with molecule names from the collection as keys.

    Shape of df is 20 rows (two 10 sphere slices for each half of the molecule) with 14 columns of electronic and indicator
    RDFs

    This should be applicable to nitrogen nucleophiles with the exception that one atom list with all atoms should be passed.
    This would give an output with 10 extra rows that could be trimmed or just removed later with variance threshold.

    """
    mol_rdfs = {}  # Going to store dfs in here with name for retrieval for now
    for mol in col:
        rdf_df = pd.DataFrame(index=["sphere_" + str(i) for i in range(10)])
        rdf_df.name = mol.name
        labels = [f.symbol for f in mol.atoms]
        try:
            br_atom = mol.get_atoms_by_symbol(symbol="Br")[0]
        except:
            raise Exception(
                f"Looks like Br RDF was called on a non-bromide, structure {mol.name}"
            )
        br_idx = mol.atoms.index(br_atom)
        conn = mol.get_connected_atoms(br_atom)
        if len(conn) != 1:
            raise Exception(
                "More than one group found bonded to Br atom. Check structures"
            )
        elif len(conn) == 1:
            ipso_atom = list(conn)[0]
        else:
            print("foundglitch")
        ipso_idx = mol.atoms.index(ipso_atom)
        rdk_mol = Chem.MolFromMol2Block(mol.to_mol2(), sanitize=False)
        if rdk_mol == None:
            obconv = openbabel.OBConversion()
            obconv.SetInAndOutFormats("mol2", "smi")
            obmol = openbabel.OBMol()
            with open("buffer.mol2", "w") as g:
                g.write(mol.to_mol2())
            obconv.ReadFile(obmol, "buffer.mol2")
            obconv.Convert()
            smi = obconv.WriteString(obmol).split()[0]
            if "([N](=O)[O-])" in smi:
                smi = smi.replace("([N](=O)[O-])", "([N+](=O)[O-])")
            rdk_mol = Chem.MolFromSmiles(smi)
        leftref = get_left_reference(rdk_mol, ipso_idx, br_idx)
        conf_rdfs = {}
        for k, conf in enumerate(mol.conformers):
            df = pd.DataFrame.from_dict(apd[mol.name][k], orient="columns")
            coords = conf.coord
            a, b, c, d = get_molplane(coords, br_idx, ipso_idx, leftref)
            e, f, g, h = get_orthogonal_plane(
                coords, br_idx, ipso_idx, a, b, c, leftref
            )
            h1, h2 = sort_into_halves(mol, conf, e, f, g, h)
            for prop in df.index:
                rdf_ser_1 = get_rdf(
                    coords,
                    br_idx,
                    h1,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                rdf_ser_2 = get_rdf(
                    coords,
                    br_idx,
                    h2,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.80,
                )
                if prop in conf_rdfs.keys():
                    conf_rdfs[prop].append([list(rdf_ser_1), list(rdf_ser_2)])
                else:
                    conf_rdfs[prop] = [[list(rdf_ser_1), list(rdf_ser_2)]]
            rdf_ser_3 = get_atom_ind_rdf(
                mol.atoms, coords, br_idx, h1, inc_size=increment, first_int=1.80
            )
            rdf_ser_4 = get_atom_ind_rdf(
                mol.atoms, coords, br_idx, h2, inc_size=increment, first_int=1.80
            )
        for sym, _3, _4 in zip(symbols_for_rdf, rdf_ser_3, rdf_ser_4):
            conf_rdfs[sym + "_rdf"] = [[_3, _4]]
        desc_df = pd.DataFrame()
        for prop, values in conf_rdfs.items():
            array_ = np.array(values)
            avg_array = np.mean(array_, axis=0)
            desc_df[prop] = pd.concat([pd.Series(f) for f in avg_array], axis=0)
        desc_df.index = ["slice_" + str(f + 1) for f in range(20)]
        mol_rdfs[mol.name] = desc_df
    return mol_rdfs


def get_amine_ref_n(mol: ml.Molecule):
    """
    Returns the reference atom index for the nitrogen with an H (assumes only one)
    """
    nit_atm = False
    for atm in mol.get_atoms_by_symbol(symbol="N"):
        nbrs = mol.get_connected_atoms(atm)
        for nbr in nbrs:
            if nbr.symbol == "H":
                nit_atm = atm
                return nit_atm


def retrieve_amine_rdf_descriptors(
    col, apd, increment: float = 1.1, radial_scale: int = 0, ref_atoms=None
):
    """
    Takes collection and json-type atom property descriptors (generated by scraping function built for xTB outputs)

    outputs dataframe for each molecule with conformer-averaged descriptor columns and spherical slices for indices

    These are put into a dictionary with molecule names from the collection as keys.

    Shape of df is 20 rows (two 10 sphere slices for each half of the molecule) with 14 columns of electronic and indicator
    RDFs

    This should be applicable to nitrogen nucleophiles with the exception that one atom list with all atoms should be passed.
    This would give an output with 10 extra rows that could be trimmed or just removed later with variance threshold.

    """
    molecule_rdfs = {}  # Going to store dfs in here with name for retrieval for now
    if ref_atoms is not None:
        try:
            assert isinstance(ref_atoms, list)
            assert all(isinstance(f, ml.dtypes.Atom) or f == "-" for f in ref_atoms)
            reference = []
            for i, k in enumerate(ref_atoms):
                if k == "-":
                    reference.append(get_amine_ref_n(col.molecules[i]))
                elif isinstance(k, ml.dtypes.Atom):
                    reference.append(k)
                else:
                    ValueError()
        except:
            warnings.warn(
                "Calculating RDF descriptors failed because a ref_atom \
argument was passed which was neither an atom class nor an atom index. None is a valid case, \
and will result in a guess of the reference halide atom (Br > Cl)."
            )
    elif ref_atoms is None:
        reference = []
        for mol in col:
            ref_ = get_amine_ref_n(mol)
            reference.append(ref_)
    else:
        warnings.warn(
            "Passed an improper input for ref_atoms during new substrate descriptor calculation."
        )
        reference = []
        for mol in col:
            ref_ = get_amine_ref_n(mol)
            reference.append(ref_)
    assert len(col.molecules) == len(reference)
    for mol, ref in zip(col, reference):
        rdf_df = pd.DataFrame(index=["sphere_" + str(i) for i in range(10)])
        rdf_df.name = mol.name
        n_idx = mol.atoms.index(ref)
        assert ref.symbol == "N"
        conf_rdfs = {}
        a_idx_l = [mol.atoms.index(f) for f in mol.atoms]
        for k, conf in enumerate(mol.conformers):
            df = pd.DataFrame.from_dict(apd[mol.name][k], orient="columns")
            coords = conf.coord
            for prop in df.index:
                rdf_ser_1 = get_rdf(
                    coords,
                    n_idx,
                    a_idx_l,
                    df.loc[prop],
                    radial_scaling=radial_scale,
                    inc_size=increment,
                    first_int=1.20,
                )
                if prop in conf_rdfs.keys():
                    conf_rdfs[prop].append([list(rdf_ser_1)])
                else:
                    conf_rdfs[prop] = [[list(rdf_ser_1)]]
            rdf_ser_3 = get_atom_ind_rdf(
                mol.atoms, coords, n_idx, a_idx_l, inc_size=increment, first_int=1.20
            )
        for sym, _3 in zip(symbols_for_rdf, rdf_ser_3):
            conf_rdfs[sym + "_rdf"] = [[_3]]
        desc_df = pd.DataFrame()
        for prop, values in conf_rdfs.items():
            array_ = np.array(values)
            avg_array = np.mean(array_, axis=0)
            desc_df[prop] = pd.concat([pd.Series(f) for f in avg_array], axis=0)
        desc_df.index = ["slice_" + str(f + 1) for f in range(10)]
        molecule_rdfs[mol.name] = desc_df
    return molecule_rdfs


def get_rdf(
    coords: ml.dtypes.CartesianGeometry,
    reference_idx: int,
    atom_list,
    all_atoms_property_list: list,
    inc_size=0.90,
    first_int: float = 1.80,
    radial_scaling: int or None = 0,
):
    """
    Takes coordinates for molecule, reference atom index, list of atom indices to compute for, and property list ordered by atom idx

    radial_scaling is an exponent for 1/(r^n) scaling the descriptors - whatever they may be

    """
    al = []
    bl = []
    cl = []
    dl = []
    el = []
    fl = []
    gl = []
    hl = []
    il = []
    jl = []
    central_atom = coords[reference_idx]
    for x in atom_list:
        point = coords[x]
        dist = sqrt(
            (
                (float(central_atom[0]) - float(point[0])) ** 2
                + (float(central_atom[1]) - float(point[1])) ** 2
                + (float(central_atom[2]) - float(point[2])) ** 2
            )
        )
        property = list(all_atoms_property_list)[x]
        try:
            property_ = float(property)
        except:
            property_ = 4.1888 * vdw_dict[property] ** 3
        const = first_int
        if radial_scaling == 0 or radial_scaling == None:
            pass
        elif type(radial_scaling) is int and radial_scaling != 0:
            property_ = property_ / (dist**radial_scaling)
        else:
            raise ValueError("radial scaling exponent should be an integer or None")
        if dist <= const + inc_size:
            al.append(property_)
        elif dist > const + inc_size and dist <= const + inc_size * 2:
            bl.append(property_)
        elif dist > const + inc_size * 2 and dist <= const + inc_size * 3:
            cl.append(property_)
        elif dist > const + inc_size * 3 and dist <= const + inc_size * 4:
            dl.append(property_)
        elif dist > const + inc_size * 4 and dist <= const + inc_size * 5:
            el.append(property_)
        elif dist > const + inc_size * 5 and dist <= const + inc_size * 6:
            fl.append(property_)
        elif dist > const + inc_size * 6 and dist <= const + inc_size * 7:
            gl.append(property_)
        elif dist > const + inc_size * 7 and dist <= const + inc_size * 8:
            hl.append(property_)
        elif dist > const + inc_size * 8 and dist <= const + inc_size * 9:
            il.append(property_)
        elif dist > const + inc_size * 9:
            jl.append(property_)
    series_ = pd.Series(
        [
            sum(al),
            sum(bl),
            sum(cl),
            sum(dl),
            sum(el),
            sum(fl),
            sum(gl),
            sum(hl),
            sum(il),
            sum(jl),
        ],
        index=["sphere_" + str(f + 1) for f in range(10)],
    )
    """
    print al
    print bl
    print cl
    print dl
    print el
    print fl
    print gl
    print hl
    print il
    print jl
    """
    return series_


def get_atom_ind_rdf(
    atoms: list[ml.dtypes.Atom],
    coords: ml.dtypes.CartesianGeometry,
    reference_idx: int,
    atom_list,
    first_int: float = 1.80,
    inc_size=0.90,
):
    """
    Takes atoms and returns simple binary indicator for presence of specific atom types. These are not distance-weighted.
    """
    atomtypes = ["C", "N", "S", "O", "F"]
    outlist = []
    for symb in atomtypes:
        al = []
        bl = []
        cl = []
        dl = []
        el = []
        fl = []
        gl = []
        hl = []
        il = []
        jl = []
        central_atom = coords[reference_idx]
        for x in atom_list:
            point = coords[x]
            symbol = atoms[x].symbol
            if symbol != symb:
                continue
            dist = sqrt(
                (
                    (float(central_atom[0]) - float(point[0])) ** 2
                    + (float(central_atom[1]) - float(point[1])) ** 2
                    + (float(central_atom[2]) - float(point[2])) ** 2
                )
            )
            const = first_int
            if dist <= const + inc_size:
                al.append(1)
            elif dist > const + inc_size and dist <= const + inc_size * 2:
                bl.append(1)
            elif dist > const + inc_size * 2 and dist <= const + inc_size * 3:
                cl.append(1)
            elif dist > const + inc_size * 3 and dist <= const + inc_size * 4:
                dl.append(1)
            elif dist > const + inc_size * 4 and dist <= const + inc_size * 5:
                el.append(1)
            elif dist > const + inc_size * 5 and dist <= const + inc_size * 6:
                fl.append(1)
            elif dist > const + inc_size * 6 and dist <= const + inc_size * 7:
                gl.append(1)
            elif dist > const + inc_size * 7 and dist <= const + inc_size * 8:
                hl.append(1)
            elif dist > const + inc_size * 8 and dist <= const + inc_size * 9:
                il.append(1)
            elif dist > const + inc_size * 9:
                jl.append(1)
        series_ = [
            sum(al),
            sum(bl),
            sum(cl),
            sum(dl),
            sum(el),
            sum(fl),
            sum(gl),
            sum(hl),
            sum(il),
            sum(jl),
        ]
        outlist.append(series_)
    output = outlist
    return output


def get_molplane(coords: np.array, ref_1, ref_2, ref_3):
    """
    Makes plane of molecule. Bromide for bromides, nitrogen for amines as ref atom.

    Br, ipso, leftref
    """
    # Setting up reference points
    p1 = np.array(coords[ref_1])
    p2 = np.array(coords[ref_2])
    p3 = np.array(coords[ref_3])
    # Setting up reference vectors - Br-C bond, leftref - br
    v1 = p2 - p1
    v2 = p3 - p1
    # Calculating plane's vector to get direction
    cp = np.cross(v1, v2)
    a, b, c = cp
    # Calculating position for plane
    d = np.dot(cp, p1)
    return a, b, c, d

def get_orthogonal_plane(coords: np.array, ref_1, ref_2, a, b, c, leftref):
    """
    DEV VERSION - MAKING COMPATIBLE WITH CL CALCULATIONS
    Retrieve orthogonal plane to molecule, but containing reactive atom
    ref1 is the reactive atom (br or n)
    ref2 is the atom attached to it (for making a direction towards the molecule)
    """
    p1 = np.array(coords[ref_1])  # Halogen
    p2 = np.array(coords[ref_2])  # Ipso atom
    p4 = np.array(coords[leftref])  # for "positive" direction left/right
    v1 = p2 - p1  # ipso to halogen
    v2 = np.array([a, b, c])  # vector for molecular plane
    cp = np.cross(v1, v2)
    e, f, g = cp
    vc = np.array([e, f, g])  # vector for orthogonal plane
    if np.dot(vc, p4) > 0:
        h = np.dot(vc, p1)
        return e, f, g, h
    elif np.dot(vc, p4) < 0:
        cp = np.cross(v2, v1)
        e, f, g = cp
        vc = np.array([e, f, g])
        h = np.dot(vc, p1)
        return e, f, g, h
    else:
        return None


def sort_into_halves(mol: ml.Molecule, conf: ml.dtypes.CartesianGeometry, e, f, g, h):
    """
    This function takes in the atom list and spits out a list of lists with atoms sorted
    into octants. This is done with the three orthonormal planes defined by get_orthogonal_planes
    """
    coords: np.array = conf.coord
    oct1 = []
    oct2 = []
    cp = np.array([e, f, g])
    for i, pos in enumerate(coords):
        direction_ = (np.tensordot(pos, cp, axes=1) - h) / abs(sqrt(e**2 + f**2 + g**2))
        if direction_ > 0.15:
            oct1.append(i)
        elif direction_ < -0.15:
            oct2.append(i)
    return [oct1, oct2]


def select_left_reference(mol: ml.Molecule, ipso_atom, halide_atom):
    """
    select left reference for molecule

    [new version - no RDKit dependency]
    """
    ortho_atoms, meta_atoms, tertiary_atoms = get_ortho_meta_symbols(
        mol, ipso_atom, halide_atom
    )


def evaluate_atom_heirarchy(
    mol: ml.Molecule, halide: ml.dtypes.Atom, ortho: list, meta: list
):
    """
    Pass an ortho atom list or meta atom list
    """
    from mendeleev import element

    o_elms = [element(f.symbol) for f in ortho]
    o_aos = [f.atomic_number for f in o_elms]
    if o_aos[0] == o_aos[1]:  # Not simple; have to look through graph
        ortho_substitutents, meta_ring_atoms = sort_graph_search_atoms(
            mol, halide, meta
        )
        a, b = [set(mol.get_bonds_with_atom(meta_ring_atoms[f])) for f in range(2)]
        c, d = [set(mol.get_bonds_with_atom(ortho_substituents[f])) for f in range(2)]

        meta_symbol_sets = [set([g.symbol for g in f]) for f in meta]
        m_elms = [[element(f.symbol).atomic_number for f in g] for g in meta]
        m_avg = [sum(k) for k in m_elms]
        if m_avg[0] != m_avg[1]:  # One side is more branched than the other
            return ortho[m_avg.index(max(m_avg))]  # Side with most branching
        else:  # Ortho substitutions are identical - need to look at meta position
            m_en = [
                [element(f.symbol).electronegativity("allen") for f in g] for g in meta
            ]
            m_en_max = [max(k) for k in m_elms]
            if (
                m_en_max[0] != m_en_max[1]
            ):  # One meta position is more electronegative (e.g., N vs C)
                return ortho[m_en_max.index(max(m_en_max))]

    else:  # Ortho atoms are not both carbon
        sym = [f.symbol for f in ortho]
        if "N" in sym:  # Highest priority (arbitrary)
            return ortho[sym.index("N")]
        if "S" in sym:  # Second priority (arbitrary)
            return ortho[sym.index("S")]
        if "O" in sym:  # Third priority (arbitrary)
            return ortho[sym.index("O")]
        else:  # If not one of those heteroatoms, use the highest MW (e.g., weird Se heterocycle, etc.)
            return ortho[
                o_aos.index(max(o_aos))
            ]  # get ortho atom with largest ao if both sides differ


def sort_graph_search_atoms(mol: ml.Molecule, halide: ml.dtypes.Atom, meta: list):
    """
    Takes atoms 2 Manhattan steps from the ipso aryl atom and returns atoms
    which are an ortho substitutent separately from the meta ring atoms.
    This is done using distance from the reference atom.
    """
    halide_coords = mol.geom.get_coord(mol.get_atom_idx(halide))
    distances = [
        [
            np.linalg.norm(halide_coords - mol.geom.get_coord(mol.get_atom_idx(f)))
            for f in m
        ]
        for m in meta
    ]
    ## Building dist_arr to have 2 columns: distances, and atoms
    dist_arr = np.concatenate(
        np.array(
            [
                [p for p in zip(np.array(f)[0], np.array(f)[1])]
                for f in zip(distances, meta)
            ]
        ),
        axis=0,
    )
    sorted_dist = dist_arr[dist_arr[:, 0].argsort()]
    meta_ring = sorted_dist[-2:, 1].flatten().to_list()
    ortho_subst = sorted_dist[:2, 1].flatten().to_list()
    return ortho_subst, meta_ring


def get_ortho_meta_symbols(
    mol: ml.Molecule, ipso: ml.dtypes.Atom, halide: ml.dtypes.Atom
):
    """
    molli-based molecular graph walk through (hetero)aryl ring system to get symbols of
    ortho and meta ring atoms. These are also returned with branching of carbons.
    """
    ortho_atoms = [f for f in mol.get_connected_atoms(ipso) if f != halide]
    meta_atoms = []
    for orth in ortho_atoms:
        meta_candidates = [f for f in mol.get_connected_atoms(orth) if f != ipso]
        meta_atoms.append(meta_candidates)
    tertiary = []
    for meta in meta_atoms:
        tert = [
            [f for f in mol.get_connected_atoms(g) if f not in ortho_atoms]
            for g in meta
        ]
        tertiary.append(tert)
    return ortho_atoms, meta_atoms, tertiary


def get_left_reference(mol: Chem.rdchem.Mol, ipso_idx, br_idx):
    """
    return leftref
    """
    ipso_reference = mol.GetAtomWithIdx(ipso_idx)
    br_ref = mol.GetAtomWithIdx(br_idx)
    ortho_het, meta_het = _get_ortho_meta_symbols(mol, ipso_idx)
    if len(ortho_het) == 0:  # no ortho heteroatoms
        less_sub = get_less_substituted_ortho(mol, ipso_idx)
        if less_sub == None:  # ortho both the same
            if len(meta_het) == 0:  # no meta het, so using substitution
                less_meta_sub = get_less_substituted_meta(mol, ipso_idx)
                if less_meta_sub == None:
                    nbrs = [
                        f for f in ipso_reference.GetNeighbors() if f.GetIdx() != br_idx
                    ]
                    leftref = nbrs[0].GetIdx()  # arbitrary; symmetric
                elif (
                    less_meta_sub != None
                ):  # using less substituted meta atom for left reference
                    leftref = less_meta_sub
            elif len(meta_het) == 1:  # list of tuples (symbol, idx, atomic num)
                leftref = meta_het[0][1]
            elif len(meta_het) == 2:
                if (
                    meta_het[0][2] > meta_het[1][2]
                ):  # atomic number of first greater than atomic number of second
                    leftref = meta_het[0][1]
                elif meta_het[0][2] < meta_het[1][2]:
                    leftref = meta_het[1][1]
                elif meta_het[0][2] == meta_het[1][2]:
                    leftref = meta_het[0][1]  # arbitrary if they are the same
        elif less_sub != None:
            leftref = less_sub  # If one side is less substituted AND no heteroatoms were found
    elif len(ortho_het) == 1:
        leftref = ortho_het[0][1]  # heteroatom in ortho defines
    elif len(ortho_het) == 2:  # both ortho are het
        if (
            ortho_het[0][2] > ortho_het[1][2]
        ):  # atomic number of first greater than atomic number of second
            leftref = ortho_het[0][1]
        elif ortho_het[0][2] < ortho_het[1][2]:
            leftref = ortho_het[1][1]
        elif ortho_het[0][2] == ortho_het[1][2]:
            leftref = ortho_het[0][1]  # arbitrary if they are the same
    else:
        pass
    return leftref


def _get_ortho_meta_symbols(mol: Chem.rdchem.Mol, aryl_ref):
    """
    Finds out if and what heteroatoms are in the ortho-positions of aniline-type amines
    Returns list of ortho heteroatoms and then meta heteroatoms
    Form is: tuple (symbol,index,atomicnumber)
    Third value can be used to sort these by importance

    This should work for bromides!!!
    Uses ipso carbon as reference atom.
    """
    pt = Chem.GetPeriodicTable()
    ar_atm = get_aromatic_atoms(mol)  # all aryl atoms
    if aryl_ref not in ar_atm:
        return None  # This is weird; error here if this happens RDKIT BREAKS HERE - NON-ARYL HETEROARENES
    het_ar_atm = []  # list of tuples describing heteroarene heteroatoms, empty if none
    for atm in ar_atm:  # Loop over aromatic atoms to find heteroaromatic atoms
        symb = mol.GetAtomWithIdx(atm).GetSymbol()
        if symb != "C":
            het_ar_atm.append(tuple([symb, atm]))
    refatom = mol.GetAtomWithIdx(aryl_ref)
    nbrs = refatom.GetNeighbors()
    ortho_het = []
    meta_het = []
    for nbr in nbrs:  # This looks at ortho atoms
        test_value = tuple([nbr.GetSymbol(), nbr.GetIdx()])
        if test_value in het_ar_atm:
            ortho_het.append(
                tuple([f for f in test_value] + [pt.GetAtomicNumber(test_value[0])])
            )
        nbr2 = [f for f in nbr.GetNeighbors() if f not in nbrs]
        for nbrr in nbr2:  # This looks at one further atom out from ortho
            test_val_2 = tuple([nbrr.GetSymbol(), nbrr.GetIdx()])
            if test_val_2 in het_ar_atm:
                meta_het.append(
                    tuple([f for f in test_val_2] + [pt.GetAtomicNumber(test_val_2[0])])
                )
    return (
        ortho_het,
        meta_het,
    )  # Should find ortho heteroatoms and meta heteroatoms based on aromaticity, BUT will fail generally for fused systems and ones that RDKit doesn't recognize as aromatic


def get_aromatic_atoms(mol: Chem.rdchem.Mol):
    """
    Self explanatory: retrieves atom indices for aromatic atoms in mol

    NOTE: for multi-nuclear arenes (i.e. naphthalene, etc), this will give ALL aromatic atoms
    Therefore, the primary aromatic ring atoms will be included in this, but the list will not exclusively contain them.

    Some heterocycles (particularly pi-rich) are not parsed properly - this is an RDKit problem. It will be solved eventually.
    """

    q = rdqueries.IsAromaticQueryAtom()
    return [x.GetIdx() for x in mol.GetAtomsMatchingQuery(q)]


def get_less_substituted_ortho(mol: Chem.rdchem.Mol, atomidx):
    """
    Basic idea: retrieve aromatic atom in primary ring which is ortho to C-Br

    This is used to define left/right halves of molecule for RDF

    DEFINITELY will fail/not work generally for all molecules.
    """

    atomref = mol.GetAtomWithIdx(atomidx)
    nbrs = atomref.GetNeighbors()
    nbrs_ = [
        f
        for f in nbrs
        if f.GetSymbol() != "H"
        and f.GetSymbol() != "Br"
        and f.GetSymbol() != "N"
        and f.GetSymbol() != "Cl"
    ]  # No H, Br, Cl, or N
    nbrlist = [[k.GetSymbol() for k in f.GetNeighbors()] for f in nbrs_]
    cntlist = [f.count("H") for f in nbrlist]
    if cntlist.count(cntlist[0]) == len(cntlist):
        return None  # This means H count is same
    min_v = min(cntlist)
    min_indx = cntlist.index(min_v)
    lesssub = nbrs_[min_indx].GetIdx()
    return lesssub


def get_less_substituted_meta(mol: Chem.rdchem.Mol, ipsoidx):
    """
    Basic idea: retrieve aromatic atom in primary ring meta to C-Br which is less substituted

    This is used to define left vs right halves

    """

    atomref = mol.GetAtomWithIdx(ipsoidx)
    nbrs = atomref.GetNeighbors()
    nbrs_ = [f for f in nbrs if f.GetSymbol() != "H"]
    atomrings = mol.GetRingInfo().AtomRings()
    for ring in atomrings:
        if ipsoidx in ring:
            mainring = ring
    meta_ = [
        [
            k
            for k in f.GetNeighbors()
            if k.GetIdx() not in [p.GetIdx() for p in nbrs_]
            and k.GetSymbol != "H"
            and k.GetIdx() in mainring
            and k.GetIdx() != ipsoidx
        ]
        for f in nbrs_
    ]
    meta_ = [p for p in meta_ if len(p) != 0]
    meta_type_list = [
        [k.GetSymbol() for k in f[0].GetNeighbors()] for f in meta_
    ]  # List with one item; need index in nested inner list
    cntlist = [f.count("H") for f in meta_type_list]
    if cntlist.count(cntlist[0]) == len(cntlist):
        return None  # This means H count is same
    min_v = min(cntlist)
    min_indx = cntlist.index(min_v)
    lesssub = meta_[min_indx][0].GetIdx()
    return lesssub
