import molli as ml
from molli.dtypes.molecule import Atom
import pandas as pd
import numpy as np
from pandas._libs.missing import NA
import json
from glob import glob
import os
from scipy.spatial.transform import Rotation as R

"""
This should be updated with the most recently tested version [4.11.2023]. The ligand library from RB
was a pretty robust test. 

"""

##hydrogen
H = "H"
##carbon
C_SP3 = "C.3"
C_SP2 = "C.2"
C_SP = "C.1"
C_AR = "C.ar"
C_CO2 = "C.co2"
##nitrogen
N_SP3 = "N.3"
N_SP2 = "N.2"
N_SP = "N.1"
N_PL3 = "N.pl3"  ##planar N
N_AR = "N.ar"  ##aryl N
N_AM = "N.am"  ##amide N
N_4 = "N.4"  ##NR4+
##oxygen
O_SP3 = "O.3"
O_SP2 = "O.2"
O_CARBOXY = "O.co2"  ##NOTE: SHOULDNT NEED TO IMPLEMENT
##sulfur
S_SP3 = "S.3"
S_SP2 = "S.2"
S_O = "S.o"  ##sulfoxide
S_O2 = "S.o2"  ##sulfoxide
##phosphorus
P_SP3 = "P.3"
##halogens
F = "F"
Cl = "Cl"
Br = "Br"
I = "I"
# misc
Si = "Si"
Na = "Na"
##vdW parameters for each atom type
a_i = "alpha-i"
N_i = "N-i"
A_i = "A_i"
G_i = "G_i"
rad = "rad"

vdw_dict = {}
# note: no carbonyl differentiation -Jeremy H., 10/28/2015
##hydrogen
vdw_dict[H] = {a_i: 0.250, N_i: 0.800, A_i: 4.200, G_i: 1.209, rad: 1.20}
##carbon
vdw_dict[C_SP3] = {a_i: 1.050, N_i: 2.490, A_i: 3.890, G_i: 1.282, rad: 1.70}
vdw_dict[C_SP2] = {a_i: 1.350, N_i: 2.490, A_i: 3.890, G_i: 1.282, rad: 1.70}
vdw_dict[C_SP] = {a_i: 1.300, N_i: 2.490, A_i: 3.890, G_i: 1.282, rad: 1.70}
vdw_dict[C_AR] = {a_i: 1.350, N_i: 2.490, A_i: 3.890, G_i: 1.282, rad: 1.70}
vdw_dict[C_CO2] = {a_i: 1.350, N_i: 2.490, A_i: 3.890, G_i: 1.282, rad: 1.70}
##nitrogen
vdw_dict[N_SP3] = {a_i: 1.150, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_SP2] = {a_i: 0.900, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_SP] = {a_i: 1.000, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_PL3] = {a_i: 0.850, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_AR] = {a_i: 0.850, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_AM] = {a_i: 1.000, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
vdw_dict[N_4] = {a_i: 1.000, N_i: 2.820, A_i: 3.890, G_i: 1.282, rad: 1.55}
# oxygen
vdw_dict[O_SP3] = {a_i: 0.700, N_i: 3.150, A_i: 3.890, G_i: 1.282, rad: 1.52}
vdw_dict[O_SP2] = {a_i: 0.650, N_i: 3.150, A_i: 3.890, G_i: 1.282, rad: 1.52}
vdw_dict[O_CARBOXY] = {a_i: 0.650, N_i: 3.150, A_i: 3.890, G_i: 1.282, rad: 1.52}
# sulfur
vdw_dict[S_SP3] = {a_i: 3.000, N_i: 4.800, A_i: 3.320, G_i: 1.345, rad: 1.80}
vdw_dict[S_SP2] = {a_i: 3.900, N_i: 4.800, A_i: 3.320, G_i: 1.345, rad: 1.80}
vdw_dict[S_O] = {a_i: 2.700, N_i: 4.800, A_i: 3.320, G_i: 1.345, rad: 1.80}
vdw_dict[S_O2] = {a_i: 2.100, N_i: 4.800, A_i: 3.320, G_i: 1.345, rad: 1.80}
# phosphorus
vdw_dict[P_SP3] = {a_i: 3.600, N_i: 4.500, A_i: 3.320, G_i: 1.345, rad: 1.80}
# halogens
vdw_dict[F] = {a_i: 0.350, N_i: 3.480, A_i: 3.890, G_i: 1.282, rad: 1.47}
vdw_dict[Cl] = {a_i: 2.300, N_i: 5.100, A_i: 3.320, G_i: 1.345, rad: 1.75}
vdw_dict[Br] = {a_i: 3.400, N_i: 6.000, A_i: 3.190, G_i: 1.359, rad: 1.85}
vdw_dict[I] = {a_i: 5.500, N_i: 6.950, A_i: 3.080, G_i: 1.404, rad: 1.98}
# misc
vdw_dict[Si] = {a_i: 4.500, N_i: 4.200, A_i: 3.320, G_i: 1.345, rad: 2.10}
vdw_dict[Na] = {rad: 2.27}


def get_ref_atoms(mol):
    """
    Takes molecule in, and returns atom instances for the appropraite atoms as a list

    This is P, Ni, C (attached to P as ipso carbon of biaryl)

    These come from a reference document
    """
    name_ = mol.name.rsplit("_", 1)[0]
    p, ni, c = align_dict[name_]
    atms = [mol.atoms[int(p) - 1], mol.atoms[int(ni) - 1], mol.atoms[int(c) - 1]]
    return atms


def get_closest_gpts(coords, grid, atom):
    """
    Retrieves mask with "True" for gridpoints touching query atom

    This is a boolean query array.

    This takes coordinates in given conformer geometry, the grid, and the atom
    object (to retrieve appropriate vdw radius)
    """
    gpts = grid.gridpoints
    vdw = vdw_dict[atom.atom_type]["rad"]
    dist = np.sqrt(np.sum((gpts - coords) ** 2, axis=1))
    mask = dist < vdw
    return mask


def get_closest_atom(gpt, conf_geom, mol):
    """
    Gridpoint to query, conformer cartesian geometry coords, mol object

    This gets the closest atom number to a given gridpoint

    Returns atom, atom index
    """
    atoms = mol.atoms
    dist = np.sqrt(np.sum((conf_geom - gpt) ** 2, axis=1))
    clos_idx = np.argmin(dist)
    closest_atom = atoms[clos_idx]
    return closest_atom, clos_idx, np.amin(dist)


def intersect_boolean(lst: list[np.array]):
    """
    Intersect boolean arrays for gridpoint masks across all atoms. This provides one boolean array which has "True"
    if any conformer "touched" a gridpoint. Has shape of grid, and can serve as instruction for building a single
    conformer's grid indicator field.

    All individual boolean arrays are formed into one 2D array. Then, the logical_or function is called on the reduced
    2D array (this is a way to pass more than 2 arguments to logical_or)

    Returns boolean array of shape of the grid
    """
    mtx = np.array(tuple(lst))
    intersection = np.logical_or.reduce(mtx)
    return intersection


def calculate_ASO(grid: ml.Grid, mol: ml.Molecule):
    """
    Calculator function for ASO. Takes molecule from collection and grid, then iterates over atoms and
    conformers. Does not iterate over entire grid; just atoms, and finds gridpoints each atom "touches."

    """
    atoms = mol.atoms
    confs = mol.conformers
    confs_list = []
    for conf in confs:
        boolean_list = []
        for idx, atom in enumerate(atoms):
            msk = get_closest_gpts(conf.coord[idx], grid, atom)
            boolean_list.append(msk)
        conf_boolean = intersect_boolean(boolean_list)
        conf_SIF = conf_boolean.astype(int)
        confs_list.append(conf_SIF)
    mol_array = np.array(tuple(confs_list), dtype=np.float32)
    aso_array = mol_array.mean(axis=0)
    return aso_array


def trim_nico3(mol: ml.Molecule):
    atoms = mol.atoms
    bonds = mol.bonds
    conformers = mol.conformers
    ni_atom = mol.get_atoms_by_symbol("Ni")[0]
    p_atom = mol.get_atoms_by_symbol("P")[0]
    ni_c_atoms = mol.get_connected_atoms(ni_atom)
    ni_c_atoms_ = [f for f in ni_c_atoms if "P" not in f.label]
    ni_o_atoms = []
    for atom in ni_c_atoms_:
        o_atom = mol.get_connected_atoms(atom)
        ni_o_atoms.extend(o_atom)
    ni_o_atoms_ = [f for f in ni_o_atoms if "Ni" not in f.symbol]
    remove_list = ni_c_atoms_ + ni_o_atoms_ + [ni_atom]
    mol.remove_atoms(*remove_list)

#### Leaving this in case someone wants to run this on their own set of mols ####
# if __name__ == "__main__":
#     mol2dir = r"catalyst_descriptors/mol2s/"
#     outdir = r"catalyst_descriptors/aligned_buchwald_ligands/"
#     align_dict = {}
#     os.makedirs(mol2dir, exist_ok=True)
#     os.makedirs(outdir, exist_ok=True)
#     with open("alignment_guide.txt", "r") as g:
#         lines = g.readlines()
#         for line in lines:
#             spl = line.strip().split(",")
#             align_dict[spl[0]] = spl[1:]
#     mols = []
#     mol2s = glob(mol2dir + "*.mol2")
#     for mol2 in mol2s:
#         mol = ml.Molecule.from_mol2(mol2)
#         mols.append(mol)
#     col = ml.Collection(name="alignment", molecules=mols)
#     refmol = col[0]
#     pr, nir, cr = get_ref_atoms(refmol)
#     refmol.geom.set_origin(refmol.get_atom_idx(pr))
#     refgeom = refmol.get_subgeom([pr, nir, cr])
#     stdout = open("align_stdout.txt", "w")
#     stdout.write("molname,alignment_rmsd\n")
#     for mol in col:
#         p, ni, c = get_ref_atoms(mol)
#         mol.geom.set_origin(mol.get_atom_idx(p))
#         sub_inst = mol.get_subgeom([p, ni, c])
#         rot, rmsd = R.align_vectors(sub_inst.coord, refgeom.coord)
#         rot_mat = rot.as_matrix()
#         mol.geom.transform(rot_mat)
#         output = [mol.name, str(float(rmsd))]
#         stdout.write(",".join(output) + "\n")
#         mol2str = mol.to_mol2()
#         with open(outdir + mol.name + ".mol2", "w") as k:
#             k.write(mol2str)
#     col.to_zip("catalyst_descriptors/aligned_buchwald_conformers.zip")
