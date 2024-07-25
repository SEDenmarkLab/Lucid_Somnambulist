import molli as ml
import warnings

###         This is for active development of core functionality, classes used for debugging/checks/extra features



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


from keras.layers import Layer
from keras.activations import sigmoid, softmax, leaky_relu, relu
from keras.initializers import Constant
import tensorflow as tf


# class CancelOut(Layer):
#     """
#     CancelOut layer, keras implementation. This is not implemented, and is under development. 

#     #### THIS IS NOT ORIGINAL WORK, see reference: https://github.com/unnir/CancelOut
#     #### Layer designed for "top" of NN stack, right after input, and has constant weights which are trainable. Those can then be used
#     as feature importances later on.

#     """

#     def __init__(
#         self, activation="sigmoid", cancelout_loss=True, lambda_1=0.002, lambda_2=0.001
#     ):
#         super(CancelOut, self).__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.cancelout_loss = cancelout_loss

#         if activation == "sigmoid":
#             self.activation = sigmoid
#         if activation == "softmax":
#             self.activation = softmax
#         if activation == "leaky_relu":
#             self.activation = leaky_relu
#         if activation == "relu":
#             self.activation = relu

#     def build(self, input_shape):
#         self.w = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer=Constant(1),
#             trainable=True,
#         )

#     def call(self, inputs):
#         if self.cancelout_loss:
#             self.add_loss(
#                 self.lambda_1 * tf.norm(self.w, ord=1)
#                 + self.lambda_2 * tf.norm(self.w, ord=2)
#             )
#         return tf.math.multiply(inputs, self.activation(self.w))

#     def get_config(self):
#         return {"activation": self.activation}
