import molli as ml
from attrs import define, field


@define
class _entry:
    """
    Abstract class for a database entry.

    Optional name, CAS, smiles can be passed.
    """

    name: str = field()
    CAS: str = field()
    smiles: str = field()
    role: str = field()
    features = field(factory=list)


@define
class substrate(_entry):
    ...


@define
class catalyst(_entry):
    ...


@define
class reagent(substrate):
    ...


@define
class reaction:
    """
    Database for storing reaction data.

    -Entry names should be reaction handles.
    -CAS and smiles should be for product
    -features should be fully concatenated for the reaction, so ready for partitioning

    """

    nuc: substrate = field()
    el: substrate = field()
    cat: catalyst = field()
    solv: reagent = field()
    base: reagent = field()
    handle: str = field(init=False)

    def __attrs_post_init__(self):
        self.handle = f"{self.nuc.name}_{self.el.name}_{self.cat.name}_{self.solv.name}_{self.base.name}"
