![midjourney_1](https://user-images.githubusercontent.com/46203561/189413673-c6e184e9-218c-417e-9402-461f284eabc2.png)
Image credit: Image was created using the official Midjourney server and is being used here under a creative commons license. [see https://www.midjourney.com/]
# Lucid_Somnambulist
Welcome to the public repository for code developed in collaboration between Hoffmann-La Roche (Process Chemistry & Catalysis in Basel, Switzerland) and the Scott E. Denmark Laboratory (University of Illinois at Urbana Champaign) to provide experimentalists rapid access to predicted reaction yields for Buchwald-Hartwig couplings. 

This work has a dependency on code developed within the Denmark Laboratories independent of this collaboration, and an alpha-test version of that code is contained herein. A new and improved version of that work can be found here: [molli 1.0]

# Philosophy
This package is designed with the following ecosystem in mind: 
1. Reactant in silico libraries
Large, dynamic libraries of reactant structures are intended to be stored as serialized collection objects (see molli package). When a user seeks predictions for new compounds, these can be input via a chemdraw structure (.cdxml), and appended to the existing database of reactants. Nucleophiles and electrophiles are treated separately. Code designed to calculate descriptors used in this work should be applicable to any nucleophile which has an N-H bond (which will react in a Buchwald-Hartwig coupling) and an electrophile with a single (hetero)aryl-C-Br bond. 

2. [Backend] Catalyst/Condition libraries
This space is designed to not be dynamic, but instead fixed as a dictionary of allowable conditions. These are intended to represent the relevant dimensions broadly, and thus should provide reasonable coverage of known reactivity trends. The chemical descriptors calculated here are essentially static, until one desires to add new data to an existing dataset. 

3. Reaction Space
Using unsupervised learning, each reactant type (nucleophile and electrophile) is clustered into groups of "like" reactants. These clusters of reactants then define a "reaction space" - an array of every possible combination of each cluster of each reactant. This space broadly represents the feature space of the in silico library of reactants currently described in the ecosystem. This space is independent of static catalyst/condition libraries. The goal of this representation is to visualize the applicability domain achievable with modeling in terms of reactant structures. This directly relates to the generalizability of models trained on the dataset. 

4. [Backend] Database and Dataset Partitions
This is where reaction data is stored. It is intended to be a dynamic, serializable structure which can be recalled rapidly when training models. Pandas dataframes are serialized in the hdf5 format (subject to change). 

5. [Backend] Modeling
Modeling takes two main forms: (1) establishing the coverage of reaction space by a current snapshot of a dataset by broadly searching model architectures and hyperparameter configurations, and (2) rapidly generating inferences with ensembles of re-trained, pre-validated model architectures and hyperparameter configurations. The first step is intended to be used by the developers/overseers of a reaction system, and the second is intended to be usable by any experimentalist. Periodically, new data acquired should be added to the dataset, and step (1) should be redone to create a new library of validated architectures and configurations for inference-modeling (2). 

6. [Backend] Descriptor Calculations for Ligands and Reactants
This process is automated for ease-of-use. Grimme's xTB python package is used to (1) generate initial 3D geometries from 2D graphs input by the user, (2) search for and screen conformers using Grimme's CREST package, and (3) scrape relevant atomic property data and serialize it for conformer ensembles of structures in a heirarchical dictionary (for reactants) or align and store 3D structures (for catalysts). Then, reactant descriptors are computed using 3D coordinates and atomic properties to (1) bin atoms into radial bins for each conformer of each reactant, and (2) tabulate the relevant descriptor from the relevant atomic property of said atom. Finally, values are averaged across conformers and serialized. Finally, catalyst descriptors are calculated by using the alignment to define a plane of forced symmetry, construct a grid around the catalyst conformer ensembles, then prune that grid to half of the 3D structures. Finally, grid-based descriptors developed in the Denmark Laboratories and described elsewhere are calculated and serialized for each catalyst. The descriptor calculation for reactants is more rapid, as these are intended to be generalizable parameters. 


Temporary sketch:
![Lucid Somnambulist_3_1_2023](https://user-images.githubusercontent.com/46203561/222112280-61d2485f-3b9b-43a3-9af2-7b3ae113cb98.svg)


