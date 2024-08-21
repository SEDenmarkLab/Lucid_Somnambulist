(C) 2023 N. Ian Rinehart and the Denmark laboratory

# Lucid_Somnambulist
Welcome to the public repository for code developed in collaboration between Hoffmann-La Roche (Process Chemistry & Catalysis in Basel, Switzerland) and the Scott E. Denmark Laboratory (University of Illinois at Urbana Champaign) to provide experimentalists rapid access to predicted reaction yields for Buchwald-Hartwig couplings. 

This repository is under development.

This work has a dependency on code developed within the Denmark Laboratories, which can be found here: https://github.com/SEDenmarkLab/molli_firstgen.git. 

# Description
 This repository contains a working project for a command-line tool that is meant for a non-expert. That tool still requires a basic understanding of how to install and run a command-line program on a Linux machine. The input of structures is accomplished using smiles strings (which can be obtained from most molecular drawing programs), cdxml (Chemdraw XML), or mol2 files. 
 Note that expectations of performance should be based on the discussion in the manuscript (10.26434/chemrxiv-2022-hspwv-v2). Specifically, requesting predictions for couplings far outside of the scope of the dataset will produce suboptimal performance. The best way to understand this is by looking at the supporting information, where a list of dataset couplings is provided along with a one-page summary for each coupling of the models’ prediction performance on each coupling.
  
# Requirements

  This repository can be used with a public container image, and for generating predictions, the hardware requirements are minimal (CPU only is fine, and modern laptops can handle this with at least 16GB RAM). 
  
  This code was developed on Ubuntu OS (used with 20.04 and 22.04), and has been tested some on wsl2. Further testing is planned, with the ultimate goal of releasing a container image with the necessary programs, but we recommend using Ubuntu 20.04 if possible. 
  For **training new models, specifically**, NVidia GPUs with a compute capability above 8.6 are recommended, and will require installation of the appropriate drivers. We strongly encourage a user to see tensorflow documentation to look for latest tested linux GPU builds to ensure that the CUDA and cuDNN versions match a tested build configuration. [https://www.tensorflow.org/install/source#tested_build_configurations]

 # Getting Started

 The simplest way to use this tool is to access the dockerhub container registry image ianrinehart/somn:1.0, and to run a docker container using this image. 

## Input file
 To request predictions, you must generate an input .csv file with the following format:

 ```bash
 user, nuc, el, nuc_name, el_name, nuc_idx, el_idx
 {arbitrary request(or) name}, {nucleophile SMILES string}, {electrophile SMILES string}, {nucleophile name}, {electrophile name},-,-
 ```
 This input file can contain any number of rows with the bracketed information for each individual coupling request. Note that the hyphen "-" indicates that somn should automatically detect a reaction site (use this for molecules with only one reaction site). **If multiple reaction sites are present in a reactant, then follow the next step and replace the "-" for that reactant idx in the input file with the atom index obtained.**

**Multiple reaction sites**
 For multiple reaction sites, a utility is built in for somn to allow you to generate a mol2 file which will capture the atom ordering that somn will use on its backend. Using that mol2 file, you can identify the desired reactive atom index and specify it in your request. For example, the nucleophile shown below (SMILES string O=C(C)NC1=CC(N)=CC(C(C)=O)=C1) has two nitrogen atoms which are not tertiary, but the primary nitrogen will react selectively. To ensure that somn provides predictions for the right reaction site, we can "tell" it to do so. 

 Example SMILES: O=C(C)NC1=CC(N)=CC(C(C)=O)=C1 
 
 Example structure:
 
 ![example_image](https://github.com/user-attachments/assets/39208e13-52b2-47b8-8187-f87c6c59e8b5)
 
 Run the following somn command:

 ```bash
 somn generate "O=C(C)NC1=CC(N)=CC(C(C)=O)=C1" {output_dir}/
 ```
 The command will run, and will generate a .mol2 file at the following path:

 ./somn_mol_buffer/{output_dir}/{filename}.mol2

 The 3d structure will look like this:

 ![3dstruc](https://github.com/user-attachments/assets/d6076f73-ca19-43dd-8632-259d34763c0e)

 The mol2 file atom block will look like this:

 ```
@<TRIPOS>ATOM
     1 O       1.0424     0.0638     0.3520 O.2        1 UNL1 0.0
     2 C       2.2109    -0.1823     0.0753 C.2        1 UNL1 0.0
     3 C       2.9969    -1.1900     0.8735 C.3        1 UNL1 0.0
     4 N       2.9528     0.3661    -0.9592 N.am       1 UNL1 0.0
     5 C       2.5567     1.4263    -1.8062 C.ar       1 UNL1 0.0
     6 C       1.4529     2.2391    -1.5359 C.ar       1 UNL1 0.0
     7 C       1.0834     3.2739    -2.3986 C.ar       1 UNL1 0.0
     8 N       0.0680     4.1642    -2.0303 N.pl3      1 UNL1 0.0
     9 C       1.8935     3.5470    -3.5006 C.ar       1 UNL1 0.0
    10 C       3.0103     2.7483    -3.8009 C.ar       1 UNL1 0.0
    11 C       3.7851     3.1000    -5.0357 C.2        1 UNL1 0.0
    12 C       4.9884     2.3019    -5.4667 C.3        1 UNL1 0.0
    13 O       3.4282     4.0639    -5.7154 O.2        1 UNL1 0.0
    14 C       3.3367     1.6893    -2.9423 C.ar       1 UNL1 0.0
    15 H       2.3899    -1.5406     1.7139 H          1 UNL1 0.0
    16 H       3.9026    -0.7242     1.2716 H          1 UNL1 0.0
    17 H       3.2584    -2.0449     0.2450 H          1 UNL1 0.0
    18 H       3.8400    -0.0758    -1.1648 H          1 UNL1 0.0
    19 H       0.8673     2.0987    -0.6317 H          1 UNL1 0.0
    20 H      -0.6340     3.7523    -1.4211 H          1 UNL1 0.0
    21 H      -0.3455     4.6667    -2.8094 H          1 UNL1 0.0
    22 H       1.6523     4.3969    -4.1364 H          1 UNL1 0.0
    23 H       5.2114     1.4461    -4.8349 H          1 UNL1 0.0
    24 H       5.8627     2.9611    -5.4683 H          1 UNL1 0.0
    25 H       4.8129     1.9288    -6.4813 H          1 UNL1 0.0
    26 H       4.1937     1.0561    -3.1381 H          1 UNL1 0.0
 ```

 Of the two nitrogen atoms, the "N.pl3"-type (planar, 3-coordinate) is the one that we are interested in making predictions for, not the "N.am" (amide-type). In this example, **the reaction site _index_ would be 7, corresponding to the 8th atom.**

Thus, the following prediction and input file would be used:

![example_coupling](https://github.com/user-attachments/assets/39a6112c-cb1f-4f91-8abd-6b441752530a)

 ```bash
 user,nuc,el,nuc_name,el_name,nuc_idx,el_idx
 example-request,O=C(C)NC1=CC(N)=CC(C(C)=O)=C1,BrC1=NC2=C(C=CC=C2)S1,n1,e1,7,-
 ```

cat {your local input file} | docker exec -i {your running container} sh -c 'cat > /tmp/somn_root/somn_scratch/IID-Models-2024/scratch/test_request.csv'

## Running the job

The input file described above must be in the location: 

/tmp/somn_root/somn_scratch/IID-Models-2024/scratch/ 

...and must have a filename that ends with "_request.csv".

To do this, first run the following commands using the docker CLI from your host machine:

```bash
docker exec {your container name} sh -c 'rm /tmp/somn_root/somn_scratch/IID-Models-2024/scratch/test_request.csv'
docker cp /path/to/your/input/file/{your input file name}_request.csv {your container name}:/tmp/somn_root/somn_scratch/IID-Models-2024/scratch/
```

These commands will remove the "dummy" input file that comes with the image and then add your desired input file into the container. Alternatively, you can just overwrite the "dummy" file with your desired contents:

```bash
cat {your local input file} | docker exec -i {your running container} sh -c 'cat > /tmp/somn_root/somn_scratch/IID-Models-2024/scratch/test_request.csv'
```

To run the prediction workflow, the following command will be used inside of the somn container:

```bash
docker exec -i {your running container} micromamba run somn predict last latest {name your prediction set}
```

After finishing, the outputs will be in the location:

/tmp/somn_root/somn_scratch/IID-Models-2024/outputs/{your prediction set name}/couplings/

A heatmap of the predictions will look like this:

![n1_e1_heatmap_average](https://github.com/user-attachments/assets/f7c99b4d-8b23-4c77-b8bb-c2ba678e6cab)
  
# Installation of package as a standalone

 This repository contains the code necessary to "restart" the active learning workflow described in the associated preprint article with this work: 10.26434/chemrxiv-2022-hspwv-v2. Doing so can improve predictive power for new types of couplings, as described in the manuscript. To do this, clone this repository and read the documentation on how to use the "workflows" section. Briefly, this process involves calculating descriptors for reactants using the somn CLI, and updating your local somn package to include those descriptors and any reaction data you have obtained with them. Then, a new set of modeling partitions must be created, and new models must be trained on those partitions in a new somn project. That new project can then be used to generate predictions with models which will have learned from any newly incorporated data. 

This code was developed with miniconda as a package manager, but has also been tested with micromamba. Examples here will be given for miniconda. An export of the environment is provided, and this can be installed with the following command:
  
 ```bash
 conda env create --name somn --file somn.yml
 ```

Then, activate the environment:

```bash
conda activate somn
```

 Next, install molli_firstgen using pip:

- Download molli_firstgen, branch 0.2.3 [https://github.com/SEDenmarkLab/molli_firstgen.git].
- Install using pip into the somn environment following installation instructions.

Finally, download and install this package by running the following command in the top-level directory (/path/to/package/Lucid_Somnambulist/):
 
 ```bash
 pip install .
 ```
 
 ## Test an import:
 
  ```bash
 conda activate somn
 
 python -c "import somn"
 ```
 
