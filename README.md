# MEDIC
This is the code repository for MEDIC (**M**onitoring for **E**vent **D**ata **I**ntegrity and **C**onsistency) network, based on the paper titled **[MEDIC: a network for monitoring data quality in collider experiments](https://arxiv.com)**. The main data for training MEDIC is generated through **[Delphes-BlindCalorimeter](https://github.com/chattopadhyayA/Delphes-BlindCalorimeter)**, a fork of **[Delphes](https://delphes.github.io)** that introduces a blind calorimeter feature, allowing users to define insensitive bins in the calorimeter.


# Model Architecture

MEDIC treats the three branches (Tracks, Towers, and MET) in three separate channels that encode detector inputs through linear projections, transformer encoders and attention pooling. This is followed by a series of convolutional layers, then a global average pooling and a fully connected classifier to return
probabilities.

![alt text](https://github.com/chattopadhyayA/Medic/blob/main/Images/MEDIC_CNN.png "Title")


# File overview

**medica.py** is the utility file for **Data_create.ipynb** and **medic_network.ipynb**. 


**Data_create.ipynb**: From the JSON file with detector data (should be in a folder named Data) it creates the training dataset for MEDIC.

**medic_network.ipynb**: This notebook reads the prepared dataset and trains the MEDIC network as well as test it with differenct metrics.

**pp_jj_data_gen.ipynb**: This notebook reads the MADGRAPH+PYTHIA+DELPHES outputs and merges all the output in a single file with proper labelling.

**delphes_cards**: This folder contains all the parameter cards used to run Delphes for simulating different glitches

**Images**: A foldes to contain images to display in the main page.


## License and Citation

![Creative Commons License](https://mirrors.creativecommons.org/presskit/buttons/80x15/png/by.png)

This work is licensed under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

We encourage use of these codes \& data in derivative works. If you use the material provided here, please cite the paper using the reference:

```
@article{to be updated soon}
```