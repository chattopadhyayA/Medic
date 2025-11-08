# MEDIC
This is the code repository for MEDIC (**M**onitoring for **E**vent **D**ata **I**ntegrity and **C**onsistency) network. This network is aimed to be used for Data Quality Monitoring at particle accelarators

# File overview

**medica.py** is the utility file for **Data_create.ipynb** and **medic_network.ipynb**. 


**Data_create.ipynb**: From the JSON file with detector data (should be in a folder named Data) it creates the training dataset for MEDIC.

**medic_network.ipynb**: This notebook reads the prepared dataset and trains the MEDIC network as well as test it with differenct metrics.

**pp_jj_data_gen.ipynb**: This notebook reads the MADGRAPH+PYTHIA+DELPHES outputs and merges all the output in a single file with proper labelling.

**delphes_cards**: This folder contains all the parameter cards used to run Delphes for simulating different glitches