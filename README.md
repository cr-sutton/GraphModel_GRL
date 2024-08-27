# GraphModel_GRL
This is a repository for the code used in the random walk particle tracking model submitted to Geophysical Research Letters

There are three folders each with their respective python files and supporting documents. The workflow is as follows:
1. Download the .raw PET data from
2. Set the path to the .raw PET with other supporting files from the petToGraphFiles folder and run "petToGraph_GRL.py". This makes the max concentration data from the raw PET data. Figure 1 and 3 are made in this file. It's important to note that figure 3 uses .npy files which can be made in the particle tracking code and any combination of these files can be produced.
3. The clusteringFiles folder contains the HDBSCAN python code and the supporting files. This will create the different graph networks from the max concentration PET data which are written as .txt files and used in the model run files.
4. The modelRunFiles contains the python code for running the particle tracking model as well as the functions that are the main part of the model code. Included in this is the .txt files made in the last step as well as the processed effluent data from the radiation sensors as .csv files. 
