# Inarix

## Repository hierarchy
* jupyter: jupyter notebook
	* data_exploration,
	* transfer_learning,
* lib: python package code
    * database: paths and dataset management,
    * run_exp: tensor Flow code for training,
* data: original data base not present under github
* data_split: pre-processing original data base not present under github

## Data base: data_split
This database contains the original images:
* hist equalize on b&w,
* cut in subimages of 512 by 512 pixels,
* save with 3 identical channels (rgb),
* one repertory by graintype_device.

For the mix repository:
* subimages from each original image are saved in a repository with the name of original image,
* labels.csv is saved in the mix repository.

data_split is not under github for memory reasons:
* I give you this file database via a zip file,
* or generate it with data_exploration.ipynb and original data base "data".

