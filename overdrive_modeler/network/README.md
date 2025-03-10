# Network

Order of operations:
1. `dataset/create_symlinks.py` Plug in here the paths to your data folders and run this to create symlinks to the data.
2. `dataset/resample_db.py` If you have not done it yet, resample the database to 32kHz. This will be placed in the symlinked folder dataset_32000
3. `./create_dataframe.py` Create a dataframe with the paths to the files and the labels.