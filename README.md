# DeepMIL

This is the repo where we will insert the code that we will be using for this project.

### Setup virtual environment
To install the necessary modules, run
```
pip install -r requirements.txt
```

It is recommended to work in a virtual enviroment, make one by:
```
virtualenv -ppython3 venv
```
and then activate/deactive it by
```
source venv/bin/activate
```
or
```
deactivate
```

(OPTIONAL) If new modules are being added to the project, update requirements.txt by:
```
pip3 freeze > requirements.txt
```

### LUNGMASK
In order to run simple_ct_viewer.py one need to install the lungmask package. The best way of doing this I found was doing:
```
git clone https://github.com/JoHof/lungmask.git
python lungmask/setup.py install
```

If any errors do occur, try running with python3 instead

Then you can remove the lungmask directory.

## Directions:
### Directory Setup:
1. Create data directories and subdirectories as below.
```
+-- {DATA_DIR}/
|   +-- positive/
|   |   +-- my_positive_file_1.nii.gz
|   |   +-- my_positive_file_2.nii.gz
|   |   +-- [...]
|   |   +-- my_positive_file_n.nii.gz
|   +-- negative/
|   |   +-- my_negative_file_1.nii.gz
|   |   +-- my_negative_file_2.nii.gz
|   |   +-- [...]
|   |   +-- my_negative_file_m.nii.gz
``` 

### Generate Data:
2. Create training/val/test data by processing CT and saving them in a suitable format, by running
```
python create_data.py path_to_datagen_config.ini
```

The file.ini is the config file with all relevant parameters for generating the dataset on your setup. Please change this before usage as params, e.g. paths, might be different on your machine. The one I use can be found in the python-folder.

### Train:
3. Train your model running
```
pythont train.py path_to_training_config.ini
```

The file.ini is the config file with all relevant parameters for training your model(s) on your setup. Please change this before usage as params, e.g. paths, might be different on your machine


### Workspace setup
```
+-- {DeepMIL folder}/
|   +-- python/
|   |   +-- create_data.py
|   |   +-- train.py
|   |   +-- [...]
|   |   +-- other_relevant_scripts_or_folders(.py or /)
|   +-- data/
|   |   +-- generated_data_binary_2d/
|   |   +-- generated_data_binary_3d/
|   |   +-- [...]
|   |   +-- some_training_datas_generated/
|   +-- output/
|   |   +-- models/
|   |   |   +-- actual_produced_trained_model_2d.h5
|   |   |   +-- dataset_produced_trained_model_3d.h5
|   |   |   +-- [...]
|   |   +-- datasets/
|   |   |   +-- correspoding_generated_dataset_for_trained_model_2d.h5
|   |   |   +-- corresponding_generated_dataset_for_trained_model__3d.h5
|   |   |   +-- [...]
|   |   +-- history/
|   |   |   +-- corresponding_generated_history_for_trained_model_2d.h5
|   |   |   +-- corresponding_generated_history_for_trained_model_3d.h5
|   |   |   +-- [...]
|   |   +-- configs/
|   |   |   +-- corresponding_generated_configurations_for_trained_model_2d.h5
|   |   |   +-- corresponding_generated_configurations_for_trained_model_3d.h5
|   |   |   +-- [...]
```


### Generated dataset setup
```
+--- {some created dataset}/
|   +--- class_label_CT1/
|   |    +--- slice_file_1.h5
|   |    +--- slice_file_2.h5
|   |    +--- [...]
|   |    +--- slice_file_m1.h5
|   +--- class_label_CT2/
|   |    +--- slice_file_1.h5
|   |    [...]
|   |    +--- slice_file_m2.h5
|   +--- [...]
|   +--- class_label_CTN/
|   +--- [...]
```



