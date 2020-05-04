# DeepMIL

This is the repo where we will insert the code that we will be using for this project.

### Setup virtual environment
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

To install the necessary modules, run
```
pip install -r requirements.txt
```

(OPTIONAL) If new modules are being added to the project, update requirements.txt by, and push:
```
pip3 freeze > requirements.txt
```

### LUNGMASK
In order to run simple_ct_viewer.py as well as lungmask_predict.py one need to install the lungmask package.
```
pip install git+https://github.com/JoHof/lungmask
```

Note that lungmask_predict.py shouldnt be necessary to run, as these lungmasks should have been made available in the repo.

## Directions:
### Directory Setup:
1. Create data directories and subdirectories as below.
```
+-- {DATA_DIR}/
|   +-- Healthy/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- [...]
|   |   +-- file_n1.nii.gz
|   +-- Sick/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- [...]
|   |   +-- file_n2.nii.gz
|   +-- Healthy-Emphysema/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- [...]
|   |   +-- file_n3.nii.gz
|   +-- Sick-Emphysema/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- [...]
|   |   +-- file_n4.nii.gz
[...]
|   +-- any_other_class_subdirectory/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- [...]
|   |   +-- file_m.nii.gz
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
|   +--- class_label_1/
|   |    +--- CT1/
|   |    |    +--- CT1.h5
|   |    +--- CT2/
|   |    |    +--- CT2.h5
[...]
|   |    +--- CTn/
|   |    |    +--- CTn.h5
|   +--- class_label_2/
|   |    +--- CT1/
|   |    |    +--- CT1.h5
|   |    +--- CT2/
|   |    |    +--- CT2.h5
[...]
|   |    +--- CTm/
|   |    |    +--- CTm.h5
```

Note that each CT-folder only contains *one* .h5-file that is the corresponding preprocessed CT-data in user-defined form. This folder isn't really necessary, but there is some legacy code that expects this structure of the data (to be changed in the future). Note that all .h5-files has been given the name "1.h5", such that path is: "/path_to_CT_folder/CT_name/1.h5".



