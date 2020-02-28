# DeepMIL

This is the repo where we will insert the code that we will be using for this project.

If new modules are being added to the project, update requirements.txt doing:
```
pip3 freeze > requirements.txt
```

To install the necessary modules, run
```
pip install -r requirements.txt
```

It is recommended to work in a virtual enviroment, make one by:
```
virtualenv -ppython3 venv
```
and then activate/deactive it doing
```
source venv/bin/activate
```
or
```
deactivate
```

### LUNGMASK
In order to run simple_ct_viewer.py one need to install the lungmask package. The best way of doing this I found was doing:
```
git clone https://github.com/JoHof/lungmask.git
python lungmask/setup.py install
```

If any errors do occur, try running with python3 instead

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
python create_data.py
```

### Train:
3. Train your model running
```
pythont train.py
```

NOTE: For both 2. and 3., paths are hardcoded for my (André's) setup. Hence, one might need to change some paths (to be handled better in the future)


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
|   |   |   +-- corresponding_genereated_history_for_trained_model_3d.h5
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



