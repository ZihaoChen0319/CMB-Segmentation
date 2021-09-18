# CMB-Segmentation
![Framework](./figure/framework_overall.png)

## Setup
* Clone this respository
```
git clone https://github.com/zihaochen0319/CMB-Segmentation
cd CMB-Segmentation
```
* Check dependencies in requirements.txt
```
pip install -r requirements.txt
```

## Data Preparation
* Access the data from [VALDO 2021](https://valdo.grand-challenge.org/Description/) and save it in ```./data/Task2/```
* Run ```python ./data_preprocess_v2.py``` to preprocess raw data, then the processed data would be saved in ```./data/Task2_processed/Preprocessed_v2/```
* (Optional) Run ```python ./data_augmentation.py/``` to augment data, which can be added to training set by setting the ```aug_num``` in configuration of ```train_xxx.py``` files
