# CMB-Segmentation
![Framework](./figure/framework_overall.png)

## Setup
* Clone this repository
```
git clone https://github.com/zihaochen0319/CMB-Segmentation
cd CMB-Segmentation
```
* Check dependencies in requirements.txt
```
pip install -r requirements.txt
```

## Data Preparation
* Access the data from [VALDO 2021](https://valdo.grand-challenge.org/Description/) and save it in ```./data/Task2/```.
* To preprocess the raw data, run 
```
python data_preprocess_v2.py
```
The processed data would be saved in ```./data/Task2_processed/Preprocessed_v2/```.
* (Optional) To augment training data, run
```
python data_augmentation.py
```
* The division of training and validation data can be modified in ```./fold_division.py```.

## Training
This framework consists of three stages: screening, discrimination and segmentation. Each stage should be trained independently.
* To train screening network, run:
```
python train_screen.py -f FOLD
```
* To train discrimination network, run:
```
python train_discri.py -f FOLD
```
* To train segmentation network, run:
```
python train_unet.py -f FOLD
```
* The trained models are saved in ```./models

## Evaluation
* Specify the names of screening, discrimination and segmentation model 
