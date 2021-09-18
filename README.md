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
* To preprocess the raw data, run:
```
python data_preprocess_v2.py
```
The processed data would be saved in ```./data/Task2_processed/Preprocessed_v2/```.
* (Optional) To augment training data, run:
```
python data_augmentation.py
```
* The division of training and validation data can be modified in ```./fold_division.py```.

## Training
This framework consists of three stages: screening, discrimination and segmentation. Each stage should be trained independently.
* To train screening network, run:
```
python train_screen.py -f FOLD -sn SCREEN_NAME
```
* To train discrimination network, run:
```
python train_discri.py -f FOLD -sn SCREEN_NAME -dn DISCRI_NAME
```
* To train segmentation network, run:
```
python train_unet.py -f FOLD -sn SCREEN_NAME -dn DISCRI_NAME -un UNET_NAME
```
* ```FOLD``` is the fold index(0, 1, 2, 3, 4 under 5-fold cross-validation) you want to train. ```SCREEN_NAME``` ```DISCRI_NAME``` ```UNET_NAME``` should be replaced by the names of model you want to use. The trained models would be saved in ```./models/MODEL_NAME/FOLD/```.

## Evaluation
Run:
```
python evaluate.py -f FOLD -sn SCREEN_NAME -dn DISCRI_NAME -un UNET_NAME
```
The final segmentation output and the preprocessed volume would be saved in ```./results/```.
