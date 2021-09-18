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
The final segmentation outputs and their corresponding preprocessed volume would be saved in ```./results/```.

## Acknowledgement
This repository is based on paper:

Q. Dou et al., "Automatic Detection of Cerebral Microbleeds From MR Images via 3D Convolutional Neural Networks," in IEEE Transactions on Medical Imaging, vol. 35, no. 5, pp. 1182-1195, May 2016, doi: 10.1109/TMI.2016.2528129. \[[paper](https://ieeexplore.ieee.org/abstract/document/7403984)\]


Part of the code in this repository are copied or revised from:

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

[Loss functions for image segmentation](https://github.com/JunMa11/SegLoss)

[Focal-Loss-Pytorch](https://github.com/yatengLG/Focal-Loss-Pytorch)
